# src/modeling/ensemble.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.config import ProjectConfig
from src.modeling.trainer import load_trained_model


@dataclass(frozen=True)
class EnsembleSpec:
    run_names: List[str]
    weights: np.ndarray  # shape (K,)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_summary(cfg: ProjectConfig) -> dict:
    path = cfg.ensemble.summary_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"walk_forward_summary.json non trovato: {path}. "
            f"Esegui train_wf oppure imposta ensemble.run_names."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_weight(score: float, power: float) -> float:
    # clamp a 0 per evitare pesi negativi
    s = max(0.0, float(score))
    return float(s ** float(power))


def build_ensemble_spec(cfg: ProjectConfig) -> EnsembleSpec:
    """
    Decide quali run usare e con quali pesi.
    """
    ec = cfg.ensemble

    # 1) se l'utente ha specificato run_names, usa quelli
    if ec.run_names:
        run_names = list(ec.run_names)
        weights = np.ones(len(run_names), dtype=float)
        weights /= weights.sum() if weights.sum() > 0 else 1.0
        return EnsembleSpec(run_names=run_names, weights=weights)

    # 2) altrimenti auto-selezione dal summary
    if not ec.use_summary_auto_select:
        raise ValueError("Ensemble: run_names vuoto e use_summary_auto_select=False. Non so cosa ensemble-are.")

    summary = _load_summary(cfg)
    folds = summary.get("folds", [])
    if not folds:
        raise ValueError("Ensemble: summary non contiene folds.")

    # filtra fold con min trades in validation
    candidates = []
    for f in folds:
        val = f.get("best_threshold_val", {})
        trades = int(val.get("trades", 0))
        if trades < int(ec.min_val_trades):
            continue

        run_name = f.get("run_name", "")
        if not run_name:
            continue

        # score in base al weight_mode
        if ec.weight_mode == "val_profit_factor":
            score = _safe_float(val.get("profit_factor", 0.0), 0.0)
        elif ec.weight_mode == "val_expectancy":
            score = _safe_float(val.get("expectancy_net_pips", 0.0), 0.0)
        else:
            score = 1.0  # equal

        candidates.append((run_name, score))

    if not candidates:
        raise ValueError(
            "Ensemble: nessun fold candidato (min_val_trades troppo alto o summary vuoto). "
            "Abbassa ensemble.min_val_trades oppure imposta ensemble.run_names."
        )

    # ordina per score desc e prendi top_k
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[: int(ec.top_k)]

    run_names = [r for r, _ in candidates]

    if ec.weight_mode == "equal":
        weights = np.ones(len(run_names), dtype=float)
    else:
        w = np.array([_compute_weight(s, ec.weight_power) for _, s in candidates], dtype=float)
        # se tutti zero, fallback equal
        weights = w if w.sum() > 0 else np.ones(len(run_names), dtype=float)

    weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

    return EnsembleSpec(run_names=run_names, weights=weights)


def _combine_probs(P: np.ndarray, method: str, trimmed_frac: float) -> np.ndarray:
    """
    P shape: (K, N, 2)
    return shape: (N, 2)
    """
    method = (method or "mean").lower()

    if method == "mean":
        return P.mean(axis=0)

    if method == "median":
        return np.median(P, axis=0)

    if method == "trimmed_mean":
        tfc = float(trimmed_frac)
        tfc = min(max(tfc, 0.0), 0.49)
        K = P.shape[0]
        if K < 3 or tfc <= 0:
            return P.mean(axis=0)

        lo = int(np.floor(K * tfc))
        hi = int(np.ceil(K * (1.0 - tfc)))
        # ordina lungo asse modelli (K)
        Ps = np.sort(P, axis=0)
        Ps = Ps[lo:hi, :, :]
        return Ps.mean(axis=0)

    raise ValueError(f"Ensemble method non supportato: {method}")


def predict_proba_ensemble(
    cfg: ProjectConfig,
    X_list: List[np.ndarray],
    spec: EnsembleSpec,
) -> np.ndarray:
    """
    Carica i modelli e predice proba (N,2) combinando i fold.
    """
    ec = cfg.ensemble
    run_names = spec.run_names
    weights = spec.weights  # (K,)

    probs = []
    for rn in run_names:
        model = load_trained_model(cfg, run_name=rn)
        p = model.predict(X_list, verbose=0).astype(np.float32)  # (N,2)
        probs.append(p)

    P = np.stack(probs, axis=0)  # (K,N,2)

    # se mean pesato: applica pesi; altrimenti usa _combine_probs
    if (ec.method or "mean").lower() == "mean" and weights is not None and len(weights) == P.shape[0]:
        w = weights.reshape(-1, 1, 1).astype(np.float32)
        return (P * w).sum(axis=0)

    return _combine_probs(P, ec.method, ec.trimmed_frac)

def build_ensemble_spec_from_scores(
    run_names: List[str],
    scores: List[float],
    weight_mode: str = "val_expectancy",
    weight_power: float = 1.0,
) -> EnsembleSpec:
    """
    Crea una EnsembleSpec da run_names + scores (es. expectancy o PF).
    Se weight_mode == "equal" => pesi uguali.
    Altrimenti i pesi sono max(0, score)^power normalizzati.
    """
    if len(run_names) == 0:
        raise ValueError("run_names vuoto")

    if weight_mode == "equal":
        w = np.ones(len(run_names), dtype=float)
        w /= w.sum()
        return EnsembleSpec(run_names=run_names, weights=w)

    if len(scores) != len(run_names):
        raise ValueError("scores e run_names devono avere la stessa lunghezza")

    w = np.array([max(0.0, float(s)) ** float(weight_power) for s in scores], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(run_names), dtype=float)
    w /= w.sum()
    return EnsembleSpec(run_names=run_names, weights=w)