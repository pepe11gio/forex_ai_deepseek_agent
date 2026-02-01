# src/analysis/signal_generator.py
from __future__ import annotations

import os 
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import ProjectConfig
from src.features.normalizer import FeatureScaler
from src.features.windowing import build_multiscale_windows
from src.modeling.trainer import load_feature_columns, load_trained_model
from src.evaluation.threshold_search import decide_actions

from src.modeling.ensemble import build_ensemble_spec, predict_proba_ensemble


@dataclass(frozen=True)
class TradeSignal:
    time: str
    action: int
    side: str
    p_long: float
    p_short: float
    p_min: float
    tp_net_pips: float
    sl_net_pips: float
    cost_pips: float
    rr_t: float
    horizon_t: int
    expectancy_long_pips: float
    expectancy_short_pips: float
    model_used: str  # "single:<run>" or "ensemble:<runs>"


def _load_scaler(cfg: ProjectConfig, run_name_for_scaler: str) -> FeatureScaler:
    export_dir = f"{cfg.model_dir}/exports/{run_name_for_scaler}"
    scaler_path = f"{export_dir}/feature_scaler.json"

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler non trovato: {scaler_path}")

    # ✅ API pulita: metodo d'istanza esplicito
    return FeatureScaler(cfg.derived.scaler_type).load_into_self(scaler_path)


def generate_signal(
    cfg: ProjectConfig,
    df_full: pd.DataFrame,
    run_name: str,
    p_min: float,
    t: Optional[int] = None,
) -> TradeSignal:
    if t is None:
        t = len(df_full) - 1

    # Feature columns: per consistenza, prendiamo quelli del run_name passato
    feature_cols = load_feature_columns(f"{cfg.model_dir}/exports/{run_name}/feature_columns.json")

    # Scaler: per semplicità usiamo quello di run_name (single) o del primo run ensemble.
    if cfg.ensemble.enabled:
        spec = build_ensemble_spec(cfg)
        scaler_run = spec.run_names[0]
    else:
        spec = None
        scaler_run = run_name

    scaler = _load_scaler(cfg, scaler_run)

    # Scale
    df_scaled = scaler.transform(df_full)

    # Robustezza NaN sulle ultime max_window righe
    wmax = int(max(cfg.windows.windows)) if cfg.windows.windows else 0
    tail = df_scaled.iloc[max(0, t - wmax + 1) : t + 1]
    if tail[feature_cols].isna().any().any():
        tm = str(df_full["time"].iat[t]) if "time" in df_full.columns else ""
        return TradeSignal(
            time=tm,
            action=0,
            side="NONE",
            p_long=0.0,
            p_short=0.0,
            p_min=float(p_min),
            tp_net_pips=float("nan"),
            sl_net_pips=float("nan"),
            cost_pips=float(cfg.market.spread_pips),
            rr_t=float("nan"),
            horizon_t=int(cfg.market.horizon_bars_fixed),
            expectancy_long_pips=float("nan"),
            expectancy_short_pips=float("nan"),
            model_used="nan_features",
        )

    # Windowing
    ds = build_multiscale_windows(df_scaled, feature_cols, cfg)

    # trova sample che corrisponde a t
    pos = np.where(ds.t_index == t)[0]
    if pos.size == 0:
        tm = str(df_full["time"].iat[t]) if "time" in df_full.columns else ""
        return TradeSignal(
            time=tm,
            action=0,
            side="NONE",
            p_long=0.0,
            p_short=0.0,
            p_min=float(p_min),
            tp_net_pips=float("nan"),
            sl_net_pips=float("nan"),
            cost_pips=float(cfg.market.spread_pips),
            rr_t=float("nan"),
            horizon_t=int(cfg.market.horizon_bars_fixed),
            expectancy_long_pips=float("nan"),
            expectancy_short_pips=float("nan"),
            model_used="insufficient_history",
        )

    j = int(pos[-1])

    X_list = [x[j : j + 1] for x in ds.X_list]  # batch=1

    # Predict
    if cfg.ensemble.enabled:
        assert spec is not None
        p1 = predict_proba_ensemble(cfg, X_list, spec=spec)[0]  # (2,)
        model_used = "ensemble:" + ",".join(spec.run_names)
    else:
        model = load_trained_model(cfg, run_name=run_name)
        p1 = model.predict(X_list, verbose=0).astype(np.float32)[0]
        model_used = "single:" + run_name

    # Decision su singola barra
    df_one = df_full.iloc[[t]].copy()
    if "valid_label" not in df_one.columns:
        df_one["valid_label"] = True
    if "cost_pips" not in df_one.columns:
        df_one["cost_pips"] = float(cfg.market.spread_pips)

    action = int(decide_actions(df_one, p1.reshape(1, 2), cfg, p_min=float(p_min))[0])
    side = "LONG" if action == 1 else "SHORT" if action == -1 else "NONE"

    tp = float(pd.to_numeric(df_one["tp_net_pips"], errors="coerce").iat[0]) if "tp_net_pips" in df_one.columns else float("nan")
    sl = float(pd.to_numeric(df_one["sl_net_pips"], errors="coerce").iat[0]) if "sl_net_pips" in df_one.columns else float("nan")
    cost = float(pd.to_numeric(df_one["cost_pips"], errors="coerce").iat[0])

    p_long = float(p1[0])
    p_short = float(p1[1])

    E_long = p_long * (tp - cost) - (1.0 - p_long) * (sl + cost) if np.isfinite(tp) and np.isfinite(sl) else float("nan")
    E_short = p_short * (tp - cost) - (1.0 - p_short) * (sl + cost) if np.isfinite(tp) and np.isfinite(sl) else float("nan")

    rr_t = float(pd.to_numeric(df_one["rr_t"], errors="coerce").iat[0]) if "rr_t" in df_one.columns else float(cfg.market.rr_fixed)
    h_t = int(pd.to_numeric(df_one["horizon_t"], errors="coerce").iat[0]) if "horizon_t" in df_one.columns else int(cfg.market.horizon_bars_fixed)

    tm = str(df_full["time"].iat[t]) if "time" in df_full.columns else ""
    return TradeSignal(
        time=tm,
        action=action,
        side=side,
        p_long=p_long,
        p_short=p_short,
        p_min=float(p_min),
        tp_net_pips=tp,
        sl_net_pips=sl,
        cost_pips=cost,
        rr_t=rr_t,
        horizon_t=h_t,
        expectancy_long_pips=float(E_long),
        expectancy_short_pips=float(E_short),
        model_used=model_used,
    )
