# src/main.py
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from typing import List

import numpy as np
import pandas as pd

from src.config import default_config, ProjectConfig
from src.io.data_loader import load_mt5_csvs
from src.features.derived_features import compute_derived_features, get_model_feature_columns, apply_nan_policy
from src.labeling.tp_sl_labeler import add_tp_sl_labels
from src.features.normalizer import FeatureScaler
from src.features.windowing import build_multiscale_windows
from src.modeling.tf_model import build_multiscale_model, compile_model
from src.modeling.trainer import train_model, load_trained_model
from src.evaluation.walk_forward import generate_walk_forward_splits, describe_split
from src.evaluation.threshold_search import search_best_threshold
from src.evaluation.backtester import run_backtest
from src.analysis.signal_generator import generate_signal
from src.modeling.ensemble import predict_proba_ensemble, build_ensemble_spec_from_scores, EnsembleSpec



def ensure_dirs(cfg: ProjectConfig) -> None:
    os.makedirs(cfg.data_raw_dir, exist_ok=True)
    os.makedirs(cfg.data_processed_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.reports_dir, exist_ok=True)

def _predict_probs(model, X_list: List[np.ndarray]) -> np.ndarray:
    p = model.predict(X_list, verbose=0)
    return p.astype(np.float32)

def _drop_invalid_labels(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    # elimina le righe con label invalide (tipicamente ultime H e dove manca forward)
    if "valid_label" in df.columns:
        df = df[df["valid_label"].astype(bool)].copy()
    # inoltre droppa NaN nelle colonne target
    if "y_long" in df.columns and "y_short" in df.columns:
        df = df[df["y_long"].notna() & df["y_short"].notna()].copy()
    return df.reset_index(drop=True)

def _score_from_fold_report(fold_report: dict, weight_mode: str) -> float:
    val = fold_report.get("best_threshold_val", {}) or {}
    if weight_mode == "val_profit_factor":
        return float(val.get("profit_factor", 0.0))
    if weight_mode == "val_expectancy":
        return float(val.get("expectancy_net_pips", 0.0))
    return 1.0

def _build_past_only_ensemble_spec(
    fold_reports: list,
    current_fold: int,
    top_k: int,
    min_val_trades: int,
    weight_mode: str,
    weight_power: float,
) -> EnsembleSpec | None:
    """
    Usa solo fold <= current_fold già addestrati.
    Filtra per min_val_trades.
    Se non ci sono abbastanza modelli, ritorna None.
    """
    candidates = []
    for fr in fold_reports:
        if int(fr.get("fold", 10**9)) > int(current_fold):
            continue
        val = fr.get("best_threshold_val", {}) or {}
        if int(val.get("trades", 0)) < int(min_val_trades):
            continue
        rn = fr.get("run_name", "")
        if not rn:
            continue
        score = _score_from_fold_report(fr, weight_mode)
        candidates.append((rn, score))

    if len(candidates) < 2:
        return None  # con 1 modello non è ensemble

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[: int(top_k)]

    run_names = [r for r, _ in candidates]
    scores = [s for _, s in candidates]

    return build_ensemble_spec_from_scores(
        run_names=run_names,
        scores=scores,
        weight_mode=weight_mode,
        weight_power=weight_power,
    )

def _safe_get(d: dict, path: list, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_ensemble_comparison_rows(fold_reports: list) -> list:
    rows = []
    for fr in fold_reports:
        fold = int(fr.get("fold", -1))
        run_name = str(fr.get("run_name", ""))

        # SINGLE
        s_val_pmin = _safe_get(fr, ["best_threshold_val", "p_min"], None)
        s_val_trades = _safe_get(fr, ["best_threshold_val", "trades"], None)
        s_val_exp = _safe_get(fr, ["best_threshold_val", "expectancy_net_pips"], None)
        s_val_pf = _safe_get(fr, ["best_threshold_val", "profit_factor"], None)

        s_test_trades = _safe_get(fr, ["test_metrics", "trades"], None)
        s_test_wr = _safe_get(fr, ["test_metrics", "winrate"], None)
        s_test_exp = _safe_get(fr, ["test_metrics", "expectancy_net_pips"], None)
        s_test_pf = _safe_get(fr, ["test_metrics", "profit_factor"], None)
        s_test_mdd = _safe_get(fr, ["test_metrics", "max_drawdown_pips"], None)

        # ENSEMBLE (optional)
        ens = fr.get("ensemble", None)
        if isinstance(ens, dict):
            e_runs = ens.get("ensemble_runs", [])
            e_weights = ens.get("ensemble_weights", [])

            e_val_pmin = _safe_get(ens, ["best_threshold_val_ensemble", "p_min"], None)
            e_val_trades = _safe_get(ens, ["best_threshold_val_ensemble", "trades"], None)
            e_val_exp = _safe_get(ens, ["best_threshold_val_ensemble", "expectancy_net_pips"], None)
            e_val_pf = _safe_get(ens, ["best_threshold_val_ensemble", "profit_factor"], None)

            e_test_trades = _safe_get(ens, ["test_metrics_ensemble", "trades"], None)
            e_test_wr = _safe_get(ens, ["test_metrics_ensemble", "winrate"], None)
            e_test_exp = _safe_get(ens, ["test_metrics_ensemble", "expectancy_net_pips"], None)
            e_test_pf = _safe_get(ens, ["test_metrics_ensemble", "profit_factor"], None)
            e_test_mdd = _safe_get(ens, ["test_metrics_ensemble", "max_drawdown_pips"], None)
        else:
            e_runs = []
            e_weights = []
            e_val_pmin = e_val_trades = e_val_exp = e_val_pf = None
            e_test_trades = e_test_wr = e_test_exp = e_test_pf = e_test_mdd = None

        # deltas (ensemble - single)
        def dsub(a, b):
            if a is None or b is None:
                return None
            try:
                return float(a) - float(b)
            except Exception:
                return None

        row = {
            "fold": fold,
            "run_name": run_name,

            # single val
            "single_val_p_min": s_val_pmin,
            "single_val_trades": s_val_trades,
            "single_val_expectancy": s_val_exp,
            "single_val_profit_factor": s_val_pf,

            # single test
            "single_test_trades": s_test_trades,
            "single_test_winrate": s_test_wr,
            "single_test_expectancy": s_test_exp,
            "single_test_profit_factor": s_test_pf,
            "single_test_max_drawdown_pips": s_test_mdd,

            # ensemble meta
            "ensemble_enabled": isinstance(ens, dict),
            "ensemble_runs": ",".join([str(x) for x in e_runs]) if e_runs else "",
            "ensemble_weights": ",".join([f"{float(x):.6f}" for x in e_weights]) if e_weights else "",

            # ensemble val
            "ensemble_val_p_min": e_val_pmin,
            "ensemble_val_trades": e_val_trades,
            "ensemble_val_expectancy": e_val_exp,
            "ensemble_val_profit_factor": e_val_pf,

            # ensemble test
            "ensemble_test_trades": e_test_trades,
            "ensemble_test_winrate": e_test_wr,
            "ensemble_test_expectancy": e_test_exp,
            "ensemble_test_profit_factor": e_test_pf,
            "ensemble_test_max_drawdown_pips": e_test_mdd,

            # deltas test
            "delta_test_expectancy": dsub(e_test_exp, s_test_exp),
            "delta_test_profit_factor": dsub(e_test_pf, s_test_pf),
            "delta_test_max_drawdown_pips": dsub(e_test_mdd, s_test_mdd),
            "delta_test_trades": dsub(e_test_trades, s_test_trades),
            "delta_test_winrate": dsub(e_test_wr, s_test_wr),
        }
        rows.append(row)

    return rows


def write_ensemble_comparison_reports(cfg: ProjectConfig, fold_reports: list) -> None:
    """
    Genera:
      - reports/ensemble_comparison.csv
      - reports/ensemble_comparison.json
    usando fold_reports del walk-forward.
    """
    rows = _build_ensemble_comparison_rows(fold_reports)

    os.makedirs(cfg.reports_dir, exist_ok=True)

    csv_path = os.path.join(cfg.reports_dir, "ensemble_comparison.csv")
    json_path = os.path.join(cfg.reports_dir, "ensemble_comparison.json")

    dfc = pd.DataFrame(rows)

    # ordinamento colonne per leggibilità
    preferred = [
        "fold", "run_name",
        "ensemble_enabled", "ensemble_runs", "ensemble_weights",
        "single_val_p_min", "ensemble_val_p_min",
        "single_test_trades", "ensemble_test_trades", "delta_test_trades",
        "single_test_expectancy", "ensemble_test_expectancy", "delta_test_expectancy",
        "single_test_profit_factor", "ensemble_test_profit_factor", "delta_test_profit_factor",
        "single_test_max_drawdown_pips", "ensemble_test_max_drawdown_pips", "delta_test_max_drawdown_pips",
        "single_test_winrate", "ensemble_test_winrate", "delta_test_winrate",
        "single_val_trades", "ensemble_val_trades",
        "single_val_expectancy", "ensemble_val_expectancy",
        "single_val_profit_factor", "ensemble_val_profit_factor",
    ]
    cols = [c for c in preferred if c in dfc.columns] + [c for c in dfc.columns if c not in preferred]
    dfc = dfc[cols]

    dfc.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(f"[REPORT] Saved ensemble comparison CSV: {csv_path}")
    print(f"[REPORT] Saved ensemble comparison JSON: {json_path}")


def run_train_walk_forward(cfg: ProjectConfig, args) -> None:
    ensure_dirs(cfg)

    # 1) Load all CSVs
    df = load_mt5_csvs(cfg)

    # 2) Derived features (includes S/R + breakout features if enabled)
    df = compute_derived_features(df, cfg)

    # 3) Label TP/SL (dynamic rr/h + costs)
    df = add_tp_sl_labels(df, cfg)

    # 4) NaN policy (features)
    df = apply_nan_policy(df, cfg)

    # 5) Walk-forward splits (with embargo)
    splits = generate_walk_forward_splits(df, cfg)
    if cfg.verbose:
        for s in splits:
            print(describe_split(df, s))

    fold_reports = []
    best_fold = None
    best_score = -1e18

    if hasattr(args, "regime") and args.regime:
        rd = dict(cfg.regime.__dict__)
        rd["enabled"] = True
        if getattr(args, "regime_mode", None):
            rd["mode"] = str(args.regime_mode)
        cfg = cfg.__class__(**{**cfg.__dict__, "regime": cfg.regime.__class__(**rd)})

    print("\n========== CURRENT CONFIG ==========")
    try:
        print(json.dumps(asdict(cfg), indent=2, default=str))
    except Exception:
        print(cfg)
    print("====================================\n")

    for s in splits:
        fold = s.fold
        run_name = f"wf_fold_{fold}"
        print(f"\n=== WALK-FORWARD FOLD {fold} ({run_name}) ===")

        df_train = df.iloc[s.train_start:s.train_end].reset_index(drop=True)
        df_val = df.iloc[s.val_start:s.val_end].reset_index(drop=True)
        df_test = df.iloc[s.test_start:s.test_end].reset_index(drop=True)

        # drop invalid labels per training/val/test
        df_train = _drop_invalid_labels(df_train, cfg)
        df_val = _drop_invalid_labels(df_val, cfg)
        df_test = _drop_invalid_labels(df_test, cfg)

        feature_cols = get_model_feature_columns(df_train, cfg)

        # Fit scaler on train only
        scaler = FeatureScaler(cfg.derived.scaler_type).fit(df_train, feature_cols)
        df_train_s = scaler.transform(df_train)
        df_val_s = scaler.transform(df_val)
        df_test_s = scaler.transform(df_test)

        # Windowing
        ds_train = build_multiscale_windows(df_train_s, feature_cols, cfg)
        ds_val = build_multiscale_windows(df_val_s, feature_cols, cfg)
        ds_test = build_multiscale_windows(df_test_s, feature_cols, cfg)

        # align dfs to windowed samples
        df_val_aligned = df_val.iloc[ds_val.t_index].reset_index(drop=True)
        df_test_aligned = df_test.iloc[ds_test.t_index].reset_index(drop=True)

        # Build/compile model
        n_features = len(feature_cols)
        model = build_multiscale_model(cfg, n_features=n_features)
        model = compile_model(model, cfg)

        artifacts = train_model(
            cfg=cfg,
            model=model,
            scaler=scaler,
            feature_columns=feature_cols,
            X_train_list=ds_train.X_list,
            y_train=ds_train.y,
            X_val_list=ds_val.X_list,
            y_val=ds_val.y,
            run_name=run_name,
        )

        trained_model = load_trained_model(cfg, run_name=run_name)

        # Predict on validation
        p_val = _predict_probs(trained_model, ds_val.X_list)

        # Threshold search on aligned validation df
        best_thr, thr_grid = search_best_threshold(df_val_aligned, p_val, cfg)

        # Predict on test & backtest using aligned test df
        p_test = _predict_probs(trained_model, ds_test.X_list)
        metrics_test, trades_test = run_backtest(df_test_aligned, p_test, cfg, p_min=best_thr.p_min)

        ensemble_report = None

        if args.ensemble_test:
            # costruisci spec (past-only): usa fold già visti (incluso quello corrente)
            spec = _build_past_only_ensemble_spec(
                fold_reports=fold_reports,   # include fold già completati; ATTENZIONE: aggiungi fold_report dopo questo blocco
                current_fold=fold,
                top_k=int(args.ens_topk),
                min_val_trades=int(args.ens_min_val_trades),
                weight_mode=str(args.ens_weight),
                weight_power=float(args.ens_weight_power),
            )

            # per includere anche il modello corrente, aggiungilo manualmente come candidato “temporaneo”
            # se non è ancora in fold_reports
            if spec is None:
                # prova con (current + precedenti migliori)
                # costruiamo una lista temporanea: fold_reports + current fold info
                tmp_reports = fold_reports + [{
                    "fold": fold,
                    "run_name": run_name,
                    "best_threshold_val": best_thr.to_dict(),
                }]
                spec = _build_past_only_ensemble_spec(
                    fold_reports=tmp_reports,
                    current_fold=fold,
                    top_k=int(args.ens_topk),
                    min_val_trades=int(args.ens_min_val_trades),
                    weight_mode=str(args.ens_weight),
                    weight_power=float(args.ens_weight_power),
                )

            if spec is not None:
                # override method per l’ensemble
                cfg_ens = cfg
                ed = dict(cfg.ensemble.__dict__) if hasattr(cfg, "ensemble") else {}
                ed["enabled"] = True
                ed["method"] = str(args.ens_method)
                ed["weight_mode"] = str(args.ens_weight)
                ed["weight_power"] = float(args.ens_weight_power)
                ed["backtest_enabled"] = True
                ed["backtest_past_only"] = True
                cfg_ens = cfg.__class__(**{**cfg.__dict__, "ensemble": cfg.ensemble.__class__(**ed)})

                # predizioni ensemble su VAL e soglia ensemble
                p_val_ens = predict_proba_ensemble(cfg_ens, ds_val.X_list, spec=spec)
                best_thr_ens, thr_grid_ens = search_best_threshold(df_val_aligned, p_val_ens, cfg_ens)

                # predizioni ensemble su TEST e backtest ensemble
                p_test_ens = predict_proba_ensemble(cfg_ens, ds_test.X_list, spec=spec)
                metrics_test_ens, trades_test_ens = run_backtest(df_test_aligned, p_test_ens, cfg_ens, p_min=best_thr_ens.p_min)

                # salva artefatti ensemble
                fold_dir = os.path.join(cfg.reports_dir, "backtests", run_name)
                os.makedirs(fold_dir, exist_ok=True)
                thr_grid_ens.to_csv(os.path.join(fold_dir, "val_threshold_grid_ensemble.csv"), index=False)
                trades_test_ens.to_csv(os.path.join(fold_dir, "test_trades_ensemble.csv"), index=False)

                with open(os.path.join(fold_dir, "best_threshold_ensemble.json"), "w", encoding="utf-8") as f:
                    json.dump(best_thr_ens.to_dict(), f, indent=2)

                with open(os.path.join(fold_dir, "test_metrics_ensemble.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics_test_ens.to_dict(), f, indent=2)

                ensemble_report = {
                    "ensemble_runs": list(spec.run_names),
                    "ensemble_weights": [float(x) for x in spec.weights.tolist()],
                    "best_threshold_val_ensemble": best_thr_ens.to_dict(),
                    "test_metrics_ensemble": metrics_test_ens.to_dict(),
                }

                print(
                    f"[Fold {fold}] ENSEMBLE using {ensemble_report['ensemble_runs']} "
                    f"p_min={best_thr_ens.p_min:.2f} TEST exp={metrics_test_ens.expectancy_net_pips:.2f} PF={metrics_test_ens.profit_factor:.2f} trades={metrics_test_ens.trades}"
                )
            else:
                print(f"[Fold {fold}] ENSEMBLE skipped (not enough past models meeting criteria).")


        print(
            f"[Fold {fold}] Best threshold on VAL: p_min={best_thr.p_min:.2f} "
            f"VAL trades={best_thr.trades} VAL exp={best_thr.expectancy_net_pips:.2f} PF={best_thr.profit_factor:.2f}"
        )
        print(
            f"[Fold {fold}] TEST: trades={metrics_test.trades} winrate={metrics_test.winrate:.2%} "
            f"exp={metrics_test.expectancy_net_pips:.2f} PF={metrics_test.profit_factor:.2f} "
            f"MDD={metrics_test.max_drawdown_pips:.2f}"
        )

        fold_dir = os.path.join(cfg.reports_dir, "backtests", run_name)
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "best_threshold.json"), "w", encoding="utf-8") as f:
            json.dump(best_thr.to_dict(), f, indent=2)

        with open(os.path.join(fold_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_test.to_dict(), f, indent=2)

        trades_test.to_csv(os.path.join(fold_dir, "test_trades.csv"), index=False)
        thr_grid.to_csv(os.path.join(fold_dir, "val_threshold_grid.csv"), index=False)

        fold_report = {
            "fold": fold,
            "run_name": run_name,
            "best_threshold_val": best_thr.to_dict(),
            "test_metrics": metrics_test.to_dict(),
        }

        if ensemble_report is not None:
            fold_report["ensemble"] = ensemble_report

        fold_reports.append(fold_report)

        if metrics_test.trades >= 50 and metrics_test.expectancy_net_pips > best_score:
            best_score = metrics_test.expectancy_net_pips
            best_fold = fold_report
            best_fold["_selected_by"] = "test_expectancy_net_pips_with_min_trades_50"

    summary_path = os.path.join(cfg.reports_dir, "walk_forward_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"folds": fold_reports, "best_fold": best_fold}, f, indent=2)

    # --- NEW: riepilogo unico single vs ensemble ---
    if getattr(args, "ensemble_test", False):
        print("=============        write_ensemble_comparison_reports =======================\n")
        write_ensemble_comparison_reports(cfg, fold_reports)

    print("\n=== WALK-FORWARD DONE ===")
    print(f"Summary saved: {summary_path}")
    if best_fold:
        print(f"Best fold: {best_fold['run_name']} (score={best_score:.2f})")
    else:
        print("No fold met the min trades criteria; check reports and adjust thresholds/filters.")

def run_signal(cfg: ProjectConfig, run_name: str, p_min: float, args) -> None:
    ensure_dirs(cfg)

    df = load_mt5_csvs(cfg)
    df = compute_derived_features(df, cfg)
    df = add_tp_sl_labels(df, cfg)
    df = apply_nan_policy(df, cfg)

    # override ensemble da CLI
    if args.ensemble:
        # abilita ensemble
        cfg = cfg.__class__(**{**cfg.__dict__, "ensemble": cfg.ensemble.__class__(**{**cfg.ensemble.__dict__, "enabled": True})})

    if cfg.ensemble.enabled:
        ed = dict(cfg.ensemble.__dict__)
        if args.ens_topk is not None:
            ed["top_k"] = int(args.ens_topk)
        if args.ens_method:
            ed["method"] = str(args.ens_method)
        if args.ens_weight:
            ed["weight_mode"] = str(args.ens_weight)
        if args.ens_runs:
            runs = [x.strip() for x in str(args.ens_runs).split(",") if x.strip()]
            ed["run_names"] = tuple(runs)
            ed["use_summary_auto_select"] = False

        cfg = cfg.__class__(**{**cfg.__dict__, "ensemble": cfg.ensemble.__class__(**ed)})

    sig = generate_signal(cfg, df_full=df, run_name=run_name, p_min=float(p_min), t=None)
    print(json.dumps(sig.__dict__, indent=2, default=str))


def main():
    cfg = default_config()

    parser = argparse.ArgumentParser(description="FOREX TF TP/SL multiscale - training & analysis only")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wf = sub.add_parser("train_wf", help="Train with walk-forward and evaluate")
    p_wf.add_argument("--ensemble_test", action="store_true", help="Esegue anche backtest ensemble per ogni fold")
    p_wf.add_argument("--ens_topk", type=int, default=3, help="Quanti modelli includere nell’ensemble (past-only)")
    p_wf.add_argument("--ens_method", type=str, default="mean", help="mean|median|trimmed_mean")
    p_wf.add_argument("--ens_weight", type=str, default="val_expectancy", help="equal|val_expectancy|val_profit_factor")
    p_wf.add_argument("--ens_weight_power", type=float, default=1.0, help="Amplifica differenze tra pesi")
    p_wf.add_argument("--ens_min_val_trades", type=int, default=30, help="Min trades in validation per includere un fold nell’ensemble")
    p_wf.add_argument("--regime", action="store_true", help="Abilita regime filter")
    p_wf.add_argument("--regime_mode", type=str, default="trend_only", help="trend_only|trend_or_range|no_chaos")

    p_sig = sub.add_parser("signal", help="Generate signal from trained run")
    p_sig.add_argument("--run", type=str, default="wf_fold_0", help="run_name in models/exports/")
    p_sig.add_argument("--pmin", type=float, required=True, help="threshold p_min chosen (from validation search)")
    p_sig.add_argument("--ensemble", action="store_true", help="Usa ensemble (se abilitato o forzato)")
    p_sig.add_argument("--ens_topk", type=int, default=None, help="Override ensemble.top_k")
    p_sig.add_argument("--ens_method", type=str, default=None, help="Override ensemble.method: mean|median|trimmed_mean")
    p_sig.add_argument("--ens_weight", type=str, default=None, help="Override ensemble.weight_mode: equal|val_expectancy|val_profit_factor")
    p_sig.add_argument("--ens_runs", type=str, default=None, help="Lista run separati da virgola, es: wf_fold_1,wf_fold_3")

    args = parser.parse_args()

    if args.cmd == "train_wf":
        run_train_walk_forward(cfg, args)
    elif args.cmd == "signal":
        run_signal(cfg, run_name=args.run, p_min=args.pmin, args=args)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
