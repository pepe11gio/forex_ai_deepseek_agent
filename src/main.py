# src/main.py
from __future__ import annotations

import argparse
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


def run_train_walk_forward(cfg: ProjectConfig) -> None:
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
        fold_reports.append(fold_report)

        if metrics_test.trades >= 50 and metrics_test.expectancy_net_pips > best_score:
            best_score = metrics_test.expectancy_net_pips
            best_fold = fold_report
            best_fold["_selected_by"] = "test_expectancy_net_pips_with_min_trades_50"

    summary_path = os.path.join(cfg.reports_dir, "walk_forward_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"folds": fold_reports, "best_fold": best_fold}, f, indent=2)

    print("\n=== WALK-FORWARD DONE ===")
    print(f"Summary saved: {summary_path}")
    if best_fold:
        print(f"Best fold: {best_fold['run_name']} (score={best_score:.2f})")
    else:
        print("No fold met the min trades criteria; check reports and adjust thresholds/filters.")


def run_signal(cfg: ProjectConfig, run_name: str, p_min: float) -> None:
    ensure_dirs(cfg)

    df = load_mt5_csvs(cfg)
    df = compute_derived_features(df, cfg)
    df = add_tp_sl_labels(df, cfg)
    df = apply_nan_policy(df, cfg)

    sig = generate_signal(cfg, df_full=df, run_name=run_name, p_min=float(p_min), t=None)
    print(json.dumps(sig.to_dict(), indent=2, default=str))


def main():
    cfg = default_config()

    parser = argparse.ArgumentParser(description="FOREX TF TP/SL multiscale - training & analysis only")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train_wf", help="Train with walk-forward and evaluate")

    p_sig = sub.add_parser("signal", help="Generate signal from trained run")
    p_sig.add_argument("--run", type=str, default="wf_fold_0", help="run_name in models/exports/")
    p_sig.add_argument("--pmin", type=float, required=True, help="threshold p_min chosen (from validation search)")

    args = parser.parse_args()

    if args.cmd == "train_wf":
        run_train_walk_forward(cfg)
    elif args.cmd == "signal":
        run_signal(cfg, run_name=args.run, p_min=args.pmin)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
