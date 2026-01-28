# src/features/breakout_features.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig


def add_breakout_features(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    if not cfg.breakout.enabled:
        return df

    out = df.copy()

    # Required base columns
    needed = {"open", "high", "low", "close", "atr_14_pips", "log_return_1"}
    missing = [c for c in needed if c not in out.columns]
    if missing:
        # Se manca log_return_1 o altro, meglio non esplodere: torna senza queste feature.
        if cfg.verbose:
            print(f"[Breakout] Missing columns, skipping breakout features: {missing}")
        return out

    # 1) close position in range (0..1)
    rng = (out["high"] - out["low"]).replace(0.0, np.nan)
    out["close_pos_in_range"] = ((out["close"] - out["low"]) / rng).clip(0.0, 1.0)

    # 2) ATR ratio
    w_atr = int(cfg.breakout.atr_ratio_window)
    atr = out["atr_14_pips"].astype(float)
    atr_sma = atr.rolling(window=w_atr, min_periods=w_atr).mean()
    out[f"atr_ratio_{w_atr}"] = (atr / atr_sma).replace([np.inf, -np.inf], np.nan)

    # 3) range / ATR and its SMA
    range_pips = (out["high"] - out["low"]) / float(cfg.market.pip_size)
    out["range_by_atr_inst"] = (range_pips / atr.replace(0.0, np.nan))
    w_rng = int(cfg.breakout.range_atr_sma_window)
    out[f"range_by_atr_sma_{w_rng}"] = out["range_by_atr_inst"].rolling(window=w_rng, min_periods=w_rng).mean()

    # 4) impulse: rolling sums of returns
    for w in cfg.breakout.ret_sum_windows:
        ww = int(w)
        out[f"ret_sum_{ww}"] = out["log_return_1"].rolling(window=ww, min_periods=ww).sum()

    # 5) pressure near S/R using dist_*_atr if available
    near_thr = float(cfg.breakout.near_thr_atr)
    lookback = int(cfg.breakout.pressure_lookback)

    if "dist_res1_atr" in out.columns:
        near_res = (out["dist_res1_atr"].astype(float) <= near_thr).astype(float)
        out[f"near_res_count_{lookback}"] = near_res.rolling(window=lookback, min_periods=lookback).sum()

    if "dist_sup1_atr" in out.columns:
        near_sup = (out["dist_sup1_atr"].astype(float) <= near_thr).astype(float)
        out[f"near_sup_count_{lookback}"] = near_sup.rolling(window=lookback, min_periods=lookback).sum()

    # 6) retest logic (soft, feature-based)
    # We approximate "broke above resistance recently" as:
    # dist_res1_atr < 0 (meaning close above nearest resistance) by margin
    recent = int(cfg.breakout.break_recent_lookback)
    margin = float(cfg.breakout.break_margin_atr)

    if "dist_res1_atr" in out.columns:
        broke_above = (out["dist_res1_atr"].astype(float) < -margin).astype(float)
        out[f"break_above_res_recent_{recent}"] = broke_above.rolling(window=recent, min_periods=recent).max()

        # retest now: we are near resistance and we broke above recently
        near_now = (out["dist_res1_atr"].astype(float) <= near_thr).astype(float)
        out["retest_res_now"] = (near_now * out[f"break_above_res_recent_{recent}"]).clip(0.0, 1.0)

    if "dist_sup1_atr" in out.columns:
        broke_below = (out["dist_sup1_atr"].astype(float) < -margin).astype(float)
        out[f"break_below_sup_recent_{recent}"] = broke_below.rolling(window=recent, min_periods=recent).max()

        near_now = (out["dist_sup1_atr"].astype(float) <= near_thr).astype(float)
        out["retest_sup_now"] = (near_now * out[f"break_below_sup_recent_{recent}"]).clip(0.0, 1.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
