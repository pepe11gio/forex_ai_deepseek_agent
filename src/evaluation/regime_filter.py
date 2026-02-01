# src/evaluation/regime_filter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.config import ProjectConfig


@dataclass(frozen=True)
class RegimeInfo:
    name: str
    allowed_mask: np.ndarray  # bool array len(df)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").astype(float)


def compute_trade_allowed_mask(df: pd.DataFrame, cfg: ProjectConfig) -> RegimeInfo:
    """
    Proxy regime veloce:
      - trend_strength in [0,1] (da derived_features: (ADX-20)/20 clipped)
      - atr_percentile in [0,1] (da derived_features)

    Regole:
      TREND: trend_strength >= thr_trend AND atr_percentile >= thr_atr
      RANGE_OK: trend_strength < thr_trend AND atr_percentile between [atr_low, atr_high]
      CHAOS: atr_percentile very high AND trend_strength low  (spesso whipsaw)
    """
    rc = cfg.regime

    trend = _col(df, "trend_strength")
    atrp = _col(df, "atr_percentile")

    # fallback se non presenti: non filtrare (così non rompi nulla)
    if trend.isna().all() or atrp.isna().all():
        return RegimeInfo(name="no_regime_features", allowed_mask=np.ones(len(df), dtype=bool))

    thr_trend = float(rc.trend_thr)
    thr_atr = float(rc.atr_thr)

    atr_low = float(rc.atr_low)
    atr_high = float(rc.atr_high)

    is_trend = (trend >= thr_trend) & (atrp >= thr_atr)
    is_range_ok = (trend < thr_trend) & (atrp >= atr_low) & (atrp <= atr_high)

    # chaos: vol alta ma trend basso -> spesso false rotture
    is_chaos = (trend < thr_trend) & (atrp >= float(rc.chaos_atr_thr))

    mode = (rc.mode or "trend_only").lower()

    if mode == "trend_only":
        allowed = is_trend
        name = "trend_only"
    elif mode == "trend_or_range":
        allowed = is_trend | is_range_ok
        name = "trend_or_range"
    elif mode == "no_chaos":
        allowed = ~is_chaos
        name = "no_chaos"
    else:
        # default
        allowed = is_trend
        name = "trend_only"

    # min warmup: evita i primi punti dove atr_percentile non è affidabile
    warmup = int(rc.warmup_bars)
    if warmup > 0 and len(df) > warmup:
        allowed.iloc[:warmup] = False

    return RegimeInfo(name=name, allowed_mask=allowed.to_numpy(dtype=bool))
