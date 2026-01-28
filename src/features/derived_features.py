# src/features/derived_features.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.features.support_resistance import add_support_resistance
from src.features.breakout_features import add_breakout_features


# -------------------------
# Helpers
# -------------------------
def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(0.0, 1.0)


def _ensure_datetime_time(out: pd.DataFrame) -> pd.DataFrame:
    if "time" in out.columns and not np.issubdtype(out["time"].dtype, np.datetime64):
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out


# -------------------------
# Time features
# -------------------------
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_datetime_time(out)
    if "time" not in out.columns:
        return out

    hour = out["time"].dt.hour.astype(float)
    dow = out["time"].dt.dayofweek.astype(float)

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out


# -------------------------
# Candle anatomy (ATR-normalized)
# -------------------------
def _add_candle_anatomy_normalized_by_atr(df: pd.DataFrame, pip_size: float) -> pd.DataFrame:
    out = df.copy()
    atr = pd.to_numeric(out["atr_14_pips"], errors="coerce").astype(float).replace(0.0, np.nan)

    rng_pips = (pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")) / pip_size
    body_pips = (pd.to_numeric(out["close"], errors="coerce") - pd.to_numeric(out["open"], errors="coerce")) / pip_size

    upper_wick_pips = (pd.to_numeric(out["high"], errors="coerce") - out[["open", "close"]].astype(float).max(axis=1)) / pip_size
    lower_wick_pips = (out[["open", "close"]].astype(float).min(axis=1) - pd.to_numeric(out["low"], errors="coerce")) / pip_size

    out["range_by_atr"] = (rng_pips / atr)
    out["body_by_atr"] = (body_pips / atr)
    out["body_abs_by_atr"] = (body_pips.abs() / atr)
    out["upper_wick_by_atr"] = (upper_wick_pips / atr)
    out["lower_wick_by_atr"] = (lower_wick_pips / atr)

    rng = (pd.to_numeric(out["high"], errors="coerce") - pd.to_numeric(out["low"], errors="coerce")).replace(0.0, np.nan)
    out["body_to_range"] = (pd.to_numeric(out["close"], errors="coerce") - pd.to_numeric(out["open"], errors="coerce")).abs() / rng

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# -------------------------
# Returns & volatility
# -------------------------
def _add_returns_and_volatility(df: pd.DataFrame, lags: Tuple[int, ...], vol_windows: Tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)

    out["log_return_1"] = np.log(close / close.shift(1))

    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            continue
        if lag == 1:
            out[f"log_return_{lag}"] = out["log_return_1"]
        else:
            out[f"log_return_{lag}"] = np.log(close / close.shift(lag))

    for w in vol_windows:
        w = int(w)
        if w <= 1:
            continue
        out[f"roll_vol_{w}"] = out["log_return_1"].rolling(window=w, min_periods=w).std()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# -------------------------
# Indicator slopes / trend strength
# -------------------------
def _add_indicator_slopes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ema_50" in out.columns:
        out["ema_50_slope"] = pd.to_numeric(out["ema_50"], errors="coerce").astype(float).diff()
    if "rsi_14" in out.columns:
        out["rsi_14_slope"] = pd.to_numeric(out["rsi_14"], errors="coerce").astype(float).diff()
    if "atr_14_pips" in out.columns:
        out["atr_14_pips_slope"] = pd.to_numeric(out["atr_14_pips"], errors="coerce").astype(float).diff()

    # Extra: trend_strength in [0,1] from ADX (same semantics used by dynamic rr/h)
    if "adx_14" in out.columns:
        adx = pd.to_numeric(out["adx_14"], errors="coerce").astype(float)
        out["trend_strength"] = _clip01((adx.fillna(20.0) - 20.0) / 20.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# -------------------------
# ATR percentile (vol_rank)
# -------------------------
def _add_atr_percentile(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Produces:
      - atr_percentile in [0,1]
    rolling apply is O(N*window) but with window ~200 on H1 it’s ok.
    """
    out = df.copy()
    w = int(window)
    if w <= 10:
        w = 10

    atr = pd.to_numeric(out["atr_14_pips"], errors="coerce").astype(float)

    def _pct_rank(x: np.ndarray) -> float:
        v = x[-1]
        if not np.isfinite(v):
            return np.nan
        return float(np.sum(x <= v) / len(x))

    out["atr_percentile"] = atr.rolling(window=w, min_periods=w).apply(_pct_rank, raw=True)
    out["atr_percentile"] = _clip01(out["atr_percentile"])
    return out


# -------------------------
# Main pipeline
# -------------------------
def compute_derived_features(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    Calcola feature derivate in Python (tutte causali):
    - returns/vol
    - slopes + trend_strength
    - candle anatomy
    - atr_percentile (vol_rank)
    - time features
    - support/resistance (fast causale o slow)
    - breakout features (dipendono da dist_*_atr se S/R attivo)
    """
    out = df.copy()
    pip_size = float(cfg.market.pip_size)

    out = _ensure_datetime_time(out)

    out = _add_returns_and_volatility(out, cfg.derived.return_lags, cfg.derived.vol_windows)
    out = _add_indicator_slopes(out)

    if cfg.derived.include_candle_anatomy:
        out = _add_candle_anatomy_normalized_by_atr(out, pip_size=pip_size)

    # IMPORTANT: per rr/h dinamici serve sempre atr_percentile
    # Se l’utente lo disattiva, lo calcoliamo comunque quando rr/h sono dinamici
    need_atr_pct = bool(cfg.derived.include_atr_percentile)
    if getattr(cfg.market, "rr_mode", "fixed").lower() == "dynamic" or getattr(cfg.market, "horizon_mode", "fixed").lower() == "dynamic":
        need_atr_pct = True

    if need_atr_pct:
        out = _add_atr_percentile(out, cfg.derived.atr_percentile_window)

    if cfg.derived.include_time_features:
        out = _add_time_features(out)

    # Support/Resistance (fast causale)
    out = add_support_resistance(out, cfg)

    # Breakout features
    out = add_breakout_features(out, cfg)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# -------------------------
# Feature columns for model
# -------------------------
def get_model_feature_columns(df: pd.DataFrame, cfg: ProjectConfig) -> List[str]:
    """
    Ritorna colonne usate come feature per il modello.
    Esclude target/labeling e colonne di servizio.
    """
    cols: List[str] = ["open", "high", "low", "close", "volume"]

    cols += [
        "ema_20",
        "ema_50",
        "rsi_14",
        "atr_14_pips",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
    ]

    # colonne “non-feature” che vogliamo escludere dal “derived_cols”
    base_exclude = {
        "time",
        # targets + diagnostics
        "y_long",
        "y_short",
        "no_touch",
        "valid_label",
        # per-trade params
        "sl_net_pips",
        "tp_net_pips",
        "sl_gross_pips",
        "tp_gross_pips",
        "cost_pips",
        "rr_t",
        "horizon_t",
    }

    derived_cols = [c for c in df.columns if c not in set(cols) and c not in base_exclude]
    derived_cols = sorted(derived_cols)

    cols += derived_cols

    # keep order, only existing cols
    seen = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen and c in df.columns:
            ordered.append(c)
            seen.add(c)
    return ordered


# -------------------------
# NaN policy (FEATURES ONLY)
# -------------------------
def apply_nan_policy(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    Importante:
    - Non deve droppare righe solo perché le label sono NaN (es. ultime H barre).
    - Deve ripulire NaN sulle colonne feature (e colonne base) perché windowing/TF non accetta NaN.

    Quindi:
    - ffill opzionale
    - dropna solo su feature columns + colonne base minime (OHLC/indicators/ATR/time)
    """
    out = df.copy()

    if cfg.derived.nan_policy == "ffill_then_drop":
        out = out.ffill()

    # colonne base minime
    base_needed = [
        "time",
        "open", "high", "low", "close",
        "ema_20", "ema_50", "rsi_14",
        "atr_14_pips",
        "adx_14", "plus_di_14", "minus_di_14",
    ]
    base_needed = [c for c in base_needed if c in out.columns]

    # feature columns (esclude label)
    feature_cols = get_model_feature_columns(out, cfg)

    drop_subset = list(dict.fromkeys(base_needed + feature_cols))  # unique preserve order
    out = out.dropna(subset=drop_subset).reset_index(drop=True)
    return out
