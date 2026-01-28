# src/features/support_resistance_fast.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

from src.config import ProjectConfig


def _bin_size_pips_for_slice(atr_pips: np.ndarray, cfg: ProjectConfig) -> float:
    sr = cfg.sr
    if not sr.use_dynamic_bin:
        return float(sr.bin_size_pips)

    # bin dinamico = max(bin_size_pips, atr_bin_mult * median_atr)
    med = float(np.nanmedian(atr_pips)) if np.isfinite(np.nanmedian(atr_pips)) else float(sr.bin_size_pips)
    dyn = float(sr.atr_bin_mult) * med
    return float(max(sr.bin_size_pips, dyn))


def _levels_from_hist(prices: np.ndarray, pip_size: float, bin_size_pips: float, top_k: int) -> List[float]:
    """
    Histogram "binning" in pips. Ritorna top_k livelli (in prezzo) ordinati per frequenza desc.
    """
    if prices.size == 0:
        return []

    pips = prices / pip_size
    b = float(bin_size_pips)
    # assegna bin (intero)
    bins = np.floor(pips / b).astype(np.int64)

    # count bins
    uniq, cnt = np.unique(bins, return_counts=True)
    if uniq.size == 0:
        return []

    idx = np.argsort(-cnt)
    uniq = uniq[idx][:top_k]

    # livello = centro del bin
    levels_pips = (uniq.astype(float) + 0.5) * b
    levels_price = levels_pips * pip_size
    return list(levels_price.astype(float))


def _compute_sr_for_time(
    close_t: float,
    swing_prices: np.ndarray,
    atr_t_pips: float,
    pip_size: float,
    bin_size_pips: float,
    top_k: int,
) -> Tuple[List[float], List[float], float, float]:
    """
    Ritorna:
    - res_levels (sopra close) top_k
    - sup_levels (sotto close) top_k
    - dist_res1_atr
    - dist_sup1_atr

    dist_*_atr = (level - close)/ATR_price  (positivo se level sopra close)
    """
    if swing_prices.size == 0 or not np.isfinite(close_t) or not np.isfinite(atr_t_pips) or atr_t_pips <= 0:
        return [], [], np.nan, np.nan

    res_candidates = swing_prices[swing_prices >= close_t]
    sup_candidates = swing_prices[swing_prices <= close_t]

    res_levels = _levels_from_hist(res_candidates, pip_size, bin_size_pips, top_k)
    sup_levels = _levels_from_hist(sup_candidates, pip_size, bin_size_pips, top_k)

    # ordina livelli: res crescente (più vicino prima), sup decrescente (più vicino prima)
    res_levels = sorted(res_levels)[:top_k]
    sup_levels = sorted(sup_levels, reverse=True)[:top_k]

    atr_price = atr_t_pips * pip_size

    dist_res1_atr = np.nan
    dist_sup1_atr = np.nan

    if len(res_levels) > 0:
        dist_res1_atr = (float(res_levels[0]) - close_t) / atr_price
    if len(sup_levels) > 0:
        dist_sup1_atr = (close_t - float(sup_levels[0])) / atr_price

    return res_levels, sup_levels, float(dist_res1_atr), float(dist_sup1_atr)


def add_sr_fast(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    FAST e CAUSALE:
    - per ogni t (a step recalc_every) calcola livelli su slice [t-lookback+1 : t] (solo passato)
    - forward-fill fino al prossimo recalc
    """
    if not cfg.sr.enabled:
        return df

    out = df.copy()
    needed = {"high", "low", "close", "atr_14_pips"}
    if any(c not in out.columns for c in needed):
        if cfg.verbose:
            print("[SR_FAST] Missing base columns, skipping.")
        return out

    sr = cfg.sr
    lookback = int(sr.lookback_bars)
    step = int(max(1, sr.recalc_every))
    top_k = int(sr.top_k_levels)
    pip_size = float(cfg.market.pip_size)

    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    atr_pips = pd.to_numeric(out["atr_14_pips"], errors="coerce").to_numpy(dtype=float)

    n = len(out)

    # output columns
    for i in range(1, top_k + 1):
        out[f"res{i}"] = np.nan
        out[f"sup{i}"] = np.nan
    out["dist_res1_atr"] = np.nan
    out["dist_sup1_atr"] = np.nan

    # Swing proxy CAUSALE: usiamo high e low della slice (non fractal con futuro).
    # Prendiamo “candidate levels” come:
    # - quantili alti/bassi della slice + estremi locali su w (solo backward)
    # Qui facciamo semplice: usiamo prezzi high+low della slice e lasciamo che l’istogramma scelga.
    # (funziona sorprendentemente bene, ed è 100% causale)
    for t in range(n):
        if t < 10:
            continue
        if t % step != 0 and t != n - 1:
            continue

        start = max(0, t - lookback + 1)
        sl_high = high[start : t + 1]
        sl_low = low[start : t + 1]
        sl_close = close[start : t + 1]
        sl_atr = atr_pips[start : t + 1]

        # candidate prices: highs+lows (filtra NaN)
        cand = np.concatenate([sl_high, sl_low, sl_close])
        cand = cand[np.isfinite(cand)]
        if cand.size == 0:
            continue

        bin_size_pips = _bin_size_pips_for_slice(sl_atr, cfg)

        res_levels, sup_levels, dres, dsup = _compute_sr_for_time(
            close_t=close[t],
            swing_prices=cand,
            atr_t_pips=atr_pips[t],
            pip_size=pip_size,
            bin_size_pips=bin_size_pips,
            top_k=top_k,
        )

        # write at t
        for i in range(1, top_k + 1):
            out.at[out.index[t], f"res{i}"] = res_levels[i - 1] if i - 1 < len(res_levels) else np.nan
            out.at[out.index[t], f"sup{i}"] = sup_levels[i - 1] if i - 1 < len(sup_levels) else np.nan

        out.at[out.index[t], "dist_res1_atr"] = dres
        out.at[out.index[t], "dist_sup1_atr"] = dsup

    # forward fill levels/distances
    cols_ff = [c for c in out.columns if c.startswith("res") or c.startswith("sup") or c.startswith("dist_")]
    out[cols_ff] = out[cols_ff].ffill()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
