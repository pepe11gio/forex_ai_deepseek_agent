# src/features/support_resistance_slow.py
from __future__ import annotations

import time
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import ProjectConfig


def _find_swings(high: np.ndarray, low: np.ndarray, w: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(high)
    sh = np.zeros(T, dtype=bool)
    sl = np.zeros(T, dtype=bool)
    if T < (2 * w + 1):
        return sh, sl

    for i in range(w, T - w):
        if high[i] == np.max(high[i - w : i + w + 1]):
            sh[i] = True
        if low[i] == np.min(low[i - w : i + w + 1]):
            sl[i] = True
    return sh, sl


def _cluster_levels_pips(levels_pips: np.ndarray, bin_size_pips: float) -> List[Tuple[float, int]]:
    if levels_pips.size == 0 or bin_size_pips <= 0:
        return []

    bins = np.floor(levels_pips / bin_size_pips).astype(int)
    counts: Dict[int, int] = {}
    sums: Dict[int, float] = {}

    for b, v in zip(bins, levels_pips):
        counts[b] = counts.get(b, 0) + 1
        sums[b] = sums.get(b, 0.0) + float(v)

    clusters = []
    for b in counts:
        clusters.append((sums[b] / counts[b], counts[b]))

    clusters.sort(key=lambda x: x[1], reverse=True)
    return clusters


def _select_nearest_levels(
    clusters_pips: List[Tuple[float, int]],
    price_pips: float,
    side: str,
    k: int
) -> List[Tuple[float, int]]:
    if not clusters_pips:
        return []

    if side == "support":
        c = [(lvl, cnt) for (lvl, cnt) in clusters_pips if lvl <= price_pips]
        c.sort(key=lambda x: (price_pips - x[0]))
    else:
        c = [(lvl, cnt) for (lvl, cnt) in clusters_pips if lvl >= price_pips]
        c.sort(key=lambda x: (x[0] - price_pips))
    return c[:k]


def add_sr_slow(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    ⚠️ VERSIONE LENTA – SOLO DIAGNOSTICA ⚠️

    - Ricalcola swing + clustering ad ogni barra t
    - Complessità ~ O(T²)
    - NON usare in produzione
    """
    out = df.copy()
    T = len(out)
    pip_size = float(cfg.market.pip_size)

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    atr_pips = out["atr_14_pips"].to_numpy(dtype=float)

    high_pips = high / pip_size
    low_pips = low / pip_size
    close_pips = close / pip_size

    w = int(cfg.sr.swing_window)
    lookback = int(cfg.sr.lookback_bars)
    k = int(cfg.sr.top_k_levels)

    # init columns
    for i in range(1, k + 1):
        out[f"dist_sup{i}_atr"] = np.nan
        out[f"dist_res{i}_atr"] = np.nan
        out[f"sup{i}_strength"] = np.nan
        out[f"res{i}_strength"] = np.nan

    t0 = time.time()

    progress_every = cfg.sr.progress_every
    if progress_every <= 0:
        progress_every = max(200, T // 100)

    for t in range(T):
        start = max(0, t - lookback)
        if t - start < (2 * w + 1):
            continue

        # dynamic bin size per-bar
        bin_size = cfg.sr.bin_size_pips
        if cfg.sr.use_dynamic_bin and np.isfinite(atr_pips[t]) and atr_pips[t] > 0:
            bin_size = max(bin_size, cfg.sr.atr_bin_mult * atr_pips[t])

        sh, sl = _find_swings(
            high_pips[start : t + 1],
            low_pips[start : t + 1],
            w
        )

        swing_high = high_pips[start : t + 1][sh]
        swing_low = low_pips[start : t + 1][sl]

        res_clusters = _cluster_levels_pips(swing_high, bin_size)
        sup_clusters = _cluster_levels_pips(swing_low, bin_size)

        sup_max = max((c for _, c in sup_clusters), default=1)
        res_max = max((c for _, c in res_clusters), default=1)

        atr = atr_pips[t]
        if not np.isfinite(atr) or atr <= 0:
            continue

        px = close_pips[t]

        sups = _select_nearest_levels(sup_clusters, px, "support", k)
        ress = _select_nearest_levels(res_clusters, px, "resistance", k)

        for i, (lvl, cnt) in enumerate(sups, 1):
            out.iat[t, out.columns.get_loc(f"dist_sup{i}_atr")] = (px - lvl) / atr
            out.iat[t, out.columns.get_loc(f"sup{i}_strength")] = cnt / sup_max

        for i, (lvl, cnt) in enumerate(ress, 1):
            out.iat[t, out.columns.get_loc(f"dist_res{i}_atr")] = (lvl - px) / atr
            out.iat[t, out.columns.get_loc(f"res{i}_strength")] = cnt / res_max

        # progress log
        if cfg.verbose and t > 0 and t % progress_every == 0:
            elapsed = time.time() - t0
            pct = (t / (T - 1)) * 100
            eta = (elapsed / t) * (T - t)
            print(f"[S/R SLOW] {pct:6.2f}% elapsed={elapsed:,.1f}s ETA~{eta:,.1f}s")

    if cfg.verbose:
        print(f"[S/R SLOW] Done in {time.time() - t0:,.1f}s")

    return out
