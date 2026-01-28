# src/labeling/tp_sl_labeler.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig


class LabelingError(Exception):
    pass


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _compute_vol_rank(out: pd.DataFrame, cfg: ProjectConfig) -> np.ndarray:
    """
    Ritorna vol_rank in [0,1]. Preferisce atr_percentile se presente,
    altrimenti usa rolling rank grezzo su ATR (più lento ma ok).
    """
    if "atr_percentile" in out.columns:
        v = pd.to_numeric(out["atr_percentile"], errors="coerce").to_numpy(dtype=float)
        return _clip01(np.nan_to_num(v, nan=0.5))

    w = int(cfg.derived.atr_percentile_window)
    atr = pd.to_numeric(out["atr_14_pips"], errors="coerce")
    # rolling percentile rank (approx): rank of last vs window
    # per semplicità: usa quantile of rolling distribution via pandas rank in window (più costoso)
    # fallback robusto: normalizza con min/max rolling
    rmin = atr.rolling(w, min_periods=w).min()
    rmax = atr.rolling(w, min_periods=w).max()
    v = ((atr - rmin) / (rmax - rmin)).replace([np.inf, -np.inf], np.nan).fillna(0.5).to_numpy(dtype=float)
    return _clip01(v)


def _compute_trend_strength(out: pd.DataFrame) -> np.ndarray:
    """
    Trend strength in [0,1] usando ADX.
    ADX ~20: inizio trend, ~40+ trend forte.
    """
    if "adx_14" not in out.columns:
        return np.full(len(out), 0.5, dtype=float)
    adx = pd.to_numeric(out["adx_14"], errors="coerce").fillna(20.0).to_numpy(dtype=float)
    s = (adx - 20.0) / 20.0
    return _clip01(s)


def compute_dynamic_rr_h(out: pd.DataFrame, cfg: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Policy dinamica semplice e stabile:
    - rr aumenta con trend_strength e vol_rank
    - horizon diminuisce con vol_rank (alta vol -> raggiunge TP/SL prima) e aumenta con trend_strength

    Ritorna:
    - rr_t (float)
    - h_t (int)
    """
    mc = cfg.market

    vol_rank = _compute_vol_rank(out, cfg)            # 0..1
    trend = _compute_trend_strength(out)              # 0..1

    # score complessivo 0..1
    score_rr = _clip01(0.6 * trend + 0.4 * vol_rank)
    rr_t = mc.rr_min + (mc.rr_max - mc.rr_min) * score_rr

    # horizon: alta vol -> più corto, trend -> più lungo
    score_h = _clip01(0.55 * trend + 0.45 * (1.0 - vol_rank))
    h_float = mc.horizon_bars_min + (mc.horizon_bars_max - mc.horizon_bars_min) * score_h
    h_t = np.rint(h_float).astype(int)
    h_t = np.clip(h_t, mc.horizon_bars_min, mc.horizon_bars_max)

    return rr_t.astype(float), h_t.astype(int)


def add_tp_sl_labels(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    Aggiunge colonne:
    - rr_t, horizon_t
    - sl_net_pips, tp_net_pips
    - cost_pips (spread)
    - sl_gross_pips, tp_gross_pips
    - y_long, y_short (0/1) oppure NaN se invalid
    - valid_label (bool)
    - no_touch (opzionale)

    Assunzioni:
    - df ha colonne: close, high, low, atr_14_pips
    - df è ordinato per time crescente
    """
    if "atr_14_pips" not in df.columns:
        raise LabelingError("Colonna richiesta mancante: atr_14_pips (da MT5 o convertita).")

    out = df.copy()

    mc = cfg.market
    pip = float(mc.pip_size)
    spread = float(mc.spread_pips)
    k = float(mc.sl_atr_k)

    n = len(out)

    # RR/H per barra
    if mc.rr_mode.lower() == "dynamic" or mc.horizon_mode.lower() == "dynamic":
        rr_t, h_t = compute_dynamic_rr_h(out, cfg)
    else:
        rr_t = np.full(n, float(mc.rr_fixed), dtype=float)
        h_t = np.full(n, int(mc.horizon_bars_fixed), dtype=int)

    out["rr_t"] = rr_t
    out["horizon_t"] = h_t

    # SL/TP netti in pips
    out["sl_net_pips"] = k * pd.to_numeric(out["atr_14_pips"], errors="coerce").astype(float)
    out["tp_net_pips"] = out["sl_net_pips"] * out["rr_t"].astype(float)

    # Costi (spread round-turn)
    out["cost_pips"] = spread

    # SL/TP gross (per “tocco” severo: includi costi)
    out["sl_gross_pips"] = out["sl_net_pips"] + out["cost_pips"]
    out["tp_gross_pips"] = out["tp_net_pips"] + out["cost_pips"]

    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(out["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(out["low"], errors="coerce").to_numpy(dtype=float)

    # target come float per supportare NaN
    y_long = np.full(n, np.nan, dtype=float) if cfg.labeling.invalid_as_nan else np.zeros(n, dtype=float)
    y_short = np.full(n, np.nan, dtype=float) if cfg.labeling.invalid_as_nan else np.zeros(n, dtype=float)
    valid_label = np.zeros(n, dtype=np.int8)
    no_touch = np.zeros(n, dtype=np.int8)

    for t in range(n):
        H = int(h_t[t])
        if H <= 0:
            continue

        # Serve almeno H barre forward disponibili (t+1..t+H)
        if t + H >= n:
            if cfg.labeling.compute_no_touch_flag:
                no_touch[t] = 1
            continue

        entry = close[t]
        if not np.isfinite(entry):
            continue

        slg = float(out["sl_gross_pips"].iat[t]) * pip
        tpg = float(out["tp_gross_pips"].iat[t]) * pip

        # LONG levels
        up_long = entry + tpg
        down_long = entry - slg

        # SHORT levels
        down_short = entry - tpg
        up_short = entry + slg

        hf = high[t + 1 : t + 1 + H]
        lf = low[t + 1 : t + 1 + H]

        # LONG outcome (worst-case tie -> SL first)
        hit_tp_long = hf >= up_long
        hit_sl_long = lf <= down_long
        idx_tp = np.where(hit_tp_long)[0]
        idx_sl = np.where(hit_sl_long)[0]

        if idx_tp.size == 0 and idx_sl.size == 0:
            y_long[t] = 0.0
        elif idx_tp.size == 0:
            y_long[t] = 0.0
        elif idx_sl.size == 0:
            y_long[t] = 1.0
        else:
            if idx_tp[0] < idx_sl[0]:
                y_long[t] = 1.0
            elif idx_sl[0] < idx_tp[0]:
                y_long[t] = 0.0
            else:
                y_long[t] = 0.0

        # SHORT outcome (worst-case tie -> SL first)
        hit_tp_short = lf <= down_short
        hit_sl_short = hf >= up_short
        idx_tp_s = np.where(hit_tp_short)[0]
        idx_sl_s = np.where(hit_sl_short)[0]

        if idx_tp_s.size == 0 and idx_sl_s.size == 0:
            y_short[t] = 0.0
        elif idx_tp_s.size == 0:
            y_short[t] = 0.0
        elif idx_sl_s.size == 0:
            y_short[t] = 1.0
        else:
            if idx_tp_s[0] < idx_sl_s[0]:
                y_short[t] = 1.0
            elif idx_sl_s[0] < idx_tp_s[0]:
                y_short[t] = 0.0
            else:
                y_short[t] = 0.0

        valid_label[t] = 1

        if cfg.labeling.compute_no_touch_flag:
            any_touch = (idx_tp.size > 0) or (idx_sl.size > 0) or (idx_tp_s.size > 0) or (idx_sl_s.size > 0)
            no_touch[t] = 0 if any_touch else 1

    out["y_long"] = y_long
    out["y_short"] = y_short
    out["valid_label"] = valid_label.astype(bool)
    if cfg.labeling.compute_no_touch_flag:
        out["no_touch"] = no_touch

    return out


def min_sl_net_filter_mask(df: pd.DataFrame, cfg: ProjectConfig) -> pd.Series:
    """
    Filtro anti-spread: SL_net >= multiple * spread.
    (spread round-turn)
    """
    spread = float(cfg.market.spread_pips)
    mult = float(cfg.market.min_sl_net_multiple_of_spread)

    if "sl_net_pips" not in df.columns:
        raise LabelingError("sl_net_pips non presente: chiama add_tp_sl_labels() prima.")

    return pd.to_numeric(df["sl_net_pips"], errors="coerce") >= (mult * spread)
