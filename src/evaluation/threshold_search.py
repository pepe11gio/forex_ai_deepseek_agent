# src/evaluation/threshold_search.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.labeling.tp_sl_labeler import min_sl_net_filter_mask


@dataclass(frozen=True)
class BestThreshold:
    p_min: float
    trades: int
    winrate: float
    expectancy_net_pips: float
    profit_factor: float

    def to_dict(self) -> dict:
        return {
            "p_min": self.p_min,
            "trades": self.trades,
            "winrate": self.winrate,
            "expectancy_net_pips": self.expectancy_net_pips,
            "profit_factor": self.profit_factor,
        }


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def decide_actions(
    df: pd.DataFrame,
    p: np.ndarray,
    cfg: ProjectConfig,
    p_min: float,
) -> np.ndarray:
    """
    Decide azioni:
      -1 short, +1 long, 0 no trade
    Regola: scegli direzione con expectancy migliore, se > buffer e p_dir >= p_min
    Expectancy per barra:
      E = p* (tp_net - cost) - (1-p)*(sl_net + cost)
    """
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("p deve avere shape (N,2) = [p_win_long, p_win_short]")

    # filtri: label valida e SL minimo
    valid = df["valid_label"].astype(bool).to_numpy() if "valid_label" in df.columns else np.ones(len(df), dtype=bool)
    sl_mask = min_sl_net_filter_mask(df, cfg).to_numpy(dtype=bool)

    sl = _safe_col(df, "sl_net_pips", default=np.nan)
    tp = _safe_col(df, "tp_net_pips", default=np.nan)
    cost = _safe_col(df, "cost_pips", default=float(cfg.market.spread_pips))

    p_long = p[:, 0].astype(float)
    p_short = p[:, 1].astype(float)

    # E_long/E_short per barra
    # win: + (tp - cost), loss: - (sl + cost)
    E_long = p_long * (tp - cost) - (1.0 - p_long) * (sl + cost)
    E_short = p_short * (tp - cost) - (1.0 - p_short) * (sl + cost)

    buffer = float(cfg.market.expectancy_buffer_pips)

    actions = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df)):
        if not valid[i]:
            continue
        if not sl_mask[i]:
            continue
        if not np.isfinite(sl[i]) or not np.isfinite(tp[i]) or sl[i] <= 0 or tp[i] <= 0:
            continue

        eL = E_long[i]
        eS = E_short[i]
        if not np.isfinite(eL) or not np.isfinite(eS):
            continue

        if eL >= eS:
            if p_long[i] >= p_min and eL > buffer:
                actions[i] = 1
        else:
            if p_short[i] >= p_min and eS > buffer:
                actions[i] = -1

    return actions


def _metrics_from_actions(df: pd.DataFrame, actions: np.ndarray, cfg: ProjectConfig) -> Tuple[int, float, float, float]:
    """
    Ritorna: trades, winrate, expectancy_net_pips, profit_factor
    """
    y_long = _safe_col(df, "y_long", default=np.nan)
    y_short = _safe_col(df, "y_short", default=np.nan)
    sl = _safe_col(df, "sl_net_pips", default=np.nan)
    tp = _safe_col(df, "tp_net_pips", default=np.nan)
    cost = _safe_col(df, "cost_pips", default=float(cfg.market.spread_pips))

    wins = 0
    pnls: List[float] = []
    gross_profit = 0.0
    gross_loss = 0.0

    for i, a in enumerate(actions):
        if a == 0:
            continue

        if a == 1:
            if not np.isfinite(y_long[i]):
                continue
            is_win = int(y_long[i] == 1.0)
        else:
            if not np.isfinite(y_short[i]):
                continue
            is_win = int(y_short[i] == 1.0)

        if not (np.isfinite(sl[i]) and np.isfinite(tp[i]) and np.isfinite(cost[i])):
            continue

        pnl = (tp[i] - cost[i]) if is_win else -(sl[i] + cost[i])
        pnls.append(float(pnl))
        wins += int(is_win)

        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += -pnl

    trades = len(pnls)
    if trades == 0:
        return 0, 0.0, 0.0, 0.0

    winrate = wins / trades
    expectancy = float(np.mean(pnls))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    return trades, float(winrate), expectancy, profit_factor


def search_best_threshold(
    df_val: pd.DataFrame,
    p_val: np.ndarray,
    cfg: ProjectConfig,
    p_grid: np.ndarray | None = None,
) -> Tuple[BestThreshold, pd.DataFrame]:
    """
    Cerca p_min migliore massimizzando expectancy su validation.
    """
    if p_grid is None:
        p_grid = np.round(np.arange(0.20, 0.81, 0.02), 2)

    rows = []
    best = BestThreshold(p_min=float(p_grid[0]), trades=0, winrate=0.0, expectancy_net_pips=-1e18, profit_factor=0.0)

    for p_min in p_grid:
        actions = decide_actions(df_val, p_val, cfg, p_min=float(p_min))
        trades, winrate, exp, pf = _metrics_from_actions(df_val, actions, cfg)

        rows.append(
            {
                "p_min": float(p_min),
                "trades": int(trades),
                "winrate": float(winrate),
                "expectancy_net_pips": float(exp),
                "profit_factor": float(pf),
            }
        )

        # vincolo minimo trades per evitare overfit soglia
        if trades >= 30 and exp > best.expectancy_net_pips:
            best = BestThreshold(p_min=float(p_min), trades=int(trades), winrate=float(winrate),
                                 expectancy_net_pips=float(exp), profit_factor=float(pf))

    grid_df = pd.DataFrame(rows).sort_values("p_min").reset_index(drop=True)
    return best, grid_df
