# src/evaluation/backtester.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.evaluation.threshold_search import decide_actions


@dataclass(frozen=True)
class BacktestMetrics:
    trades: int
    winrate: float
    expectancy_net_pips: float
    profit_factor: float
    gross_profit_pips: float
    gross_loss_pips: float
    max_drawdown_pips: float
    avg_win_pips: float
    avg_loss_pips: float

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "winrate": self.winrate,
            "expectancy_net_pips": self.expectancy_net_pips,
            "profit_factor": self.profit_factor,
            "gross_profit_pips": self.gross_profit_pips,
            "gross_loss_pips": self.gross_loss_pips,
            "max_drawdown_pips": self.max_drawdown_pips,
            "avg_win_pips": self.avg_win_pips,
            "avg_loss_pips": self.avg_loss_pips,
        }


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd))


def run_backtest(
    df: pd.DataFrame,
    p: np.ndarray,
    cfg: ProjectConfig,
    p_min: float,
) -> Tuple[BacktestMetrics, pd.DataFrame]:
    """
    Backtest “event-based” usando etichette y_long/y_short e PnL coerente con costi:
      win  -> +(tp_net - cost)
      loss -> -(sl_net + cost)

    Richiede nel df:
      - y_long, y_short
      - tp_net_pips, sl_net_pips, cost_pips
      - valid_label (opzionale)
    """
    actions = decide_actions(df, p, cfg, p_min=p_min)

    y_long = _safe_col(df, "y_long", default=np.nan)
    y_short = _safe_col(df, "y_short", default=np.nan)
    tp = _safe_col(df, "tp_net_pips", default=np.nan)
    sl = _safe_col(df, "sl_net_pips", default=np.nan)
    cost = _safe_col(df, "cost_pips", default=float(cfg.market.spread_pips))

    times = df["time"].astype(str).to_numpy() if "time" in df.columns else np.array([""] * len(df), dtype=object)

    rows = []
    pnls = []
    wins = 0

    for i, a in enumerate(actions):
        if a == 0:
            continue

        if not (np.isfinite(tp[i]) and np.isfinite(sl[i]) and np.isfinite(cost[i])):
            continue
        if tp[i] <= 0 or sl[i] <= 0:
            continue

        if a == 1:
            if not np.isfinite(y_long[i]):
                continue
            is_win = int(y_long[i] == 1.0)
            side = "LONG"
        else:
            if not np.isfinite(y_short[i]):
                continue
            is_win = int(y_short[i] == 1.0)
            side = "SHORT"

        pnl = (tp[i] - cost[i]) if is_win else -(sl[i] + cost[i])

        pnls.append(float(pnl))
        wins += int(is_win)

        rows.append(
            {
                "time": times[i],
                "i": int(i),
                "side": side,
                "p_long": float(p[i, 0]),
                "p_short": float(p[i, 1]),
                "tp_net_pips": float(tp[i]),
                "sl_net_pips": float(sl[i]),
                "cost_pips": float(cost[i]),
                "is_win": int(is_win),
                "pnl_net_pips": float(pnl),
            }
        )

    trades = len(pnls)
    if trades == 0:
        metrics = BacktestMetrics(
            trades=0,
            winrate=0.0,
            expectancy_net_pips=0.0,
            profit_factor=0.0,
            gross_profit_pips=0.0,
            gross_loss_pips=0.0,
            max_drawdown_pips=0.0,
            avg_win_pips=0.0,
            avg_loss_pips=0.0,
        )
        return metrics, pd.DataFrame(rows)

    pnls_arr = np.array(pnls, dtype=float)
    equity = np.cumsum(pnls_arr)

    gross_profit = float(np.sum(pnls_arr[pnls_arr > 0]))
    gross_loss = float(np.sum(-pnls_arr[pnls_arr < 0]))
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_win = float(np.mean(pnls_arr[pnls_arr > 0])) if np.any(pnls_arr > 0) else 0.0
    avg_loss = float(np.mean(pnls_arr[pnls_arr < 0])) if np.any(pnls_arr < 0) else 0.0

    metrics = BacktestMetrics(
        trades=int(trades),
        winrate=float(wins / trades),
        expectancy_net_pips=float(np.mean(pnls_arr)),
        profit_factor=pf,
        gross_profit_pips=gross_profit,
        gross_loss_pips=gross_loss,
        max_drawdown_pips=_max_drawdown(equity),
        avg_win_pips=avg_win,
        avg_loss_pips=avg_loss,
    )
    return metrics, pd.DataFrame(rows)
