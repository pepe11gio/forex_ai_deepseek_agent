# src/analysis/signal_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import ProjectConfig
from src.features.normalizer import FeatureScaler
from src.features.windowing import build_multiscale_windows
from src.modeling.trainer import load_feature_columns, load_trained_model
from src.evaluation.threshold_search import decide_actions


@dataclass(frozen=True)
class TradeSignal:
    time: str
    action: int                 # -1 short, 0 none, +1 long
    side: str                   # "SHORT"|"LONG"|"NONE"
    p_long: float
    p_short: float
    p_min: float
    tp_net_pips: float
    sl_net_pips: float
    cost_pips: float
    rr_t: float
    horizon_t: int
    expectancy_long_pips: float
    expectancy_short_pips: float

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "action": self.action,
            "side": self.side,
            "p_long": self.p_long,
            "p_short": self.p_short,
            "p_min": self.p_min,
            "tp_net_pips": self.tp_net_pips,
            "sl_net_pips": self.sl_net_pips,
            "cost_pips": self.cost_pips,
            "rr_t": self.rr_t,
            "horizon_t": self.horizon_t,
            "expectancy_long_pips": self.expectancy_long_pips,
            "expectancy_short_pips": self.expectancy_short_pips,
        }


def _load_scaler(cfg: ProjectConfig, run_name: str) -> FeatureScaler:
    from src.modeling.trainer import _export_paths  # local import

    paths = _export_paths(cfg, run_name)
    sc = FeatureScaler(cfg.derived.scaler_type)
    sc.load(paths["scaler_path"])
    return sc


def generate_signal(
    cfg: ProjectConfig,
    df_full: pd.DataFrame,
    run_name: str,
    p_min: float,
    t: Optional[int] = None,
) -> TradeSignal:
    """
    Genera un segnale su una barra t (default ultima barra).
    df_full deve contenere:
      - feature columns
      - colonne labeling: tp_net_pips, sl_net_pips, cost_pips, rr_t, horizon_t, valid_label
    """
    if t is None:
        t = len(df_full) - 1

    model = load_trained_model(cfg, run_name=run_name)
    feature_cols = load_feature_columns(f"{cfg.model_dir}/exports/{run_name}/feature_columns.json")
    scaler = _load_scaler(cfg, run_name)

    # scala
    df_scaled = scaler.transform(df_full)

    # robustezza: NaN check sugli ultimi max_window step
    wmax = int(max(cfg.windows.windows)) if cfg.windows.windows else 0
    tail = df_scaled.iloc[max(0, t - wmax + 1) : t + 1]
    if tail[feature_cols].isna().any().any():
        tm = str(df_full["time"].iat[t]) if "time" in df_full.columns else ""
        return TradeSignal(
            time=tm,
            action=0,
            side="NONE",
            p_long=0.0,
            p_short=0.0,
            p_min=float(p_min),
            tp_net_pips=float("nan"),
            sl_net_pips=float("nan"),
            cost_pips=float(cfg.market.spread_pips),
            rr_t=float("nan"),
            horizon_t=int(cfg.market.horizon_bars_fixed),
            expectancy_long_pips=float("nan"),
            expectancy_short_pips=float("nan"),
        )

    # windowing: costruisci dataset e prendi sample corrispondente a t
    ds = build_multiscale_windows(df_scaled, feature_cols, cfg)

    # trova indice sample che punta a t
    # ds.t_index contiene gli indici "t" del df_scaled usati come label/target
    pos = np.where(ds.t_index == t)[0]
    if pos.size == 0:
        # non abbastanza storico per finestra
        tm = str(df_full["time"].iat[t]) if "time" in df_full.columns else ""
        return TradeSignal(
            time=tm,
            action=0,
            side="NONE",
            p_long=0.0,
            p_short=0.0,
            p_min=float(p_min),
            tp_net_pips=float("nan"),
            sl_net_pips=float("nan"),
            cost_pips=float(cfg.market.spread_pips),
            rr_t=float("nan"),
            horizon_t=int(cfg.market.horizon_bars_fixed),
            expectancy_long_pips=float("nan"),
            expectancy_short_pips=float("nan"),
        )

    j = int(pos[-1])

    X_list = [x[j : j + 1] for x in ds.X_list]
    p = model.predict(X_list, verbose=0).astype(np.float32)[0]  # (2,)

    # decision su singola barra usando stessa logica del backtest
    df_one = df_full.iloc[[t]].copy()
    # per decide_actions serve valid_label e colonne tp/sl/cost
    if "valid_label" not in df_one.columns:
        df_one["valid_label"] = True
    if "cost_pips" not in df_one.columns:
        df_one["cost_pips"] = float(cfg.market.spread_pips)

    action = int(decide_actions(df_one, p.reshape(1, 2), cfg, p_min=float(p_min))[0])
    side = "LONG" if action == 1 else "SHORT" if action == -1 else "NONE"

    tp = float(pd.to_numeric(df_one["tp_net_pips"], errors="coerce").iat[0]) if "tp_net_pips" in df_one.columns else float("nan")
    sl = float(pd.to_numeric(df_one["sl_net_pips"], errors="coerce").iat[0]) if "sl_net_pips" in df_one.columns else float("nan")
    cost = float(pd.to_numeric(df_one["cost_pips"], errors="coerce").iat[0])

    p_long = float(p[0])
    p_short = float(p[1])

    E_long = p_long * (tp - cost) - (1.0 - p_long) * (sl + cost) if np.isfinite(tp) and np.isfinite(sl) else float("nan")
    E_short = p_short * (tp - cost) - (1.0 - p_short) * (sl + cost) if np.isfinite(tp) and np.isfinite(sl) else float("nan")

    rr_t = float(pd.to_numeric(df_one["rr_t"], errors="coerce").iat[0]) if "rr_t" in df_one.columns else float(cfg.market.rr_fixed)
    h_t = int(pd.to_numeric(df_one["horizon_t"], errors="coerce").iat[0]) if "horizon_t" in df_one.columns else int(cfg.market.horizon_bars_fixed)

    tm = str(df_full["time"].iat[t]) if "time" in df_full.columns else ""
    return TradeSignal(
        time=tm,
        action=action,
        side=side,
        p_long=p_long,
        p_short=p_short,
        p_min=float(p_min),
        tp_net_pips=tp,
        sl_net_pips=sl,
        cost_pips=cost,
        rr_t=rr_t,
        horizon_t=h_t,
        expectancy_long_pips=float(E_long),
        expectancy_short_pips=float(E_short),
    )
