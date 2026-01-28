# src/evaluation/walk_forward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.config import ProjectConfig


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def _compute_embargo(cfg: ProjectConfig) -> int:
    """
    Embargo minimo per evitare leakage:
    - labels usano forward fino a H (dinamico -> usa H_max)
    - windowing usa max_window-1 lookback
    """
    H = int(cfg.market.horizon_bars_max if cfg.market.horizon_mode.lower() == "dynamic" else cfg.market.horizon_bars_fixed)
    wmax = int(max(cfg.windows.windows)) if cfg.windows.windows else 0
    embargo = max(H, max(0, wmax - 1))
    embargo += int(cfg.market.extra_embargo_bars)
    return int(max(0, embargo))


def generate_walk_forward_splits(df: pd.DataFrame, cfg: ProjectConfig) -> List[WalkForwardSplit]:
    wc = cfg.walk_forward
    n = len(df)

    train_bars = int(wc.train_bars)
    val_bars = int(wc.val_bars)
    test_bars = int(wc.test_bars)
    step_bars = int(wc.step_bars)

    embargo = _compute_embargo(cfg)

    splits: List[WalkForwardSplit] = []
    fold = 0
    start = 0

    # Schema:
    # train: [start, start+train)
    # embargo gap
    # val:   [..)
    # embargo gap
    # test:  [..)
    while True:
        train_start = start
        train_end = train_start + train_bars

        val_start = train_end + embargo
        val_end = val_start + val_bars

        test_start = val_end + embargo
        test_end = test_start + test_bars

        if test_end > n:
            break

        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold += 1
        start += step_bars

        if fold >= int(wc.min_folds) and (start + train_bars + embargo + val_bars + embargo + test_bars > n):
            break

    return splits


def describe_split(df: pd.DataFrame, s: WalkForwardSplit) -> str:
    t0 = df["time"].iloc[s.train_start] if "time" in df.columns else None
    t1 = df["time"].iloc[s.train_end - 1] if "time" in df.columns else None
    v0 = df["time"].iloc[s.val_start] if "time" in df.columns else None
    v1 = df["time"].iloc[s.val_end - 1] if "time" in df.columns else None
    te0 = df["time"].iloc[s.test_start] if "time" in df.columns else None
    te1 = df["time"].iloc[s.test_end - 1] if "time" in df.columns else None

    return (
        f"Fold {s.fold}: "
        f"train[{s.train_start}:{s.train_end}) {t0}->{t1} | "
        f"val[{s.val_start}:{s.val_end}) {v0}->{v1} | "
        f"test[{s.test_start}:{s.test_end}) {te0}->{te1}"
    )
