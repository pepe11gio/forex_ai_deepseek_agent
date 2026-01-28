# src/features/support_resistance.py
from __future__ import annotations

import pandas as pd

from src.config import ProjectConfig
from src.features.support_resistance_fast import add_sr_fast
from src.features.support_resistance_slow import add_sr_slow


def add_support_resistance(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    if not cfg.sr.enabled:
        return df

    mode = (cfg.sr.mode or "fast").lower()
    if mode == "slow":
        return add_sr_slow(df, cfg)
    return add_sr_fast(df, cfg)
