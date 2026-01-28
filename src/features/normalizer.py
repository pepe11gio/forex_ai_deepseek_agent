# src/features/normalizer.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import ProjectConfig


@dataclass
class ScalerState:
    scaler_type: str
    columns: List[str]
    params: Dict[str, Dict[str, float]]  # col -> stats


class FeatureScaler:
    """
    Scaler semplice e deterministico (no sklearn).
    Fit SOLO su train.
    """

    def __init__(self, scaler_type: str = "zscore"):
        if scaler_type not in ("zscore", "robust"):
            raise ValueError("scaler_type must be 'zscore' or 'robust'")
        self.scaler_type = scaler_type
        self.columns: List[str] = []
        self.params: Dict[str, Dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "FeatureScaler":
        self.columns = list(columns)
        self.params = {}

        for c in self.columns:
            x = pd.to_numeric(df[c], errors="coerce").astype(float)

            if self.scaler_type == "zscore":
                mean = float(np.nanmean(x))
                std = float(np.nanstd(x))
                if std == 0.0 or not np.isfinite(std):
                    std = 1.0
                self.params[c] = {"mean": mean, "std": std}

            else:  # robust
                med = float(np.nanmedian(x))
                q1 = float(np.nanpercentile(x, 25))
                q3 = float(np.nanpercentile(x, 75))
                iqr = q3 - q1
                if iqr == 0.0 or not np.isfinite(iqr):
                    iqr = 1.0
                self.params[c] = {"median": med, "iqr": iqr}

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.columns or not self.params:
            raise RuntimeError("Scaler not fitted or loaded.")

        out = df.copy()
        for c in self.columns:
            x = pd.to_numeric(out[c], errors="coerce").astype(float)

            if self.scaler_type == "zscore":
                mean = self.params[c]["mean"]
                std = self.params[c]["std"]
                out[c] = (x - mean) / std
            else:
                med = self.params[c]["median"]
                iqr = self.params[c]["iqr"]
                out[c] = (x - med) / iqr

        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def to_state(self) -> ScalerState:
        return ScalerState(
            scaler_type=self.scaler_type,
            columns=list(self.columns),
            params={k: dict(v) for k, v in self.params.items()},
        )

    @staticmethod
    def from_state(state: ScalerState) -> "FeatureScaler":
        sc = FeatureScaler(state.scaler_type)
        sc.columns = list(state.columns)
        sc.params = {k: dict(v) for k, v in state.params.items()}
        return sc

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = self.to_state()
        payload = {
            "scaler_type": state.scaler_type,
            "columns": state.columns,
            "params": state.params,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load(filepath: str) -> "FeatureScaler":
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        state = ScalerState(
            scaler_type=payload["scaler_type"],
            columns=list(payload["columns"]),
            params={k: dict(v) for k, v in payload["params"].items()},
        )
        return FeatureScaler.from_state(state)


def scaler_default_path(cfg: ProjectConfig, run_name: str = "default") -> str:
    return os.path.join(cfg.model_dir, "exports", run_name, "feature_scaler.json")
