# src/features/windowing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig


@dataclass
class WindowedDataset:
    """
    Container dataset multiscale pronto per TF.
    """
    X_list: List[np.ndarray]   # each: (samples, window, n_features)
    y: np.ndarray              # (samples, 2)
    t_index: np.ndarray        # (samples,) index in original df
    feature_columns: List[str]
    windows: Tuple[int, ...]


def _ensure_float32(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        return a.astype(np.float32)
    return a


def build_multiscale_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    cfg: ProjectConfig,
    target_columns: Tuple[str, str] = ("y_long", "y_short"),
) -> WindowedDataset:
    """
    Crea dataset multiscale:
    - Per ogni t, crea finestre [t-N+1 .. t] per ciascun N in cfg.windows.windows
    - y(t) = [y_long(t), y_short(t)]

    Requisiti:
    - df ordinato temporalmente
    - df contiene target_columns
    - df non deve contenere NaN nelle colonne feature/target (consigliato: apply_nan_policy prima)
    """
    windows = tuple(cfg.windows.windows)
    stride = int(cfg.windows.stride)

    if stride <= 0:
        raise ValueError("stride deve essere >= 1")

    for tc in target_columns:
        if tc not in df.columns:
            raise ValueError(f"Target mancante: {tc}")

    # Converti feature matrix
    X_all = df[feature_columns].to_numpy(dtype=np.float32)  # (T, F)
    y_all = df[list(target_columns)].to_numpy(dtype=np.int8)  # (T, 2)

    T, F = X_all.shape
    max_w = max(windows)

    # Indici validi: t deve avere abbastanza storia per max window
    start_t = max_w - 1
    # e i target devono essere validi (es. labeling già mette 0 negli ultimi H; ok comunque)
    t_candidates = np.arange(start_t, T, stride, dtype=np.int32)

    # Pre-alloc: numero sample = len(t_candidates)
    n_samples = len(t_candidates)

    X_list: List[np.ndarray] = []
    for w in windows:
        Xw = np.empty((n_samples, w, F), dtype=np.float32)
        X_list.append(Xw)

    y = np.empty((n_samples, 2), dtype=np.int8)
    t_index = np.empty((n_samples,), dtype=np.int32)

    # Fill
    for i, t in enumerate(t_candidates):
        # Targets at t
        y[i, :] = y_all[t, :]
        t_index[i] = t

        # Each window
        for j, w in enumerate(windows):
            start = t - w + 1
            end = t + 1
            X_list[j][i, :, :] = X_all[start:end, :]

    # Optionally enforce require_full_windows (già garantito da start_t)
    # Sanity
    for j in range(len(X_list)):
        if np.isnan(X_list[j]).any():
            raise ValueError("NaN presenti nelle finestre: applica apply_nan_policy o controlla feature.")

    return WindowedDataset(
        X_list=[_ensure_float32(x) for x in X_list],
        y=y.astype(np.float32),  # per BCE in TF è più comodo float32
        t_index=t_index,
        feature_columns=list(feature_columns),
        windows=windows,
    )


def split_by_index(
    dataset: WindowedDataset,
    idx_from: int,
    idx_to: int,
) -> WindowedDataset:
    """
    Slice del dataset per range [idx_from:idx_to] sui sample (non sul tempo originale).
    Utile se fai split dopo windowing.
    """
    X_list = [x[idx_from:idx_to] for x in dataset.X_list]
    y = dataset.y[idx_from:idx_to]
    t_index = dataset.t_index[idx_from:idx_to]

    return WindowedDataset(
        X_list=X_list,
        y=y,
        t_index=t_index,
        feature_columns=dataset.feature_columns,
        windows=dataset.windows,
    )
