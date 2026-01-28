# src/io/data_loader.py
from __future__ import annotations

import os
import glob
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import ProjectConfig


class DataLoaderError(Exception):
    pass


def _parse_time_column(series: pd.Series) -> pd.Series:
    """
    Parse robusto per:
    - ISO string
    - epoch seconds/ms
    """
    dt = pd.to_datetime(series, errors="coerce", utc=False)

    if dt.isna().mean() > 0.5:
        s_num = pd.to_numeric(series, errors="coerce")
        if s_num.notna().mean() > 0.5:
            median = float(s_num.dropna().median())
            unit = "ms" if median > 1e12 else "s"
            dt2 = pd.to_datetime(s_num, errors="coerce", unit=unit, utc=False)
            if dt2.isna().mean() < dt.isna().mean():
                dt = dt2

    return dt


def _validate_required_columns(df: pd.DataFrame, cfg: ProjectConfig) -> None:
    m = cfg.mapping

    required_ohlc = [m.time, m.open, m.high, m.low, m.close]
    missing_ohlc = [c for c in required_ohlc if c not in df.columns]
    if missing_ohlc:
        raise DataLoaderError(f"Colonne OHLC mancanti: {missing_ohlc}")

    required_inds = [m.ema_20, m.ema_50, m.rsi_14, m.adx_14, m.plus_di_14, m.minus_di_14]
    missing_inds = [c for c in required_inds if c not in df.columns]
    if missing_inds:
        raise DataLoaderError(
            f"Indicatori MT5 mancanti: {missing_inds}. "
            f"Attesi almeno: EMA_20, EMA_50, RSI_14, ADX_14, PLUS_DI_14, MINUS_DI_14"
        )

    has_atr_pips = m.atr_14_pips in df.columns
    has_atr_price = m.atr_14_price in df.columns
    if not (has_atr_pips or has_atr_price):
        raise DataLoaderError("Manca ATR: serve ATR_14_PIPS oppure ATR_14 (in prezzo).")


def _standardize_one_df(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """
    Standardizza un singolo CSV in colonne interne:
    time, open, high, low, close, volume,
    ema_20, ema_50, rsi_14, atr_14_pips, adx_14, plus_di_14, minus_di_14
    """
    m = cfg.mapping
    _validate_required_columns(df, cfg)

    # Volume optional
    if m.volume not in df.columns:
        df[m.volume] = 0.0

    has_atr_pips = m.atr_14_pips in df.columns
    has_atr_price = m.atr_14_price in df.columns

    # Keep only relevant columns
    cols = [
        m.time, m.open, m.high, m.low, m.close, m.volume,
        m.ema_20, m.ema_50, m.rsi_14,
        m.adx_14, m.plus_di_14, m.minus_di_14,
        m.atr_14_pips if has_atr_pips else m.atr_14_price
    ]
    df = df[cols].copy()

    # Parse time
    df[m.time] = _parse_time_column(df[m.time])
    df = df.dropna(subset=[m.time])

    # Numeric cast
    for c in cols:
        if c != m.time:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[m.open, m.high, m.low, m.close])

    # Rename to internal standard
    rename_map = {
        m.time: "time",
        m.open: "open",
        m.high: "high",
        m.low: "low",
        m.close: "close",
        m.volume: "volume",
        m.ema_20: "ema_20",
        m.ema_50: "ema_50",
        m.rsi_14: "rsi_14",
        m.adx_14: "adx_14",
        m.plus_di_14: "plus_di_14",
        m.minus_di_14: "minus_di_14",
    }
    if has_atr_pips:
        rename_map[m.atr_14_pips] = "atr_14_pips"
    else:
        rename_map[m.atr_14_price] = "atr_14_price"

    df = df.rename(columns=rename_map)

    # Fix OHLC inconsistencies conservatively
    bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    bad = bad_high | bad_low
    if bad.any():
        df.loc[bad, "high"] = df.loc[bad, ["high", "open", "close", "low"]].max(axis=1)
        df.loc[bad, "low"] = df.loc[bad, ["low", "open", "close", "high"]].min(axis=1)

    # Volume
    df["volume"] = df["volume"].fillna(0.0)

    # ATR in pips
    if "atr_14_pips" not in df.columns:
        df["atr_14_pips"] = df["atr_14_price"] / cfg.market.pip_size
        df = df.drop(columns=["atr_14_price"])

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def _discover_csv_files(directory: str) -> List[str]:
    # include .csv and .CSV
    files = glob.glob(os.path.join(directory, "*.csv")) + glob.glob(os.path.join(directory, "*.CSV"))
    files = sorted(set(files))
    return files


def load_mt5_csvs(
    cfg: ProjectConfig,
    directory: Optional[str] = None,
    pattern: Optional[str] = None,
) -> pd.DataFrame:
    """
    Carica e concatena TUTTI i CSV trovati nella directory.
    - directory: default cfg.data_raw_dir
    - pattern: opzionale, es. "EURUSD_H1_*.csv"

    Gestisce:
    - file multipli, anche con overlap
    - dedup su time (keep last)
    - sorting per time
    """
    raw_dir = directory or cfg.data_raw_dir
    if not os.path.isdir(raw_dir):
        raise DataLoaderError(f"Directory data_raw_dir non trovata: {raw_dir}")

    if pattern:
        files = sorted(set(glob.glob(os.path.join(raw_dir, pattern))))
    else:
        files = _discover_csv_files(raw_dir)

    if not files:
        raise DataLoaderError(f"Nessun CSV trovato in {raw_dir} (pattern={pattern})")

    dfs: List[pd.DataFrame] = []
    errors: List[str] = []

    for fp in files:
        try:
            df0 = pd.read_csv(fp)
            df1 = _standardize_one_df(df0, cfg)
            dfs.append(df1)
            if cfg.verbose:
                print(f"[DataLoader] OK  {os.path.basename(fp)} -> {len(df1):,} rows")
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")

    if not dfs:
        msg = "Impossibile caricare i CSV. Errori:\n" + "\n".join(errors[:20])
        raise DataLoaderError(msg)

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Clean NaT, sort, dedupe
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    # Remove non-positive prices (corrupt)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)].reset_index(drop=True)

    if cfg.verbose:
        print(f"[DataLoader] Loaded total {len(df):,} rows from {len(dfs)} files (dir={raw_dir})")
        print(f"[DataLoader] Time range: {df['time'].min()} -> {df['time'].max()}")
        if errors:
            print(f"[DataLoader] WARNING: {len(errors)} file(s) skipped due to errors.")
            for line in errors[:5]:
                print(f"  - {line}")

    return df
