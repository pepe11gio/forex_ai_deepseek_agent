# src/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# =========================
# CSV COLUMN MAPPING
# =========================
@dataclass(frozen=True)
class ColumnMapping:
    # OHLCV
    time: str = "time"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"

    # MT5 indicators (input CSV)
    ema_20: str = "EMA_20"
    ema_50: str = "EMA_50"
    rsi_14: str = "RSI_14"

    # ATR: preferred in pips; fallback in price
    atr_14_pips: str = "ATR_14_PIPS"
    atr_14_price: str = "ATR_14"

    adx_14: str = "ADX_14"
    plus_di_14: str = "PLUS_DI_14"
    minus_di_14: str = "MINUS_DI_14"


# =========================
# MARKET CONFIG
# =========================
@dataclass(frozen=True)
class MarketConfig:
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    pip_size: float = 0.0001

    # spread/costi (pips)
    # Interpretiamo spread_pips come costo "round-turn" per trade.
    spread_pips: float = 1.5

    # ---- TP/SL policy ----
    # SL in ATR (sempre)
    sl_atr_k: float = 1.0

    # RR e H: fixed o dynamic
    rr_mode: str = "dynamic"        # "fixed" | "dynamic"
    horizon_mode: str = "dynamic"   # "fixed" | "dynamic"

    # default (se fixed, o fallback)
    rr_fixed: float = 4.0
    horizon_bars_fixed: int = 48

    # bounds per dynamic
    rr_min: float = 2.5
    rr_max: float = 5.5
    horizon_bars_min: int = 24
    horizon_bars_max: int = 96

    # risk filters
    min_sl_net_multiple_of_spread: float = 2.0
    expectancy_buffer_pips: float = 0.0

    # walk-forward purge/embargo
    # embargo = max(horizon_bars_max, windows_max-1) di default (vedi walk_forward.py)
    extra_embargo_bars: int = 0


# =========================
# DERIVED FEATURES
# =========================
@dataclass(frozen=True)
class DerivedFeatureConfig:
    return_lags: Tuple[int, ...] = (1, 3, 6, 12, 24)
    vol_windows: Tuple[int, ...] = (24, 48)

    include_candle_anatomy: bool = True
    include_time_features: bool = True

    include_atr_percentile: bool = True
    atr_percentile_window: int = 200

    scaler_type: str = "zscore"            # "zscore" | "robust"
    nan_policy: str = "drop"               # "drop" | "ffill_then_drop"


# =========================
# SUPPORT / RESISTANCE
# =========================
@dataclass(frozen=True)
class SupportResistanceConfig:
    enabled: bool = False

    # "fast" (prod) | "slow" (diagnostic)
    mode: str = "fast"

    swing_window: int = 5
    lookback_bars: int = 3000

    bin_size_pips: float = 10.0
    use_dynamic_bin: bool = True
    atr_bin_mult: float = 0.30

    top_k_levels: int = 3

    # NEW: recalc cadence for fast-causal
    recalc_every: int = 10

    # slow-only diagnostics
    progress_every: int = 500

    # plotting
    plot_levels: bool = False
    plot_last_n_bars: int = 800


# =========================
# BREAKOUT FEATURES
# =========================
@dataclass(frozen=True)
class BreakoutFeatureConfig:
    enabled: bool = False

    atr_ratio_window: int = 24
    range_atr_sma_window: int = 24

    near_thr_atr: float = 0.25
    pressure_lookback: int = 50

    ret_sum_windows: Tuple[int, ...] = (6, 12)

    break_recent_lookback: int = 24
    break_margin_atr: float = 0.10


# =========================
# WINDOWING
# =========================
@dataclass(frozen=True)
class WindowConfig:
    windows: Tuple[int, ...] = (48, 128, 256)
    stride: int = 1
    require_full_windows: bool = True


# =========================
# LABELING
# =========================
@dataclass(frozen=True)
class LabelConfig:
    # lascia qui solo flag e controlli; i parametri SL sono in MarketConfig
    compute_no_touch_flag: bool = True

    # NEW: quando t+H supera dataset, y diventa invalid (NaN) invece che 0
    invalid_as_nan: bool = True


# =========================
# MODEL CONFIG
# =========================
@dataclass(frozen=True)
class ModelConfig:
    conv_filters: int = 32
    conv_kernel_size: int = 5
    conv_dropout: float = 0.15

    gru_units: int = 32
    gru_dropout: float = 0.15

    dense_units: int = 64
    head_dropout: float = 0.20

    output_dim: int = 2


# =========================
# TRAIN CONFIG
# =========================
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42

    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3

    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4

    loss_type: str = "bce"
    use_class_weights: bool = False


# =========================
# WALK FORWARD
# =========================
@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 18000
    val_bars: int = 3500
    test_bars: int = 3500
    step_bars: int = 3500
    min_folds: int = 2


# =========================
# PROJECT CONFIG
# =========================
@dataclass(frozen=True)
class ProjectConfig:
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    model_dir: str = "models"
    reports_dir: str = "reports"

    raw_csv_filename: str = "EURUSD_H1.csv"

    mapping: ColumnMapping = field(default_factory=ColumnMapping)
    market: MarketConfig = field(default_factory=MarketConfig)

    derived: DerivedFeatureConfig = field(default_factory=DerivedFeatureConfig)
    sr: SupportResistanceConfig = field(default_factory=SupportResistanceConfig)
    breakout: BreakoutFeatureConfig = field(default_factory=BreakoutFeatureConfig)

    windows: WindowConfig = field(default_factory=WindowConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    verbose: bool = True


def default_config() -> ProjectConfig:
    return ProjectConfig()
