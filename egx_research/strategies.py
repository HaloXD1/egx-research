from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from egx_research.indicators import adx, aroon, atr, bollinger_bands, cci, cmf, ema, kama, macd, mfi, moving_average, obv, rsi, sma


PARAMETER_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "trend": {
        "fast_ma": {"type": "int", "low": 10, "high": 80, "step": 1},
        "slow_ma": {"type": "int", "low": 50, "high": 250, "step": 1},
        "ma_type": {"type": "categorical", "choices": ["SMA", "EMA"]},
        "adx_len": {"type": "int", "low": 7, "high": 28, "step": 1},
        "adx_threshold": {"type": "float", "low": 15.0, "high": 35.0, "step": 0.5},
        "atr_stop": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
    },
    "mean_reversion": {
        "regime_ma": {"type": "int", "low": 100, "high": 250, "step": 1},
        "rsi_len": {"type": "int", "low": 2, "high": 20, "step": 1},
        "rsi_entry": {"type": "float", "low": 15.0, "high": 40.0, "step": 0.5},
        "rsi_exit": {"type": "float", "low": 45.0, "high": 70.0, "step": 0.5},
        "bb_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "bb_std": {"type": "float", "low": 1.5, "high": 3.0, "step": 0.1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
    },
    "breakout": {
        "entry_lookback": {"type": "int", "low": 20, "high": 120, "step": 1},
        "exit_lookback": {"type": "int", "low": 5, "high": 60, "step": 1},
        "regime_ma": {"type": "int", "low": 100, "high": 250, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 4.0, "step": 0.1},
    },
    "fdi_supertrend": {
        "per": {"type": "int", "low": 10, "high": 60, "step": 1},
        "speed": {"type": "int", "low": 5, "high": 40, "step": 1},
        "mult": {"type": "float", "low": 1.0, "high": 6.0, "step": 0.1},
        "adapt": {"type": "categorical", "choices": [True, False]},
    },
    "kama_cci_atr_allocation": {
        "kama_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 60, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "cci_threshold": {"type": "float", "low": 0.0, "high": 100.0, "step": 5.0},
        "allocation_ladder": {"type": "categorical", "choices": ["100_50_0", "100_25_0", "100_60_20"]},
        "atr_len": {"type": "int", "low": 7, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "dca_tactical_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.7, 0.75, 0.8]},
        "kama_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 60, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "cci_threshold": {"type": "float", "low": 0.0, "high": 50.0, "step": 5.0},
        "sleeve_ladder": {"type": "categorical", "choices": ["100_60_20", "100_50_0", "100_25_0"]},
        "atr_len": {"type": "int", "low": 7, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "dca_zone_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.7, 0.75, 0.8]},
        "kama_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 60, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "buy_mild": {"type": "float", "low": -50.0, "high": -5.0, "step": 5.0},
        "buy_deep": {"type": "float", "low": -200.0, "high": -50.0, "step": 5.0},
        "trim_mild": {"type": "float", "low": 50.0, "high": 150.0, "step": 5.0},
        "trim_hard": {"type": "float", "low": 100.0, "high": 250.0, "step": 5.0},
        "trend_buffer_atr": {"type": "float", "low": 0.0, "high": 2.0, "step": 0.1},
        "trim_buffer_atr": {"type": "float", "low": 0.0, "high": 3.0, "step": 0.1},
        "zone_profile": {"type": "categorical", "choices": ["100_75_50_25_0", "100_60_40_20_0", "100_50_25_10_0"]},
        "atr_len": {"type": "int", "low": 7, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "dca_pullback_topup": {
        "core_weight": {"type": "categorical", "choices": [0.7, 0.75, 0.8]},
        "reserve_weight": {"type": "categorical", "choices": [0.2, 0.25, 0.3]},
        "kama_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 60, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "buy_mild": {"type": "float", "low": -100.0, "high": -5.0, "step": 5.0},
        "buy_deep": {"type": "float", "low": -250.0, "high": -50.0, "step": 5.0},
        "trend_buffer_atr": {"type": "float", "low": 0.0, "high": 2.0, "step": 0.1},
        "topup_profile": {"type": "categorical", "choices": ["25_50_100", "33_66_100", "50_75_100"]},
        "atr_len": {"type": "int", "low": 7, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "dca_pullback_only": {
        "kama_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 60, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "buy_threshold": {"type": "float", "low": -150.0, "high": -5.0, "step": 5.0},
        "trend_buffer_atr": {"type": "float", "low": 0.0, "high": 2.0, "step": 0.1},
        "atr_len": {"type": "int", "low": 7, "high": 30, "step": 1},
    },
    "hierarchy_combo": {
        "hierarchy": {"type": "categorical", "choices": ["trend_only", "trend_momentum", "trend_volume", "trend_momentum_volume"]},
        "trend_indicator": {"type": "categorical", "choices": ["supertrend", "kama", "aroon"]},
        "momentum_indicator": {"type": "categorical", "choices": ["rsi", "macd", "cci"]},
        "volume_indicator": {"type": "categorical", "choices": ["cmf", "mfi", "obv"]},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_dynamic_momentum": {
        "stoch_len": {"type": "int", "low": 34, "high": 144, "step": 1},
        "kd_smooth": {"type": "int", "low": 2, "high": 7, "step": 1},
        "mom_fast": {"type": "int", "low": 8, "high": 21, "step": 1},
        "mom_slow": {"type": "int", "low": 21, "high": 55, "step": 1},
        "mom_signal": {"type": "int", "low": 3, "high": 13, "step": 1},
        "trend_len": {"type": "int", "low": 55, "high": 200, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_multi_bbands": {
        "bb_len": {"type": "int", "low": 14, "high": 40, "step": 1},
        "inner_mult": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.1},
        "outer_mult": {"type": "float", "low": 2.0, "high": 3.6, "step": 0.1},
        "trend_ma": {"type": "int", "low": 55, "high": 200, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_zlema_band": {
        "zlema_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "band_mult": {"type": "float", "low": 0.5, "high": 2.5, "step": 0.1},
        "trend_ma": {"type": "int", "low": 55, "high": 200, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_ichimoku": {
        "tenkan_len": {"type": "int", "low": 7, "high": 12, "step": 1},
        "kijun_len": {"type": "int", "low": 20, "high": 34, "step": 1},
        "senkou_b_len": {"type": "int", "low": 44, "high": 78, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_ravi": {
        "fast_len": {"type": "int", "low": 5, "high": 21, "step": 1},
        "slow_len": {"type": "int", "low": 20, "high": 60, "step": 1},
        "bias_len": {"type": "int", "low": 100, "high": 250, "step": 1},
        "ravi_entry": {"type": "float", "low": 0.5, "high": 5.0, "step": 0.1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_cci_rsi": {
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "rsi_len": {"type": "int", "low": 5, "high": 21, "step": 1},
        "rsi_signal": {"type": "int", "low": 3, "high": 10, "step": 1},
        "cci_entry": {"type": "float", "low": -150.0, "high": 0.0, "step": 5.0},
        "cci_exit": {"type": "float", "low": 0.0, "high": 200.0, "step": 5.0},
        "trend_ma": {"type": "int", "low": 55, "high": 200, "step": 1},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
    "blackcat_superj": {
        "stoch_len": {"type": "int", "low": 9, "high": 34, "step": 1},
        "kd_smooth": {"type": "int", "low": 2, "high": 5, "step": 1},
        "j_smooth": {"type": "int", "low": 3, "high": 10, "step": 1},
        "trigger_len": {"type": "int", "low": 3, "high": 13, "step": 1},
        "trend_ma": {"type": "int", "low": 55, "high": 200, "step": 1},
        "oversold": {"type": "float", "low": 5.0, "high": 30.0, "step": 1.0},
        "atr_stop": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 0.5, "high": 4.0, "step": 0.1},
    },
}

TREND_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "supertrend": {
        "st_len": {"type": "int", "low": 7, "high": 30, "step": 1},
        "st_mult": {"type": "float", "low": 1.5, "high": 5.0, "step": 0.1},
    },
    "kama": {
        "kama_len": {"type": "int", "low": 10, "high": 50, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 5, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 60, "step": 1},
    },
    "aroon": {
        "aroon_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "aroon_threshold": {"type": "float", "low": 50.0, "high": 90.0, "step": 1.0},
    },
}

MOMENTUM_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "rsi": {
        "mom_rsi_len": {"type": "int", "low": 5, "high": 30, "step": 1},
        "mom_rsi_threshold": {"type": "float", "low": 45.0, "high": 65.0, "step": 1.0},
    },
    "macd": {
        "macd_fast": {"type": "int", "low": 5, "high": 20, "step": 1},
        "macd_slow": {"type": "int", "low": 15, "high": 40, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 15, "step": 1},
    },
    "cci": {
        "cci_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "cci_threshold": {"type": "float", "low": 0.0, "high": 150.0, "step": 5.0},
    },
}

VOLUME_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "cmf": {
        "cmf_len": {"type": "int", "low": 10, "high": 40, "step": 1},
        "cmf_threshold": {"type": "float", "low": -0.1, "high": 0.2, "step": 0.01},
    },
    "mfi": {
        "mfi_len": {"type": "int", "low": 5, "high": 30, "step": 1},
        "mfi_threshold": {"type": "float", "low": 40.0, "high": 70.0, "step": 1.0},
    },
    "obv": {
        "obv_ema_len": {"type": "int", "low": 5, "high": 60, "step": 1},
    },
}

HIERARCHY_ACTIVE = {
    "trend_only": {"momentum": False, "volume": False},
    "trend_momentum": {"momentum": True, "volume": False},
    "trend_volume": {"momentum": False, "volume": True},
    "trend_momentum_volume": {"momentum": True, "volume": True},
}

ALLOCATION_LADDERS = {
    "100_50_0": (1.0, 0.5, 0.0),
    "100_25_0": (1.0, 0.25, 0.0),
    "100_60_20": (1.0, 0.6, 0.2),
}

ZONE_PROFILES = {
    "100_75_50_25_0": (1.0, 0.75, 0.5, 0.25, 0.0),
    "100_60_40_20_0": (1.0, 0.6, 0.4, 0.2, 0.0),
    "100_50_25_10_0": (1.0, 0.5, 0.25, 0.1, 0.0),
}

TOPUP_PROFILES = {
    "25_50_100": (0.25, 0.5, 1.0),
    "33_66_100": (0.33, 0.66, 1.0),
    "50_75_100": (0.5, 0.75, 1.0),
}


def _cast_param(spec: dict[str, Any], value: Any) -> Any:
    if spec["type"] == "int":
        return int(round(float(value)))
    if spec["type"] == "float":
        return float(value)
    first_choice = spec["choices"][0]
    if isinstance(first_choice, bool):
        return bool(value)
    if isinstance(first_choice, int):
        return int(round(float(value)))
    if isinstance(first_choice, float):
        return float(value)
    return str(value)


def _default_from_spec(spec: dict[str, Any]) -> Any:
    if spec["type"] == "int":
        return int(spec["low"])
    if spec["type"] == "float":
        return float(spec["low"])
    return spec["choices"][0]


def _rma_variable(values: pd.Series, lengths: pd.Series) -> pd.Series:
    output = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values)):
        value = float(values.iloc[i])
        length = max(1, int(round(float(lengths.iloc[i]))))
        if i == 0 or np.isnan(output[i - 1]):
            output[i] = value
        else:
            output[i] = (value - output[i - 1]) * (1.0 / length) + output[i - 1]
    return pd.Series(output, index=values.index, dtype=float)


def _fdi_speed(src: pd.Series, per: int, speed_in: int) -> pd.Series:
    values = src.astype(float).to_numpy()
    output = np.full(len(values), np.nan, dtype=float)
    if per <= 1:
        return pd.Series(np.full(len(values), float(max(1, speed_in))), index=src.index, dtype=float)

    for end in range(len(values)):
        start = end - per + 1
        if start < 0:
            continue
        window = values[start : end + 1]
        fmax = np.nanmax(window)
        fmin = np.nanmin(window)
        denom = fmax - fmin
        if denom == 0 or np.isnan(denom):
            output[end] = float(max(1, speed_in))
            continue

        norm = (window - fmin) / denom
        length = 0.0
        step = 1.0 / (per**2)
        for i in range(1, len(norm)):
            length += np.sqrt(((norm[i] - norm[i - 1]) ** 2) + step)
        if length <= 0:
            output[end] = float(max(1, speed_in))
            continue

        fdi = 1.0 + (np.log(length) + np.log(2.0)) / np.log(2.0 * per)
        denom_fdi = 2.0 - fdi
        if denom_fdi == 0:
            output[end] = float(max(1, speed_in))
            continue

        trail_dim = 1.0 / denom_fdi
        alpha = trail_dim / 2.0
        output[end] = float(max(1, int(round(speed_in * alpha))))

    return pd.Series(output, index=src.index, dtype=float).ffill().fillna(float(max(1, speed_in)))


def _rolling_midpoint(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    highest = high.rolling(window=window, min_periods=window).max()
    lowest = low.rolling(window=window, min_periods=window).min()
    return (highest + lowest) / 2.0


def _stochastic_rsv(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    highest = high.rolling(window=window, min_periods=window).max()
    lowest = low.rolling(window=window, min_periods=window).min()
    return 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)


def _vwma(series: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    numerator = (series * volume.fillna(0.0)).rolling(window=window, min_periods=window).sum()
    denominator = volume.fillna(0.0).rolling(window=window, min_periods=window).sum()
    return numerator / denominator.replace(0, np.nan)


def _zero_lag_ema(series: pd.Series, window: int) -> pd.Series:
    lag = max(1, int((window - 1) / 2))
    adjusted = series + (series - series.shift(lag))
    return ema(adjusted, window)


def _ravi(close: pd.Series, fast_len: int, slow_len: int, bias_len: int) -> pd.Series:
    fast = sma(close, fast_len)
    slow = sma(close, slow_len)
    bias = sma(close, bias_len)
    return 100.0 * (fast - slow) / bias.replace(0, np.nan)


def _pine_supertrend(
    src: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    factor: float,
    atr_periods: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_series = _rma_variable(tr, atr_periods)

    upper_raw = src + factor * atr_series
    lower_raw = src - factor * atr_series

    upper = np.full(len(src), np.nan, dtype=float)
    lower = np.full(len(src), np.nan, dtype=float)
    supertrend = np.full(len(src), np.nan, dtype=float)
    direction = np.full(len(src), np.nan, dtype=float)

    for i in range(len(src)):
        raw_upper = float(upper_raw.iloc[i])
        raw_lower = float(lower_raw.iloc[i])
        prev_upper = upper[i - 1] if i > 0 else raw_upper
        prev_lower = lower[i - 1] if i > 0 else raw_lower
        prev_close = float(close.iloc[i - 1]) if i > 0 else float(close.iloc[i])

        lower[i] = raw_lower if (raw_lower > prev_lower or prev_close < prev_lower) else prev_lower
        upper[i] = raw_upper if (raw_upper < prev_upper or prev_close > prev_upper) else prev_upper

        if i == 0 or np.isnan(atr_series.iloc[i - 1]):
            direction[i] = 1.0
        elif supertrend[i - 1] == prev_upper:
            direction[i] = -1.0 if float(close.iloc[i]) > upper[i] else 1.0
        else:
            direction[i] = 1.0 if float(close.iloc[i]) < lower[i] else -1.0

        supertrend[i] = lower[i] if direction[i] == -1.0 else upper[i]

    return (
        pd.Series(supertrend, index=src.index, dtype=float),
        pd.Series(direction, index=src.index, dtype=float),
    )


def _hierarchy_specs(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "hierarchy": PARAMETER_SPACES["hierarchy_combo"]["hierarchy"],
        "trend_indicator": PARAMETER_SPACES["hierarchy_combo"]["trend_indicator"],
        "atr_stop": PARAMETER_SPACES["hierarchy_combo"]["atr_stop"],
        "atr_trail": PARAMETER_SPACES["hierarchy_combo"]["atr_trail"],
    }
    trend_indicator = params.get("trend_indicator")
    hierarchy = params.get("hierarchy", "trend_only")
    if trend_indicator in TREND_SPECS:
        specs.update(TREND_SPECS[trend_indicator])

    active = HIERARCHY_ACTIVE.get(str(hierarchy), HIERARCHY_ACTIVE["trend_only"])
    if active["momentum"]:
        specs["momentum_indicator"] = PARAMETER_SPACES["hierarchy_combo"]["momentum_indicator"]
        momentum_indicator = params.get("momentum_indicator")
        if momentum_indicator in MOMENTUM_SPECS:
            specs.update(MOMENTUM_SPECS[momentum_indicator])
    if active["volume"]:
        specs["volume_indicator"] = PARAMETER_SPACES["hierarchy_combo"]["volume_indicator"]
        volume_indicator = params.get("volume_indicator")
        if volume_indicator in VOLUME_SPECS:
            specs.update(VOLUME_SPECS[volume_indicator])
    return specs


def sample_params(trial: Any, family: str) -> dict[str, Any]:
    if family == "hierarchy_combo":
        params: dict[str, Any] = {
            "hierarchy": trial.suggest_categorical("hierarchy", PARAMETER_SPACES[family]["hierarchy"]["choices"]),
            "trend_indicator": trial.suggest_categorical("trend_indicator", PARAMETER_SPACES[family]["trend_indicator"]["choices"]),
            "atr_stop": trial.suggest_float("atr_stop", 1.0, 5.0, step=0.1),
            "atr_trail": trial.suggest_float("atr_trail", 0.5, 4.0, step=0.1),
        }
        for key, spec in TREND_SPECS[params["trend_indicator"]].items():
            if spec["type"] == "int":
                params[key] = trial.suggest_int(key, spec["low"], spec["high"], step=spec["step"])
            else:
                params[key] = trial.suggest_float(key, spec["low"], spec["high"], step=spec["step"])

        active = HIERARCHY_ACTIVE[params["hierarchy"]]
        if active["momentum"]:
            params["momentum_indicator"] = trial.suggest_categorical("momentum_indicator", PARAMETER_SPACES[family]["momentum_indicator"]["choices"])
            for key, spec in MOMENTUM_SPECS[params["momentum_indicator"]].items():
                if spec["type"] == "int":
                    params[key] = trial.suggest_int(key, spec["low"], spec["high"], step=spec["step"])
                else:
                    params[key] = trial.suggest_float(key, spec["low"], spec["high"], step=spec["step"])
        if active["volume"]:
            params["volume_indicator"] = trial.suggest_categorical("volume_indicator", PARAMETER_SPACES[family]["volume_indicator"]["choices"])
            for key, spec in VOLUME_SPECS[params["volume_indicator"]].items():
                if spec["type"] == "int":
                    params[key] = trial.suggest_int(key, spec["low"], spec["high"], step=spec["step"])
                else:
                    params[key] = trial.suggest_float(key, spec["low"], spec["high"], step=spec["step"])
        return normalize_params(family, params)

    params = {}
    for name, spec in PARAMETER_SPACES[family].items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec["step"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec["step"])
        else:
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return normalize_params(family, params)


def normalize_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(params)
    specs = PARAMETER_SPACES[family] if family != "hierarchy_combo" else _hierarchy_specs(normalized)
    if family == "hierarchy_combo":
        normalized = {key: value for key, value in normalized.items() if key in specs}
        for name, spec in specs.items():
            if name not in normalized or pd.isna(normalized[name]):
                normalized[name] = _default_from_spec(spec)
    for name, spec in specs.items():
        if name not in normalized:
            continue
        if pd.isna(normalized[name]):
            normalized[name] = _default_from_spec(spec)
        normalized[name] = _cast_param(spec, normalized[name])

    if family == "trend" and normalized["fast_ma"] >= normalized["slow_ma"]:
        normalized["fast_ma"] = max(PARAMETER_SPACES[family]["fast_ma"]["low"], normalized["slow_ma"] - 1)
    if family == "mean_reversion" and normalized["rsi_entry"] >= normalized["rsi_exit"]:
        normalized["rsi_exit"] = min(PARAMETER_SPACES[family]["rsi_exit"]["high"], normalized["rsi_entry"] + 5.0)
    if family == "breakout" and normalized["exit_lookback"] >= normalized["entry_lookback"]:
        normalized["exit_lookback"] = max(PARAMETER_SPACES[family]["exit_lookback"]["low"], normalized["entry_lookback"] // 2)
    if family == "kama_cci_atr_allocation" and normalized["kama_fast"] >= normalized["kama_slow"]:
        normalized["kama_fast"] = max(PARAMETER_SPACES[family]["kama_fast"]["low"], normalized["kama_slow"] - 1)
    if family == "dca_tactical_overlay" and normalized["kama_fast"] >= normalized["kama_slow"]:
        normalized["kama_fast"] = max(PARAMETER_SPACES[family]["kama_fast"]["low"], normalized["kama_slow"] - 1)
    if family == "dca_zone_overlay":
        if normalized["kama_fast"] >= normalized["kama_slow"]:
            normalized["kama_fast"] = max(PARAMETER_SPACES[family]["kama_fast"]["low"], normalized["kama_slow"] - 1)
        if normalized["buy_deep"] > normalized["buy_mild"]:
            normalized["buy_deep"] = normalized["buy_mild"] - 5.0
        if normalized["trim_hard"] < normalized["trim_mild"]:
            normalized["trim_hard"] = normalized["trim_mild"] + 5.0
    if family == "dca_pullback_topup":
        if normalized["kama_fast"] >= normalized["kama_slow"]:
            normalized["kama_fast"] = max(PARAMETER_SPACES[family]["kama_fast"]["low"], normalized["kama_slow"] - 1)
        if normalized["buy_deep"] > normalized["buy_mild"]:
            normalized["buy_deep"] = normalized["buy_mild"] - 5.0
        if normalized["core_weight"] + normalized["reserve_weight"] > 1.0:
            normalized["reserve_weight"] = round(1.0 - normalized["core_weight"], 2)
    if family == "dca_pullback_only" and normalized["kama_fast"] >= normalized["kama_slow"]:
        normalized["kama_fast"] = max(PARAMETER_SPACES[family]["kama_fast"]["low"], normalized["kama_slow"] - 1)
    if family == "hierarchy_combo":
        if normalized.get("trend_indicator") == "kama" and "kama_fast" in normalized and "kama_slow" in normalized and normalized["kama_fast"] >= normalized["kama_slow"]:
            normalized["kama_fast"] = max(TREND_SPECS["kama"]["kama_fast"]["low"], normalized["kama_slow"] - 1)
        if normalized.get("momentum_indicator") == "macd" and "macd_fast" in normalized and "macd_slow" in normalized and normalized["macd_fast"] >= normalized["macd_slow"]:
            normalized["macd_fast"] = max(MOMENTUM_SPECS["macd"]["macd_fast"]["low"], normalized["macd_slow"] - 1)
    if family == "blackcat_dynamic_momentum" and normalized["mom_fast"] >= normalized["mom_slow"]:
        normalized["mom_fast"] = max(PARAMETER_SPACES[family]["mom_fast"]["low"], normalized["mom_slow"] - 1)
    if family == "blackcat_multi_bbands" and normalized["inner_mult"] >= normalized["outer_mult"]:
        normalized["outer_mult"] = min(PARAMETER_SPACES[family]["outer_mult"]["high"], normalized["inner_mult"] + 0.5)
    if family == "blackcat_ichimoku":
        if normalized["tenkan_len"] >= normalized["kijun_len"]:
            normalized["tenkan_len"] = max(PARAMETER_SPACES[family]["tenkan_len"]["low"], normalized["kijun_len"] - 1)
        if normalized["kijun_len"] >= normalized["senkou_b_len"]:
            normalized["kijun_len"] = max(PARAMETER_SPACES[family]["kijun_len"]["low"], normalized["senkou_b_len"] - 1)
    if family == "blackcat_ravi":
        if normalized["fast_len"] >= normalized["slow_len"]:
            normalized["fast_len"] = max(PARAMETER_SPACES[family]["fast_len"]["low"], normalized["slow_len"] - 1)
        if normalized["slow_len"] >= normalized["bias_len"]:
            normalized["slow_len"] = max(PARAMETER_SPACES[family]["slow_len"]["low"], normalized["bias_len"] - 1)
    if family == "blackcat_cci_rsi" and normalized["cci_entry"] >= normalized["cci_exit"]:
        normalized["cci_exit"] = min(PARAMETER_SPACES[family]["cci_exit"]["high"], normalized["cci_entry"] + 25.0)
    return normalized


def build_neighbors(family: str, params: dict[str, Any], steps: int = 1) -> list[dict[str, Any]]:
    base = normalize_params(family, params)
    neighbors: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    specs = PARAMETER_SPACES[family] if family != "hierarchy_combo" else _hierarchy_specs(base)

    for name, spec in specs.items():
        if name not in base:
            continue
        if spec["type"] == "categorical":
            for choice in spec["choices"]:
                if choice == base[name]:
                    continue
                candidate = deepcopy(base)
                candidate[name] = choice
                candidate = normalize_params(family, candidate)
                key = tuple(sorted(candidate.items()))
                if key not in seen:
                    seen.add(key)
                    neighbors.append(candidate)
            continue

        delta = spec["step"] * steps
        for direction in (-1, 1):
            candidate = deepcopy(base)
            candidate[name] = candidate[name] + delta * direction
            candidate[name] = max(spec["low"], min(spec["high"], candidate[name]))
            candidate = normalize_params(family, candidate)
            if candidate == base:
                continue
            key = tuple(sorted(candidate.items()))
            if key not in seen:
                seen.add(key)
                neighbors.append(candidate)
    return neighbors


def _trend_condition(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    src = (high + low) / 2.0
    indicator = params["trend_indicator"]

    if indicator == "supertrend":
        periods = pd.Series(float(params["st_len"]), index=df.index)
        _, direction = _pine_supertrend(src, close, high, low, params["st_mult"], periods)
        return direction == -1
    if indicator == "kama":
        line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        return (close > line) & (line > line.shift(1))
    if indicator == "aroon":
        up, down = aroon(high, low, params["aroon_len"])
        return (up > down) & (up > params["aroon_threshold"])
    raise ValueError(indicator)


def _momentum_condition(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    indicator = params["momentum_indicator"]

    if indicator == "rsi":
        value = rsi(close, params["mom_rsi_len"])
        return value > params["mom_rsi_threshold"]
    if indicator == "macd":
        line, signal, hist = macd(close, params["macd_fast"], params["macd_slow"], params["macd_signal"])
        return (line > signal) & (hist > 0)
    if indicator == "cci":
        value = cci(high, low, close, params["cci_len"])
        return value > params["cci_threshold"]
    raise ValueError(indicator)


def _volume_condition(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    indicator = params["volume_indicator"]

    if indicator == "cmf":
        value = cmf(high, low, close, volume, params["cmf_len"])
        return value > params["cmf_threshold"]
    if indicator == "mfi":
        value = mfi(high, low, close, volume, params["mfi_len"])
        return value > params["mfi_threshold"]
    if indicator == "obv":
        line = obv(close, volume)
        return line > ema(line, params["obv_ema_len"])
    raise ValueError(indicator)


def build_blackcat_gate_signal(data: pd.DataFrame, family: str, params: dict[str, Any]) -> pd.Series:
    params = normalize_params(family, params)
    df = data.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    if family == "blackcat_ichimoku":
        tenkan = _rolling_midpoint(high, low, params["tenkan_len"])
        kijun = _rolling_midpoint(high, low, params["kijun_len"])
        span_a = (tenkan + kijun) / 2.0
        span_b = _rolling_midpoint(high, low, params["senkou_b_len"])
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        return ((tenkan > kijun) & (close > cloud_top)).fillna(False)

    if family == "blackcat_ravi":
        ravi_value = _ravi(close, params["fast_len"], params["slow_len"], params["bias_len"])
        bias_line = sma(close, params["bias_len"])
        return ((ravi_value > params["ravi_entry"]) & (close > bias_line) & (ravi_value > ravi_value.shift(1))).fillna(False)

    if family == "blackcat_zlema_band":
        zlema_line = _zero_lag_ema(close, params["zlema_len"])
        trend_line = sma(close, params["trend_ma"])
        return ((close > zlema_line) & (zlema_line > zlema_line.shift(1)) & (close > trend_line)).fillna(False)

    raise ValueError(f"Unsupported blackcat gate family: {family}")


def build_blackcat_gate_score(data: pd.DataFrame, family: str, params: dict[str, Any]) -> pd.Series:
    params = normalize_params(family, params)
    df = data.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    if family == "blackcat_ichimoku":
        tenkan = _rolling_midpoint(high, low, params["tenkan_len"])
        kijun = _rolling_midpoint(high, low, params["kijun_len"])
        span_a = (tenkan + kijun) / 2.0
        span_b = _rolling_midpoint(high, low, params["senkou_b_len"])
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        cloud_score = pd.Series(
            np.where(
                close > cloud_top,
                1.0,
                np.where(close >= cloud_bottom, 0.5, 0.0),
            ),
            index=df.index,
            dtype=float,
        )
        trend_score = (tenkan > kijun).astype(float)
        slope_score = (kijun > kijun.shift(1)).astype(float)
        score = 0.5 * cloud_score + 0.3 * trend_score + 0.2 * slope_score
        return score.clip(0.0, 1.0).fillna(0.0)

    if family == "blackcat_zlema_band":
        zlema_line = _zero_lag_ema(close, params["zlema_len"])
        deviation = close.rolling(params["zlema_len"], min_periods=params["zlema_len"]).std(ddof=0)
        upper_band = zlema_line + deviation * params["band_mult"]
        trend_line = sma(close, params["trend_ma"])

        score = (
            0.35 * (close > zlema_line).astype(float)
            + 0.25 * (close > trend_line).astype(float)
            + 0.25 * (zlema_line > zlema_line.shift(1)).astype(float)
            + 0.15 * (close > upper_band).astype(float)
        )
        return score.clip(0.0, 1.0).fillna(0.0)

    raise ValueError(f"Unsupported blackcat gate score family: {family}")


def build_strategy_frame(data: pd.DataFrame, family: str, params: dict[str, Any]) -> pd.DataFrame:
    params = normalize_params(family, params)
    df = data.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    if family == "trend":
        fast = moving_average(close, params["fast_ma"], params["ma_type"])
        slow = moving_average(close, params["slow_ma"], params["ma_type"])
        trend_adx = adx(high, low, close, params["adx_len"])
        df["entry_signal"] = (fast > slow) & (trend_adx > params["adx_threshold"])
        df["exit_signal"] = fast < slow
        df["atr"] = atr(high, low, close, params["adx_len"])
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "mean_reversion":
        regime_ma = sma(close, params["regime_ma"])
        rsi_value = rsi(close, params["rsi_len"])
        lower, middle, _ = bollinger_bands(close, params["bb_len"], params["bb_std"])
        df["entry_signal"] = (close > regime_ma) & (rsi_value < params["rsi_entry"]) & (close < lower)
        df["exit_signal"] = (rsi_value > params["rsi_exit"]) | (close > middle)
        df["atr"] = atr(high, low, close, max(14, params["rsi_len"]))
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = 0.0

    elif family == "breakout":
        regime_ma = sma(close, params["regime_ma"])
        entry_level = high.rolling(params["entry_lookback"], min_periods=params["entry_lookback"]).max().shift(1)
        exit_level = low.rolling(params["exit_lookback"], min_periods=params["exit_lookback"]).min().shift(1)
        df["entry_signal"] = (close > entry_level) & (close > regime_ma)
        df["exit_signal"] = close < exit_level
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "fdi_supertrend":
        src = (high + low) / 2.0
        adaptive_len = _fdi_speed(src, params["per"], params["speed"])
        atr_periods = adaptive_len if params["adapt"] else pd.Series(float(params["per"]), index=df.index)
        _, direction = _pine_supertrend(src, close, high, low, params["mult"], atr_periods)
        df["entry_signal"] = (direction == -1) & (direction.shift(1) == 1)
        df["exit_signal"] = (direction == 1) & (direction.shift(1) == -1)
        df["atr"] = 0.0
        df["stop_mult"] = 0.0
        df["trail_mult"] = 0.0

    elif family == "kama_cci_atr_allocation":
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        cci_value = cci(high, low, close, params["cci_len"])
        strong_alloc, neutral_alloc, bad_alloc = ALLOCATION_LADDERS[params["allocation_ladder"]]
        strong_regime = (close > kama_line) & (kama_line > kama_line.shift(1)) & (cci_value > params["cci_threshold"])
        neutral_regime = close > kama_line
        allocations = np.where(strong_regime, strong_alloc, np.where(neutral_regime, neutral_alloc, bad_alloc))
        df["target_allocation"] = pd.Series(allocations, index=df.index, dtype=float)
        df["entry_signal"] = df["target_allocation"] > 0.0
        df["exit_signal"] = df["target_allocation"] <= 0.0
        df["atr"] = atr(high, low, close, params["atr_len"])
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "dca_tactical_overlay":
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        cci_value = cci(high, low, close, params["cci_len"])
        strong_alloc, neutral_alloc, bad_alloc = ALLOCATION_LADDERS[params["sleeve_ladder"]]
        strong_regime = (close > kama_line) & (kama_line > kama_line.shift(1)) & (cci_value > params["cci_threshold"])
        neutral_regime = close > kama_line
        sleeve_alloc = np.where(strong_regime, strong_alloc, np.where(neutral_regime, neutral_alloc, bad_alloc))
        total_alloc = params["core_weight"] + (1.0 - params["core_weight"]) * sleeve_alloc
        df["target_allocation"] = pd.Series(total_alloc, index=df.index, dtype=float)
        df["floor_allocation"] = float(params["core_weight"])
        df["entry_signal"] = df["target_allocation"] > 0.0
        df["exit_signal"] = False
        df["atr"] = atr(high, low, close, params["atr_len"])
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "dca_zone_overlay":
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        cci_value = cci(high, low, close, params["cci_len"])
        atr_value = atr(high, low, close, params["atr_len"])
        deep_buy_alloc, mild_buy_alloc, base_alloc, trim_alloc, hard_trim_alloc = ZONE_PROFILES[params["zone_profile"]]

        trend_intact = close > (kama_line - atr_value * params["trend_buffer_atr"])
        trend_broken = (close < kama_line) & (kama_line < kama_line.shift(1))
        stretched_up = close > (kama_line + atr_value * params["trim_buffer_atr"])

        sleeve_alloc = np.where(
            trend_broken,
            hard_trim_alloc,
            np.where(
                trend_intact & (cci_value <= params["buy_deep"]),
                deep_buy_alloc,
                np.where(
                    trend_intact & (cci_value <= params["buy_mild"]),
                    mild_buy_alloc,
                    np.where(
                        stretched_up & (cci_value >= params["trim_hard"]),
                        hard_trim_alloc,
                        np.where(
                            (cci_value >= params["trim_mild"]) | stretched_up,
                            trim_alloc,
                            base_alloc,
                        ),
                    ),
                ),
            ),
        )
        total_alloc = params["core_weight"] + (1.0 - params["core_weight"]) * sleeve_alloc
        df["target_allocation"] = pd.Series(total_alloc, index=df.index, dtype=float)
        df["floor_allocation"] = float(params["core_weight"])
        df["entry_signal"] = df["target_allocation"] > 0.0
        df["exit_signal"] = False
        df["atr"] = atr_value
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "dca_pullback_topup":
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        cci_value = cci(high, low, close, params["cci_len"])
        atr_value = atr(high, low, close, params["atr_len"])
        mild_topup, deep_topup, max_topup = TOPUP_PROFILES[params["topup_profile"]]

        trend_intact = close > (kama_line - atr_value * params["trend_buffer_atr"])
        mild_pullback = trend_intact & (cci_value <= params["buy_mild"])
        deep_pullback = trend_intact & (cci_value <= params["buy_deep"])
        reserve_target = np.where(deep_pullback, max_topup, np.where(mild_pullback, deep_topup, mild_topup))
        reserve_target = np.where(trend_intact, reserve_target, 0.0)
        total_alloc = params["core_weight"] + params["reserve_weight"] * reserve_target
        df["target_allocation"] = pd.Series(total_alloc, index=df.index, dtype=float)
        df["floor_allocation"] = float(params["core_weight"])
        df["entry_signal"] = df["target_allocation"] > 0.0
        df["exit_signal"] = False
        df["atr"] = atr_value
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "dca_pullback_only":
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        cci_value = cci(high, low, close, params["cci_len"])
        atr_value = atr(high, low, close, params["atr_len"])
        trend_intact = close > (kama_line - atr_value * params["trend_buffer_atr"])
        pullback_signal = trend_intact & (cci_value <= params["buy_threshold"])
        df["deploy_fraction"] = pd.Series(np.where(pullback_signal, 1.0, 0.0), index=df.index, dtype=float)
        df["entry_signal"] = pullback_signal
        df["exit_signal"] = False
        df["atr"] = 0.0
        df["stop_mult"] = 0.0
        df["trail_mult"] = 0.0

    elif family == "blackcat_dynamic_momentum":
        rsv = _stochastic_rsv(high, low, close, params["stoch_len"])
        k_line = sma(rsv, params["kd_smooth"])
        d_line = sma(k_line, params["kd_smooth"])
        momentum = ema(close, params["mom_fast"]) - ema(close, params["mom_slow"])
        momentum_signal = ema(momentum, params["mom_signal"])
        trend_line = sma(close, params["trend_len"])
        df["entry_signal"] = (k_line > d_line) & (k_line.shift(1) <= d_line.shift(1)) & (momentum > momentum_signal) & (close > trend_line)
        df["exit_signal"] = (k_line < d_line) | (momentum < momentum_signal) | (close < trend_line)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "blackcat_multi_bbands":
        lower_inner, midline, upper_inner = bollinger_bands(close, params["bb_len"], params["inner_mult"])
        lower_outer, _, upper_outer = bollinger_bands(close, params["bb_len"], params["outer_mult"])
        trend_line = sma(close, params["trend_ma"])
        touched_pullback = low.shift(1) < lower_inner.shift(1)
        df["entry_signal"] = (close > trend_line) & touched_pullback & (close > lower_inner)
        df["exit_signal"] = (close > upper_outer) | (close < midline) | (close < trend_line) | (close < lower_outer)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "blackcat_zlema_band":
        zlema_line = _zero_lag_ema(close, params["zlema_len"])
        deviation = close.rolling(params["zlema_len"], min_periods=params["zlema_len"]).std(ddof=0)
        upper_band = zlema_line + deviation * params["band_mult"]
        trend_line = sma(close, params["trend_ma"])
        breakout = (close > upper_band) & (close.shift(1) <= upper_band.shift(1))
        trend_reclaim = (close > zlema_line) & (close.shift(1) <= zlema_line.shift(1))
        df["entry_signal"] = (breakout | trend_reclaim) & (zlema_line > zlema_line.shift(1)) & (close > trend_line)
        df["exit_signal"] = (close < zlema_line) | (zlema_line < zlema_line.shift(1)) | (close < trend_line)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "blackcat_ichimoku":
        tenkan = _rolling_midpoint(high, low, params["tenkan_len"])
        kijun = _rolling_midpoint(high, low, params["kijun_len"])
        span_a = (tenkan + kijun) / 2.0
        span_b = _rolling_midpoint(high, low, params["senkou_b_len"])
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        bullish_cross = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        cloud_breakout = (close > cloud_top) & (close.shift(1) <= cloud_top.shift(1))
        bearish_cross = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
        df["entry_signal"] = ((tenkan > kijun) & (close > cloud_top)) & (bullish_cross | cloud_breakout)
        df["exit_signal"] = bearish_cross | (close < cloud_bottom)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "blackcat_ravi":
        ravi_value = _ravi(close, params["fast_len"], params["slow_len"], params["bias_len"])
        bias_line = sma(close, params["bias_len"])
        df["entry_signal"] = (ravi_value > params["ravi_entry"]) & (close > bias_line) & (ravi_value > ravi_value.shift(1))
        df["exit_signal"] = (ravi_value < 0.0) | (close < bias_line)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "blackcat_cci_rsi":
        cci_value = cci(high, low, close, params["cci_len"])
        rsi_value = rsi(close, params["rsi_len"])
        rsi_signal = sma(rsi_value, params["rsi_signal"])
        trend_line = sma(close, params["trend_ma"])
        rsi_cross = (rsi_value > rsi_signal) & (rsi_value.shift(1) <= rsi_signal.shift(1))
        df["entry_signal"] = (close > trend_line) & (cci_value > params["cci_entry"]) & (rsi_value > rsi_signal) & rsi_cross
        df["exit_signal"] = (cci_value > params["cci_exit"]) | (rsi_value < rsi_signal) | (close < trend_line)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "blackcat_superj":
        rsv = _stochastic_rsv(high, low, close, params["stoch_len"])
        k_line = ema(rsv, params["kd_smooth"])
        d_line = ema(k_line, params["kd_smooth"])
        j_raw = 3.0 * k_line - 2.0 * d_line
        j_line = _vwma(j_raw, df["volume"], params["j_smooth"]).fillna(ema(j_raw, params["j_smooth"]))
        trigger = ema(j_line, params["trigger_len"])
        trend_line = sma(close, params["trend_ma"])
        washed_out = j_line.shift(1) < params["oversold"]
        df["entry_signal"] = washed_out & (j_line > trigger) & (j_line.shift(1) <= trigger.shift(1)) & (close > trend_line)
        df["exit_signal"] = (j_line < trigger) | (j_line > 90.0) | (close < trend_line)
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    elif family == "hierarchy_combo":
        active = HIERARCHY_ACTIVE[params["hierarchy"]]
        trend_ok = _trend_condition(df, params)
        momentum_ok = _momentum_condition(df, params) if active["momentum"] else pd.Series(True, index=df.index)
        volume_ok = _volume_condition(df, params) if active["volume"] else pd.Series(True, index=df.index)
        combo_ok = trend_ok & momentum_ok & volume_ok
        df["entry_signal"] = combo_ok
        df["exit_signal"] = ~combo_ok
        df["atr"] = atr(high, low, close, 14)
        df["stop_mult"] = params["atr_stop"]
        df["trail_mult"] = params["atr_trail"]

    else:
        raise ValueError(f"Unsupported family: {family}")

    df["entry_signal"] = df["entry_signal"].fillna(False)
    df["exit_signal"] = df["exit_signal"].fillna(False)
    return df
