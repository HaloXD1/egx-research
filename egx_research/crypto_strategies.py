from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from egx_research.indicators import (
    adx,
    aroon,
    atr,
    bollinger_bands,
    cci,
    cmf,
    ema,
    kama,
    macd,
    mfi,
    moving_average,
    obv,
    rsi,
    sma,
)


CRYPTO_PARAMETER_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "crypto_price_trend": {
        "fast_ema": {"type": "int", "low": 10, "high": 80, "step": 1},
        "slow_ema": {"type": "int", "low": 80, "high": 260, "step": 1},
        "trend_ma": {"type": "int", "low": 100, "high": 320, "step": 5},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_trend_adx": {
        "fast_ma": {"type": "int", "low": 8, "high": 80, "step": 1},
        "slow_ma": {"type": "int", "low": 60, "high": 260, "step": 1},
        "trend_ma": {"type": "int", "low": 80, "high": 320, "step": 5},
        "ma_type": {"type": "categorical", "choices": ["EMA", "SMA"]},
        "adx_len": {"type": "int", "low": 7, "high": 35, "step": 1},
        "adx_threshold": {"type": "float", "low": 10.0, "high": 40.0, "step": 0.5},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_donchian_breakout": {
        "entry_lookback": {"type": "int", "low": 20, "high": 180, "step": 1},
        "exit_lookback": {"type": "int", "low": 5, "high": 90, "step": 1},
        "regime_ma": {"type": "int", "low": 80, "high": 320, "step": 5},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_supertrend_combo": {
        "st_len": {"type": "int", "low": 7, "high": 40, "step": 1},
        "st_mult": {"type": "float", "low": 1.0, "high": 6.0, "step": 0.1},
        "confirmation_ma": {"type": "int", "low": 40, "high": 240, "step": 5},
        "rsi_len": {"type": "int", "low": 5, "high": 30, "step": 1},
        "rsi_floor": {"type": "float", "low": 35.0, "high": 60.0, "step": 1.0},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_pullback_combo": {
        "regime_ma": {"type": "int", "low": 80, "high": 320, "step": 5},
        "bb_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "bb_std": {"type": "float", "low": 1.2, "high": 3.5, "step": 0.1},
        "rsi_len": {"type": "int", "low": 2, "high": 30, "step": 1},
        "rsi_entry": {"type": "float", "low": 15.0, "high": 45.0, "step": 0.5},
        "rsi_exit": {"type": "float", "low": 45.0, "high": 75.0, "step": 0.5},
        "cci_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "cci_entry": {"type": "float", "low": -250.0, "high": -25.0, "step": 5.0},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_dca_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.5, 0.6, 0.7, 0.8]},
        "kama_len": {"type": "int", "low": 15, "high": 60, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 8, "step": 1},
        "kama_slow": {"type": "int", "low": 30, "high": 100, "step": 1},
        "cci_len": {"type": "int", "low": 14, "high": 60, "step": 1},
        "buy_threshold": {"type": "float", "low": -220.0, "high": -20.0, "step": 5.0},
        "trend_buffer_atr": {"type": "float", "low": 0.0, "high": 3.0, "step": 0.1},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_onchain_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.5, 0.6, 0.7, 0.8]},
        "mvrv_buy": {"type": "float", "low": 0.8, "high": 1.8, "step": 0.05},
        "mvrv_trim": {"type": "float", "low": 2.0, "high": 4.5, "step": 0.1},
        "activity_window": {"type": "int", "low": 30, "high": 180, "step": 5},
        "flow_window": {"type": "int", "low": 14, "high": 90, "step": 1},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_sentiment_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.5, 0.6, 0.7, 0.8]},
        "fear_buy": {"type": "float", "low": 10.0, "high": 40.0, "step": 1.0},
        "greed_trim": {"type": "float", "low": 60.0, "high": 90.0, "step": 1.0},
        "trend_ma": {"type": "int", "low": 100, "high": 320, "step": 5},
        "cci_len": {"type": "int", "low": 14, "high": 60, "step": 1},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_macro_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.5, 0.6, 0.7, 0.8]},
        "btc_trend_ma": {"type": "int", "low": 100, "high": 320, "step": 5},
        "macro_ma": {"type": "int", "low": 40, "high": 220, "step": 5},
        "risk_reduce": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.05},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_ensemble_overlay": {
        "core_weight": {"type": "categorical", "choices": [0.5, 0.6, 0.7, 0.8]},
        "fast_ema": {"type": "int", "low": 10, "high": 80, "step": 1},
        "slow_ema": {"type": "int", "low": 80, "high": 260, "step": 1},
        "cci_len": {"type": "int", "low": 14, "high": 60, "step": 1},
        "fear_buy": {"type": "float", "low": 10.0, "high": 40.0, "step": 1.0},
        "greed_trim": {"type": "float", "low": 60.0, "high": 90.0, "step": 1.0},
        "mvrv_buy": {"type": "float", "low": 0.8, "high": 1.8, "step": 0.05},
        "mvrv_trim": {"type": "float", "low": 2.0, "high": 4.5, "step": 0.1},
        "activity_window": {"type": "int", "low": 30, "high": 180, "step": 5},
        "macro_ma": {"type": "int", "low": 40, "high": 220, "step": 5},
        "price_weight": {"type": "float", "low": 0.25, "high": 0.7, "step": 0.05},
        "onchain_weight": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
        "sentiment_weight": {"type": "float", "low": 0.0, "high": 0.35, "step": 0.05},
        "macro_weight": {"type": "float", "low": 0.0, "high": 0.35, "step": 0.05},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_hierarchy_combo": {
        "trend_indicator": {"type": "categorical", "choices": ["ema", "supertrend", "kama", "aroon"]},
        "momentum_indicator": {"type": "categorical", "choices": ["none", "rsi", "macd", "cci"]},
        "volume_indicator": {"type": "categorical", "choices": ["none", "cmf", "mfi", "obv"]},
        "combo_rule": {"type": "categorical", "choices": ["all", "majority"]},
        "min_confirmations": {"type": "int", "low": 1, "high": 3, "step": 1},
        "fast_ema": {"type": "int", "low": 8, "high": 80, "step": 1},
        "slow_ema": {"type": "int", "low": 60, "high": 260, "step": 1},
        "st_len": {"type": "int", "low": 7, "high": 40, "step": 1},
        "st_mult": {"type": "float", "low": 1.0, "high": 6.0, "step": 0.1},
        "kama_len": {"type": "int", "low": 10, "high": 80, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 8, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 120, "step": 1},
        "aroon_len": {"type": "int", "low": 10, "high": 80, "step": 1},
        "aroon_threshold": {"type": "float", "low": 45.0, "high": 90.0, "step": 1.0},
        "mom_rsi_len": {"type": "int", "low": 5, "high": 30, "step": 1},
        "mom_rsi_threshold": {"type": "float", "low": 40.0, "high": 70.0, "step": 1.0},
        "macd_fast": {"type": "int", "low": 5, "high": 24, "step": 1},
        "macd_slow": {"type": "int", "low": 15, "high": 60, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 18, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "cci_threshold": {"type": "float", "low": -50.0, "high": 150.0, "step": 5.0},
        "cmf_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "cmf_threshold": {"type": "float", "low": -0.2, "high": 0.3, "step": 0.01},
        "mfi_len": {"type": "int", "low": 5, "high": 40, "step": 1},
        "mfi_threshold": {"type": "float", "low": 35.0, "high": 75.0, "step": 1.0},
        "obv_ema_len": {"type": "int", "low": 5, "high": 80, "step": 1},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
    "crypto_multisignal_score": {
        "core_weight": {"type": "categorical", "choices": [0.0, 0.25, 0.5, 0.6]},
        "fast_ema": {"type": "int", "low": 8, "high": 80, "step": 1},
        "slow_ema": {"type": "int", "low": 60, "high": 260, "step": 1},
        "st_len": {"type": "int", "low": 7, "high": 40, "step": 1},
        "st_mult": {"type": "float", "low": 1.0, "high": 6.0, "step": 0.1},
        "kama_len": {"type": "int", "low": 10, "high": 80, "step": 1},
        "kama_fast": {"type": "int", "low": 2, "high": 8, "step": 1},
        "kama_slow": {"type": "int", "low": 20, "high": 120, "step": 1},
        "rsi_len": {"type": "int", "low": 5, "high": 30, "step": 1},
        "rsi_threshold": {"type": "float", "low": 40.0, "high": 70.0, "step": 1.0},
        "macd_fast": {"type": "int", "low": 5, "high": 24, "step": 1},
        "macd_slow": {"type": "int", "low": 15, "high": 60, "step": 1},
        "macd_signal": {"type": "int", "low": 5, "high": 18, "step": 1},
        "cci_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "cci_threshold": {"type": "float", "low": -50.0, "high": 150.0, "step": 5.0},
        "cmf_len": {"type": "int", "low": 10, "high": 60, "step": 1},
        "mfi_len": {"type": "int", "low": 5, "high": 40, "step": 1},
        "obv_ema_len": {"type": "int", "low": 5, "high": 80, "step": 1},
        "fear_buy": {"type": "float", "low": 10.0, "high": 40.0, "step": 1.0},
        "greed_trim": {"type": "float", "low": 60.0, "high": 90.0, "step": 1.0},
        "mvrv_buy": {"type": "float", "low": 0.8, "high": 1.8, "step": 0.05},
        "mvrv_trim": {"type": "float", "low": 2.0, "high": 4.5, "step": 0.1},
        "activity_window": {"type": "int", "low": 30, "high": 180, "step": 5},
        "flow_window": {"type": "int", "low": 14, "high": 90, "step": 1},
        "macro_ma": {"type": "int", "low": 40, "high": 220, "step": 5},
        "price_weight": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.05},
        "momentum_weight": {"type": "float", "low": 0.0, "high": 0.5, "step": 0.05},
        "volume_weight": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
        "onchain_weight": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
        "sentiment_weight": {"type": "float", "low": 0.0, "high": 0.35, "step": 0.05},
        "macro_weight": {"type": "float", "low": 0.0, "high": 0.35, "step": 0.05},
        "allocation_threshold": {"type": "float", "low": 0.2, "high": 0.8, "step": 0.05},
        "atr_len": {"type": "int", "low": 10, "high": 30, "step": 1},
        "atr_stop": {"type": "float", "low": 1.5, "high": 6.0, "step": 0.1},
        "atr_trail": {"type": "float", "low": 1.0, "high": 5.0, "step": 0.1},
    },
}


def _default_from_spec(spec: dict[str, Any]) -> Any:
    if spec["type"] == "int":
        return int(spec["low"])
    if spec["type"] == "float":
        return float(spec["low"])
    return spec["choices"][0]


def _cast_param(spec: dict[str, Any], value: Any) -> Any:
    if spec["type"] == "int":
        return int(round(float(value)))
    if spec["type"] == "float":
        return float(value)
    first = spec["choices"][0]
    if isinstance(first, float):
        return float(value)
    if isinstance(first, int):
        return int(round(float(value)))
    return value


def normalize_crypto_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    if family not in CRYPTO_PARAMETER_SPACES:
        raise ValueError(f"Unsupported crypto family: {family}")
    normalized = deepcopy(params)
    specs = CRYPTO_PARAMETER_SPACES[family]
    for name, spec in specs.items():
        if name not in normalized or pd.isna(normalized[name]):
            normalized[name] = _default_from_spec(spec)
        normalized[name] = _cast_param(spec, normalized[name])

    for fast_key, slow_key in (
        ("fast_ema", "slow_ema"),
        ("fast_ma", "slow_ma"),
        ("macd_fast", "macd_slow"),
        ("kama_fast", "kama_slow"),
    ):
        if fast_key in normalized and slow_key in normalized and normalized[fast_key] >= normalized[slow_key]:
            normalized[fast_key] = max(1, normalized[slow_key] - 1)
    if "exit_lookback" in normalized and "entry_lookback" in normalized and normalized["exit_lookback"] >= normalized["entry_lookback"]:
        normalized["exit_lookback"] = max(2, normalized["entry_lookback"] // 2)
    if "rsi_entry" in normalized and "rsi_exit" in normalized and normalized["rsi_entry"] >= normalized["rsi_exit"]:
        normalized["rsi_exit"] = normalized["rsi_entry"] + 5.0
    if "mvrv_buy" in normalized and "mvrv_trim" in normalized and normalized["mvrv_buy"] >= normalized["mvrv_trim"]:
        normalized["mvrv_trim"] = normalized["mvrv_buy"] + 0.5
    if "fear_buy" in normalized and "greed_trim" in normalized and normalized["fear_buy"] >= normalized["greed_trim"]:
        normalized["greed_trim"] = normalized["fear_buy"] + 10.0
    return normalized


def sample_crypto_params(trial: Any, family: str) -> dict[str, Any]:
    params = {}
    for name, spec in CRYPTO_PARAMETER_SPACES[family].items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec["step"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], step=spec["step"])
        else:
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return normalize_crypto_params(family, params)


def build_crypto_neighbors(family: str, params: dict[str, Any], steps: int = 1) -> list[dict[str, Any]]:
    base = normalize_crypto_params(family, params)
    specs = CRYPTO_PARAMETER_SPACES[family]
    neighbors: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()

    for name, spec in specs.items():
        if spec["type"] == "categorical":
            choices = spec["choices"]
            for choice in choices:
                if choice == base[name]:
                    continue
                candidate = deepcopy(base)
                candidate[name] = choice
                candidate = normalize_crypto_params(family, candidate)
                key = tuple(sorted(candidate.items()))
                if key not in seen:
                    seen.add(key)
                    neighbors.append(candidate)
            continue

        delta = spec["step"] * steps
        for direction in (-1, 1):
            candidate = deepcopy(base)
            candidate[name] = max(spec["low"], min(spec["high"], candidate[name] + delta * direction))
            candidate = normalize_crypto_params(family, candidate)
            if candidate == base:
                continue
            key = tuple(sorted(candidate.items()))
            if key not in seen:
                seen.add(key)
                neighbors.append(candidate)
    return neighbors


def _numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _score_to_allocation(core_weight: float, score: pd.Series) -> pd.Series:
    score = score.clip(0.0, 1.0).fillna(0.5)
    return core_weight + (1.0 - core_weight) * score


def _stateful_target(entry: pd.Series, exit_: pd.Series, index: pd.Index) -> pd.Series:
    target = np.zeros(len(index), dtype=float)
    state = 0.0
    entry = entry.fillna(False).astype(bool)
    exit_ = exit_.fillna(False).astype(bool)
    for i in range(len(index)):
        if bool(exit_.iloc[i]):
            state = 0.0
        if bool(entry.iloc[i]):
            state = 1.0
        target[i] = state
    return pd.Series(target, index=index, dtype=float)


def _supertrend_bullish(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
    multiplier: float,
) -> pd.Series:
    mid = (high + low) / 2.0
    atr_value = atr(high, low, close, window)
    upper_basic = mid + multiplier * atr_value
    lower_basic = mid - multiplier * atr_value
    upper = np.full(len(close), np.nan, dtype=float)
    lower = np.full(len(close), np.nan, dtype=float)
    bullish = np.full(len(close), False, dtype=bool)

    for i in range(len(close)):
        if pd.isna(upper_basic.iloc[i]) or pd.isna(lower_basic.iloc[i]):
            bullish[i] = bullish[i - 1] if i > 0 else False
            continue
        if i == 0 or pd.isna(upper[i - 1]) or pd.isna(lower[i - 1]):
            upper[i] = float(upper_basic.iloc[i])
            lower[i] = float(lower_basic.iloc[i])
            bullish[i] = False
            continue

        prev_close = float(close.iloc[i - 1])
        upper[i] = (
            float(upper_basic.iloc[i])
            if float(upper_basic.iloc[i]) < upper[i - 1] or prev_close > upper[i - 1]
            else upper[i - 1]
        )
        lower[i] = (
            float(lower_basic.iloc[i])
            if float(lower_basic.iloc[i]) > lower[i - 1] or prev_close < lower[i - 1]
            else lower[i - 1]
        )
        if bool(bullish[i - 1]):
            bullish[i] = not (float(close.iloc[i]) < lower[i])
        else:
            bullish[i] = float(close.iloc[i]) > upper[i]

    return pd.Series(bullish, index=close.index, dtype=bool)


def _price_score(df: pd.DataFrame, fast_ema: int, slow_ema: int) -> pd.Series:
    close = df["close"]
    fast = ema(close, fast_ema)
    slow = ema(close, slow_ema)
    trend = pd.Series(np.where((fast > slow) & (close > slow), 1.0, np.where(close > slow, 0.6, 0.2)), index=df.index)
    lower, middle, _ = bollinger_bands(close, 20, 2.0)
    pullback = (close < middle) & (close > lower)
    return pd.Series(np.where(pullback & (trend >= 0.6), 1.0, trend), index=df.index, dtype=float)


def _onchain_score(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    mvrv = _numeric(df, "CapMVRVCur")
    addresses = _numeric(df, "AdrActCnt")
    tx_count = _numeric(df, "TxCnt")
    hash_rate = _numeric(df, "HashRate")
    inflow = _numeric(df, "FlowInExUSD", 0.0)
    outflow = _numeric(df, "FlowOutExUSD", 0.0)
    window = params.get("activity_window", 90)
    flow_window = params.get("flow_window", 30)

    mvrv_score = pd.Series(
        np.where(
            mvrv <= params.get("mvrv_buy", 1.2),
            1.0,
            np.where(mvrv >= params.get("mvrv_trim", 3.0), 0.15, 0.65),
        ),
        index=df.index,
        dtype=float,
    )
    activity_score = (
        (addresses > addresses.rolling(window, min_periods=max(10, window // 3)).mean()).astype(float)
        + (tx_count > tx_count.rolling(window, min_periods=max(10, window // 3)).mean()).astype(float)
        + (hash_rate > hash_rate.rolling(window, min_periods=max(10, window // 3)).mean()).astype(float)
    ) / 3.0
    netflow = inflow - outflow
    netflow_z = (netflow - netflow.rolling(flow_window, min_periods=max(7, flow_window // 2)).mean()) / netflow.rolling(
        flow_window, min_periods=max(7, flow_window // 2)
    ).std(ddof=0).replace(0, np.nan)
    flow_score = pd.Series(np.where(netflow_z > 1.0, 0.25, np.where(netflow_z < -1.0, 1.0, 0.65)), index=df.index)
    return (0.45 * mvrv_score + 0.35 * activity_score + 0.2 * flow_score).clip(0.0, 1.0).fillna(0.55)


def _sentiment_score(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    value = _numeric(df, "fear_greed_value")
    close = df["close"]
    trend_line = sma(close, params.get("trend_ma", 200))
    cci_value = cci(df["high"], df["low"], close, params.get("cci_len", 30))
    trend_intact = close > trend_line * 0.85
    fear_buy = value <= params.get("fear_buy", 25.0)
    greed_trim = value >= params.get("greed_trim", 75.0)
    pullback = cci_value < -50.0
    score = pd.Series(
        np.where(
            trend_intact & fear_buy,
            1.0,
            np.where(greed_trim, 0.15, np.where(trend_intact & pullback, 0.85, 0.55)),
        ),
        index=df.index,
        dtype=float,
    )
    return score.fillna(0.55)


def _macro_score(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    macro_ma = params.get("macro_ma", 120)
    nasdaq = _numeric(df, "macro_nasdaq")
    us10y = _numeric(df, "macro_us10y")
    dollar = _numeric(df, "macro_dollar")
    liquidity = _numeric(df, "macro_fed_liquidity")
    vix = _numeric(df, "macro_vix")

    nasdaq_ok = (nasdaq > sma(nasdaq, macro_ma)).astype(float)
    dollar_ok = (dollar <= sma(dollar, macro_ma)).astype(float)
    yield_ok = (us10y <= sma(us10y, macro_ma)).astype(float)
    liquidity_ok = (liquidity >= sma(liquidity, macro_ma)).astype(float)
    vix_ok = (vix <= sma(vix, max(20, macro_ma // 2))).astype(float)
    score = 0.35 * nasdaq_ok + 0.2 * dollar_ok + 0.2 * yield_ok + 0.15 * liquidity_ok + 0.1 * vix_ok
    return score.clip(0.0, 1.0).fillna(0.55)


def _trend_condition(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    indicator = params["trend_indicator"]
    if indicator == "ema":
        fast = ema(close, params["fast_ema"])
        slow = ema(close, params["slow_ema"])
        return (fast > slow) & (close > slow)
    if indicator == "supertrend":
        return _supertrend_bullish(high, low, close, params["st_len"], params["st_mult"])
    if indicator == "kama":
        line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        return (close > line) & (line > line.shift(1))
    if indicator == "aroon":
        up, down = aroon(high, low, params["aroon_len"])
        return (up > down) & (up > params["aroon_threshold"])
    raise ValueError(f"Unsupported crypto trend indicator: {indicator}")


def _momentum_condition(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    indicator = params["momentum_indicator"]
    if indicator == "none":
        return pd.Series(True, index=df.index)
    if indicator == "rsi":
        return rsi(close, params["mom_rsi_len"]) > params["mom_rsi_threshold"]
    if indicator == "macd":
        line, signal, hist = macd(close, params["macd_fast"], params["macd_slow"], params["macd_signal"])
        return (line > signal) & (hist > 0)
    if indicator == "cci":
        return cci(high, low, close, params["cci_len"]) > params["cci_threshold"]
    raise ValueError(f"Unsupported crypto momentum indicator: {indicator}")


def _volume_condition(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    indicator = params["volume_indicator"]
    if indicator == "none":
        return pd.Series(True, index=df.index)
    if indicator == "cmf":
        return cmf(high, low, close, volume, params["cmf_len"]) > params["cmf_threshold"]
    if indicator == "mfi":
        return mfi(high, low, close, volume, params["mfi_len"]) > params["mfi_threshold"]
    if indicator == "obv":
        line = obv(close, volume)
        return line > ema(line, params["obv_ema_len"])
    raise ValueError(f"Unsupported crypto volume indicator: {indicator}")


def _momentum_score(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    line, signal, hist = macd(close, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    parts = [
        (rsi(close, params["rsi_len"]) > params["rsi_threshold"]).astype(float),
        ((line > signal) & (hist > 0)).astype(float),
        (cci(df["high"], df["low"], close, params["cci_len"]) > params["cci_threshold"]).astype(float),
    ]
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def _volume_score(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    close = df["close"]
    volume = df["volume"]
    obv_line = obv(close, volume)
    parts = [
        (cmf(df["high"], df["low"], close, volume, params["cmf_len"]) > 0).astype(float),
        (mfi(df["high"], df["low"], close, volume, params["mfi_len"]) > 50).astype(float),
        (obv_line > ema(obv_line, params["obv_ema_len"])).astype(float),
    ]
    return pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)


def build_crypto_strategy_frame(data: pd.DataFrame, family: str, params: dict[str, Any]) -> pd.DataFrame:
    params = normalize_crypto_params(family, params)
    df = data.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    atr_value = atr(high, low, close, params.get("atr_len", 14))

    if family == "crypto_price_trend":
        fast = ema(close, params["fast_ema"])
        slow = ema(close, params["slow_ema"])
        trend_line = sma(close, params["trend_ma"])
        target = pd.Series(np.where((fast > slow) & (close > trend_line), 1.0, 0.0), index=df.index, dtype=float)
        df["target_allocation"] = target
        df["floor_allocation"] = 0.0

    elif family == "crypto_trend_adx":
        fast = moving_average(close, params["fast_ma"], params["ma_type"])
        slow = moving_average(close, params["slow_ma"], params["ma_type"])
        trend_line = sma(close, params["trend_ma"])
        trend_strength = adx(high, low, close, params["adx_len"])
        target = (
            (fast > slow)
            & (close > trend_line)
            & (trend_strength >= params["adx_threshold"])
        ).astype(float)
        df["target_allocation"] = target
        df["floor_allocation"] = 0.0

    elif family == "crypto_donchian_breakout":
        entry_channel = high.rolling(params["entry_lookback"], min_periods=params["entry_lookback"]).max().shift(1)
        exit_channel = low.rolling(params["exit_lookback"], min_periods=params["exit_lookback"]).min().shift(1)
        regime_line = sma(close, params["regime_ma"])
        entry = (close > entry_channel) & (close > regime_line)
        exit_ = (close < exit_channel) | (close < regime_line)
        df["target_allocation"] = _stateful_target(entry, exit_, df.index)
        df["floor_allocation"] = 0.0

    elif family == "crypto_supertrend_combo":
        bullish = _supertrend_bullish(high, low, close, params["st_len"], params["st_mult"])
        confirmation = close > sma(close, params["confirmation_ma"])
        momentum_ok = rsi(close, params["rsi_len"]) > params["rsi_floor"]
        df["target_allocation"] = (bullish & confirmation & momentum_ok).astype(float)
        df["floor_allocation"] = 0.0

    elif family == "crypto_pullback_combo":
        lower, middle, upper = bollinger_bands(close, params["bb_len"], params["bb_std"])
        rsi_value = rsi(close, params["rsi_len"])
        cci_value = cci(high, low, close, params["cci_len"])
        regime_ok = close > sma(close, params["regime_ma"])
        entry = regime_ok & (
            (close <= lower)
            | (rsi_value <= params["rsi_entry"])
            | (cci_value <= params["cci_entry"])
        )
        exit_ = (~regime_ok) | (rsi_value >= params["rsi_exit"]) | (close >= middle)
        df["target_allocation"] = _stateful_target(entry, exit_, df.index)
        df["floor_allocation"] = 0.0

    elif family == "crypto_dca_overlay":
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        cci_value = cci(high, low, close, params["cci_len"])
        trend_intact = close > (kama_line - atr_value * params["trend_buffer_atr"])
        strong = (close > kama_line) & (kama_line > kama_line.shift(1))
        pullback = trend_intact & (cci_value <= params["buy_threshold"])
        sleeve = pd.Series(np.where(pullback, 1.0, np.where(strong, 0.8, np.where(trend_intact, 0.45, 0.1))), index=df.index)
        df["target_allocation"] = _score_to_allocation(params["core_weight"], sleeve)
        df["floor_allocation"] = float(params["core_weight"])

    elif family == "crypto_onchain_overlay":
        score = _onchain_score(df, params)
        df["target_allocation"] = _score_to_allocation(params["core_weight"], score)
        df["floor_allocation"] = float(params["core_weight"])

    elif family == "crypto_sentiment_overlay":
        score = _sentiment_score(df, params)
        df["target_allocation"] = _score_to_allocation(params["core_weight"], score)
        df["floor_allocation"] = float(params["core_weight"])

    elif family == "crypto_macro_overlay":
        btc_trend = (close > sma(close, params["btc_trend_ma"])).astype(float)
        macro = _macro_score(df, params)
        score = np.where(macro >= 0.6, btc_trend, btc_trend * params["risk_reduce"])
        df["target_allocation"] = _score_to_allocation(params["core_weight"], pd.Series(score, index=df.index))
        df["floor_allocation"] = float(params["core_weight"])

    elif family == "crypto_ensemble_overlay":
        price = _price_score(df, params["fast_ema"], params["slow_ema"])
        onchain = _onchain_score(df, params)
        sentiment = _sentiment_score(df, params)
        macro = _macro_score(df, params)
        weights = np.array(
            [
                params["price_weight"],
                params["onchain_weight"],
                params["sentiment_weight"],
                params["macro_weight"],
            ],
            dtype=float,
        )
        if weights.sum() <= 0:
            weights = np.array([1.0, 0.0, 0.0, 0.0])
        weights = weights / weights.sum()
        score = weights[0] * price + weights[1] * onchain + weights[2] * sentiment + weights[3] * macro
        df["target_allocation"] = _score_to_allocation(params["core_weight"], score)
        df["floor_allocation"] = float(params["core_weight"])
        df["crypto_price_score"] = price
        df["crypto_onchain_score"] = onchain
        df["crypto_sentiment_score"] = sentiment
        df["crypto_macro_score"] = macro
        df["crypto_ensemble_score"] = score

    elif family == "crypto_hierarchy_combo":
        trend_ok = _trend_condition(df, params)
        conditions = [trend_ok]
        if params["momentum_indicator"] != "none":
            conditions.append(_momentum_condition(df, params))
        if params["volume_indicator"] != "none":
            conditions.append(_volume_condition(df, params))
        if params["combo_rule"] == "all":
            combo = conditions[0]
            for condition in conditions[1:]:
                combo = combo & condition
        else:
            threshold = min(int(params["min_confirmations"]), len(conditions))
            combo = pd.concat([condition.astype(float) for condition in conditions], axis=1).sum(axis=1) >= threshold
        df["target_allocation"] = combo.fillna(False).astype(float)
        df["floor_allocation"] = 0.0

    elif family == "crypto_multisignal_score":
        ema_part = ((ema(close, params["fast_ema"]) > ema(close, params["slow_ema"])) & (close > ema(close, params["slow_ema"]))).astype(float)
        st_part = _supertrend_bullish(high, low, close, params["st_len"], params["st_mult"]).astype(float)
        kama_line = kama(close, params["kama_len"], params["kama_fast"], params["kama_slow"])
        kama_part = ((close > kama_line) & (kama_line > kama_line.shift(1))).astype(float)
        price = pd.concat([ema_part, st_part, kama_part], axis=1).mean(axis=1).fillna(0.0)
        momentum = _momentum_score(df, params)
        volume = _volume_score(df, params)
        onchain = _onchain_score(df, params)
        sentiment = _sentiment_score(df, params)
        macro = _macro_score(df, params)
        weights = np.array(
            [
                params["price_weight"],
                params["momentum_weight"],
                params["volume_weight"],
                params["onchain_weight"],
                params["sentiment_weight"],
                params["macro_weight"],
            ],
            dtype=float,
        )
        if weights.sum() <= 0:
            weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        weights = weights / weights.sum()
        score = (
            weights[0] * price
            + weights[1] * momentum
            + weights[2] * volume
            + weights[3] * onchain
            + weights[4] * sentiment
            + weights[5] * macro
        ).clip(0.0, 1.0)
        deploy_score = score.where(score >= params["allocation_threshold"], 0.0)
        df["target_allocation"] = _score_to_allocation(params["core_weight"], deploy_score)
        df["floor_allocation"] = float(params["core_weight"])
        df["crypto_price_score"] = price
        df["crypto_momentum_score"] = momentum
        df["crypto_volume_score"] = volume
        df["crypto_onchain_score"] = onchain
        df["crypto_sentiment_score"] = sentiment
        df["crypto_macro_score"] = macro
        df["crypto_ensemble_score"] = score

    else:
        raise ValueError(f"Unsupported crypto family: {family}")

    df["target_allocation"] = df["target_allocation"].clip(df["floor_allocation"], 1.0).fillna(df["floor_allocation"])
    df["entry_signal"] = df["target_allocation"] > df["floor_allocation"]
    df["exit_signal"] = df["target_allocation"] <= df["floor_allocation"]
    df["atr"] = atr_value
    df["stop_mult"] = float(params.get("atr_stop", 0.0))
    df["trail_mult"] = float(params.get("atr_trail", 0.0))
    return df
