from __future__ import annotations

import math

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights) / weights.sum()),
        raw=True,
    )


def moving_average(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    if ma_type.lower() == "ema":
        return ema(series, window)
    return sma(series, window)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(close: pd.Series, window: int = 20, stddev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    dev = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + dev * stddev
    lower = mid - dev * stddev
    return lower, mid, upper


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    atr_series = atr(high, low, close, window)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr_series.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr_series.replace(0, np.nan)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def kama(series: pd.Series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    change = series.diff(er_window).abs()
    volatility = series.diff().abs().rolling(er_window, min_periods=er_window).sum()
    efficiency_ratio = change / volatility.replace(0, np.nan)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    smoothing = ((efficiency_ratio * (fast_sc - slow_sc)) + slow_sc) ** 2

    output = np.full(len(series), np.nan, dtype=float)
    for i in range(len(series)):
        price = float(series.iloc[i])
        if i == 0 or np.isnan(output[i - 1]):
            output[i] = price
            continue
        sc = float(smoothing.iloc[i]) if pd.notna(smoothing.iloc[i]) else slow_sc**2
        output[i] = output[i - 1] + sc * (price - output[i - 1])
    return pd.Series(output, index=series.index, dtype=float)


def hma(series: pd.Series, window: int) -> pd.Series:
    half = max(1, int(window / 2))
    sqrt_window = max(1, int(math.sqrt(window)))
    return wma(2 * wma(series, half) - wma(series, window), sqrt_window)


def aroon(high: pd.Series, low: pd.Series, window: int = 25) -> tuple[pd.Series, pd.Series]:
    def _days_since_high(values: np.ndarray) -> float:
        return float(window - 1 - int(np.argmax(values)))

    def _days_since_low(values: np.ndarray) -> float:
        return float(window - 1 - int(np.argmin(values)))

    periods_since_high = high.rolling(window=window, min_periods=window).apply(_days_since_high, raw=True)
    periods_since_low = low.rolling(window=window, min_periods=window).apply(_days_since_low, raw=True)
    aroon_up = ((window - periods_since_high) / window) * 100.0
    aroon_down = ((window - periods_since_low) / window) * 100.0
    return aroon_up, aroon_down


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    macd_line = fast_line - slow_line
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    mean = sma(typical_price, window)
    mean_dev = typical_price.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    return (typical_price - mean) / (0.015 * mean_dev.replace(0, np.nan))


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    denominator = (high - low).replace(0, np.nan)
    multiplier = ((close - low) - (high - close)) / denominator
    money_flow = multiplier.fillna(0.0) * volume.fillna(0.0)
    return money_flow.rolling(window=window, min_periods=window).sum() / volume.rolling(window=window, min_periods=window).sum().replace(0, np.nan)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume.fillna(0.0)
    direction = typical_price.diff()
    positive = raw_money_flow.where(direction > 0, 0.0)
    negative = raw_money_flow.where(direction < 0, 0.0)
    positive_sum = positive.rolling(window=window, min_periods=window).sum()
    negative_sum = negative.rolling(window=window, min_periods=window).sum()
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + money_ratio))


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    flow = volume.fillna(0.0) * direction
    return flow.cumsum()
