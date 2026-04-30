from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(rows: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    trend = np.linspace(0, 40, rows)
    cycle = 8 * np.sin(np.linspace(0, 18 * np.pi, rows))
    noise = rng.normal(0, 1.2, rows).cumsum() * 0.15
    close = 100 + trend + cycle + noise
    open_ = close * (1 + rng.normal(0, 0.002, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, rows))
    volume = rng.integers(1000, 5000, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture()
def synthetic_ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv()
