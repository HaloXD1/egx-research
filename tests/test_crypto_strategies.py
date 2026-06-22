from __future__ import annotations

import numpy as np
import pandas as pd

from egx_research.config import BacktestConfig
from egx_research.crypto_research import run_weekly_dca_benchmark
from egx_research.crypto_strategies import CRYPTO_PARAMETER_SPACES, build_crypto_strategy_frame, normalize_crypto_params


def _crypto_frame(rows: int = 420) -> pd.DataFrame:
    idx = np.arange(rows)
    close = 100 + idx * 0.2 + 8 * np.sin(idx / 18)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "open": close * 0.998,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000 + idx,
            "CapMVRVCur": 1.3 + 0.2 * np.sin(idx / 50),
            "AdrActCnt": 500000 + idx * 100,
            "TxCnt": 250000 + idx * 50,
            "HashRate": 1000000 + idx * 200,
            "FlowInExUSD": 1000000 + 1000 * np.sin(idx / 10),
            "FlowOutExUSD": 1100000 + 1000 * np.cos(idx / 10),
            "fear_greed_value": 50 + 30 * np.sin(idx / 25),
            "macro_nasdaq": 10000 + idx * 3,
            "macro_us10y": 4.0 - idx * 0.001,
            "macro_dollar": 100 - idx * 0.002,
            "macro_fed_liquidity": 8000 + idx,
            "macro_vix": 20 + np.sin(idx / 15),
        }
    )


def test_crypto_strategy_families_emit_valid_frames() -> None:
    data = _crypto_frame()
    for family in CRYPTO_PARAMETER_SPACES:
        params = normalize_crypto_params(family, {})
        frame = build_crypto_strategy_frame(data, family, params)
        assert {"target_allocation", "entry_signal", "exit_signal", "atr", "stop_mult", "trail_mult"}.issubset(frame.columns)
        assert frame["target_allocation"].between(0.0, 1.0).all()
        assert frame["target_allocation"].notna().all()
        assert frame["floor_allocation"].between(0.0, 1.0).all()
        assert (frame["target_allocation"] >= frame["floor_allocation"]).all()


def test_weekly_dca_uses_fractional_btc_precision() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        }
    )
    config = BacktestConfig(initial_cash=0.0, monthly_contribution=100.0, fee_bps=0.0, slippage_bps=0.0, share_precision=8)
    result = run_weekly_dca_benchmark(data, 0, len(data) - 1, config)
    assert result.trades["shares"].sum() == 0.5
    assert result.metrics["final_equity"] == 50.0
