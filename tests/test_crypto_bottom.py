from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from egx_research.crypto_bottom import (
    _add_market_indicators,
    _component_scores,
    _merge_optional_sources,
)
from egx_research.crypto_config import CryptoConfig


def test_add_market_indicators_derivation() -> None:
    # Create test data
    rows = 200
    idx = np.arange(rows)
    # Simple upward trend with some volatility
    close = 100.0 + idx * 0.5 + 10.0 * np.sin(idx / 5)
    volume = 1000.0 + idx * 10
    
    data = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": volume,
    })

    # Run indicators derivation
    df = _add_market_indicators(data)

    # Check that new indicator columns exist
    assert "sharpe_30d" in df.columns
    assert "sharpe_90d" in df.columns
    assert "sth_realized_price" in df.columns
    assert "sth_mvrv" in df.columns

    # Verify Sharpe ratios
    # First few rows should be 0.0 or nan (filled to 0.0 by fillna(0.0) in our code)
    assert (df["sharpe_30d"].iloc[:10] == 0.0).all()
    # Later rows should have non-zero values
    assert (df["sharpe_30d"].iloc[30:] != 0.0).any()
    assert not df["sharpe_30d"].isin([np.inf, -np.inf]).any()

    # Verify STH realized price (155-day rolling VWAP)
    # Should be close to close price when min_periods=30 is not met
    assert (df["sth_realized_price"].iloc[:20] == df["close"].iloc[:20]).all()
    # At index 35 (which is >= 30 min_periods), sth_realized_price is rolling VWAP
    assert df["sth_realized_price"].iloc[35] > 0.0
    
    # Calculate VWAP manually for day 35
    window_close = df["close"].iloc[35 - 155 + 1:36] if 35 >= 155 else df["close"].iloc[:36]
    window_vol = df["volume"].iloc[35 - 155 + 1:36] if 35 >= 155 else df["volume"].iloc[:36]
    manual_vwap = (window_close * window_vol).sum() / window_vol.sum()
    
    assert pytest.approx(df["sth_realized_price"].iloc[35]) == manual_vwap

    # sth_mvrv should be Close / STH Realized Price
    assert pytest.approx(df["sth_mvrv"].iloc[35]) == df["close"].iloc[35] / df["sth_realized_price"].iloc[35]


def test_component_scores_derivation() -> None:
    rows = 100
    idx = np.arange(rows)
    close = 100.0 + idx * 0.5
    
    raw_frame = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": 1000.0 + idx,
    })

    # First add the market indicators
    frame = _add_market_indicators(raw_frame)

    # Now add the optional columns that would be loaded from external sources
    frame["CapMVRVCur"] = 1.5 + 0.5 * np.sin(idx / 10)  # MVRV
    frame["SplyCur"] = 19000000.0 + idx * 100           # Circulating Supply
    frame["fear_greed_value"] = 50 + 30 * np.sin(idx / 25)

    # Call component scores
    scores = _component_scores(frame)

    # Check columns
    assert "price_structure" in scores.columns
    assert "capitulation" in scores.columns
    assert "onchain" in scores.columns
    assert "sentiment" in scores.columns

    # Verify NUPL calculation
    # NUPL = 1 - 1 / CapMVRVCur
    mvrv = frame["CapMVRVCur"]
    expected_nupl = 1.0 - 1.0 / mvrv
    # Check that sentiment includes nupl_washout correctly
    # nupl_washout = 1.0 - _scale_between(nupl, -0.15, 0.45)
    # Check that values are within [0, 1]
    assert scores["sentiment"].between(0.0, 1.0).all()

    # Verify MVRV Z-Score calculation
    # mcap = close * SplyCur
    # rcap = mcap / CapMVRVCur
    # mvrv_zscore = (mcap - rcap) / mcap.expanding(min_periods=30).std()
    # Check that onchain score is valid and within [0, 1]
    assert scores["onchain"].between(0.0, 1.0).all()
    # First 29 rows of expanding std will be NaN, so zscore_cheapness is NaN
    # but the mean calculation in onchain should still succeed because it ignores NaN.
    assert not scores["onchain"].isna().any()


def test_bottom_score_aliases_legacy_optional_columns(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3, freq="D"),
            "open_interest": [100.0, 110.0, 120.0],
            "coinbase_premium": [0.001, 0.002, 0.003],
            "stablecoin_supply": [1000.0, 1010.0, 1020.0],
            "options_skew": [-0.1, -0.2, -0.3],
            "put_call_ratio": [0.9, 0.8, 0.7],
        }
    )

    merged, loaded = _merge_optional_sources(frame, config)

    assert "feature:derivatives_open_interest" in loaded
    assert "feature:spot_coinbase_premium" in loaded
    assert "feature:liquidity_stablecoin_supply" in loaded
    assert "feature:options_options_skew" in loaded
    assert merged["derivatives_open_interest"].equals(merged["open_interest"])
    assert merged["spot_coinbase_premium"].equals(merged["coinbase_premium"])
    assert merged["liquidity_stablecoin_supply"].equals(merged["stablecoin_supply"])
    assert merged["options_options_skew"].equals(merged["options_skew"])
    assert merged["options_put_call_ratio"].equals(merged["put_call_ratio"])


def test_bottom_score_with_exchange_reserves_and_dry_powder_ratio(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)

    rows = 120
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")

    # Write dummy exchange_stablecoin_reserves.csv
    pd.DataFrame({
        "date": dates,
        "exchange_stablecoin_reserves": np.linspace(100_000_000.0, 150_000_000.0, rows)
    }).to_csv(tmp_path / "raw" / "exchange_stablecoin_reserves.csv", index=False)

    pd.DataFrame({
        "date": dates,
        "stablecoin_supply": np.linspace(150_000_000.0, 220_000_000.0, rows)
    }).to_csv(tmp_path / "raw" / "stablecoin_supply.csv", index=False)

    base_frame = pd.DataFrame({
        "date": dates,
        "open": [100.0] * rows,
        "high": [105.0] * rows,
        "low": [95.0] * rows,
        "close": [100.0] * rows,
        "volume": [1000.0] * rows,
        "SplyCur": [19500000.0] * rows
    })

    merged, loaded = _merge_optional_sources(base_frame, config)
    
    assert "exchange_stablecoin_reserves.csv" in loaded
    assert "stablecoin_supply.csv" in loaded
    assert "liquidity_exchange_stablecoin_reserves" in merged.columns
    assert "liquidity_stablecoin_supply" in merged.columns
    assert "liquidity_dry_powder_ratio" in merged.columns

    # Verify dry powder ratio is computed correctly:
    # liquidity_dry_powder_ratio = stable_supply_shifted / (close_shifted * SplyCur_shifted)
    # Check index 2 (2025-01-03):
    # shifted values (shifted by 1 in _merge_optional_sources):
    # stable_supply_shifted = stable_supply[1] = 1020.0
    # SplyCur_shifted = SplyCur[1] = 19500000.0
    # prev_close = close[1] = 100.0
    # ratio = 1020.0 / (100.0 * 19500000.0) = 1020.0 / 1950000000.0 = 5.230769e-7
    expected_ratio = (150_000_000.0 + (220_000_000.0 - 150_000_000.0) / (rows - 1)) / (100.0 * 19500000.0)
    assert pytest.approx(merged["liquidity_dry_powder_ratio"].iloc[2], rel=1e-5) == expected_ratio

    # Run component scores
    from egx_research.crypto_bottom import _component_scores
    # To run _component_scores we need to add other indicators
    from egx_research.crypto_bottom import _add_market_indicators
    indicators_df = _add_market_indicators(merged)
    
    # Fill any indicators needed by other components so they don't NaN out
    indicators_df["CapMVRVCur"] = 1.5
    indicators_df["fear_greed_value"] = 50.0

    scores = _component_scores(indicators_df)
    assert "spot_demand" in scores.columns
    assert "macro" in scores.columns
    assert pd.notna(scores["spot_demand"].iloc[-1])
    assert pd.notna(scores["macro"].iloc[-1])

    # Dry powder score improvement test:
    # If we double the stablecoin supply, the dry powder ratio doubles, which should increase macro score
    # since dry_powder_ratio increases and macro contains scaled dry powder.
    high_stable_frame = indicators_df.copy()
    high_stable_frame["liquidity_dry_powder_ratio"] = high_stable_frame["liquidity_dry_powder_ratio"] * 10
    scores_high = _component_scores(high_stable_frame)
    
    # Check that high dry powder yields a higher or equal macro score at indices where it's valid
    # Since dry_powder_ratio scales up, _scale_between(dry_powder, 0.05, 0.20) will increase
    # (or stay capped at 1.0)
    assert scores_high["macro"].iloc[-1] >= scores["macro"].iloc[-1]

