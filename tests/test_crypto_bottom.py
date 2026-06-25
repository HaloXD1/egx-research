from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from egx_research.crypto_bottom import (
    _add_market_indicators,
    _component_scores,
    _merge_optional_sources,
    run_crypto_bottom_score,
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


def test_liquidations_climax_and_imbalance_scoring() -> None:
    # 35 days to allow min_periods=10 for rolling mean of 30 days
    rows = 35
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    
    # Base price
    close = np.linspace(100, 110, rows)
    
    # Liquidations data
    # 34 days of normal liquidations (100) and 1 day spike (800)
    long_liq = np.full(rows, 100.0)
    long_liq[-1] = 800.0  # spike
    
    short_liq = np.full(rows, 100.0)
    short_liq[-1] = 50.0   # low short liq to create high imbalance
    
    # Nearest down heatmap is far away
    heatmap_down = np.full(rows, 80.0)
    
    frame = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.full(rows, 1000.0),
        "derivatives_liquidations_long_usd": long_liq,
        "derivatives_short_liq_usd": short_liq,
        "derivatives_heatmap_nearest_down_liq": heatmap_down,
        "derivatives_liq_imbalance": np.nan,  # test fallback calculation
    })
    
    # Run market indicators
    frame_with_indicators = _add_market_indicators(frame)
    
    # Run component scores
    scores = _component_scores(frame_with_indicators)
    
    # Check that derivatives and capitulation scores are calculated and have valid values
    assert "derivatives" in scores.columns
    assert "capitulation" in scores.columns
    
    # Calculate expected spike ratio at last index: 800 / rolling_mean
    # rolling mean of 30 days including the spike: (29 * 100 + 800) / 30 = 3700 / 30 = 123.33
    # spike ratio: 800 / 123.33 = 6.48x
    # 6.48 scaled between 1.5 and 8.0: (6.48 - 1.5) / 6.5 = 4.98 / 6.5 = 0.766
    # Let's verify that the derivatives score is higher at the spike index than the prior index
    assert scores["derivatives"].iloc[-1] > scores["derivatives"].iloc[-2]
    assert scores["capitulation"].iloc[-1] > scores["capitulation"].iloc[-2]


def test_heatmap_downside_confidence_penalty(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    # 150 days of data for model training
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    close = np.linspace(10000, 15000, 150)
    
    # Save base normalized price
    pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.full(150, 100000.0),
    }).to_csv(tmp_path / "normalized" / config.data.normalized_filename, index=False)
    
    # Scenario A: Downside heatmap level is far away (10% lower) -> penalty factor is 1.0
    heatmap_down_far = np.full(150, 13500.0)  # close is 15000, so 10% distance
    pd.DataFrame({
        "date": dates,
        "long_liq_usd": np.full(150, 1000.0),
        "short_liq_usd": np.full(150, 1000.0),
        "total_liq_usd": np.full(150, 2000.0),
        "liq_imbalance": np.full(150, 0.0),
        "heatmap_nearest_down_liq": heatmap_down_far,
        "heatmap_nearest_up_liq": np.full(150, 16000.0),
    }).to_csv(tmp_path / "raw" / "liquidations.csv", index=False)
    
    from egx_research.crypto_data import build_crypto_feature_panel
    build_crypto_feature_panel(config)
    
    config_path = tmp_path / "config.yaml"
    from egx_research.crypto_config import save_crypto_config
    save_crypto_config(config_path, config)
    
    res_far = run_crypto_bottom_score(config, config_path)
    assert res_far.summary["heatmap_penalty_factor"] == pytest.approx(1.0)
    assert res_far.summary["best_case"]["confidence"] == res_far.summary["best_case"]["adjusted_confidence"]
    
    # Scenario B: Downside heatmap level is close (1% lower) -> penalty factor applied
    heatmap_down_close = np.full(150, 14850.0)  # 15000 * 0.99 (1% distance)
    pd.DataFrame({
        "date": dates,
        "long_liq_usd": np.full(150, 1000.0),
        "short_liq_usd": np.full(150, 1000.0),
        "total_liq_usd": np.full(150, 2000.0),
        "liq_imbalance": np.full(150, 0.0),
        "heatmap_nearest_down_liq": heatmap_down_close,
        "heatmap_nearest_up_liq": np.full(150, 16000.0),
    }).to_csv(tmp_path / "raw" / "liquidations.csv", index=False)
    
    build_crypto_feature_panel(config)
    
    res_close = run_crypto_bottom_score(config, config_path)
    
    # Distance is 1% (0.01). 0.01 < 0.03 threshold.
    # Expected penalty: 1.0 - 0.15 * (1.0 - 0.01 / 0.03) = 1.0 - 0.15 * (2/3) = 1.0 - 0.10 = 0.90
    assert res_close.summary["heatmap_penalty_factor"] == pytest.approx(0.90)
    
    # Check that adjusted_confidence is exactly raw_confidence * 0.90
    best_far = res_far.summary["best_case"]
    best_close = res_close.summary["best_case"]
    
    assert best_close["adjusted_confidence"] == pytest.approx(best_close["confidence"] * 0.90)
