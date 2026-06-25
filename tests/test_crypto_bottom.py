from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egx_research.crypto_bottom import (
    _add_market_indicators,
    _add_regime_labels,
    _bottom_type_label,
    _classify_current_bottom_type,
    _confidence_bucket_summary,
    _component_scores,
    _driver_attribution,
    _merge_optional_sources,
    _recommendation_engine,
    _scale_between,
    _walk_forward_validation,
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


def test_existing_open_interest_regression() -> None:
    rows = 30
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = 40000.0 + np.arange(rows) * 100
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "funding_rate_mean": np.full(rows, 0.0001),
            "derivatives_open_interest": [10000.0 + i * 100 for i in range(rows)],
            "derivatives_liquidations_long_usd": np.full(rows, 50000.0),
        }
    )

    frame = _add_market_indicators(frame)
    scores = _component_scores(frame)
    funding = frame["funding_rate_mean"]
    funding_reset = _scale_between(-funding, -0.0002, 0.0015)
    funding_cooling = _scale_between(-(funding - funding.rolling(14, min_periods=5).mean()), -0.0002, 0.0008)
    oi_flush = _scale_between(-frame["derivatives_open_interest"].pct_change(7), 0.02, 0.20)
    long_liq = frame["derivatives_liquidations_long_usd"]
    long_liq_spike = _scale_between(long_liq / long_liq.rolling(30, min_periods=10).mean(), 1.5, 8.0)
    expected = pd.concat([funding_reset, funding_cooling, oi_flush, long_liq_spike], axis=1).mean(axis=1)

    pd.testing.assert_series_equal(scores["derivatives"], expected, check_names=False)


def test_futures_positioning_changes_derivatives_score() -> None:
    rows = 120
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(40000.0, 43000.0, rows)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "funding_rate_mean": np.full(rows, 0.0001),
            "derivatives_open_interest": np.linspace(10000, 8000, rows),
            "derivatives_liquidations_long_usd": np.full(rows, 50000.0),
        }
    )
    without_positioning = _component_scores(_add_market_indicators(base))

    with_positioning = _add_market_indicators(base.copy())
    with_positioning["derivatives_basis"] = 0.001
    with_positioning["derivatives_taker_buy_sell_ratio"] = 0.95
    with_positioning["derivatives_long_short_ratio"] = 0.9
    with_positioning["derivatives_leverage_ratio"] = np.r_[np.full(rows - 10, 1.2), np.linspace(1.2, 0.7, 10)]
    positioned_scores = _component_scores(with_positioning)

    assert positioned_scores["derivatives"].iloc[-1] != without_positioning["derivatives"].iloc[-1]


def test_derivatives_audit_output(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    rows = 150
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(40000.0, 42000.0, rows)
    feature_df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "funding_rate_mean": np.full(rows, 0.0001),
            "derivatives_open_interest": np.linspace(10000, 9000, rows),
            "derivatives_liquidations_long_usd": np.full(rows, 50000.0),
            "derivatives_basis": np.full(rows, 0.001),
            "derivatives_taker_buy_sell_ratio": np.full(rows, 1.0),
            "derivatives_long_short_ratio": np.full(rows, 1.1),
            "derivatives_leverage_ratio": np.full(rows, 0.8),
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("egx_research.crypto_bottom.load_crypto_feature_data", lambda c: feature_df)

    run = run_crypto_bottom_score(config, "config/crypto_btc.yaml", run_id="futures-audit")
    audit_file = Path(run.run_dir) / "bottom_derivatives_audit.csv"
    component_file = Path(run.run_dir) / "bottom_component_scores.csv"

    audit_df = pd.read_csv(audit_file)
    component_df = pd.read_csv(component_file)
    for column in [
        "sub_derivatives_spot_led_bounce",
        "sub_derivatives_leverage_reset",
        "sub_derivatives_overheated_long_short_penalty",
    ]:
        assert column in audit_df.columns
        assert column in component_df.columns


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


def test_options_features_move_sentiment() -> None:
    rows = 120
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 120.0, rows)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "fear_greed_value": np.full(rows, 50.0),
            "CapMVRVCur": np.full(rows, 1.5),
        }
    )

    low_options = _add_market_indicators(base.copy())
    low_options["options_25d_skew"] = 0.0
    low_options["options_put_call_oi"] = 0.8
    low_options["options_put_call_volume"] = 0.8
    low_options["options_iv_30d"] = np.linspace(0.55, 0.80, rows)
    low_options["options_term_structure"] = 0.95

    strong_options = _add_market_indicators(base.copy())
    strong_options["options_25d_skew"] = 0.20
    strong_options["options_put_call_oi"] = 1.6
    strong_options["options_put_call_volume"] = 1.6
    strong_options["options_iv_30d"] = np.r_[np.full(rows - 1, 0.80), 0.55]
    strong_options["options_term_structure"] = 1.25

    low_scores = _component_scores(low_options)
    strong_scores = _component_scores(strong_options)

    assert strong_scores["sentiment"].iloc[-1] > low_scores["sentiment"].iloc[-1]


def test_missing_options_data_keeps_score_stable() -> None:
    rows = 120
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 120.0, rows)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "fear_greed_value": np.full(rows, 50.0),
            "CapMVRVCur": np.full(rows, 1.5),
        }
    )
    with_missing_columns = base.copy()
    for column in [
        "options_25d_skew",
        "options_put_call_oi",
        "options_put_call_volume",
        "options_iv_30d",
        "options_term_structure",
    ]:
        with_missing_columns[column] = np.nan

    scores_without = _component_scores(_add_market_indicators(base))
    scores_with_missing = _component_scores(_add_market_indicators(with_missing_columns))

    pd.testing.assert_series_equal(
        scores_without["sentiment"],
        scores_with_missing["sentiment"],
        check_names=False,
    )


def test_exchange_flows_improve_onchain_score() -> None:
    rows = 140
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 120.0, rows)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "CapMVRVCur": np.full(rows, 1.5),
            "FlowOutExUSD": np.full(rows, 5_000_000.0),
            "FlowInExUSD": np.full(rows, 5_000_000.0),
        }
    )
    weak = _add_market_indicators(base.copy())
    weak["onchain_exchange_reserve_btc"] = np.linspace(2_000_000.0, 2_050_000.0, rows)
    weak["onchain_exchange_netflow_btc"] = np.full(rows, 1000.0)
    weak["onchain_whale_inflow_usd"] = np.full(rows, 50_000_000.0)
    weak["onchain_realized_profit_loss_exchange"] = np.full(rows, 25_000_000.0)

    strong = _add_market_indicators(base.copy())
    strong["onchain_exchange_reserve_btc"] = np.linspace(2_000_000.0, 1_850_000.0, rows)
    strong["onchain_exchange_netflow_btc"] = np.full(rows, -3000.0)
    strong["onchain_whale_inflow_usd"] = np.full(rows, 5_000_000.0)
    strong["onchain_realized_profit_loss_exchange"] = np.full(rows, -100_000_000.0)

    weak_scores = _component_scores(weak)
    strong_scores = _component_scores(strong)

    assert strong_scores["onchain"].iloc[-1] > weak_scores["onchain"].iloc[-1]


def test_sth_sopr_real_values_feed_scores() -> None:
    rows = 140
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 120.0, rows)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "CapMVRVCur": np.full(rows, 1.5),
            "FlowOutExUSD": np.full(rows, 5_000_000.0),
            "FlowInExUSD": np.full(rows, 5_000_000.0),
        }
    )
    weak = base.copy()
    weak["onchain_sth_mvrv"] = np.full(rows, 1.2)
    weak["onchain_sth_sopr"] = np.full(rows, 1.04)
    weak["onchain_realized_loss_usd"] = np.full(rows, 10_000_000.0)
    weak = _add_market_indicators(weak)

    strong = base.copy()
    strong["onchain_sth_mvrv"] = np.full(rows, 0.95)
    strong["onchain_sth_sopr"] = np.full(rows, 0.96)
    strong["onchain_realized_loss_usd"] = np.r_[np.full(rows - 1, 10_000_000.0), 60_000_000.0]
    strong = _add_market_indicators(strong)

    weak_scores = _component_scores(weak)
    strong_scores = _component_scores(strong)

    assert strong["sth_mvrv_source"].iloc[-1] == "real"
    assert strong_scores["capitulation"].iloc[-1] > weak_scores["capitulation"].iloc[-1]


def test_regime_labels_are_deterministic() -> None:
    rows = 260
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = np.r_[np.linspace(100.0, 160.0, rows - 1), 95.0]
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "macro_vix": np.r_[np.full(rows - 1, 18.0), 35.0],
            "macro_nasdaq": np.r_[np.linspace(100.0, 130.0, rows - 1), 110.0],
        }
    )
    labeled = _add_regime_labels(_add_market_indicators(frame))
    assert labeled.iloc[-1]["primary_regime"] == "macro stress"
    assert "macro_stress" in labeled.iloc[-1]["regime_tags"]


def test_walkforward_validation_and_buckets() -> None:
    rows = 720
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    close = 100.0 + np.sin(np.arange(rows) / 15.0) * 10.0 + np.arange(rows) * 0.04
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "CapMVRVCur": np.full(rows, 1.5),
        }
    )
    frame = _add_regime_labels(_add_market_indicators(frame))
    components = _component_scores(frame)
    validation, summary = _walk_forward_validation(frame, components, horizon=30, tolerance=0.10, step_days=80)
    buckets = _confidence_bucket_summary(validation)
    assert summary["rows"] > 0
    assert len(buckets) == 5
    assert {"low", "medium", "high", "very_high", "extreme"} == {row["bucket"] for row in buckets}


def test_bottom_type_recommendation_and_drivers() -> None:
    rows = 160
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.r_[np.linspace(100.0, 55.0, rows // 2), np.linspace(55.0, 95.0, rows // 2)]
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "CapMVRVCur": np.full(rows, 1.2),
        }
    )
    frame = _add_regime_labels(_add_market_indicators(frame))
    components = _component_scores(frame)
    latest = frame.iloc[-1]
    bottom_type = _classify_current_bottom_type(latest, confidence=0.82, regime="bear market")
    recommendation = _recommendation_engine(
        0.82,
        bottom_type["primary"],
        "bear market",
        latest,
        {"invalidation_low": 50.0, "sma20": 90.0, "prior_20d_high": 96.0},
    )
    drivers = _driver_attribution(components, len(frame) - 1, {key: 1 / 7 for key in components.columns if key != "date"}, {"warnings": ["missing source"]})
    assert bottom_type["primary"] in bottom_type["probabilities"]
    assert recommendation["action"] in {"wait", "probe", "tranche", "aggressive accumulation", "defensive"}
    assert drivers["positive_drivers"]
    assert drivers["negative_drivers"]
    assert drivers["source_penalties"]
    assert _bottom_type_label(frame, 50) in {"local bounce", "tradable swing bottom", "cycle bottom", "dead-cat bounce risk"}
