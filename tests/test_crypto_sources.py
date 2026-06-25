import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from egx_research.crypto_config import CryptoConfig
from egx_research.crypto_sources import (
    get_source_api_key,
    get_source_env_var,
    is_source_enabled,
    is_source_required,
    run_data_quality_checks,
    SOURCE_REGISTRY,
)
from egx_research.crypto_data import sync_crypto_data, fetch_btc_etf_flows
from egx_research.crypto_bottom import run_crypto_bottom_score


def test_get_source_api_key(monkeypatch) -> None:
    assert get_source_api_key("coinmetrics", "") is None
    
    monkeypatch.setenv("COINMETRICS_API_KEY", "secret_key_123")
    assert get_source_api_key("coinmetrics", "COINMETRICS_API_KEY") == "secret_key_123"


def test_is_source_enabled_and_required() -> None:
    config = CryptoConfig()
    
    # Defaults
    assert is_source_enabled(config, "coinmetrics") is True
    assert is_source_required(config, "coinmetrics") is False

    # Explicit override enabled=False
    config.sources.optional_sources["coinmetrics"]["enabled"] = False
    assert is_source_enabled(config, "coinmetrics") is False

    # Explicit override required=True
    config.sources.optional_sources["coinmetrics"]["required"] = True
    assert is_source_required(config, "coinmetrics") is True


def test_sync_skips_disabled_source(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    # Mock fetchers to return dummy df
    dummy_price = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "open": [10, 11, 12, 13, 14],
        "high": [11, 12, 13, 14, 15],
        "low": [9, 10, 11, 12, 13],
        "close": [10, 11, 12, 13, 14],
        "volume": [100, 110, 120, 130, 140],
    })
    monkeypatch.setattr("egx_research.crypto_data.fetch_binance_klines", lambda c: dummy_price)

    # Disable coinmetrics
    config.sources.optional_sources["coinmetrics"]["enabled"] = False

    # Mock other fetchers to return empty dfs to speed up
    for s in SOURCE_REGISTRY:
        if s not in ("binance", "coinmetrics"):
            monkeypatch.setattr(f"egx_research.crypto_data.fetch_{s}" if s != "options_skew" else "egx_research.crypto_data.fetch_deribit_options", lambda c: pd.DataFrame(columns=["date"]))

    sync_crypto_data(config)

    # Assert raw_dir does NOT contain coinmetrics_btc.csv
    assert not (tmp_path / "raw" / "coinmetrics_btc.csv").exists()
    
    # Check sync summary
    summary_path = tmp_path / "features" / "sync_summary.json"
    assert summary_path.exists()
    import json
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["source_statuses"]["coinmetrics"] == "disabled"


def test_sync_runs_public_source_without_credentials(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    dummy_price = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "open": [10, 11, 12, 13, 14],
        "high": [11, 12, 13, 14, 15],
        "low": [9, 10, 11, 12, 13],
        "close": [10, 11, 12, 13, 14],
        "volume": [100, 110, 120, 130, 140],
    })
    monkeypatch.setattr("egx_research.crypto_data.fetch_binance_klines", lambda c: dummy_price)

    # Ensure credential for coinmetrics is missing in environment
    monkeypatch.delenv("COINMETRICS_API_KEY", raising=False)
    coinmetrics = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "CapMVRVCur": [1.1, 1.2, 1.3, 1.4, 1.5],
        "FlowInExUSD": [10, 11, 12, 13, 14],
        "FlowOutExUSD": [11, 12, 13, 14, 15],
        "AdrActCnt": [100, 101, 102, 103, 104],
        "TxCnt": [50, 51, 52, 53, 54],
        "HashRate": [1000, 1001, 1002, 1003, 1004],
        "SplyCur": [19000, 19001, 19002, 19003, 19004],
    })
    monkeypatch.setattr(
        "egx_research.crypto_data.fetch_coinmetrics",
        lambda c: (coinmetrics, {"source": "test"}),
    )
    
    # Mock other fetchers to return empty dfs to speed up
    for s in SOURCE_REGISTRY:
        if s not in ("binance", "coinmetrics"):
            monkeypatch.setattr(f"egx_research.crypto_data.fetch_{s}" if s != "options_skew" else "egx_research.crypto_data.fetch_deribit_options", lambda c: pd.DataFrame(columns=["date"]))

    sync_crypto_data(config)

    # Check sync summary status
    summary_path = tmp_path / "features" / "sync_summary.json"
    with open(summary_path) as f:
        import json
        summary = json.load(f)
    assert summary["source_statuses"]["coinmetrics"] == "success"
    assert (tmp_path / "raw" / "coinmetrics_btc.csv").exists()


def test_config_env_var_override_is_used() -> None:
    config = CryptoConfig()
    config.sources.optional_sources["coinmetrics"]["env_var"] = "CUSTOM_COINMETRICS_KEY"
    assert get_source_env_var(config, "coinmetrics") == "CUSTOM_COINMETRICS_KEY"


def test_sync_fails_missing_credentials_if_required(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    # Require coinmetrics and make credentials missing
    config.sources.optional_sources["coinmetrics"]["required"] = True
    config.sources.optional_sources["coinmetrics"]["env_var"] = "COINMETRICS_API_KEY"
    monkeypatch.delenv("COINMETRICS_API_KEY", raising=False)

    dummy_price = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "open": [10, 11, 12, 13, 14],
        "high": [11, 12, 13, 14, 15],
        "low": [9, 10, 11, 12, 13],
        "close": [10, 11, 12, 13, 14],
        "volume": [100, 110, 120, 130, 140],
    })
    monkeypatch.setattr("egx_research.crypto_data.fetch_binance_klines", lambda c: dummy_price)

    with pytest.raises(ValueError, match="Missing credentials for required source 'coinmetrics'"):
        sync_crypto_data(config)


def test_disabled_critical_source_is_not_penalized_as_missing(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    config.sources.optional_sources["coinmetrics"]["enabled"] = False
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    panel = pd.DataFrame({
        "date": dates,
        "open": np.linspace(10, 20, 50),
        "high": np.linspace(11, 21, 50),
        "low": np.linspace(9, 19, 50),
        "close": np.linspace(10, 20, 50),
        "volume": np.linspace(100, 200, 50),
    })
    panel.to_csv(tmp_path / "normalized" / config.data.normalized_filename, index=False)

    dq = run_data_quality_checks(config, panel, dates[-1])

    assert dq["source_statuses"]["coinmetrics"]["status"] == "disabled"
    assert dq["reliability_score"] == pytest.approx(0.55)


def test_run_data_quality_checks(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    # Let's create dummy files in raw
    # binance (normalized)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    pd.DataFrame({
        "date": dates,
        "open": np.linspace(10, 20, 50),
        "high": np.linspace(11, 21, 50),
        "low": np.linspace(9, 19, 50),
        "close": np.linspace(10, 20, 50),
        "volume": np.linspace(100, 200, 50),
    }).to_csv(tmp_path / "normalized" / config.data.normalized_filename, index=False)

    # coinmetrics - fresh
    pd.DataFrame({
        "date": dates,
        "CapMVRVCur": np.linspace(1.2, 1.5, 50),
        "FlowInExUSD": np.linspace(10, 20, 50),
        "FlowOutExUSD": np.linspace(11, 21, 50),
        "AdrActCnt": np.linspace(100, 200, 50),
        "TxCnt": np.linspace(50, 100, 50),
        "HashRate": np.linspace(1000, 2000, 50),
        "SplyCur": np.linspace(19000, 20000, 50),
    }).to_csv(tmp_path / "raw" / "coinmetrics_btc.csv", index=False)

    # fear_greed - stale (last data is older than SLA)
    stale_dates = dates[:40]
    pd.DataFrame({
        "date": stale_dates,
        "fear_greed_value": np.linspace(30, 40, 40),
    }).to_csv(tmp_path / "raw" / "fear_greed.csv", index=False)

    # funding_rates - fresh
    pd.DataFrame({
        "date": dates,
        "funding_rate_mean": np.linspace(-0.0001, 0.0002, 50),
    }).to_csv(tmp_path / "raw" / "funding_rates.csv", index=False)

    # open_interest - partial/missing some columns
    pd.DataFrame({
        "date": dates,
        # open_interest column is missing, so it's missing the required columns
    }).to_csv(tmp_path / "raw" / "open_interest.csv", index=False)

    # Build the panel
    from egx_research.crypto_data import build_crypto_feature_panel
    panel = build_crypto_feature_panel(config)

    as_of = dates[-1]
    dq = run_data_quality_checks(config, panel, as_of)

    # Verify statuses
    assert dq["source_statuses"]["binance"]["status"] == "success"
    assert dq["source_statuses"]["coinmetrics"]["status"] == "success"
    assert dq["source_statuses"]["fear_greed"]["status"] == "stale"
    assert dq["source_statuses"]["open_interest"]["status"] == "missing"
    assert dq["source_statuses"]["coinbase_premium"]["status"] == "missing"  # no file on disk

    # Verify penalty rating
    # binance (success), coinmetrics (success), fear_greed (stale -0.10), open_interest (missing -0.15).
    # Expected reliability_score = 1.0 - 0.10 - 0.15 = 0.75
    assert dq["reliability_score"] == pytest.approx(0.75)
    assert dq["reliability_rating"] == "Degraded"
    assert len(dq["warnings"]) > 0


def test_bottom_score_integration_does_not_affect_math(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    # Create sufficient dummy data for bottom score to run
    dates = pd.date_range("2024-01-01", periods=150, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "open": np.linspace(10, 20, 150),
        "high": np.linspace(11, 21, 150),
        "low": np.linspace(9, 19, 150),
        "close": np.linspace(10, 20, 150),
        "volume": np.linspace(100, 200, 150),
    })
    df.to_csv(tmp_path / "normalized" / config.data.normalized_filename, index=False)

    # Build the feature panel
    from egx_research.crypto_data import build_crypto_feature_panel
    panel = build_crypto_feature_panel(config)
    panel.to_csv(tmp_path / "features" / config.data.features_filename, index=False)

    # Save the config
    from egx_research.crypto_config import save_crypto_config
    config_path = tmp_path / "crypto_btc.yaml"
    save_crypto_config(config_path, config)

    # Run bottom score
    res = run_crypto_bottom_score(config, config_path)
    
    # Assert data quality metrics exist in result summary
    assert "data_quality" in res.summary
    assert "reliability_rating" in res.summary
    assert "reliability_note" in res.summary
    
    # Confirm math did not change (confidence is NOT multiplied by reliability_score yet)
    # The default confidence with all missing optional sources is determined by the model
    # but the value should not be affected by the reliability_score.
    best_row = res.summary["best_case"]
    assert best_row["confidence_pct"] == pytest.approx(best_row["confidence"] * 100)


def test_etf_flows_public_table_shape_change(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    fallback_path = tmp_path / "raw" / "btc_etf_flows.csv"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)

    # HTML table missing required IBIT column
    bad_html = """
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>FBTC</th>
          <th>ARKB</th>
          <th>BITB</th>
          <th>GBTC</th>
          <th>Total</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>2024-06-01</td>
          <td>10.0</td>
          <td>5.0</td>
          <td>2.0</td>
          <td>-5.0</td>
          <td>12.0</td>
        </tr>
      </tbody>
    </table>
    """
    monkeypatch.setattr("egx_research.crypto_data._get_text", lambda url, params=None: bad_html)

    # Without fallback CSV, should raise error
    with pytest.raises(ValueError, match="No table in public ETF URL matched the required schema"):
        fetch_btc_etf_flows(config)

    # Write fallback CSV
    pd.DataFrame({
        "date": [pd.Timestamp("2024-06-01")],
        "etf_flow_ibit": [10_000_000.0],
        "etf_flow_fbtc": [10_000_000.0],
        "etf_flow_arkb": [5_000_000.0],
        "etf_flow_bitb": [2_000_000.0],
        "etf_flow_gbtc": [-5_000_000.0],
        "etf_net_flow_usd": [22_000_000.0]
    }).to_csv(fallback_path, index=False)

    # Should fall back to local CSV successfully
    df, summary = fetch_btc_etf_flows(config)
    assert summary["status"] == "fallback_local"
    assert len(df) == 1
    assert df.loc[0, "etf_flow_ibit"] == 10_000_000.0


def test_etf_flows_csv_fallback(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    fallback_path = tmp_path / "raw" / "btc_etf_flows.csv"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)

    # Simulate network error
    def mock_get_text(url, params=None):
        raise ConnectionError("Network failure")
    monkeypatch.setattr("egx_research.crypto_data._get_text", mock_get_text)

    # Without fallback CSV, should raise error
    with pytest.raises(ValueError, match="Public ETF fetch failed: Network failure"):
        fetch_btc_etf_flows(config)

    # Write fallback CSV
    pd.DataFrame({
        "date": [pd.Timestamp("2024-06-01")],
        "etf_flow_ibit": [10_000_000.0],
        "etf_flow_fbtc": [10_000_000.0],
        "etf_flow_arkb": [5_000_000.0],
        "etf_flow_bitb": [2_000_000.0],
        "etf_flow_gbtc": [-5_000_000.0],
        "etf_net_flow_usd": [22_000_000.0]
    }).to_csv(fallback_path, index=False)

    df, summary = fetch_btc_etf_flows(config)
    assert summary["status"] == "fallback_local"
    assert len(df) == 1


def test_etf_flows_unit_and_date_validation(tmp_path, monkeypatch) -> None:
    from datetime import UTC
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    fallback_path = tmp_path / "raw" / "btc_etf_flows.csv"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Future date validation
    tomorrow = pd.Timestamp.now(UTC).tz_localize(None).normalize() + pd.Timedelta(days=1)
    df_future = pd.DataFrame({
        "date": [tomorrow],
        "etf_flow_ibit": [1.0e6], "etf_flow_fbtc": [1.0e6], "etf_flow_arkb": [1.0e6],
        "etf_flow_bitb": [1.0e6], "etf_flow_gbtc": [1.0e6], "etf_net_flow_usd": [5.0e6]
    })
    df_future.to_csv(fallback_path, index=False)
    monkeypatch.setattr("egx_research.crypto_data._get_text", lambda u, p=None: "raise exception")
    with pytest.raises(ValueError, match="future dates"):
        fetch_btc_etf_flows(config)

    # 2. Unsorted date validation
    df_unsorted = pd.DataFrame({
        "date": [pd.Timestamp("2024-06-02"), pd.Timestamp("2024-06-01")],
        "etf_flow_ibit": [1.0e6, 1.0e6], "etf_flow_fbtc": [1.0e6, 1.0e6], "etf_flow_arkb": [1.0e6, 1.0e6],
        "etf_flow_bitb": [1.0e6, 1.0e6], "etf_flow_gbtc": [1.0e6, 1.0e6], "etf_net_flow_usd": [5.0e6, 5.0e6]
    })
    df_unsorted.to_csv(fallback_path, index=False)
    with pytest.raises(ValueError, match="not sorted"):
        fetch_btc_etf_flows(config)

    # 3. Duplicate date validation
    df_dup = pd.DataFrame({
        "date": [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-01")],
        "etf_flow_ibit": [1.0e6, 1.0e6], "etf_flow_fbtc": [1.0e6, 1.0e6], "etf_flow_arkb": [1.0e6, 1.0e6],
        "etf_flow_bitb": [1.0e6, 1.0e6], "etf_flow_gbtc": [1.0e6, 1.0e6], "etf_net_flow_usd": [5.0e6, 5.0e6]
    })
    df_dup.to_csv(fallback_path, index=False)
    with pytest.raises(ValueError, match="duplicate dates"):
        fetch_btc_etf_flows(config)

    # 4. Unit scaling check (warning only)
    df_unscaled = pd.DataFrame({
        "date": [pd.Timestamp("2024-06-01")],
        "etf_flow_ibit": [10.0], "etf_flow_fbtc": [10.0], "etf_flow_arkb": [5.0],
        "etf_flow_bitb": [2.0], "etf_flow_gbtc": [-5.0], "etf_net_flow_usd": [22.0]
    })
    df_unscaled.to_csv(fallback_path, index=False)
    df, summary = fetch_btc_etf_flows(config)
    assert summary["status"] == "fallback_local"
    assert len(summary["warnings"]) == 1
    assert "seem unscaled" in summary["warnings"][0]


def test_etf_demand_flow_regimes_and_gbtc_noise() -> None:
    # We construct a synthetic panel with 100 daily rows to test the scoring math
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    
    # Base frame
    frame = pd.DataFrame({
        "date": dates,
        "open": np.linspace(40000, 45000, 100),
        "high": np.linspace(41000, 46000, 100),
        "low": np.linspace(39000, 44000, 100),
        "close": np.linspace(40000, 45000, 100),
        "volume": np.full(100, 1000.0),
        "SplyCur": np.full(100, 19.5e6),
        "volume_ratio_20": np.full(100, 1.0),
    })

    # Add technical indicators first to make sure frame is valid for _add_market_indicators
    from egx_research.crypto_bottom import _add_market_indicators, _component_scores
    
    # 1. Positive regime: persistent positive flows across multiple issuers
    pos_frame = frame.copy()
    pos_frame["etf_flow_ibit"] = 50.0e6
    pos_frame["etf_flow_fbtc"] = 30.0e6
    pos_frame["etf_flow_arkb"] = 15.0e6
    pos_frame["etf_flow_bitb"] = 10.0e6
    pos_frame["etf_flow_gbtc"] = -5.0e6
    pos_frame["etf_net_flow_usd"] = 100.0e6 # Total: +100M
    
    pos_df = _add_market_indicators(pos_frame)
    pos_scores = _component_scores(pos_df)
    
    # 2. Negative regime: persistent negative flows
    neg_frame = frame.copy()
    neg_frame["etf_flow_ibit"] = -5.0e6
    neg_frame["etf_flow_fbtc"] = -10.0e6
    neg_frame["etf_flow_arkb"] = -5.0e6
    neg_frame["etf_flow_bitb"] = -2.0e6
    neg_frame["etf_flow_gbtc"] = -50.0e6
    neg_frame["etf_net_flow_usd"] = -72.0e6 # Total: -72M
    
    neg_df = _add_market_indicators(neg_frame)
    neg_scores = _component_scores(neg_df)
    
    # 3. GBTC noise regime: heavy GBTC outflows, but strong, persistent non-GBTC inflows
    # Total flow is negative (or near zero), but organic demand is positive and broad.
    noise_frame = frame.copy()
    noise_frame["etf_flow_ibit"] = 25.0e6  # Broad positive
    noise_frame["etf_flow_fbtc"] = 25.0e6  # Broad positive
    noise_frame["etf_flow_arkb"] = 15.0e6  # Broad positive
    noise_frame["etf_flow_bitb"] = 10.0e6  # Broad positive
    noise_frame["etf_flow_gbtc"] = -100.0e6 # Massive GBTC outflow
    noise_frame["etf_net_flow_usd"] = -25.0e6 # Total flow is negative!
    
    noise_df = _add_market_indicators(noise_frame)
    noise_scores = _component_scores(noise_df)
    
    # Ensure spot demand score:
    # Positive score should be significantly higher than negative score.
    # GBTC-noise score (negative total, but strong ex-GBTC persistent breadth) should remain constructive (e.g. > 0.4)
    # and significantly higher than pure negative regime score.
    pos_score_val = pos_scores["spot_demand"].iloc[-1]
    neg_score_val = neg_scores["spot_demand"].iloc[-1]
    noise_score_val = noise_scores["spot_demand"].iloc[-1]
    
    assert pos_score_val > neg_score_val
    assert noise_score_val > neg_score_val
    assert noise_score_val > 0.35


def test_futures_positioning_source_metadata() -> None:
    assert "futures_positioning" in SOURCE_REGISTRY
    meta = SOURCE_REGISTRY["futures_positioning"]
    assert meta["filename"] == "futures_positioning.csv"
    assert meta["critical"] is False
    assert "derivatives_taker_buy_sell_ratio" in meta["required_columns"]


def test_exchange_flows_source_metadata() -> None:
    assert "exchange_flows" in SOURCE_REGISTRY
    meta = SOURCE_REGISTRY["exchange_flows"]
    assert meta["filename"] == "exchange_flows.csv"
    assert meta["critical"] is False
    assert "onchain_exchange_netflow_btc" in meta["required_columns"]


def test_glassnode_sth_sopr_source_metadata() -> None:
    assert "glassnode_sth_sopr" in SOURCE_REGISTRY
    meta = SOURCE_REGISTRY["glassnode_sth_sopr"]
    assert meta["filename"] == "glassnode_sth_sopr.csv"
    assert meta["critical"] is False
    assert "onchain_sth_sopr" in meta["required_columns"]


def test_liquidations_optional_source_missing(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.sources.optional_sources["liquidations"]["enabled"] = True
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    dummy_price = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "open": [10, 11, 12, 13, 14],
        "high": [11, 12, 13, 14, 15],
        "low": [9, 10, 11, 12, 13],
        "close": [10, 11, 12, 13, 14],
        "volume": [100, 110, 120, 130, 140],
    })
    monkeypatch.setattr("egx_research.crypto_data.fetch_binance_klines", lambda c: dummy_price)

    # Ensure credential for liquidations is missing in environment
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)

    # Mock other fetchers to return empty dfs to speed up
    for s in SOURCE_REGISTRY:
        if s not in ("binance", "liquidations"):
            monkeypatch.setattr(
                f"egx_research.crypto_data.fetch_{s}" if s != "options_skew" else "egx_research.crypto_data.fetch_deribit_options",
                lambda c: pd.DataFrame(columns=["date"])
            )

    sync_crypto_data(config)

    # Check sync summary status
    summary_path = tmp_path / "features" / "sync_summary.json"
    with open(summary_path) as f:
        import json
        summary = json.load(f)
    assert summary["source_statuses"]["liquidations"] == "missing_optional"
    assert summary["liquidations_rows"] == 0
    assert not (tmp_path / "raw" / "liquidations.csv").exists()
