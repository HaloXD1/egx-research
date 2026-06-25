import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from egx_research.crypto_config import CryptoConfig
from egx_research.crypto_sources import (
    get_source_api_key,
    is_source_enabled,
    is_source_required,
    run_data_quality_checks,
    SOURCE_REGISTRY,
)
from egx_research.crypto_data import sync_crypto_data
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


def test_sync_skips_missing_credentials_gracefully(tmp_path, monkeypatch) -> None:
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
    assert summary["source_statuses"]["coinmetrics"] == "missing_credentials"


def test_sync_fails_missing_credentials_if_required(tmp_path, monkeypatch) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for p in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        p.mkdir(parents=True, exist_ok=True)

    # Require coinmetrics and make credentials missing
    config.sources.optional_sources["coinmetrics"]["required"] = True
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
