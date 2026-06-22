from __future__ import annotations

from pathlib import Path

import pandas as pd

from egx_research.crypto_config import CryptoConfig
from egx_research.crypto_data import (
    build_crypto_feature_panel,
    parse_binance_klines,
    parse_coinmetrics_payload,
    parse_fear_greed_payload,
    parse_fred_csv,
    parse_funding_payload,
    fetch_open_interest,
    fetch_coinbase_premium,
    fetch_stablecoin_supply,
    fetch_deribit_options,
)


def test_parse_binance_klines_aligns_utc_and_deduplicates() -> None:
    payload = [
        [1704153600000, "10", "12", "9", "11", "100", 0, "1000", 5, "0", "0", "0"],
        [1704067200000, "9", "11", "8", "10", "90", 0, "900", 4, "0", "0", "0"],
        [1704067200000, "9.5", "11.5", "8.5", "10.5", "95", 0, "950", 6, "0", "0", "0"],
    ]
    frame = parse_binance_klines(payload)
    assert list(frame["date"]) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert frame.iloc[0]["close"] == 10.5
    assert frame.iloc[0]["trade_count"] == 6


def test_parse_external_payloads() -> None:
    coinmetrics = parse_coinmetrics_payload(
        {
            "data": [
                {
                    "asset": "btc",
                    "time": "2024-01-01T00:00:00.000000000Z",
                    "CapMVRVCur": "1.2",
                    "FlowInExUSD-status": "reviewed",
                }
            ]
        }
    )
    fear = parse_fear_greed_payload({"data": [{"timestamp": "1704067200", "value": "25", "value_classification": "Fear"}]})
    fred = parse_fred_csv("observation_date,DGS10\n2024-01-01,4.1\n2024-01-02,.\n", "DGS10", "macro_us10y")
    funding = parse_funding_payload(
        [
            {"fundingTime": 1704067200000, "fundingRate": "0.0001"},
            {"fundingTime": 1704096000000, "fundingRate": "0.0002"},
        ]
    )
    assert coinmetrics.iloc[0]["CapMVRVCur"] == 1.2
    assert fear.iloc[0]["fear_greed_value"] == 25.0
    assert pd.isna(fred.iloc[1]["macro_us10y"])
    assert round(funding.iloc[0]["funding_rate_mean"], 6) == 0.00015


def test_feature_panel_shifts_external_features(tmp_path) -> None:
    config = CryptoConfig()
    config.data.raw_dir = str(tmp_path / "raw")
    config.data.normalized_dir = str(tmp_path / "normalized")
    config.data.features_dir = str(tmp_path / "features")
    for path in (tmp_path / "raw", tmp_path / "normalized", tmp_path / "features"):
        path.mkdir(parents=True, exist_ok=True)

    price = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open": [10, 11, 12, 13],
            "high": [11, 12, 13, 14],
            "low": [9, 10, 11, 12],
            "close": [10, 11, 12, 13],
            "volume": [100, 110, 120, 130],
        }
    )
    price.to_csv(tmp_path / "normalized" / config.data.normalized_filename, index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "CapMVRVCur": [1.0, 1.1, 1.2, 1.3],
        }
    ).to_csv(tmp_path / "raw" / "coinmetrics_btc.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "macro_nasdaq": [100.0, 101.0],
        }
    ).to_csv(tmp_path / "raw" / "macro_fred.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open_interest": [1000.0, 1100.0, 1200.0, 1300.0],
            "open_interest_value": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_csv(tmp_path / "raw" / "open_interest.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "coinbase_close": [10.05, 11.05, 12.05, 13.05],
            "coinbase_premium": [0.005, 0.005, 0.004, 0.004],
        }
    ).to_csv(tmp_path / "raw" / "coinbase_premium.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "stablecoin_supply": [100.0, 101.0, 102.0, 103.0],
        }
    ).to_csv(tmp_path / "raw" / "stablecoin_supply.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "options_skew": [0.1, 0.1, 0.2, 0.2],
            "put_call_ratio": [1.1, 1.1, 1.2, 1.2],
            "dvol": [50.0, 51.0, 52.0, 53.0],
        }
    ).to_csv(tmp_path / "raw" / "options_skew.csv", index=False)

    panel = build_crypto_feature_panel(config)
    assert pd.isna(panel.iloc[0]["CapMVRVCur"])
    assert panel.iloc[1]["CapMVRVCur"] == 1.0
    assert panel.iloc[3]["macro_nasdaq"] == 101.0
    
    # Assertions for the new columns showing they are present and properly lagged
    assert pd.isna(panel.iloc[0]["open_interest"])
    assert panel.iloc[1]["open_interest"] == 1000.0
    assert panel.iloc[1]["open_interest_value"] == 10.0
    assert panel.iloc[1]["coinbase_premium"] == 0.005
    assert panel.iloc[1]["stablecoin_supply"] == 100.0
    assert panel.iloc[1]["options_skew"] == 0.1
    assert panel.iloc[1]["put_call_ratio"] == 1.1
    assert panel.iloc[1]["dvol"] == 50.0


def test_fetch_open_interest(monkeypatch) -> None:
    config = CryptoConfig()
    mock_payload = [
        {"symbol": "BTCUSDT", "sumOpenInterest": "100.5", "sumOpenInterestValue": "6000000.0", "timestamp": 1704153600000}
    ]
    monkeypatch.setattr("egx_research.crypto_data._get_json", lambda url, params=None: mock_payload)
    df = fetch_open_interest(config)
    assert not df.empty
    assert list(df.columns) == ["date", "open_interest", "open_interest_value"]
    assert df.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    assert df.iloc[0]["open_interest"] == 100.5
    assert df.iloc[0]["open_interest_value"] == 6000000.0


def test_fetch_coinbase_premium(monkeypatch, tmp_path) -> None:
    config = CryptoConfig()
    config.data.normalized_dir = str(tmp_path)
    config.data.normalized_filename = "normalized.csv"
    normalized_path = Path(config.data.normalized_path)
    pd.DataFrame({
        "date": [pd.Timestamp("2024-01-02")],
        "close": [41600.0]
    }).to_csv(normalized_path, index=False)

    # Coinbase candle: [time, low, high, open, close, volume]
    mock_payload = [[1704153600, 41000.0, 42000.0, 41500.0, 41800.0, 100.0]]
    monkeypatch.setattr("egx_research.crypto_data._get_json", lambda url, params=None: mock_payload)

    df = fetch_coinbase_premium(config)
    assert not df.empty
    assert list(df.columns) == ["date", "coinbase_close", "coinbase_premium"]
    assert df.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    assert df.iloc[0]["coinbase_close"] == 41800.0
    assert abs(df.iloc[0]["coinbase_premium"] - (41800.0 - 41600.0) / 41600.0) < 1e-6


def test_fetch_stablecoin_supply(monkeypatch) -> None:
    config = CryptoConfig()
    calls = []
    def mock_get_json(url, params=None):
        calls.append(url)
        if "stablecoin=1" in url:
            return [{"date": 1704153600, "totalCirculatingUSD": {"peggedUSD": 90000000.0}}]
        elif "stablecoin=2" in url:
            return [{"date": 1704153600, "totalCirculatingUSD": {"peggedUSD": 30000000.0}}]
        return []

    monkeypatch.setattr("egx_research.crypto_data._get_json", mock_get_json)
    df = fetch_stablecoin_supply(config)
    assert not df.empty
    assert len(calls) == 2
    assert list(df.columns) == ["date", "stablecoin_supply"]
    assert df.iloc[0]["date"] == pd.Timestamp("2024-01-02")
    assert df.iloc[0]["stablecoin_supply"] == 120000000.0


def test_fetch_deribit_options(monkeypatch) -> None:
    config = CryptoConfig()
    today = pd.Timestamp.utcnow().normalize()
    today_ms = int(today.timestamp() * 1000)
    
    def mock_get_json(url, params=None):
        if "get_volatility_index_data" in url:
            return {"result": {"data": [[today_ms, 50.0, 55.0, 48.0, 52.0]]}}
        elif "get_book_summary_by_currency" in url:
            return {"result": [
                {"instrument_name": "BTC-20240102-40000-P", "open_interest": 10.0, "volume": 5.0},
                {"instrument_name": "BTC-20240102-40000-C", "open_interest": 20.0, "volume": 15.0}
            ]}
        return {}

    monkeypatch.setattr("egx_research.crypto_data._get_json", mock_get_json)
    df = fetch_deribit_options(config)
    assert not df.empty
    assert list(df.columns) == ["date", "options_skew", "put_call_ratio", "dvol"]
    assert df.iloc[0]["date"] == today.tz_convert(None)
    assert df.iloc[0]["dvol"] == 52.0
    assert df.iloc[0]["put_call_ratio"] == 0.5
    assert df.iloc[0]["options_skew"] == -0.5

