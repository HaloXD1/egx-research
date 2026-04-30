from __future__ import annotations

import pandas as pd

from egx_research.data import load_stockanalysis_history, normalize_ohlcv


def test_normalize_ohlcv_handles_investing_style_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["05/01/2024", "04/01/2024"],
            "Price": ["1,020.5", "1,000.0"],
            "Open": ["1,010", "995"],
            "High": ["1,030", "1,010"],
            "Low": ["1,000", "990"],
            "Vol.": ["10.5K", "8K"],
            "Change %": ["2.0%", "1.0%"],
        }
    )
    normalized = normalize_ohlcv(raw)
    assert list(normalized.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert normalized.iloc[0]["close"] == 1000.0
    assert normalized.iloc[1]["close"] == 1020.5
    assert normalized.iloc[1]["volume"] == 10500.0


def test_load_stockanalysis_history_parses_embedded_rows(monkeypatch) -> None:
    class DummyResponse:
        text = 'x{a:52.75,c:52.75,h:53.25,l:52.2,o:52.78,t:"2026-04-02",v:22602,ch:.53}{a:52.47,c:52.47,h:52.85,l:51.6,o:51.2,t:"2026-04-01",v:19435,ch:1.6}'

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("egx_research.data.requests.get", lambda *args, **kwargs: DummyResponse())
    df = load_stockanalysis_history("https://stockanalysis.com/quote/egx/EGX30ETF/history/")
    assert list(df.columns) == ["adj", "close", "high", "low", "open", "date", "volume"]
    assert len(df) == 2
    assert df.iloc[0]["date"] == "2026-04-02"


def test_normalize_ohlcv_clamps_invalid_high_low() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2026-04-02"],
            "Open": [10.0],
            "High": [9.5],
            "Low": [10.5],
            "Close": [11.0],
            "Volume": [100],
        }
    )
    normalized = normalize_ohlcv(raw)
    assert normalized.iloc[0]["high"] == 11.0
    assert normalized.iloc[0]["low"] == 10.0


def test_normalize_ohlcv_parses_month_first_dates() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["01/15/2015", "04/05/2026"],
            "Price": ["10.40", "53.39"],
            "Open": ["10.60", "52.75"],
            "High": ["10.85", "53.49"],
            "Low": ["10.30", "52.30"],
            "Vol.": ["375.00K", "21.62K"],
        }
    )
    normalized = normalize_ohlcv(raw)
    assert normalized.iloc[0]["date"].strftime("%Y-%m-%d") == "2015-01-15"
    assert normalized.iloc[1]["date"].strftime("%Y-%m-%d") == "2026-04-05"
