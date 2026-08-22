from __future__ import annotations

import pandas as pd
import pytest

from egx_research.crypto_universe import (
    UniversePolicy,
    build_point_in_time_panel,
    eligible_universe_as_of,
    membership_as_of,
    normalize_membership_history,
)


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_id": "bitcoin",
                "venue": "binance_spot",
                "symbol": "BTCUSDT",
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "valid_from": "2020-01-01",
                "valid_to": None,
            },
            {
                "asset_id": "oldcoin",
                "venue": "binance_spot",
                "symbol": "OLDUSDT",
                "base_asset": "OLD",
                "quote_asset": "USDT",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-05",
            },
            {
                "asset_id": "renamed",
                "venue": "binance_spot",
                "symbol": "AAAUSDT",
                "base_asset": "AAA",
                "quote_asset": "USDT",
                "valid_from": "2020-01-01",
                "valid_to": "2020-01-03",
            },
            {
                "asset_id": "renamed",
                "venue": "binance_spot",
                "symbol": "BBBUSDT",
                "base_asset": "BBB",
                "quote_asset": "USDT",
                "valid_from": "2020-01-04",
                "valid_to": None,
            },
        ]
    )


def _prices() -> pd.DataFrame:
    rows = []
    for symbol in ("BTCUSDT", "OLDUSDT", "AAAUSDT", "BBBUSDT"):
        for index, day in enumerate(pd.date_range("2020-01-01", periods=10, freq="D")):
            rows.append(
                {
                    "date": day,
                    "venue": "binance_spot",
                    "symbol": symbol,
                    "open": 10 + index,
                    "high": 11 + index,
                    "low": 9 + index,
                    "close": 10 + index,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_historical_membership_keeps_delisted_asset_without_current_survivorship() -> None:
    assert "oldcoin" in set(membership_as_of(_membership(), "2020-01-03")["asset_id"])
    assert "oldcoin" not in set(membership_as_of(_membership(), "2020-01-10")["asset_id"])


def test_point_in_time_panel_forces_delisting_exit_and_tracks_rename() -> None:
    policy = UniversePolicy(
        minimum_history_days=1,
        liquidity_lookback_days=1,
        minimum_median_quote_volume=1.0,
    )
    panel = build_point_in_time_panel(_prices(), _membership(), policy)
    old = panel.loc[panel["asset_id"] == "oldcoin"]
    assert old.iloc[-1]["forced_exit"]
    assert not old.iloc[-1]["eligible"]
    renamed = panel.loc[panel["asset_id"] == "renamed"]
    assert set(renamed["symbol"]) == {"AAAUSDT", "BBBUSDT"}
    assert renamed["date"].is_unique


def test_eligible_universe_uses_information_available_as_of_date() -> None:
    policy = UniversePolicy(
        minimum_history_days=2,
        liquidity_lookback_days=2,
        minimum_median_quote_volume=1.0,
    )
    panel = build_point_in_time_panel(_prices(), _membership(), policy)
    assets = set(eligible_universe_as_of(panel, "2020-01-06")["asset_id"])
    assert "bitcoin" in assets
    assert "oldcoin" not in assets


def test_overlapping_symbol_intervals_are_rejected() -> None:
    bad = pd.concat([_membership(), _membership().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="overlap"):
        normalize_membership_history(bad)
