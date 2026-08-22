from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from egx_research.crypto_portfolio import (
    PortfolioPolicy,
    construct_portfolio,
    portfolio_return_series,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"asset_id": "btc", "venue": "binance", "quote_asset": "USDT", "score": 1.0, "realized_volatility": 0.6, "adv_quote": 1e9, "eligible": True},
            {"asset_id": "eth", "venue": "binance", "quote_asset": "USDT", "score": 1.0, "realized_volatility": 0.7, "adv_quote": 5e8, "eligible": True},
            {"asset_id": "sol", "venue": "coinbase", "quote_asset": "USDC", "score": 0.8, "realized_volatility": 0.9, "adv_quote": 1e8, "eligible": True},
        ]
    )


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    btc = rng.normal(0.001, 0.03, 200)
    return pd.DataFrame(
        {
            "btc": btc,
            "eth": btc * 0.95 + rng.normal(0, 0.005, 200),
            "sol": rng.normal(0.001, 0.04, 200),
        }
    )


def test_portfolio_respects_asset_venue_volatility_and_turnover_limits() -> None:
    policy = PortfolioPolicy(
        maximum_asset_weight=0.35,
        maximum_venue_weight=0.60,
        target_annualized_volatility=0.30,
        maximum_one_way_turnover=0.20,
    )
    result = construct_portfolio(
        _candidates(),
        _returns(),
        {"btc": 0.2, "eth": 0.2, "sol": 0.1},
        portfolio_equity=100_000.0,
        policy=policy,
    )
    assert max(result.weights.values()) <= 0.35 + 1e-9
    assert result.venue_weights["binance"] <= 0.60 + 1e-9
    assert result.expected_annualized_volatility <= 0.30 + 1e-9
    assert result.one_way_turnover <= 0.20 + 1e-9
    assert np.isclose(sum(result.weights.values()) + result.cash_weight, 1.0)


def test_correlation_penalty_favors_independent_asset() -> None:
    result = construct_portfolio(
        _candidates(),
        _returns(),
        {},
        portfolio_equity=100_000.0,
        policy=PortfolioPolicy(
            maximum_asset_weight=1.0,
            maximum_venue_weight=1.0,
            maximum_one_way_turnover=1.0,
            target_annualized_volatility=10.0,
        ),
    )
    assert result.correlation_adjustments["sol"] > result.correlation_adjustments["btc"]


def test_capacity_limits_small_market_position_change() -> None:
    candidates = _candidates()
    candidates.loc[candidates["asset_id"] == "sol", "adv_quote"] = 1000.0
    result = construct_portfolio(
        candidates,
        _returns(),
        {},
        portfolio_equity=100_000.0,
        policy=PortfolioPolicy(
            maximum_asset_weight=1.0,
            maximum_venue_weight=1.0,
            maximum_one_way_turnover=1.0,
            maximum_adv_participation=0.01,
            target_annualized_volatility=10.0,
        ),
    )
    assert "sol" in result.capacity_limited_assets
    assert result.weights["sol"] <= 0.0001 + 1e-12


def test_empty_universe_moves_to_cash_and_return_series_aligns() -> None:
    candidates = _candidates()
    candidates["eligible"] = False
    result = construct_portfolio(
        candidates,
        _returns(),
        {},
        portfolio_equity=100_000.0,
    )
    assert result.weights == {}
    assert result.cash_weight == 1.0
    returns = portfolio_return_series(_returns(), {"btc": 0.5})
    assert np.allclose(returns, _returns()["btc"] * 0.5)


def test_duplicate_asset_execution_venues_are_rejected() -> None:
    candidates = pd.concat([_candidates(), _candidates().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one execution venue"):
        construct_portfolio(
            candidates,
            _returns(),
            {},
            portfolio_equity=100_000.0,
        )
