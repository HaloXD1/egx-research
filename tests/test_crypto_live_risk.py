from __future__ import annotations

from dataclasses import replace

import pandas as pd

from egx_research.crypto_live_risk import (
    LiveRiskPolicy,
    PreTradeContext,
    evaluate_pretrade_risk,
    validate_market_frame,
)


def _context() -> PreTradeContext:
    return PreTradeContext(
        as_of=pd.Timestamp("2026-01-02T00:05:00Z"),
        candle_close_time=pd.Timestamp("2026-01-02T00:00:00Z"),
        data_observed_at=pd.Timestamp("2026-01-02T00:01:00Z"),
        primary_price=100_000.0,
        reference_prices=(100_010.0, 99_990.0),
        feature_observed_at={"mvrv": pd.Timestamp("2026-01-01T12:00:00Z")},
        stablecoin_usd_price=1.0,
        current_target_allocation=0.40,
        proposed_target_allocation=0.45,
        portfolio_equity=10_000.0,
        realized_volatility=0.50,
        portfolio_drawdown=0.05,
    )


def test_valid_pretrade_context_is_allowed() -> None:
    decision = evaluate_pretrade_risk(_context(), LiveRiskPolicy())
    assert decision.allowed
    assert decision.approved_target_allocation == 0.45


def test_price_only_model_does_not_require_external_feature_timestamps() -> None:
    decision = evaluate_pretrade_risk(
        replace(_context(), feature_observed_at={}),
        LiveRiskPolicy(),
    )
    assert decision.allowed


def test_stale_data_and_price_divergence_fail_closed() -> None:
    context = replace(
        _context(),
        data_observed_at=pd.Timestamp("2026-01-01T23:00:00Z"),
        primary_price=120_000.0,
    )
    decision = evaluate_pretrade_risk(context, LiveRiskPolicy())
    assert not decision.allowed
    assert "data_fresh" in decision.reasons
    assert "cross_venue_price_valid" in decision.reasons
    assert decision.order_notional == 0.0


def test_kill_switch_blocks_trade(tmp_path) -> None:
    switch = tmp_path / "KILL"
    switch.touch()
    decision = evaluate_pretrade_risk(
        replace(_context(), kill_switch_path=str(switch)),
        LiveRiskPolicy(),
    )
    assert not decision.allowed
    assert "kill_switch_clear" in decision.reasons


def test_drawdown_allows_only_risk_reduction() -> None:
    policy = LiveRiskPolicy(maximum_portfolio_drawdown=0.20)
    increase = evaluate_pretrade_risk(
        replace(_context(), portfolio_drawdown=0.25, proposed_target_allocation=0.50),
        policy,
    )
    reduce = evaluate_pretrade_risk(
        replace(_context(), portfolio_drawdown=0.25, proposed_target_allocation=0.30),
        policy,
    )
    assert not increase.allowed
    assert reduce.allowed


def test_market_frame_detects_duplicates_and_invalid_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01"],
            "open": [100, 100],
            "high": [90, 101],
            "low": [95, 99],
            "close": [100, 100],
            "volume": [1, 1],
        }
    )
    reasons = validate_market_frame(frame)
    assert "duplicate_dates" in reasons
    assert "invalid_ohlc_relationship" in reasons
