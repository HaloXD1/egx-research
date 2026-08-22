from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class LiveRiskPolicy:
    maximum_target_allocation: float = 1.0
    maximum_order_notional: float = 1000.0
    maximum_data_staleness_minutes: int = 10
    maximum_cross_venue_divergence_bps: float = 50.0
    maximum_feature_age_days: int = 3
    maximum_stablecoin_depeg_bps: float = 100.0
    maximum_turnover_per_decision: float = 0.25
    volatility_target: float = 0.40
    maximum_realized_volatility: float = 1.20
    maximum_portfolio_drawdown: float = 0.25
    fail_closed: bool = True


@dataclass(frozen=True)
class PreTradeContext:
    as_of: pd.Timestamp
    candle_close_time: pd.Timestamp
    data_observed_at: pd.Timestamp
    primary_price: float
    reference_prices: tuple[float, ...]
    feature_observed_at: dict[str, pd.Timestamp]
    stablecoin_usd_price: float
    current_target_allocation: float
    proposed_target_allocation: float
    portfolio_equity: float
    realized_volatility: float
    portfolio_drawdown: float
    kill_switch_path: str | None = None


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    proposed_target_allocation: float
    approved_target_allocation: float
    order_notional: float
    reasons: tuple[str, ...]
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def validate_market_frame(frame: pd.DataFrame) -> list[str]:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return [f"missing_columns:{','.join(missing)}"]
    reasons: list[str] = []
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if dates.isna().any():
        reasons.append("invalid_dates")
    if dates.duplicated().any():
        reasons.append("duplicate_dates")
    if not dates.is_monotonic_increasing:
        reasons.append("dates_not_increasing")
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any():
        reasons.append("non_numeric_ohlcv")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        reasons.append("non_positive_price")
    if (numeric["volume"] < 0).any():
        reasons.append("negative_volume")
    if (
        (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any()
        or (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any()
    ):
        reasons.append("invalid_ohlc_relationship")
    return reasons


def evaluate_pretrade_risk(
    context: PreTradeContext,
    policy: LiveRiskPolicy,
) -> RiskDecision:
    as_of = _utc(context.as_of)
    candle_close = _utc(context.candle_close_time)
    observed = _utc(context.data_observed_at)
    reasons: list[str] = []
    checks: dict[str, bool] = {}

    checks["kill_switch_clear"] = not (
        context.kill_switch_path and Path(context.kill_switch_path).exists()
    )
    checks["candle_completed"] = candle_close <= observed <= as_of
    data_age = as_of - observed
    checks["data_fresh"] = (
        pd.Timedelta(0)
        <= data_age
        <= pd.Timedelta(minutes=policy.maximum_data_staleness_minutes)
    )
    references = [price for price in context.reference_prices if price > 0]
    if context.primary_price <= 0 or not references:
        checks["cross_venue_price_valid"] = False
    else:
        divergence = abs(context.primary_price / median(references) - 1.0) * 10_000.0
        checks["cross_venue_price_valid"] = (
            divergence <= policy.maximum_cross_venue_divergence_bps
        )

    maximum_feature_age = pd.Timedelta(days=policy.maximum_feature_age_days)
    checks["features_fresh"] = all(
        pd.Timedelta(0) <= as_of - _utc(value) <= maximum_feature_age
        for value in context.feature_observed_at.values()
    )
    stablecoin_depeg = abs(context.stablecoin_usd_price - 1.0) * 10_000.0
    checks["stablecoin_within_band"] = (
        stablecoin_depeg <= policy.maximum_stablecoin_depeg_bps
    )
    checks["realized_volatility_within_hard_limit"] = (
        0 <= context.realized_volatility <= policy.maximum_realized_volatility
    )
    checks["drawdown_within_limit"] = (
        0 <= context.portfolio_drawdown <= policy.maximum_portfolio_drawdown
        or context.proposed_target_allocation <= context.current_target_allocation
    )

    approved = min(
        max(0.0, context.proposed_target_allocation),
        policy.maximum_target_allocation,
    )
    if context.realized_volatility > 0:
        approved = min(
            approved,
            policy.maximum_target_allocation
            * policy.volatility_target
            / context.realized_volatility,
        )
    turnover = abs(approved - context.current_target_allocation)
    checks["turnover_within_limit"] = (
        turnover <= policy.maximum_turnover_per_decision
        or approved <= context.current_target_allocation
    )
    order_notional = turnover * max(0.0, context.portfolio_equity)
    checks["order_notional_within_limit"] = (
        order_notional <= policy.maximum_order_notional + 1e-9
        or approved <= context.current_target_allocation
    )
    checks["target_in_range"] = (
        0 <= context.proposed_target_allocation <= policy.maximum_target_allocation
    )

    reasons.extend(name for name, passed in checks.items() if not passed)
    allowed = not reasons
    if not allowed and policy.fail_closed:
        approved = context.current_target_allocation
        order_notional = 0.0
    return RiskDecision(
        allowed=allowed,
        proposed_target_allocation=context.proposed_target_allocation,
        approved_target_allocation=float(approved),
        order_notional=float(order_notional),
        reasons=tuple(reasons),
        checks=checks,
    )
