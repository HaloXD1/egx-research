from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Literal

import pandas as pd


Side = Literal["buy", "sell"]
Algorithm = Literal["market", "limit", "twap"]


@dataclass(frozen=True)
class ExecutionPolicy:
    algorithm: Algorithm = "twap"
    slices: int = 5
    fee_bps: float = 10.0
    half_spread_bps: float = 2.0
    base_slippage_bps: float = 3.0
    impact_bps_at_full_participation: float = 100.0
    maximum_participation_rate: float = 0.01
    minimum_notional: float = 5.0
    quantity_precision: int = 5
    price_tick: float = 0.01


@dataclass(frozen=True)
class ExecutionIntent:
    intent_id: str
    side: Side
    quantity: float
    decision_time: pd.Timestamp
    limit_price: float | None = None


@dataclass(frozen=True)
class SimulatedFill:
    timestamp: str
    quantity: float
    price: float
    fee: float
    reference_price: float
    slippage_bps: float


@dataclass(frozen=True)
class ExecutionSimulation:
    intent_id: str
    status: Literal["filled", "partially_filled", "rejected", "no_fill"]
    requested_quantity: float
    filled_quantity: float
    average_price: float | None
    total_fee: float
    arrival_price: float | None
    implementation_shortfall_bps: float | None
    rejection_reason: str | None
    fills: tuple[SimulatedFill, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fills"] = [asdict(fill) for fill in self.fills]
        return payload


@dataclass(frozen=True)
class IntradayReplayResult:
    equity: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame


def _round_quantity(value: float, precision: int) -> float:
    if precision < 0:
        raise ValueError("quantity precision cannot be negative")
    quantum = Decimal(1).scaleb(-precision)
    return float(Decimal(str(max(0.0, value))).quantize(quantum, rounding=ROUND_FLOOR))


def _round_price(value: float, tick: float, side: Side) -> float:
    if tick <= 0:
        raise ValueError("price tick must be positive")
    units = Decimal(str(value)) / Decimal(str(tick))
    rounding = ROUND_CEILING if side == "buy" else ROUND_FLOOR
    return float(units.quantize(Decimal("1"), rounding=rounding) * Decimal(str(tick)))


def _validate(policy: ExecutionPolicy, intent: ExecutionIntent) -> None:
    if intent.side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if intent.quantity <= 0:
        raise ValueError("quantity must be positive")
    if policy.slices <= 0:
        raise ValueError("execution slices must be positive")
    if not 0 < policy.maximum_participation_rate <= 1:
        raise ValueError("maximum participation rate must be in (0, 1]")
    if min(
        policy.fee_bps,
        policy.half_spread_bps,
        policy.base_slippage_bps,
        policy.impact_bps_at_full_participation,
        policy.minimum_notional,
    ) < 0:
        raise ValueError("execution costs and minimum notional cannot be negative")
    if policy.algorithm == "limit" and intent.limit_price is None:
        raise ValueError("limit execution requires a limit price")


def simulate_execution(
    intent: ExecutionIntent,
    intraday_bars: pd.DataFrame,
    policy: ExecutionPolicy,
) -> ExecutionSimulation:
    _validate(policy, intent)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(intraday_bars.columns))
    if missing:
        raise ValueError(f"intraday bars missing columns: {', '.join(missing)}")
    bars = intraday_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    decision = pd.Timestamp(intent.decision_time)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    bars = bars.loc[bars["date"] > decision].sort_values("date")
    bar_count = policy.slices if policy.algorithm == "twap" else 1
    bars = bars.head(bar_count)
    if bars.empty:
        return ExecutionSimulation(
            intent.intent_id,
            "rejected",
            intent.quantity,
            0.0,
            None,
            0.0,
            None,
            None,
            "no post-decision intraday bars",
            (),
        )

    arrival = float(bars.iloc[0]["open"])
    rounded_requested = _round_quantity(intent.quantity, policy.quantity_precision)
    if rounded_requested * arrival < policy.minimum_notional:
        return ExecutionSimulation(
            intent.intent_id,
            "rejected",
            intent.quantity,
            0.0,
            None,
            0.0,
            arrival,
            None,
            "below minimum notional",
            (),
        )

    remaining = rounded_requested
    fills: list[SimulatedFill] = []
    sign = 1.0 if intent.side == "buy" else -1.0
    for position, (_, bar) in enumerate(bars.iterrows()):
        if remaining <= 0:
            break
        remaining_slices = len(bars) - position
        desired = remaining if policy.algorithm != "twap" else remaining / remaining_slices
        volume = max(0.0, float(bar["volume"]))
        capacity = volume * policy.maximum_participation_rate
        quantity = _round_quantity(min(desired, capacity), policy.quantity_precision)
        if quantity <= 0:
            continue
        if policy.algorithm == "limit":
            crossed = (
                float(bar["low"]) <= float(intent.limit_price)
                if intent.side == "buy"
                else float(bar["high"]) >= float(intent.limit_price)
            )
            if not crossed:
                continue

        reference = float(bar.get("vwap", bar["open"]))
        participation = 0.0 if volume <= 0 else quantity / volume
        impact_bps = policy.impact_bps_at_full_participation * participation**2
        slippage_bps = (
            policy.half_spread_bps + policy.base_slippage_bps + impact_bps
        )
        raw_price = reference * (1.0 + sign * slippage_bps / 10_000.0)
        if policy.algorithm == "limit":
            raw_price = (
                min(raw_price, float(intent.limit_price))
                if intent.side == "buy"
                else max(raw_price, float(intent.limit_price))
            )
        price = _round_price(raw_price, policy.price_tick, intent.side)
        fee = quantity * price * policy.fee_bps / 10_000.0
        fills.append(
            SimulatedFill(
                timestamp=pd.Timestamp(bar["date"]).isoformat(),
                quantity=quantity,
                price=price,
                fee=fee,
                reference_price=reference,
                slippage_bps=slippage_bps,
            )
        )
        remaining = _round_quantity(
            remaining - quantity, policy.quantity_precision
        )

    filled = sum(fill.quantity for fill in fills)
    if not fills:
        status = "no_fill"
        average = shortfall = None
    else:
        status = "filled" if filled >= rounded_requested else "partially_filled"
        average = sum(fill.quantity * fill.price for fill in fills) / filled
        shortfall = sign * (average / arrival - 1.0) * 10_000.0
    return ExecutionSimulation(
        intent.intent_id,
        status,
        intent.quantity,
        filled,
        average,
        sum(fill.fee for fill in fills),
        arrival,
        shortfall,
        None,
        tuple(fills),
    )


def replay_target_allocations(
    daily_targets: pd.DataFrame,
    intraday_bars: pd.DataFrame,
    policy: ExecutionPolicy,
    *,
    initial_cash: float,
) -> IntradayReplayResult:
    required = {"decision_time", "target_allocation"}
    missing = sorted(required - set(daily_targets.columns))
    if missing:
        raise ValueError(f"daily targets missing columns: {', '.join(missing)}")
    if initial_cash <= 0:
        raise ValueError("initial cash must be positive")
    targets = daily_targets.copy()
    targets["decision_time"] = pd.to_datetime(targets["decision_time"], utc=True)
    targets = targets.sort_values("decision_time")
    bars = intraday_bars.copy()
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    cash = float(initial_cash)
    quantity = 0.0
    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        target = float(row.target_allocation)
        if not 0 <= target <= 1:
            raise ValueError("target allocation must be in [0, 1]")
        future = bars.loc[bars["date"] > row.decision_time]
        if future.empty:
            raise ValueError("no intraday execution data after a decision")
        arrival = float(future.iloc[0]["open"])
        equity_before = cash + quantity * arrival
        desired_quantity = target * equity_before / arrival
        delta = desired_quantity - quantity
        simulation: ExecutionSimulation | None = None
        if abs(delta) > 10 ** (-policy.quantity_precision):
            simulation = simulate_execution(
                ExecutionIntent(
                    intent_id=f"replay-{pd.Timestamp(row.decision_time).isoformat()}",
                    side="buy" if delta > 0 else "sell",
                    quantity=abs(delta),
                    decision_time=pd.Timestamp(row.decision_time),
                ),
                bars,
                policy,
            )
            signed_quantity = simulation.filled_quantity * (1 if delta > 0 else -1)
            cash -= signed_quantity * float(simulation.average_price or 0.0)
            cash -= simulation.total_fee
            quantity += signed_quantity
            order_rows.append(
                {
                    **simulation.to_dict(),
                    "decision_time": pd.Timestamp(row.decision_time).isoformat(),
                    "target_allocation": target,
                }
            )
            for fill in simulation.fills:
                fill_rows.append(
                    {
                        "intent_id": simulation.intent_id,
                        "side": "buy" if delta > 0 else "sell",
                        **asdict(fill),
                    }
                )
        mark_bars = policy.slices if policy.algorithm == "twap" else 1
        mark = float(future.head(mark_bars)["close"].iloc[-1])
        equity_after = cash + quantity * mark
        equity_rows.append(
            {
                "decision_time": pd.Timestamp(row.decision_time).isoformat(),
                "target_allocation": target,
                "actual_allocation": 0.0
                if equity_after <= 0
                else quantity * mark / equity_after,
                "cash": cash,
                "quantity": quantity,
                "mark_price": mark,
                "equity": equity_after,
                "order_status": simulation.status if simulation else "no_order",
            }
        )
    return IntradayReplayResult(
        pd.DataFrame(equity_rows),
        pd.DataFrame(order_rows),
        pd.DataFrame(fill_rows),
    )
