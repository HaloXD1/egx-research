from __future__ import annotations

import pandas as pd

from egx_research.crypto_execution import (
    ExecutionIntent,
    ExecutionPolicy,
    replay_target_allocations,
    simulate_execution,
)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02T00:01:00Z", periods=5, freq="min"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [100.0] * 5,
        }
    )


def test_market_execution_applies_costs_and_precision() -> None:
    intent = ExecutionIntent("one", "buy", 0.123456, pd.Timestamp("2026-01-02T00:00:00Z"))
    policy = ExecutionPolicy(algorithm="market", quantity_precision=4)
    result = simulate_execution(intent, _bars(), policy)
    assert result.status == "filled"
    assert result.filled_quantity == 0.1234
    assert result.average_price is not None and result.average_price > 100.0
    assert result.total_fee > 0


def test_twap_respects_participation_and_returns_partial_fill() -> None:
    intent = ExecutionIntent("two", "buy", 10.0, pd.Timestamp("2026-01-02T00:00:00Z"))
    policy = ExecutionPolicy(
        algorithm="twap",
        slices=5,
        maximum_participation_rate=0.01,
        quantity_precision=4,
    )
    result = simulate_execution(intent, _bars(), policy)
    assert result.status == "partially_filled"
    assert result.filled_quantity == 5.0
    assert len(result.fills) == 5


def test_limit_order_requires_a_cross() -> None:
    intent = ExecutionIntent(
        "three",
        "buy",
        1.0,
        pd.Timestamp("2026-01-02T00:00:00Z"),
        limit_price=90.0,
    )
    result = simulate_execution(
        intent,
        _bars(),
        ExecutionPolicy(algorithm="limit"),
    )
    assert result.status == "no_fill"
    assert result.filled_quantity == 0.0


def test_minimum_notional_rejects_small_order() -> None:
    intent = ExecutionIntent("four", "sell", 0.01, pd.Timestamp("2026-01-02T00:00:00Z"))
    result = simulate_execution(
        intent,
        _bars(),
        ExecutionPolicy(algorithm="market", minimum_notional=5.0),
    )
    assert result.status == "rejected"
    assert result.rejection_reason == "below minimum notional"


def test_intraday_replay_executes_target_changes_after_decisions() -> None:
    targets = pd.DataFrame(
        {
            "decision_time": [
                "2026-01-02T00:00:00Z",
                "2026-01-02T00:03:00Z",
            ],
            "target_allocation": [0.5, 0.0],
        }
    )
    replay = replay_target_allocations(
        targets,
        _bars(),
        ExecutionPolicy(
            algorithm="market",
            maximum_participation_rate=1.0,
            minimum_notional=0.0,
            quantity_precision=5,
        ),
        initial_cash=10_000.0,
    )
    assert list(replay.orders["status"]) == ["filled", "filled"]
    assert replay.equity.iloc[0]["actual_allocation"] > 0
    assert replay.equity.iloc[-1]["actual_allocation"] == 0
    assert (pd.to_datetime(replay.fills["timestamp"]) > pd.to_datetime(replay.fills["intent_id"].str.removeprefix("replay-"))).all()
