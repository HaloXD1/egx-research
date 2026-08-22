from __future__ import annotations

from pathlib import Path

from egx_research.crypto_exchange import ExchangeError, ExchangeOrder, OrderRequest
from egx_research.crypto_oms import OrderManager, OrderStore
from egx_research.crypto_reconciliation import (
    RetryPolicy,
    reconcile_exchange_state,
    retry_safe_read,
)


class RecoveringAdapter:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.read_failures = 0
        self.orphans: list[ExchangeOrder] = []
        self.exchange_status = "new"

    def submit_order(self, request: OrderRequest, client_order_id: str) -> ExchangeOrder:
        self.submit_calls += 1
        return ExchangeOrder(
            client_order_id,
            "123",
            request.symbol,
            request.side,
            self.exchange_status,
            request.quantity,
            0.0,
            0.0,
        )

    def get_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        if self.read_failures:
            self.read_failures -= 1
            raise ExchangeError("temporary")
        return ExchangeOrder(
            client_order_id,
            "123",
            symbol,
            "buy",
            self.exchange_status,
            0.1,
            0.1 if self.exchange_status == "filled" else 0.0,
            10_000.0 if self.exchange_status == "filled" else 0.0,
        )

    def cancel_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        raise NotImplementedError

    def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        return self.orphans

    def balances(self) -> dict[str, float]:
        return {"BTC": 0.1, "USDT": 9000.0}


def test_restart_reconciliation_resolves_filled_order(tmp_path: Path) -> None:
    path = tmp_path / "orders.sqlite"
    adapter = RecoveringAdapter()
    manager = OrderManager(OrderStore(path), adapter)
    request = OrderRequest("one", "BTCUSDT", "buy", 0.1)
    assert manager.submit(request).state == "acknowledged"

    adapter.exchange_status = "filled"
    restarted_store = OrderStore(path)
    report = reconcile_exchange_state(restarted_store, adapter)
    assert report.clean
    assert report.resolved_intent_ids == ("one",)
    assert restarted_store.get("one").state == "filled"  # type: ignore[union-attr]


def test_reconciliation_reports_orphans_and_balance_mismatch(tmp_path: Path) -> None:
    adapter = RecoveringAdapter()
    adapter.orphans = [
        ExchangeOrder("manual-order", "999", "BTCUSDT", "buy", "new", 1, 0, 0)
    ]
    report = reconcile_exchange_state(
        OrderStore(tmp_path / "orders.sqlite"),
        adapter,
        expected_balances={"BTC": 0.2, "USDT": 9000.0},
    )
    assert not report.clean
    assert report.orphan_client_order_ids == ("manual-order",)
    assert report.balance_mismatches["BTC"]["actual"] == 0.1


def test_safe_reads_back_off_and_recover() -> None:
    attempts = {"count": 0}
    delays: list[float] = []

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ExchangeError("temporary")
        return "ok"

    result = retry_safe_read(
        operation,
        RetryPolicy(maximum_attempts=3, initial_backoff_seconds=0.1),
        sleeper=delays.append,
    )
    assert result == "ok"
    assert delays == [0.1, 0.2]


def test_reconciliation_keeps_unresolved_state_after_read_failures(tmp_path: Path) -> None:
    adapter = RecoveringAdapter()
    store = OrderStore(tmp_path / "orders.sqlite")
    OrderManager(store, adapter).submit(OrderRequest("one", "BTCUSDT", "buy", 0.1))
    adapter.read_failures = 3
    report = reconcile_exchange_state(
        store,
        adapter,
        retry_policy=RetryPolicy(maximum_attempts=3, initial_backoff_seconds=0),
    )
    assert report.unresolved_intent_ids == ("one",)
    assert not report.clean
