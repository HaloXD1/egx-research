from __future__ import annotations

from pathlib import Path

import pytest

from egx_research.crypto_exchange import (
    BinanceSpotAdapter,
    ExchangeOrder,
    OrderRequest,
    SymbolFilters,
    UnknownExecutionState,
    conform_order,
    deterministic_client_order_id,
    sign_query,
)
from egx_research.crypto_oms import OrderManager, OrderStore


class FakeAdapter:
    def __init__(self, unknown: bool = False) -> None:
        self.calls = 0
        self.unknown = unknown

    def submit_order(self, request: OrderRequest, client_order_id: str) -> ExchangeOrder:
        self.calls += 1
        if self.unknown:
            raise UnknownExecutionState("timeout")
        return ExchangeOrder(
            client_order_id,
            "123",
            request.symbol,
            request.side,
            "new",
            request.quantity,
            0.0,
            0.0,
        )

    def get_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        raise NotImplementedError

    def cancel_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        raise NotImplementedError

    def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        return []

    def balances(self) -> dict[str, float]:
        return {}


def _request(quantity: float = 0.1) -> OrderRequest:
    return OrderRequest("signal-2026-01-01", "BTCUSDT", "buy", quantity)


def test_order_submission_is_idempotent(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    manager = OrderManager(OrderStore(tmp_path / "orders.sqlite"), adapter)
    first = manager.submit(_request())
    second = manager.submit(_request())
    assert first.state == second.state == "acknowledged"
    assert adapter.calls == 1


def test_conflicting_intent_payload_is_rejected(tmp_path: Path) -> None:
    manager = OrderManager(OrderStore(tmp_path / "orders.sqlite"), FakeAdapter())
    manager.submit(_request())
    with pytest.raises(ValueError, match="different request"):
        manager.submit(_request(quantity=0.2))


def test_unknown_submission_state_is_not_retried(tmp_path: Path) -> None:
    adapter = FakeAdapter(unknown=True)
    manager = OrderManager(OrderStore(tmp_path / "orders.sqlite"), adapter)
    assert manager.submit(_request()).state == "unknown"
    assert manager.submit(_request()).state == "unknown"
    assert adapter.calls == 1


def test_order_conformance_uses_exchange_steps() -> None:
    conformed = conform_order(
        OrderRequest("one", "BTCUSDT", "buy", 0.123456, "LIMIT", 100.019),
        SymbolFilters(0.0001, 0.0001, 0.01, 5.0),
        reference_price=100.0,
    )
    assert conformed.quantity == 0.1234
    assert conformed.price == 100.01


def test_client_ids_and_signatures_are_deterministic() -> None:
    assert deterministic_client_order_id("one") == deterministic_client_order_id("one")
    assert sign_query({"symbol": "BTCUSDT", "timestamp": 1}, "secret") == sign_query(
        {"symbol": "BTCUSDT", "timestamp": 1}, "secret"
    )


def test_binance_adapter_reads_runtime_symbol_filters() -> None:
    class Response:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {
                "symbols": [
                    {
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5.0"},
                        ]
                    }
                ]
            }

    class Session:
        @staticmethod
        def get(*args, **kwargs):
            return Response()

    filters = BinanceSpotAdapter("key", "secret", session=Session()).symbol_filters("BTCUSDT")  # type: ignore[arg-type]
    assert filters == SymbolFilters(0.00001, 0.00001, 0.01, 5.0)
