from __future__ import annotations

from pathlib import Path

import pandas as pd

from egx_research.crypto_exchange import ExchangeOrder, OrderRequest, SymbolFilters
from egx_research.crypto_live_engine import SupervisedLiveEngine
from egx_research.crypto_live_risk import LiveRiskPolicy, PreTradeContext
from egx_research.crypto_oms import OrderStore


class EngineAdapter:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.orphans: list[ExchangeOrder] = []

    def submit_order(self, request: OrderRequest, client_order_id: str) -> ExchangeOrder:
        self.submit_calls += 1
        return ExchangeOrder(client_order_id, "1", request.symbol, request.side, "new", request.quantity, 0, 0)

    def get_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        return ExchangeOrder(client_order_id, "1", symbol, "buy", "new", 0.005, 0, 0)

    def cancel_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        raise NotImplementedError

    def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        return self.orphans

    def balances(self) -> dict[str, float]:
        return {"BTC": 0.04, "USDT": 6000.0}


def _context() -> PreTradeContext:
    return PreTradeContext(
        as_of=pd.Timestamp("2026-01-02T00:05:00Z"),
        candle_close_time=pd.Timestamp("2026-01-02T00:00:00Z"),
        data_observed_at=pd.Timestamp("2026-01-02T00:01:00Z"),
        primary_price=100_000.0,
        reference_prices=(100_010.0,),
        feature_observed_at={},
        stablecoin_usd_price=1.0,
        current_target_allocation=0.40,
        proposed_target_allocation=0.45,
        portfolio_equity=10_000.0,
        realized_volatility=0.5,
        portfolio_drawdown=0.05,
    )


def _engine(tmp_path: Path, adapter: EngineAdapter) -> SupervisedLiveEngine:
    return SupervisedLiveEngine(
        OrderStore(tmp_path / "orders.sqlite"),
        adapter,
        LiveRiskPolicy(),
    )


def test_live_engine_requires_supervised_approval(tmp_path: Path) -> None:
    adapter = EngineAdapter()
    result = _engine(tmp_path, adapter).execute_target(
        intent_id="one",
        symbol="BTCUSDT",
        context=_context(),
        symbol_filters=SymbolFilters(0.00001, 0.00001, 0.01, 5.0),
        expected_balances=adapter.balances(),
        approval_reference="",
    )
    assert not result.submitted
    assert result.reason == "supervised_approval_required"
    assert adapter.submit_calls == 0


def test_live_engine_reconciles_and_submits_once(tmp_path: Path) -> None:
    adapter = EngineAdapter()
    engine = _engine(tmp_path, adapter)
    kwargs = {
        "intent_id": "one",
        "symbol": "BTCUSDT",
        "context": _context(),
        "symbol_filters": SymbolFilters(0.00001, 0.00001, 0.01, 5.0),
        "expected_balances": adapter.balances(),
        "approval_reference": "operator:ahmed",
    }
    assert engine.execute_target(**kwargs).submitted
    assert engine.execute_target(**kwargs).submitted
    assert adapter.submit_calls == 1


def test_live_engine_blocks_orphan_exchange_order(tmp_path: Path) -> None:
    adapter = EngineAdapter()
    adapter.orphans = [
        ExchangeOrder("manual", "2", "BTCUSDT", "buy", "new", 0.1, 0, 0)
    ]
    result = _engine(tmp_path, adapter).execute_target(
        intent_id="one",
        symbol="BTCUSDT",
        context=_context(),
        symbol_filters=SymbolFilters(0.00001, 0.00001, 0.01, 5.0),
        expected_balances=adapter.balances(),
        approval_reference="operator:ahmed",
    )
    assert not result.submitted
    assert result.reason == "reconciliation_not_clean"
    assert adapter.submit_calls == 0
