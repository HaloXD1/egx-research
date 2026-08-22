from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from egx_research.crypto_exchange import (
    ExchangeAdapter,
    OrderRequest,
    SymbolFilters,
    conform_order,
)
from egx_research.crypto_live_risk import (
    LiveRiskPolicy,
    PreTradeContext,
    RiskDecision,
    evaluate_pretrade_risk,
)
from egx_research.crypto_oms import OrderManager, OrderRecord, OrderStore
from egx_research.crypto_reconciliation import (
    ReconciliationReport,
    reconcile_exchange_state,
)
from egx_research.crypto_operations import AlertSink


@dataclass(frozen=True)
class LiveEngineResult:
    submitted: bool
    reason: str
    risk: RiskDecision
    reconciliation: ReconciliationReport
    order: OrderRecord | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "reason": self.reason,
            "risk": self.risk.to_dict(),
            "reconciliation": self.reconciliation.to_dict(),
            "order": asdict(self.order) if self.order else None,
        }


class SupervisedLiveEngine:
    def __init__(
        self,
        store: OrderStore,
        adapter: ExchangeAdapter,
        risk_policy: LiveRiskPolicy,
        *,
        alert_sink: AlertSink | None = None,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self.risk_policy = risk_policy
        self.alert_sink = alert_sink

    def _block(
        self,
        reason: str,
        risk: RiskDecision,
        reconciliation: ReconciliationReport,
    ) -> LiveEngineResult:
        if self.alert_sink:
            self.alert_sink.send("critical", reason, {"risk": risk.to_dict()})
        return LiveEngineResult(False, reason, risk, reconciliation, None)

    def execute_target(
        self,
        *,
        intent_id: str,
        symbol: str,
        context: PreTradeContext,
        symbol_filters: SymbolFilters,
        expected_balances: dict[str, float],
        approval_reference: str,
    ) -> LiveEngineResult:
        risk = evaluate_pretrade_risk(context, self.risk_policy)
        reconciliation = reconcile_exchange_state(
            self.store,
            self.adapter,
            expected_balances=expected_balances,
        )
        if not reconciliation.clean:
            return self._block("reconciliation_not_clean", risk, reconciliation)
        if not risk.allowed:
            return self._block("pretrade_risk_rejected", risk, reconciliation)
        if not approval_reference.strip():
            return self._block("supervised_approval_required", risk, reconciliation)
        delta = (
            risk.approved_target_allocation - context.current_target_allocation
        )
        if abs(delta) <= 1e-12:
            return LiveEngineResult(False, "target_unchanged", risk, reconciliation, None)
        request = conform_order(
            OrderRequest(
                intent_id=intent_id,
                symbol=symbol,
                side="buy" if delta > 0 else "sell",
                quantity=abs(risk.order_notional / context.primary_price),
            ),
            symbol_filters,
            context.primary_price,
        )
        order = OrderManager(self.store, self.adapter).submit(request)
        return LiveEngineResult(True, "submitted", risk, reconciliation, order)
