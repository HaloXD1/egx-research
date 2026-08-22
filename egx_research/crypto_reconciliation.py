from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

from egx_research.crypto_exchange import ExchangeAdapter, ExchangeError
from egx_research.crypto_oms import OrderRecord, OrderStore, state_from_exchange


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    maximum_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    maximum_backoff_seconds: float = 2.0


@dataclass(frozen=True)
class ReconciliationReport:
    resolved_intent_ids: tuple[str, ...]
    unresolved_intent_ids: tuple[str, ...]
    orphan_client_order_ids: tuple[str, ...]
    balance_mismatches: dict[str, dict[str, float]]
    exchange_balances: dict[str, float]

    @property
    def clean(self) -> bool:
        return not (
            self.unresolved_intent_ids
            or self.orphan_client_order_ids
            or self.balance_mismatches
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"clean": self.clean}


def retry_safe_read(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    if policy.maximum_attempts <= 0:
        raise ValueError("maximum attempts must be positive")
    delay = max(0.0, policy.initial_backoff_seconds)
    last_error: ExchangeError | None = None
    for attempt in range(policy.maximum_attempts):
        try:
            return operation()
        except ExchangeError as exc:
            last_error = exc
            if attempt + 1 == policy.maximum_attempts:
                break
            sleeper(min(delay, policy.maximum_backoff_seconds))
            delay = max(delay * 2.0, 0.001)
    assert last_error is not None
    raise last_error


def _reconcile_record(
    record: OrderRecord,
    store: OrderStore,
    adapter: ExchangeAdapter,
    retry_policy: RetryPolicy,
) -> bool:
    order = retry_safe_read(
        lambda: adapter.get_order(record.request.symbol, record.client_order_id),
        retry_policy,
    )
    state = state_from_exchange(order)
    if state == "unknown":
        return False
    store.transition(record.intent_id, state, order)
    return True


def reconcile_exchange_state(
    store: OrderStore,
    adapter: ExchangeAdapter,
    *,
    expected_balances: dict[str, float] | None = None,
    balance_tolerance: float = 1e-8,
    retry_policy: RetryPolicy | None = None,
) -> ReconciliationReport:
    policy = retry_policy or RetryPolicy()
    resolved: list[str] = []
    unresolved: list[str] = []
    local_records = store.nonterminal()
    for record in local_records:
        try:
            if _reconcile_record(record, store, adapter, policy):
                resolved.append(record.intent_id)
            else:
                unresolved.append(record.intent_id)
        except ExchangeError:
            unresolved.append(record.intent_id)

    open_orders = retry_safe_read(lambda: adapter.open_orders(), policy)
    local_client_ids = {record.client_order_id for record in local_records}
    orphans = sorted(
        order.client_order_id
        for order in open_orders
        if order.client_order_id not in local_client_ids
    )
    balances = retry_safe_read(adapter.balances, policy)
    mismatches: dict[str, dict[str, float]] = {}
    for asset, expected in (expected_balances or {}).items():
        actual = float(balances.get(asset, 0.0))
        if abs(actual - float(expected)) > balance_tolerance:
            mismatches[asset] = {"expected": float(expected), "actual": actual}
    return ReconciliationReport(
        tuple(sorted(resolved)),
        tuple(sorted(unresolved)),
        tuple(orphans),
        mismatches,
        balances,
    )
