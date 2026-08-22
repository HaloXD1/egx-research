from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from egx_research.crypto_exchange import (
    ExchangeAdapter,
    ExchangeOrder,
    OrderRequest,
    UnknownExecutionState,
    deterministic_client_order_id,
)


TERMINAL_STATES = {"filled", "cancelled", "rejected"}
TRANSITIONS = {
    "created": {"submitting", "cancelled"},
    "submitting": {
        "acknowledged",
        "partially_filled",
        "filled",
        "cancelled",
        "unknown",
        "rejected",
    },
    "acknowledged": {"partially_filled", "filled", "cancelled", "unknown"},
    "partially_filled": {"partially_filled", "filled", "cancelled", "unknown"},
    "unknown": {"acknowledged", "partially_filled", "filled", "cancelled", "rejected"},
    "filled": set(),
    "cancelled": set(),
    "rejected": set(),
}


@dataclass(frozen=True)
class OrderRecord:
    intent_id: str
    client_order_id: str
    request: OrderRequest
    state: str
    exchange_order_id: str | None
    executed_quantity: float
    cumulative_quote_quantity: float
    created_at: str
    updated_at: str


class OrderStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    intent_id TEXT PRIMARY KEY,
                    client_order_id TEXT UNIQUE NOT NULL,
                    request_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    exchange_order_id TEXT,
                    executed_quantity REAL NOT NULL DEFAULT 0,
                    cumulative_quote_quantity REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> OrderRecord:
        request = OrderRequest(**json.loads(row["request_json"]))
        return OrderRecord(
            intent_id=row["intent_id"],
            client_order_id=row["client_order_id"],
            request=request,
            state=row["state"],
            exchange_order_id=row["exchange_order_id"],
            executed_quantity=float(row["executed_quantity"]),
            cumulative_quote_quantity=float(row["cumulative_quote_quantity"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, intent_id: str) -> OrderRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM orders WHERE intent_id = ?", (intent_id,)
            ).fetchone()
        return self._record(row) if row else None

    def create(self, request: OrderRequest) -> OrderRecord:
        existing = self.get(request.intent_id)
        if existing:
            if existing.request != request:
                raise ValueError("intent id already exists with a different request")
            return existing
        now = datetime.now(UTC).isoformat()
        client_id = deterministic_client_order_id(request.intent_id)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request.intent_id,
                    client_id,
                    json.dumps(asdict(request), sort_keys=True),
                    "created",
                    None,
                    0.0,
                    0.0,
                    now,
                    now,
                ),
            )
        created = self.get(request.intent_id)
        if created is None:
            raise RuntimeError("order intent could not be persisted")
        if created.request != request:
            raise ValueError("intent id already exists with a different request")
        return created

    def transition(
        self,
        intent_id: str,
        state: str,
        order: ExchangeOrder | None = None,
    ) -> OrderRecord:
        current = self.get(intent_id)
        if current is None:
            raise KeyError(intent_id)
        if state == current.state:
            return current
        if state not in TRANSITIONS.get(current.state, set()):
            raise ValueError(f"invalid order transition {current.state} -> {state}")
        exchange_order_id = current.exchange_order_id
        executed = current.executed_quantity
        quote = current.cumulative_quote_quantity
        if order:
            exchange_order_id = order.exchange_order_id
            executed = order.executed_quantity
            quote = order.cumulative_quote_quantity
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE orders
                SET state = ?, exchange_order_id = ?, executed_quantity = ?,
                    cumulative_quote_quantity = ?, updated_at = ?
                WHERE intent_id = ?
                """,
                (
                    state,
                    exchange_order_id,
                    executed,
                    quote,
                    datetime.now(UTC).isoformat(),
                    intent_id,
                ),
            )
        return self.get(intent_id)  # type: ignore[return-value]

    def nonterminal(self) -> list[OrderRecord]:
        placeholders = ",".join("?" for _ in TERMINAL_STATES)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM orders WHERE state NOT IN ({placeholders})",
                tuple(TERMINAL_STATES),
            ).fetchall()
        return [self._record(row) for row in rows]


def _state_from_exchange(order: ExchangeOrder) -> str:
    mapping = {
        "new": "acknowledged",
        "partially_filled": "partially_filled",
        "filled": "filled",
        "canceled": "cancelled",
        "cancelled": "cancelled",
        "rejected": "rejected",
        "expired": "cancelled",
    }
    return mapping.get(order.status.lower(), "unknown")


class OrderManager:
    def __init__(self, store: OrderStore, adapter: ExchangeAdapter) -> None:
        self.store = store
        self.adapter = adapter

    def submit(self, request: OrderRequest) -> OrderRecord:
        record = self.store.create(request)
        if record.state != "created":
            return record
        record = self.store.transition(request.intent_id, "submitting")
        try:
            order = self.adapter.submit_order(request, record.client_order_id)
        except UnknownExecutionState:
            return self.store.transition(request.intent_id, "unknown")
        state = _state_from_exchange(order)
        return self.store.transition(request.intent_id, state, order)
