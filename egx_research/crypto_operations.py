from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from egx_research.utils import ensure_dir, write_json


SENSITIVE_KEYS = {"api_key", "api_secret", "secret", "signature", "authorization"}


@dataclass(frozen=True)
class OperationalSnapshot:
    research_target: float
    production_target: float
    intended_target: float
    actual_target: float
    expected_fill_price: float | None
    actual_fill_price: float | None
    expected_fee: float
    actual_fee: float
    data_fresh: bool
    reconciliation_clean: bool
    unresolved_order_count: int


@dataclass(frozen=True)
class OperationalHealth:
    healthy: bool
    alerts: tuple[str, ...]
    metrics: dict[str, float | bool | int | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertSink(Protocol):
    def send(self, severity: str, code: str, details: dict[str, Any]) -> None: ...


class FileAlertSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, severity: str, code: str, details: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": severity,
            "code": code,
            "details": redact(details),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def validate_api_key_policy(
    *,
    trading_enabled: bool,
    withdrawals_enabled: bool,
    ip_allowlist_enabled: bool,
    separate_read_only_key: bool,
) -> list[str]:
    failures = []
    if not trading_enabled:
        failures.append("trading_permission_missing")
    if withdrawals_enabled:
        failures.append("withdrawal_permission_must_be_disabled")
    if not ip_allowlist_enabled:
        failures.append("ip_allowlist_required")
    if not separate_read_only_key:
        failures.append("separate_read_only_key_required")
    return failures


def evaluate_operational_health(
    snapshot: OperationalSnapshot,
    *,
    target_tolerance: float = 1e-9,
    maximum_slippage_bps: float = 30.0,
    maximum_fee_error: float = 0.01,
) -> OperationalHealth:
    alerts: list[str] = []
    if abs(snapshot.research_target - snapshot.production_target) > target_tolerance:
        alerts.append("research_production_signal_mismatch")
    if abs(snapshot.intended_target - snapshot.actual_target) > target_tolerance:
        alerts.append("position_tracking_error")
    slippage_bps: float | None = None
    if snapshot.expected_fill_price and snapshot.actual_fill_price:
        slippage_bps = (
            abs(snapshot.actual_fill_price / snapshot.expected_fill_price - 1.0)
            * 10_000.0
        )
        if slippage_bps > maximum_slippage_bps:
            alerts.append("excessive_execution_slippage")
    fee_error = abs(snapshot.actual_fee - snapshot.expected_fee)
    if fee_error > maximum_fee_error:
        alerts.append("fee_reconciliation_error")
    if not snapshot.data_fresh:
        alerts.append("data_not_fresh")
    if not snapshot.reconciliation_clean or snapshot.unresolved_order_count:
        alerts.append("exchange_reconciliation_not_clean")
    metrics: dict[str, float | bool | int | None] = {
        "target_tracking_error": abs(snapshot.intended_target - snapshot.actual_target),
        "slippage_bps": slippage_bps,
        "fee_error": fee_error,
        "data_fresh": snapshot.data_fresh,
        "reconciliation_clean": snapshot.reconciliation_clean,
        "unresolved_order_count": snapshot.unresolved_order_count,
    }
    return OperationalHealth(not alerts, tuple(alerts), metrics)


class AccountingLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    fee_asset TEXT NOT NULL
                )
                """
            )

    def record_fill(
        self,
        *,
        fill_id: str,
        order_id: str,
        timestamp: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        fee_asset: str,
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fill_id,
                    order_id,
                    timestamp,
                    symbol,
                    side,
                    quantity,
                    price,
                    fee,
                    fee_asset,
                ),
            )

    def summary(self) -> dict[str, float | int]:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(quantity * price), 0),
                       COALESCE(SUM(fee), 0)
                FROM fills
                """
            ).fetchone()
        return {
            "fill_count": int(row[0]),
            "gross_notional": float(row[1]),
            "fees": float(row[2]),
        }


def backup_state(
    source_paths: list[str | Path],
    destination: str | Path,
) -> Path:
    target = ensure_dir(destination)
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "files": {},
    }
    for value in source_paths:
        source = Path(value)
        if not source.is_file():
            raise FileNotFoundError(source)
        copied = target / source.name
        if source.suffix in {".sqlite", ".db"}:
            with sqlite3.connect(source) as source_db, sqlite3.connect(copied) as target_db:
                source_db.backup(target_db)
        else:
            copied.write_bytes(source.read_bytes())
        digest = hashlib.sha256(copied.read_bytes()).hexdigest()
        manifest["files"][source.name] = {
            "size_bytes": copied.stat().st_size,
            "sha256": digest,
        }
    path = target / "backup_manifest.json"
    write_json(path, manifest)
    return path
