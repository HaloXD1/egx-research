from __future__ import annotations

import json
from pathlib import Path

from egx_research.crypto_operations import (
    AccountingLedger,
    FileAlertSink,
    OperationalSnapshot,
    backup_state,
    evaluate_operational_health,
    validate_api_key_policy,
)


def _snapshot() -> OperationalSnapshot:
    return OperationalSnapshot(
        research_target=0.5,
        production_target=0.5,
        intended_target=0.5,
        actual_target=0.5,
        expected_fill_price=100.0,
        actual_fill_price=100.1,
        expected_fee=1.0,
        actual_fee=1.0,
        data_fresh=True,
        reconciliation_clean=True,
        unresolved_order_count=0,
    )


def test_operational_health_detects_parity_and_reconciliation_failures() -> None:
    snapshot = OperationalSnapshot(
        **{
            **_snapshot().__dict__,
            "production_target": 0.4,
            "actual_target": 0.3,
            "reconciliation_clean": False,
        }
    )
    health = evaluate_operational_health(snapshot)
    assert not health.healthy
    assert "research_production_signal_mismatch" in health.alerts
    assert "position_tracking_error" in health.alerts
    assert "exchange_reconciliation_not_clean" in health.alerts


def test_api_key_policy_rejects_withdrawal_permission() -> None:
    failures = validate_api_key_policy(
        trading_enabled=True,
        withdrawals_enabled=True,
        ip_allowlist_enabled=True,
        separate_read_only_key=True,
    )
    assert failures == ["withdrawal_permission_must_be_disabled"]


def test_accounting_fill_is_idempotent_and_backed_up(tmp_path: Path) -> None:
    ledger = AccountingLedger(tmp_path / "ledger.sqlite")
    values = {
        "fill_id": "fill-1",
        "order_id": "order-1",
        "timestamp": "2026-01-01T00:00:00Z",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 0.1,
        "price": 100_000.0,
        "fee": 10.0,
        "fee_asset": "USDT",
    }
    ledger.record_fill(**values)
    ledger.record_fill(**values)
    assert ledger.summary() == {
        "fill_count": 1,
        "gross_notional": 10_000.0,
        "fees": 10.0,
    }
    manifest = backup_state([ledger.path], tmp_path / "backup")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["files"]["ledger.sqlite"]["sha256"]


def test_alert_file_redacts_secrets(tmp_path: Path) -> None:
    path = tmp_path / "alerts.jsonl"
    FileAlertSink(path).send(
        "critical", "order_unknown", {"api_secret": "do-not-log", "intent": "one"}
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["details"]["api_secret"] == "[REDACTED]"
