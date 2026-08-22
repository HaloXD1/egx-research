from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.crypto_model_registry import verify_model_bundle
from egx_research.utils import write_json


STAGES = ("shadow", "testnet", "supervised_micro_live", "capped_live")


@dataclass(frozen=True)
class PromotionAssessment:
    campaign_id: str
    current_stage: str
    eligible_targets: tuple[str, ...]
    elapsed_days: int
    allocation_changes: int
    round_trips: int
    clean_testnet_days: int
    clean_micro_live_days: int
    micro_live_fills: int
    blockers: tuple[str, ...]
    operational_failures: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        expected = event.pop("event_sha256")
        if event.get("previous_sha256", "") != previous or _hash(event) != expected:
            raise ValueError("forward event hash chain is invalid")
        event["event_sha256"] = expected
        events.append(event)
        previous = expected
    return events


def record_forward_event(
    campaign_path: str | Path,
    *,
    event_id: str,
    timestamp: str,
    kind: str,
    environment: str,
    payload: dict[str, Any],
    recorded_at: str | datetime | None = None,
) -> Path:
    campaign = _read_json(Path(campaign_path))
    events_path = Path(campaign["events_path"])
    event_time = pd.Timestamp(timestamp)
    event_time = event_time.tz_localize("UTC") if event_time.tzinfo is None else event_time.tz_convert("UTC")
    recorded = pd.Timestamp(recorded_at or datetime.now(UTC))
    recorded = recorded.tz_localize("UTC") if recorded.tzinfo is None else recorded.tz_convert("UTC")
    if event_time.date() < pd.Timestamp(campaign["start_date"]).date():
        raise ValueError("event predates the prospective campaign")
    if event_time > recorded:
        raise ValueError("future-dated campaign events are prohibited")
    events = _load_events(events_path)
    for existing in events:
        if existing["event_id"] == event_id:
            comparable = {
                "event_id": event_id,
                "timestamp": event_time.isoformat(),
                "kind": kind,
                "environment": environment,
                "payload": payload,
            }
            current = {key: existing[key] for key in comparable}
            if current != comparable:
                raise ValueError("event id already exists with different content")
            return events_path
    event = {
        "sequence": len(events) + 1,
        "event_id": event_id,
        "timestamp": event_time.isoformat(),
        "kind": kind,
        "environment": environment,
        "payload": payload,
        "previous_sha256": events[-1]["event_sha256"] if events else "",
    }
    event["event_sha256"] = _hash(event)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return events_path


def _maximum_consecutive_days(values: set[date]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    best = current = 1
    for previous, value in zip(ordered, ordered[1:]):
        if (value - previous).days == 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def assess_promotion(
    campaign_path: str | Path,
    *,
    as_of: str | date | datetime,
) -> PromotionAssessment:
    path = Path(campaign_path)
    campaign = _read_json(path)
    verify_model_bundle(campaign["model_bundle_path"])
    as_of_day = pd.Timestamp(as_of).date()
    events = [
        event
        for event in _load_events(Path(campaign["events_path"]))
        if pd.Timestamp(event["timestamp"]).date() <= as_of_day
    ]
    state_path = path.with_name("promotion_state.json")
    current_stage = _read_json(state_path)["stage"] if state_path.exists() else "shadow"
    elapsed = max(0, (as_of_day - pd.Timestamp(campaign["start_date"]).date()).days)
    changes = len(
        {
            event["payload"].get("signal_date", event["event_id"])
            for event in events
            if event["kind"] == "allocation_change"
        }
    )
    round_trips = sum(event["kind"] == "round_trip" for event in events)
    testnet_days = {
        pd.Timestamp(event["timestamp"]).date()
        for event in events
        if event["environment"] == "testnet"
        and event["kind"] == "daily_health"
        and event["payload"].get("clean") is True
    }
    micro_days = {
        pd.Timestamp(event["timestamp"]).date()
        for event in events
        if event["environment"] == "micro_live"
        and event["kind"] == "daily_health"
        and event["payload"].get("clean") is True
    }
    micro_fills = sum(
        event["environment"] == "micro_live" and event["kind"] == "fill"
        for event in events
    )
    failure_fields = (
        "duplicate_orders",
        "unresolved_orders",
        "signal_parity_failures",
        "balance_mismatches",
        "stale_data_trades",
        "slippage_breaches",
        "risk_violations",
    )
    failures = {
        field: sum(int(event["payload"].get(field, 0)) for event in events)
        for field in failure_fields
    }
    requirements = campaign["requirements"]
    forward_ready = (
        elapsed >= requirements["minimum_elapsed_days"]
        and changes >= requirements["minimum_allocation_changes"]
        and round_trips >= requirements["minimum_round_trips"]
    )
    testnet_clean = _maximum_consecutive_days(testnet_days)
    micro_clean = _maximum_consecutive_days(micro_days)
    operationally_clean = not any(failures.values())

    blockers: list[str] = []
    eligible: list[str] = []
    if current_stage == "shadow":
        eligible.append("testnet")
    if current_stage == "testnet":
        if testnet_clean < requirements["minimum_clean_testnet_days"]:
            blockers.append("insufficient_clean_testnet_days")
        if not forward_ready:
            blockers.append("forward_evidence_incomplete")
        if not operationally_clean:
            blockers.append("operational_failures_present")
        if not blockers:
            eligible.append("supervised_micro_live")
    if current_stage == "supervised_micro_live":
        if micro_clean < requirements["minimum_micro_live_days"]:
            blockers.append("insufficient_clean_micro_live_days")
        if micro_fills < requirements["minimum_micro_live_fills"]:
            blockers.append("insufficient_micro_live_fills")
        if not operationally_clean:
            blockers.append("operational_failures_present")
        if not blockers:
            eligible.append("capped_live")
    return PromotionAssessment(
        campaign["campaign_id"],
        current_stage,
        tuple(eligible),
        elapsed,
        changes,
        round_trips,
        testnet_clean,
        micro_clean,
        micro_fills,
        tuple(blockers),
        failures,
    )


def promote_campaign(
    campaign_path: str | Path,
    *,
    target_stage: str,
    as_of: str | date | datetime,
) -> Path:
    if target_stage not in STAGES:
        raise ValueError(f"unknown promotion stage: {target_stage}")
    assessment = assess_promotion(campaign_path, as_of=as_of)
    if target_stage not in assessment.eligible_targets:
        raise ValueError(
            f"campaign is not eligible for {target_stage}: {', '.join(assessment.blockers) or 'invalid stage transition'}"
        )
    path = Path(campaign_path)
    campaign = _read_json(path)
    receipt_content = {
        "campaign_id": campaign["campaign_id"],
        "model_bundle_sha256": campaign["model_bundle_sha256"],
        "from_stage": assessment.current_stage,
        "stage": target_stage,
        "as_of": str(pd.Timestamp(as_of).date()),
        "assessment": assessment.to_dict(),
    }
    receipt = {**receipt_content, "receipt_sha256": _hash(receipt_content)}
    receipt_path = path.with_name(f"promotion_{target_stage}.json")
    if receipt_path.exists():
        if _read_json(receipt_path) != receipt:
            raise ValueError("immutable promotion receipt already exists")
        return receipt_path
    write_json(receipt_path, receipt)
    write_json(
        path.with_name("promotion_state.json"),
        {
            "stage": target_stage,
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
    )
    return receipt_path
