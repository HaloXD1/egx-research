from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from egx_research.crypto_model_registry import freeze_crypto_model, start_forward_campaign
from egx_research.crypto_promotion import (
    assess_promotion,
    promote_campaign,
    record_forward_event,
)


def _campaign(tmp_path: Path) -> Path:
    run = tmp_path / "runs/research"
    run.mkdir(parents=True)
    (run / "crypto_research_summary.json").write_text(
        json.dumps(
            {
                "top_family": "crypto_donchian_breakout",
                "top_params": {"lookback": 100},
                "validation_status": "nested_expanding_presealed",
                "sealed_holdout_used_for_selection": False,
                "top_passed_filters": True,
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    contract = tmp_path / "live.yaml"
    contract.write_text("symbol: BTCUSDT\n", encoding="utf-8")
    bundle = freeze_crypto_model(
        "research",
        runs_root=tmp_path / "runs",
        registry_root=tmp_path / "registry",
        contract_path=contract,
        require_clean=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return start_forward_campaign(
        bundle,
        campaign_id="campaign",
        start_date="2026-01-01",
        campaigns_root=tmp_path / "campaigns",
    )


def _record_clean_evidence(campaign: Path) -> None:
    for index in range(8):
        record_forward_event(
            campaign,
            event_id=f"change-{index}",
            timestamp=f"2026-{index + 1:02d}-01T00:00:00Z",
            kind="allocation_change",
            environment="shadow",
            payload={"signal_date": f"2026-{index + 1:02d}-01"},
            recorded_at="2027-12-31",
        )
    for index in range(4):
        record_forward_event(
            campaign,
            event_id=f"trip-{index}",
            timestamp=f"2026-{index + 1:02d}-15T00:00:00Z",
            kind="round_trip",
            environment="shadow",
            payload={},
            recorded_at="2027-12-31",
        )
    for index, day in enumerate(pd.date_range("2026-11-01", periods=30, freq="D")):
        record_forward_event(
            campaign,
            event_id=f"testnet-{index}",
            timestamp=day.isoformat(),
            kind="daily_health",
            environment="testnet",
            payload={"clean": True},
            recorded_at="2027-12-31",
        )


def test_campaign_requires_real_elapsed_time_and_events(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    promote_campaign(campaign, target_stage="testnet", as_of="2026-01-02")
    early = assess_promotion(campaign, as_of="2026-12-31")
    assert "forward_evidence_incomplete" in early.blockers

    _record_clean_evidence(campaign)
    ready = assess_promotion(campaign, as_of="2027-01-02")
    assert ready.eligible_targets == ("supervised_micro_live",)


def test_capped_live_requires_micro_live_soak_and_fills(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    _record_clean_evidence(campaign)
    promote_campaign(campaign, target_stage="testnet", as_of="2026-01-02")
    promote_campaign(
        campaign, target_stage="supervised_micro_live", as_of="2027-01-02"
    )
    with pytest.raises(ValueError, match="insufficient_clean_micro_live_days"):
        promote_campaign(campaign, target_stage="capped_live", as_of="2027-01-03")

    for index, day in enumerate(pd.date_range("2027-01-03", periods=30, freq="D")):
        record_forward_event(
            campaign,
            event_id=f"micro-health-{index}",
            timestamp=day.isoformat(),
            kind="daily_health",
            environment="micro_live",
            payload={"clean": True},
            recorded_at="2027-12-31",
        )
    for index in range(4):
        record_forward_event(
            campaign,
            event_id=f"micro-fill-{index}",
            timestamp=f"2027-01-{index + 3:02d}T01:00:00Z",
            kind="fill",
            environment="micro_live",
            payload={},
            recorded_at="2027-12-31",
        )
    receipt = promote_campaign(
        campaign, target_stage="capped_live", as_of="2027-02-02"
    )
    assert json.loads(receipt.read_text(encoding="utf-8"))["stage"] == "capped_live"


def test_event_ids_are_idempotent_and_hash_chained(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    kwargs = {
        "event_id": "one",
        "timestamp": "2026-01-02T00:00:00Z",
        "kind": "daily_health",
        "environment": "testnet",
        "payload": {"clean": True},
    }
    path = record_forward_event(campaign, **kwargs)
    assert record_forward_event(campaign, **kwargs) == path
    with pytest.raises(ValueError, match="different content"):
        record_forward_event(campaign, **{**kwargs, "payload": {"clean": False}})


def test_future_dated_event_is_rejected(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    with pytest.raises(ValueError, match="future-dated"):
        record_forward_event(
            campaign,
            event_id="future",
            timestamp="2027-01-01T00:00:00Z",
            kind="daily_health",
            environment="testnet",
            payload={"clean": True},
            recorded_at="2026-12-31T23:59:59Z",
        )
