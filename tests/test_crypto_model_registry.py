from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from egx_research.crypto_model_registry import (
    freeze_crypto_model,
    register_challenger,
    set_champion,
    start_forward_campaign,
    verify_model_bundle,
)


def _research_run(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    run = runs / "research-1"
    run.mkdir(parents=True)
    (run / "crypto_research_summary.json").write_text(
        json.dumps(
            {
                "top_family": "crypto_donchian_breakout",
                "top_params": {"entry_lookback": 100},
                "validation_status": "nested_expanding_presealed",
                "sealed_holdout_used_for_selection": False,
                "top_passed_filters": True,
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps({"provenance": {"repository": {"git_commit": "abc"}}}),
        encoding="utf-8",
    )
    contract = tmp_path / "live.yaml"
    contract.write_text("symbol: BTCUSDT\n", encoding="utf-8")
    return runs, contract


def test_freeze_bundle_and_campaign_are_immutable(tmp_path: Path) -> None:
    runs, contract = _research_run(tmp_path)
    registry = tmp_path / "registry"
    bundle_path = freeze_crypto_model(
        "research-1",
        runs_root=runs,
        registry_root=registry,
        contract_path=contract,
        require_clean=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    bundle = verify_model_bundle(bundle_path)
    assert bundle["family"] == "crypto_donchian_breakout"

    campaign = start_forward_campaign(
        bundle_path,
        campaign_id="forward-1",
        start_date="2026-01-02",
        campaigns_root=tmp_path / "campaigns",
    )
    assert start_forward_campaign(
        bundle_path,
        campaign_id="forward-1",
        start_date="2026-01-02",
        campaigns_root=tmp_path / "campaigns",
    ) == campaign
    with pytest.raises(ValueError, match="different immutable terms"):
        start_forward_campaign(
            bundle_path,
            campaign_id="forward-1",
            start_date="2026-02-01",
            campaigns_root=tmp_path / "campaigns",
        )


def test_registry_preserves_rollback_champion(tmp_path: Path) -> None:
    runs, contract = _research_run(tmp_path)
    root = tmp_path / "registry"
    bundle_path = freeze_crypto_model(
        "research-1",
        runs_root=runs,
        registry_root=root,
        contract_path=contract,
        require_clean=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    model_id = verify_model_bundle(bundle_path)["bundle_sha256"][:16]
    register_challenger(bundle_path, registry_root=root)
    registry_path = set_champion(
        model_id,
        approval_reference="campaign:forward-1",
        registry_root=root,
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["champion"] == model_id
    assert registry["rollback"] is None


def test_bundle_tampering_is_detected(tmp_path: Path) -> None:
    runs, contract = _research_run(tmp_path)
    bundle_path = freeze_crypto_model(
        "research-1",
        runs_root=runs,
        registry_root=tmp_path / "registry",
        contract_path=contract,
        require_clean=False,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["params"]["entry_lookback"] = 1
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_model_bundle(bundle_path)


def test_external_model_requires_verified_point_in_time_vintages(
    tmp_path: Path,
) -> None:
    runs, contract = _research_run(tmp_path)
    summary_path = runs / "research-1" / "crypto_research_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "external_data_required": True,
            "external_vintages_verified": False,
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="point-in-time vintages"):
        freeze_crypto_model(
            "research-1",
            runs_root=runs,
            registry_root=tmp_path / "registry",
            contract_path=contract,
            require_clean=False,
        )
