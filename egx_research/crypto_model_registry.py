from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from egx_research.reproducibility import repository_state, sha256_file
from egx_research.utils import ensure_dir, write_json


DECISION_CODE_PATHS = (
    "egx_research/backtest.py",
    "egx_research/crypto_research.py",
    "egx_research/crypto_strategies.py",
    "egx_research/nested_validation.py",
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle = _read_json(Path(path))
    expected = bundle.get("bundle_sha256")
    content = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if not expected or _canonical_hash(content) != expected:
        raise ValueError("model bundle hash mismatch")
    return bundle


def freeze_crypto_model(
    research_run_id: str,
    *,
    runs_root: str | Path = "runs",
    registry_root: str | Path = "data/crypto/model_registry",
    contract_path: str | Path = "config/live_btc.yaml",
    require_clean: bool = True,
    require_accepted: bool = True,
    created_at: datetime | None = None,
) -> Path:
    run_dir = Path(runs_root) / research_run_id
    summary_path = run_dir / "crypto_research_summary.json"
    manifest_path = run_dir / "manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("research summary and manifest are required")
    contract = Path(contract_path)
    if not contract.is_file():
        raise FileNotFoundError(f"missing live contract: {contract}")

    repo = repository_state()
    if require_clean and repo.get("git_dirty") is not False:
        raise ValueError("model freezing requires a clean committed worktree")
    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)
    if summary.get("sealed_holdout_used_for_selection") is not False:
        raise ValueError("research run does not prove sealed holdout isolation")
    if require_accepted and summary.get("top_passed_filters") is not True:
        raise ValueError("research candidate did not pass its acceptance filters")
    run_repository = manifest.get("provenance", {}).get("repository", {})
    if require_clean:
        if run_repository.get("git_dirty") is not False:
            raise ValueError("research run was not produced from a clean worktree")
        if run_repository.get("git_commit") != repo.get("git_commit"):
            raise ValueError("research run commit does not match current decision code")

    code_hashes = {
        path: sha256_file(path)
        for path in DECISION_CODE_PATHS
        if Path(path).is_file()
    }
    content: dict[str, Any] = {
        "schema_version": 1,
        "created_at": (
            created_at.isoformat()
            if created_at is not None
            else str(manifest.get("created_at") or datetime.now(UTC).isoformat())
        ),
        "research_run_id": research_run_id,
        "family": summary["top_family"],
        "params": summary["top_params"],
        "validation_status": summary.get("validation_status"),
        "research_summary_sha256": sha256_file(summary_path),
        "research_manifest_sha256": sha256_file(manifest_path),
        "research_provenance": manifest.get("provenance", {}),
        "repository": repo,
        "decision_code_sha256": code_hashes,
        "live_contract_path": str(contract),
        "live_contract_sha256": sha256_file(contract),
    }
    bundle_hash = _canonical_hash(content)
    bundle = {**content, "bundle_sha256": bundle_hash}
    target = (
        ensure_dir(Path(registry_root) / "models" / bundle_hash[:16])
        / "model_bundle.json"
    )
    if target.exists():
        existing = verify_model_bundle(target)
        if existing != bundle:
            raise ValueError("immutable model bundle already exists with different content")
        return target
    write_json(target, bundle)
    return target


def start_forward_campaign(
    bundle_path: str | Path,
    *,
    campaign_id: str,
    start_date: str,
    campaigns_root: str | Path = "data/crypto/forward",
    minimum_elapsed_days: int = 365,
    minimum_allocation_changes: int = 8,
    minimum_round_trips: int = 4,
) -> Path:
    if min(
        minimum_elapsed_days,
        minimum_allocation_changes,
        minimum_round_trips,
    ) < 0:
        raise ValueError("campaign thresholds cannot be negative")
    bundle = verify_model_bundle(bundle_path)
    target = ensure_dir(Path(campaigns_root) / campaign_id) / "campaign.json"
    campaign = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "status": "active",
        "start_date": str(start_date),
        "model_bundle_path": str(bundle_path),
        "model_bundle_sha256": bundle["bundle_sha256"],
        "requirements": {
            "minimum_elapsed_days": minimum_elapsed_days,
            "minimum_allocation_changes": minimum_allocation_changes,
            "minimum_round_trips": minimum_round_trips,
            "all_required": True,
        },
        "events_path": str(target.with_name("events.jsonl")),
    }
    if target.exists():
        if _read_json(target) != campaign:
            raise ValueError("campaign id already exists with different immutable terms")
        return target
    write_json(target, campaign)
    target.with_name("events.jsonl").touch(exist_ok=True)
    return target


def register_challenger(
    bundle_path: str | Path,
    *,
    registry_root: str | Path = "data/crypto/model_registry",
) -> Path:
    bundle = verify_model_bundle(bundle_path)
    path = Path(registry_root) / "registry.json"
    registry = (
        _read_json(path)
        if path.exists()
        else {"schema_version": 1, "champion": None, "rollback": None, "challengers": []}
    )
    model_id = bundle["bundle_sha256"][:16]
    if model_id not in registry["challengers"] and model_id != registry["champion"]:
        registry["challengers"].append(model_id)
    write_json(path, registry)
    return path


def set_champion(
    model_id: str,
    *,
    approval_reference: str,
    registry_root: str | Path = "data/crypto/model_registry",
) -> Path:
    if not approval_reference.strip():
        raise ValueError("champion promotion requires an approval reference")
    path = Path(registry_root) / "registry.json"
    if not path.exists():
        raise FileNotFoundError("model registry does not exist")
    registry = _read_json(path)
    if model_id not in registry.get("challengers", []):
        raise ValueError("only a registered challenger can become champion")
    previous = registry.get("champion")
    registry["rollback"] = previous
    registry["champion"] = model_id
    registry["challengers"] = [
        item for item in registry["challengers"] if item != model_id
    ]
    registry["approval_reference"] = approval_reference
    write_json(path, registry)
    return path
