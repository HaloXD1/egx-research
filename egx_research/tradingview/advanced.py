from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.models import StrategyDefinition
from egx_research.tradingview.registry import strategy_dict
from egx_research.utils import ensure_dir, write_json


TRADE_ALIASES = {
    "entry_date": "entry_date", "entry_time": "entry_date", "entry time": "entry_date",
    "exit_date": "exit_date", "exit_time": "exit_date", "exit time": "exit_date",
    "entry_price": "entry_price", "entry price": "entry_price",
    "exit_price": "exit_price", "exit price": "exit_price",
    "shares": "shares", "qty": "shares", "quantity": "shares", "contracts": "shares",
    "profit": "pnl", "p&l": "pnl", "pnl": "pnl", "profit_usd": "pnl",
}


def compare_trades(local_path: str | Path, tradingview_path: str | Path, price_tolerance_bps: float = 10.0) -> dict[str, Any]:
    local = pd.read_csv(local_path)
    remote = pd.read_csv(tradingview_path)
    renamed = {}
    for column in remote.columns:
        key = str(column).strip().lower().replace("_", " ")
        normalized_key = key.replace(" ", "_")
        if key in TRADE_ALIASES:
            renamed[column] = TRADE_ALIASES[key]
        elif normalized_key in TRADE_ALIASES:
            renamed[column] = TRADE_ALIASES[normalized_key]
    remote = remote.rename(columns=renamed)
    required = {"entry_date", "exit_date", "entry_price", "exit_price"}
    for label, frame in (("local", local), ("TradingView", remote)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} trades missing columns: {', '.join(missing)}")
    for frame in (local, remote):
        frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce", utc=True).dt.normalize()
        frame["exit_date"] = pd.to_datetime(frame["exit_date"], errors="coerce", utc=True).dt.normalize()
        frame["entry_price"] = pd.to_numeric(frame["entry_price"], errors="coerce")
        frame["exit_price"] = pd.to_numeric(frame["exit_price"], errors="coerce")
    pairs: list[dict[str, Any]] = []
    used: set[int] = set()
    for local_index, left in local.iterrows():
        candidates = remote[(remote["entry_date"] == left["entry_date"]) & (remote["exit_date"] == left["exit_date"])]
        candidates = candidates[~candidates.index.isin(used)]
        if candidates.empty:
            pairs.append({"local_index": int(local_index), "status": "missing_in_tradingview"})
            continue
        remote_index = int(candidates.index[0])
        used.add(remote_index)
        right = remote.loc[remote_index]
        entry_bps = abs(float(left["entry_price"]) - float(right["entry_price"])) / max(abs(float(left["entry_price"])), 1e-12) * 10_000
        exit_bps = abs(float(left["exit_price"]) - float(right["exit_price"])) / max(abs(float(left["exit_price"])), 1e-12) * 10_000
        quantity_match = True
        if "shares" in local.columns and "shares" in remote.columns:
            quantity_match = abs(float(left["shares"]) - float(right["shares"])) <= 1e-8
        status = "match" if max(entry_bps, exit_bps) <= price_tolerance_bps and quantity_match else "mismatch"
        pairs.append({
            "local_index": int(local_index), "tradingview_index": remote_index, "status": status,
            "entry_price_diff_bps": entry_bps, "exit_price_diff_bps": exit_bps,
            "quantity_match": quantity_match,
        })
    for remote_index in sorted(set(remote.index) - used):
        pairs.append({"tradingview_index": int(remote_index), "status": "extra_in_tradingview"})
    mismatches = [row for row in pairs if row["status"] != "match"]
    return {
        "status": "pass" if not mismatches else "fail",
        "local_trades": len(local), "tradingview_trades": len(remote),
        "matched_trades": len(pairs) - len(mismatches), "price_tolerance_bps": price_tolerance_bps,
        "comparisons": pairs,
    }


def export_parameters(definition: StrategyDefinition, output_dir: str | Path, params: dict[str, Any] | None = None) -> Path:
    target = ensure_dir(output_dir)
    write_json(target / "strategy.json", strategy_dict(definition))
    values = dict(definition.params)
    values.update(params or {})
    write_json(target / "pine_inputs.json", values)
    pd.DataFrame([{"input": key, "value": value} for key, value in values.items()]).to_csv(target / "pine_inputs.csv", index=False)
    return target


def compare_inputs(definition: StrategyDefinition, tradingview_inputs: str | Path) -> dict[str, Any]:
    path = Path(tradingview_inputs)
    if path.suffix.lower() == ".json":
        observed = json.loads(path.read_text(encoding="utf-8"))
    else:
        frame = pd.read_csv(path)
        if not {"input", "value"}.issubset(frame.columns):
            raise ValueError("TradingView input CSV requires input and value columns")
        observed = dict(zip(frame["input"].astype(str), frame["value"], strict=False))
    rows = []

    def matches(expected: Any, actual: Any) -> bool:
        if isinstance(expected, bool):
            return str(actual).strip().lower() in ({"true", "1"} if expected else {"false", "0"})
        if isinstance(expected, (int, float)):
            try:
                return abs(float(expected) - float(actual)) <= 1e-12
            except (TypeError, ValueError):
                return False
        return str(expected) == str(actual)

    for key in sorted(set(definition.params) | set(observed)):
        expected = definition.params.get(key)
        actual = observed.get(key)
        rows.append({"input": key, "expected": expected, "actual": actual, "status": "match" if matches(expected, actual) else "mismatch"})
    return {"status": "pass" if all(row["status"] == "match" for row in rows) else "fail", "strategy_id": definition.id, "inputs": rows}


def promote_strategy(registry_path: str | Path, strategy_id: str, validation_run: str | Path, status: str) -> Path:
    if status not in {"validated", "paper", "production"}:
        raise ValueError("Promotion status must be validated, paper, or production")
    summary_path = Path(validation_run) / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("strategy_id") != strategy_id or summary.get("status") != "pass":
        raise ValueError("Promotion requires a passing validation run for the same strategy")
    path = Path(registry_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    matches = [row for row in payload.get("strategies", []) if row.get("id") == strategy_id]
    if not matches:
        raise ValueError(f"Unknown TradingView strategy: {strategy_id}")
    row = max(matches, key=lambda item: tuple(int(part) for part in str(item.get("version", "0.0.0")).split(".")))
    row["status"] = status
    row["validated_run"] = str(Path(validation_run))
    row["promoted_at"] = datetime.now(UTC).isoformat()
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def list_runs(config: TradingViewConfig) -> list[dict[str, Any]]:
    root = Path(config.runs_dir)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.stat().st_mtime, reverse=True):
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        rows.append({"run_id": path.name, "command": manifest.get("command"), "strategy_id": manifest.get("strategy_id"), "created_at": manifest.get("created_at")})
    return rows


def compare_runs(config: TradingViewConfig, run_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = Path(config.runs_dir) / run_id
        row: dict[str, Any] = {"run_id": run_id}
        for name in ("manifest.json", "metrics.json", "validation_summary.json", "parity_summary.json"):
            path = run_dir / name
            if path.exists():
                row[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
    return rows


def archive_run(config: TradingViewConfig, run_id: str, archive_dir: str | Path) -> Path:
    source = Path(config.runs_dir) / run_id
    if not source.exists():
        raise FileNotFoundError(source)
    target = ensure_dir(archive_dir) / run_id
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(source), str(target))
    return target


def clean_runs(config: TradingViewConfig, older_than_days: int, confirm: bool = False) -> list[str]:
    cutoff = datetime.now(UTC).timestamp() - max(0, older_than_days) * 86400
    root = Path(config.runs_dir)
    if not root.exists():
        return []
    candidates = [path for path in root.iterdir() if path.is_dir() and path.stat().st_mtime < cutoff]
    if confirm:
        for path in candidates:
            shutil.rmtree(path)
    return [path.name for path in candidates]
