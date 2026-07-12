from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from egx_research.tradingview.models import StrategyDefinition, SymbolMapping, strategy_from_dict, symbol_from_dict


ID_RE = re.compile(r"^[a-z0-9_]+$")
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
STATUSES = {"draft", "experimental", "validated", "paper", "production"}


def load_registry(path: str | Path) -> list[StrategyDefinition]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"TradingView registry missing: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    rows = payload.get("strategies", []) if isinstance(payload, dict) else payload
    definitions = [strategy_from_dict(row) for row in rows]
    seen: set[tuple[str, str]] = set()
    for definition in definitions:
        if not ID_RE.fullmatch(definition.id):
            raise ValueError(f"Invalid TradingView strategy id: {definition.id}")
        if not VERSION_RE.fullmatch(definition.version):
            raise ValueError(f"Invalid semantic version for {definition.id}: {definition.version}")
        if definition.script_kind not in {"strategy", "indicator"}:
            raise ValueError(f"Invalid script kind for {definition.id}: {definition.script_kind}")
        if definition.status not in STATUSES:
            raise ValueError(f"Invalid status for {definition.id}: {definition.status}")
        key = (definition.id, definition.version)
        if key in seen:
            raise ValueError(f"Duplicate TradingView strategy version: {definition.id}@{definition.version}")
        seen.add(key)
    return definitions


def load_symbols(path: str | Path) -> dict[str, SymbolMapping]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"TradingView symbol map missing: {target}")
    with target.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    rows = payload.get("symbols", []) if isinstance(payload, dict) else payload
    symbols = [symbol_from_dict(row) for row in rows]
    result: dict[str, SymbolMapping] = {}
    for symbol in symbols:
        if symbol.logical_symbol in result:
            raise ValueError(f"Duplicate TradingView logical symbol: {symbol.logical_symbol}")
        result[symbol.logical_symbol] = symbol
    return result


def find_strategy(definitions: list[StrategyDefinition], strategy_id: str, version: str | None = None) -> StrategyDefinition:
    matches = [item for item in definitions if item.id == strategy_id and (version is None or item.version == version)]
    if not matches:
        suffix = f"@{version}" if version else ""
        raise ValueError(f"Unknown TradingView strategy: {strategy_id}{suffix}")
    matches.sort(key=lambda item: tuple(int(part) for part in item.version.split(".")))
    return matches[-1]


def strategy_dict(definition: StrategyDefinition) -> dict[str, Any]:
    return {
        "id": definition.id,
        "version": definition.version,
        "display_name": definition.display_name,
        "pine_path": definition.pine_path,
        "pine_version": definition.pine_version,
        "script_kind": definition.script_kind,
        "source_mode": definition.source_mode,
        "local_family": definition.local_family,
        "asset_class": definition.asset_class,
        "logical_symbol": definition.logical_symbol,
        "timeframe": definition.timeframe,
        "timezone": definition.timezone,
        "signal_contract": {
            "signal_mode": definition.signal_contract.signal_mode,
            "target_exposure": definition.signal_contract.target_exposure,
            "actions": definition.signal_contract.actions,
            "warmup_bars": definition.signal_contract.warmup_bars,
        },
        "execution": {
            "fill_model": definition.execution.fill_model,
            "fee_bps": definition.execution.fee_bps,
            "slippage_bps": definition.execution.slippage_bps,
            "share_precision": definition.execution.share_precision,
        },
        "params": definition.params,
        "status": definition.status,
    }
