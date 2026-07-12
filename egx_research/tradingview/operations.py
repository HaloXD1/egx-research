from __future__ import annotations

import re
from datetime import UTC, datetime
from dataclasses import asdict
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.crypto_config import load_crypto_config
from egx_research.crypto_data import sync_crypto_data
from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.models import StrategyDefinition, SymbolMapping
from egx_research.tradingview.pine import inspect_pine
from egx_research.tradingview.registry import load_registry, load_symbols, strategy_dict
from egx_research.utils import ensure_dir, write_json


def refresh_symbol(config: TradingViewConfig, symbol: SymbolMapping) -> Path:
    if symbol.asset_class == "crypto" or symbol.logical_symbol == "BTCUSDT":
        crypto_config = load_crypto_config(config.crypto_config_path)
        return sync_crypto_data(crypto_config)
    raise ValueError(f"No automatic local sync is configured for {symbol.logical_symbol}; ingest its source first")


def data_status(symbol: SymbolMapping, max_stale_days: int = 2) -> dict[str, Any]:
    path = Path(symbol.local_path)
    if not path.exists():
        return {"logical_symbol": symbol.logical_symbol, "path": str(path), "status": "missing"}
    frame = pd.read_csv(path, usecols=["date"])
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return {"logical_symbol": symbol.logical_symbol, "path": str(path), "status": "empty"}
    latest = pd.Timestamp(dates.max()).normalize()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    age_days = max(0, int((today - latest).days))
    is_current_bar = latest == today
    status = "current_incomplete_bar" if is_current_bar else ("fresh" if age_days <= max_stale_days else "stale")
    return {
        "logical_symbol": symbol.logical_symbol,
        "path": str(path),
        "latest_date": str(latest.date()),
        "today_utc": str(today.date()),
        "age_days": age_days,
        "current_bar_may_be_incomplete": is_current_bar,
        "status": status,
    }


def audit_strategy(definition: StrategyDefinition) -> dict[str, Any]:
    details = inspect_pine(definition.pine_path)
    text = Path(definition.pine_path).read_text(encoding="utf-8")
    errors: list[str] = []
    warnings = list(details.get("warnings", []))
    if details["pine_version"] != definition.pine_version:
        errors.append(f"Registry Pine version {definition.pine_version} does not match source")
    if details["script_kind"] != definition.script_kind:
        errors.append(f"Registry script kind {definition.script_kind} does not match source")
    expected_close = definition.execution.fill_model == "close"
    process_match = re.search(r"process_orders_on_close\s*=\s*(true|false)", text)
    if process_match and (process_match.group(1) == "true") != expected_close:
        errors.append("Pine process_orders_on_close disagrees with registry fill_model")
    if "request.security" in text and "lookahead_on" in text:
        errors.append("Possible lookahead bias from request.security lookahead_on")
    if definition.signal_contract.signal_mode == "close_confirmed" and "barstate.isconfirmed" not in text:
        warnings.append("Close-confirmed contract relies on default bar-close execution; no explicit barstate.isconfirmed guard")
    if definition.signal_contract.warmup_bars <= 0:
        warnings.append("Signal contract does not declare warmup_bars")
    return {"strategy_id": definition.id, "valid": not errors, "errors": errors, "warnings": warnings, "pine": details, "execution": definition.execution.__dict__}


def doctor(config: TradingViewConfig, strict_data: bool = False) -> dict[str, Any]:
    """Validate the complete local TradingView configuration and registry."""
    errors: list[str] = []
    warnings: list[str] = []
    strategies: list[dict[str, Any]] = []
    symbols_status: list[dict[str, Any]] = []
    try:
        symbols = load_symbols(config.symbol_map_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        return {"status": "fail", "valid": False, "errors": [str(exc)], "warnings": [], "strategies": [], "symbols": []}
    try:
        definitions = load_registry(config.registry_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        return {"status": "fail", "valid": False, "errors": [str(exc)], "warnings": [], "strategies": [], "symbols": []}

    if not definitions:
        errors.append("TradingView registry contains no strategies")
    if not symbols:
        errors.append("TradingView symbol map contains no symbols")

    for logical_symbol, mapping in symbols.items():
        item = data_status(mapping, config.default_max_stale_days)
        symbols_status.append(item)
        if not mapping.tradingview_symbol:
            warnings.append(f"{logical_symbol}: TradingView symbol is not mapped")
        if item["status"] in {"missing", "empty"}:
            message = f"{logical_symbol}: local data is {item['status']}"
            (errors if strict_data else warnings).append(message)
        elif item["status"] == "stale":
            message = f"{logical_symbol}: local data is stale ({item['age_days']} days)"
            (errors if strict_data else warnings).append(message)

    for definition in definitions:
        if definition.logical_symbol not in symbols:
            errors.append(f"{definition.id}: unknown logical symbol {definition.logical_symbol}")
            continue
        mapping = symbols[definition.logical_symbol]
        if definition.asset_class != mapping.asset_class:
            errors.append(f"{definition.id}: asset class disagrees with symbol map")
        if definition.timeframe != mapping.timeframe:
            errors.append(f"{definition.id}: timeframe disagrees with symbol map")
        try:
            audit = audit_strategy(definition)
        except (FileNotFoundError, OSError, UnicodeError) as exc:
            audit = {"strategy_id": definition.id, "valid": False, "errors": [str(exc)], "warnings": []}
        strategies.append(audit)
        errors.extend(f"{definition.id}: {message}" for message in audit.get("errors", []))
        warnings.extend(f"{definition.id}: {message}" for message in audit.get("warnings", []))

    return {
        "status": "pass" if not errors else "fail",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "strategy_count": len(definitions),
        "symbol_count": len(symbols),
        "strategies": strategies,
        "symbols": symbols_status,
    }


def export_strategy_bundle(
    config: TradingViewConfig,
    definition: StrategyDefinition,
    symbol: SymbolMapping,
    run_id: str,
) -> Path:
    """Create a reproducible, credential-free Pine handoff bundle."""
    source = Path(definition.pine_path)
    audit = audit_strategy(definition)
    if not audit["valid"]:
        raise ValueError(f"Cannot export invalid strategy {definition.id}: {'; '.join(audit['errors'])}")
    bundle_dir = ensure_dir(Path(config.runs_dir) / run_id / "tradingview_export")
    target = bundle_dir / source.name
    shutil.copy2(source, target)
    write_json(bundle_dir / "strategy.json", strategy_dict(definition))
    write_json(bundle_dir / "symbol.json", asdict(symbol))
    write_json(bundle_dir / "audit.json", audit)
    write_json(
        bundle_dir / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "strategy_id": definition.id,
            "strategy_version": definition.version,
            "logical_symbol": symbol.logical_symbol,
            "tradingview_symbol": symbol.tradingview_symbol,
            "pine_file": target.name,
            "pine_source_hash": audit["pine"]["source_hash"],
            "timeframe": definition.timeframe,
            "timezone": definition.timezone,
            "execution": asdict(definition.execution),
            "signal_contract": asdict(definition.signal_contract),
            "live_order_execution": False,
        },
    )
    return bundle_dir
