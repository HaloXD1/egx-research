from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalContract:
    signal_mode: str = "close_confirmed"
    target_exposure: bool = False
    actions: list[str] = field(default_factory=lambda: ["buy", "sell"])
    warmup_bars: int = 0


@dataclass
class ExecutionModel:
    fill_model: str = "next_open"
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    share_precision: int = 0


@dataclass
class StrategyDefinition:
    id: str
    version: str
    display_name: str
    pine_path: str
    pine_version: int = 6
    script_kind: str = "strategy"
    source_mode: str = "hand_authored"
    local_family: str | None = None
    asset_class: str = "equity"
    logical_symbol: str = ""
    timeframe: str = "1d"
    timezone: str = "UTC"
    signal_contract: SignalContract = field(default_factory=SignalContract)
    execution: ExecutionModel = field(default_factory=ExecutionModel)
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "experimental"


@dataclass
class SymbolMapping:
    logical_symbol: str
    local_path: str
    tradingview_symbol: str = ""
    asset_class: str = "equity"
    quote_currency: str = ""
    timeframe: str = "1d"
    timezone: str = "UTC"
    session: str = ""
    data_source: str = "local"
    adjustment: str = "unknown"


def _nested(mapping: dict[str, Any], key: str, cls: type[Any]) -> Any:
    value = mapping.get(key, {}) or {}
    return cls(**value) if isinstance(value, dict) else cls()


def strategy_from_dict(mapping: dict[str, Any]) -> StrategyDefinition:
    return StrategyDefinition(
        id=str(mapping["id"]),
        version=str(mapping.get("version", "1.0.0")),
        display_name=str(mapping.get("display_name", mapping["id"])),
        pine_path=str(mapping["pine_path"]),
        pine_version=int(mapping.get("pine_version", 6)),
        script_kind=str(mapping.get("script_kind", "strategy")),
        source_mode=str(mapping.get("source_mode", "hand_authored")),
        local_family=mapping.get("local_family"),
        asset_class=str(mapping.get("asset_class", "equity")),
        logical_symbol=str(mapping.get("logical_symbol", "")),
        timeframe=str(mapping.get("timeframe", "1d")),
        timezone=str(mapping.get("timezone", "UTC")),
        signal_contract=_nested(mapping, "signal_contract", SignalContract),
        execution=_nested(mapping, "execution", ExecutionModel),
        params=dict(mapping.get("params", {}) or {}),
        status=str(mapping.get("status", "experimental")),
    )


def symbol_from_dict(mapping: dict[str, Any]) -> SymbolMapping:
    return SymbolMapping(
        logical_symbol=str(mapping["logical_symbol"]),
        local_path=str(mapping["local_path"]),
        tradingview_symbol=str(mapping.get("tradingview_symbol", "")),
        asset_class=str(mapping.get("asset_class", "equity")),
        quote_currency=str(mapping.get("quote_currency", "")),
        timeframe=str(mapping.get("timeframe", "1d")),
        timezone=str(mapping.get("timezone", "UTC")),
        session=str(mapping.get("session", "")),
        data_source=str(mapping.get("data_source", "local")),
        adjustment=str(mapping.get("adjustment", "unknown")),
    )
