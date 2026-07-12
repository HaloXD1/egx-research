from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from egx_research.config import BacktestConfig


@dataclass
class ParityConfig:
    date_tolerance_bars: int = 0
    require_full_coverage: bool = True
    max_missing_events: int = 0


@dataclass
class TradingViewConfig:
    registry_path: str = "tradingview/registry.yaml"
    symbol_map_path: str = "tradingview/symbols.yaml"
    template_dir: str = "tradingview/templates"
    runs_dir: str = "runs"
    default_timezone: str = "UTC"
    crypto_config_path: str = "config/crypto_btc.yaml"
    default_max_stale_days: int = 2
    parity: ParityConfig = field(default_factory=ParityConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    browser: dict[str, Any] = field(default_factory=lambda: {"enabled": False, "engine": "playwright"})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_tradingview_config(path: str | Path = "config/tradingview.yaml") -> TradingViewConfig:
    path = Path(path)
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    merged = _merge_dict(TradingViewConfig().to_dict(), raw)
    return TradingViewConfig(
        registry_path=str(merged["registry_path"]),
        symbol_map_path=str(merged["symbol_map_path"]),
        template_dir=str(merged["template_dir"]),
        runs_dir=str(merged["runs_dir"]),
        default_timezone=str(merged["default_timezone"]),
        crypto_config_path=str(merged["crypto_config_path"]),
        default_max_stale_days=int(merged["default_max_stale_days"]),
        parity=ParityConfig(**merged.get("parity", {})),
        backtest=BacktestConfig(**merged.get("backtest", {})),
        browser=dict(merged.get("browser", {})),
    )
