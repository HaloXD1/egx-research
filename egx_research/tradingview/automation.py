from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.models import StrategyDefinition, SymbolMapping
from egx_research.tradingview.notifications import notify_run
from egx_research.tradingview.operations import data_status, refresh_symbol
from egx_research.tradingview.paper import run_paper_track
from egx_research.tradingview.research import run_backtest, run_scan
from egx_research.utils import write_json


def run_daily_pipeline(
    config: TradingViewConfig,
    definition: StrategyDefinition,
    symbol: SymbolMapping,
    run_id: str,
    refresh: bool = False,
    notify: bool = False,
    channel: str = "webhook",
    params_json: str | None = None,
) -> Path:
    root = Path(config.runs_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    if refresh:
        path = refresh_symbol(config, symbol)
        steps.append({"step": "sync", "status": "complete", "path": str(path)})
    quality = data_status(symbol, config.default_max_stale_days)
    if quality["status"] in {"missing", "empty", "stale"}:
        raise ValueError(f"Pipeline data gate failed: {quality['status']}")
    scan_dir = run_scan(config, definition, symbol, f"{run_id}-scan", params_json)
    steps.append({"step": "scan", "status": "complete", "path": str(scan_dir)})
    paper_dir = run_paper_track(config, definition, symbol, f"{run_id}-paper", params_json=params_json)
    steps.append({"step": "paper-track", "status": "complete", "path": str(paper_dir)})
    if notify:
        notification = notify_run(f"{run_id}-paper", config.runs_dir, send=True, channel=channel)
        steps.append({"step": "notify", "status": "complete", "path": str(notification)})
    result = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "strategy_id": definition.id,
        "logical_symbol": symbol.logical_symbol,
        "status": "complete",
        "steps": steps,
    }
    write_json(root / "pipeline.json", result)
    return root


def run_batch_backtests(
    config: TradingViewConfig,
    jobs: list[tuple[StrategyDefinition, SymbolMapping]],
    run_id: str,
    params_json: str | None = None,
) -> Path:
    root = Path(config.runs_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for definition, symbol in jobs:
        child_id = f"{run_id}-{definition.id}-{symbol.logical_symbol}".lower().replace("_", "-")
        child = run_backtest(config, definition, symbol, child_id, params_json)
        rows.append({"strategy_id": definition.id, "logical_symbol": symbol.logical_symbol, "run_id": child_id, "path": str(child)})
    write_json(root / "batch.json", {"run_id": run_id, "jobs": rows, "status": "complete"})
    return root


def cron_line(config_path: str, strategy_id: str, symbol: str, hour: int, minute: int) -> str:
    command = f"egx tv schedule run --config {config_path} --strategy {strategy_id} --symbol {symbol} --run-id tv-daily-$(date +\\%Y\\%m\\%d) --refresh"
    return f"{minute} {hour} * * * cd {Path.cwd()} && {command}"
