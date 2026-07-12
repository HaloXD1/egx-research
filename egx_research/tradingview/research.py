from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.backtest import run_buy_hold_benchmark, run_dca_benchmark, run_strategy_backtest
from egx_research.config import BacktestConfig
from egx_research.crypto_research import run_weekly_dca_benchmark
from egx_research.data import load_price_data
from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.models import StrategyDefinition, SymbolMapping
from egx_research.tradingview.parity import compare_events, normalize_events
from egx_research.tradingview.signals import build_local_frame, canonical_events
from egx_research.tradingview.pine import source_hash
from egx_research.utils import ensure_dir, write_json


def _run_dir(config: TradingViewConfig, run_id: str) -> Path:
    return ensure_dir(Path(config.runs_dir) / run_id)


def _load_quality(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    required = ("date", "open", "high", "low", "close")
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Local data missing required columns: {', '.join(missing)}")
    quality = {
        "path": str(path),
        "rows_raw": len(raw),
        "duplicate_dates": int(raw.duplicated(subset=["date"]).sum()) if "date" in raw.columns else None,
        "missing_required_columns": missing,
    }
    data = load_price_data(path)
    quality.update({
        "rows_loaded": len(data),
        "start_date": str(data["date"].min().date()) if not data.empty else None,
        "end_date": str(data["date"].max().date()) if not data.empty else None,
    })
    return data, quality


def _manifest(definition: StrategyDefinition, symbol: SymbolMapping, run_id: str, data_path: str, quality: dict[str, Any], command: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "command": command,
        "strategy_id": definition.id,
        "strategy_version": definition.version,
        "pine_path": definition.pine_path,
        "pine_version": definition.pine_version,
        "pine_source_hash": source_hash(definition.pine_path) if Path(definition.pine_path).exists() else None,
        "logical_symbol": symbol.logical_symbol,
        "tradingview_symbol": symbol.tradingview_symbol,
        "local_data_path": data_path,
        "timezone": symbol.timezone,
        "timeframe": symbol.timeframe,
        "execution": {
            "fill_model": definition.execution.fill_model,
            "fee_bps": definition.execution.fee_bps,
            "slippage_bps": definition.execution.slippage_bps,
            "share_precision": definition.execution.share_precision,
        },
        "data_quality": quality,
        "validation_status": "local_only",
    }


def _params(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--params-json must contain a JSON object")
    return parsed


def run_scan(config: TradingViewConfig, definition: StrategyDefinition, symbol: SymbolMapping, run_id: str, params_json: str | None = None) -> Path:
    data, quality = _load_quality(symbol.local_path)
    frame = build_local_frame(data, definition, _params(params_json))
    events = canonical_events(frame, symbol.logical_symbol, timezone=symbol.timezone)
    run_dir = _run_dir(config, run_id)
    events.to_csv(run_dir / "signals.csv", index=False)
    latest = frame.iloc[-1]
    payload = {
        "logical_symbol": symbol.logical_symbol,
        "strategy_id": definition.id,
        "as_of": str(latest["date"]),
        "close": float(latest["close"]),
        "entry_signal": bool(latest.get("entry_signal", False)),
        "exit_signal": bool(latest.get("exit_signal", False)),
        "target_exposure": float(latest.get("target_allocation", 1.0 if latest.get("entry_signal", False) else 0.0)),
        "event_count": len(events),
    }
    write_json(run_dir / "scan.json", payload)
    write_json(run_dir / "data_quality.json", quality)
    write_json(run_dir / "manifest.json", _manifest(definition, symbol, run_id, symbol.local_path, quality, "scan"))
    return run_dir


def run_scan_many(config: TradingViewConfig, definition: StrategyDefinition, symbols: list[SymbolMapping], run_id: str, params_json: str | None = None) -> Path:
    rows: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for symbol in symbols:
        data, item_quality = _load_quality(symbol.local_path)
        frame = build_local_frame(data, definition, _params(params_json))
        latest = frame.iloc[-1]
        rows.append({
            "logical_symbol": symbol.logical_symbol,
            "as_of": str(latest["date"]),
            "close": float(latest["close"]),
            "entry_signal": bool(latest.get("entry_signal", False)),
            "exit_signal": bool(latest.get("exit_signal", False)),
            "target_exposure": float(latest.get("target_allocation", 1.0 if latest.get("entry_signal", False) else 0.0)),
        })
        quality.append({"logical_symbol": symbol.logical_symbol, **item_quality})
    run_dir = _run_dir(config, run_id)
    pd.DataFrame(rows).to_csv(run_dir / "scan.csv", index=False)
    write_json(run_dir / "scan.json", {"strategy_id": definition.id, "symbols": rows})
    write_json(run_dir / "data_quality.json", quality)
    if symbols:
        write_json(run_dir / "manifest.json", {**_manifest(definition, symbols[0], run_id, "multiple", {"symbols": quality}, "scan"), "symbols": [symbol.logical_symbol for symbol in symbols]})
    return run_dir


def run_backtest(config: TradingViewConfig, definition: StrategyDefinition, symbol: SymbolMapping, run_id: str, params_json: str | None = None) -> Path:
    data, quality = _load_quality(symbol.local_path)
    frame = build_local_frame(data, definition, _params(params_json))
    bt_config = BacktestConfig(
        initial_cash=config.backtest.initial_cash,
        monthly_contribution=config.backtest.monthly_contribution,
        monthly_contribution_day_offset=config.backtest.monthly_contribution_day_offset,
        fee_bps=definition.execution.fee_bps or config.backtest.fee_bps,
        slippage_bps=definition.execution.slippage_bps or config.backtest.slippage_bps,
        share_precision=definition.execution.share_precision,
    )
    result = run_strategy_backtest(frame, 0, len(frame) - 1, bt_config)
    monthly = run_dca_benchmark(data, 0, len(data) - 1, bt_config)
    buy_hold = run_buy_hold_benchmark(data, 0, len(data) - 1)
    benchmark_metrics: dict[str, Any] = {
        "monthly_dca": monthly.metrics,
        "buy_hold": {"total_return": float(buy_hold.iloc[-1] / buy_hold.iloc[0] - 1.0)},
    }
    weekly = None
    if symbol.asset_class == "crypto":
        weekly = run_weekly_dca_benchmark(data, 0, len(data) - 1, bt_config)
        benchmark_metrics["weekly_dca"] = weekly.metrics
    run_dir = _run_dir(config, run_id)
    canonical_events(frame, symbol.logical_symbol, timezone=symbol.timezone).to_csv(run_dir / "signals.csv", index=False)
    equity_payload = {
        "date": data["date"].iloc[: len(result.equity)].values,
        "equity": result.equity.values,
        "flows": result.flows.values,
        "monthly_dca_equity": monthly.equity.values,
        "buy_hold_equity": (buy_hold / buy_hold.iloc[0] * bt_config.initial_cash).values,
    }
    if weekly is not None:
        equity_payload["weekly_dca_equity"] = weekly.equity.values
    pd.DataFrame(equity_payload).to_csv(run_dir / "equity_curve.csv", index=False)
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    write_json(run_dir / "metrics.json", result.metrics)
    write_json(run_dir / "benchmarks.json", benchmark_metrics)
    write_json(run_dir / "data_quality.json", quality)
    write_json(run_dir / "manifest.json", _manifest(definition, symbol, run_id, symbol.local_path, quality, "backtest"))
    return run_dir


def run_parity(config: TradingViewConfig, definition: StrategyDefinition, symbol: SymbolMapping, pine_events_path: str | Path, run_id: str, params_json: str | None = None) -> Path:
    data, quality = _load_quality(symbol.local_path)
    frame = build_local_frame(data, definition, _params(params_json))
    python_events = canonical_events(frame, symbol.logical_symbol, timezone=symbol.timezone)
    pine_events = normalize_events(pine_events_path, timezone=symbol.timezone)
    summary = compare_events(python_events, pine_events, config.parity.date_tolerance_bars, data["date"])
    run_dir = _run_dir(config, run_id)
    python_events.to_csv(run_dir / "python_signals.csv", index=False)
    pine_events.to_csv(run_dir / "pine_events.csv", index=False)
    write_json(run_dir / "parity_summary.json", {**summary, "strategy_id": definition.id, "logical_symbol": symbol.logical_symbol, "data_quality": quality})
    write_json(run_dir / "data_quality.json", quality)
    write_json(run_dir / "manifest.json", _manifest(definition, symbol, run_id, symbol.local_path, quality, "parity"))
    diff = pd.DataFrame(summary["missing_python_events_in_pine"] + summary["extra_pine_events"])
    diff.to_csv(run_dir / "parity_diff.csv", index=False)
    return run_dir


def copy_pine_to_run(config: TradingViewConfig, definition: StrategyDefinition, run_id: str) -> Path:
    run_dir = _run_dir(config, run_id) / "pine"
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / Path(definition.pine_path).name
    shutil.copy2(definition.pine_path, target)
    return target
