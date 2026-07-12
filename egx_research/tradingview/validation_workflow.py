from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.backtest import run_dca_benchmark, run_strategy_backtest
from egx_research.config import AppConfig, BacktestConfig, ValidationConfig
from egx_research.data import load_price_data
from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.models import StrategyDefinition, SymbolMapping
from egx_research.tradingview.research import _load_quality, _manifest, _params
from egx_research.tradingview.signals import build_local_frame
from egx_research.utils import ensure_dir, write_json
from egx_research.validation import build_walk_forward_windows, split_holdout


def _row(metrics: dict[str, float], benchmark: dict[str, float], period: str, cost_multiplier: float = 1.0) -> dict[str, Any]:
    return {
        "period": period,
        "cost_multiplier": cost_multiplier,
        "twr_total_return": metrics["twr_total_return"],
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "excess_return_vs_dca": metrics["twr_total_return"] - benchmark["twr_total_return"],
        "dca_twr_total_return": benchmark["twr_total_return"],
        "closed_trades": metrics.get("closed_trades", 0.0),
    }


def run_validation(
    config: TradingViewConfig,
    definition: StrategyDefinition,
    symbol: SymbolMapping,
    run_id: str,
    params_json: str | None = None,
    cost_multipliers: list[float] | None = None,
) -> Path:
    data, quality = _load_quality(symbol.local_path)
    frame = build_local_frame(data, definition, _params(params_json))
    validation = AppConfig().validation
    if len(data) >= 1500:
        validation = ValidationConfig(primary_train_bars=1095, primary_test_bars=365, primary_step_bars=180, fallback_train_bars=365, fallback_test_bars=180, fallback_step_bars=90)
    research_end, holdout_bars = split_holdout(len(data), validation.holdout_ratio)
    windows, scheme = build_walk_forward_windows(research_end, validation)
    base_backtest = BacktestConfig(
        initial_cash=config.backtest.initial_cash,
        monthly_contribution=config.backtest.monthly_contribution,
        monthly_contribution_day_offset=config.backtest.monthly_contribution_day_offset,
        fee_bps=definition.execution.fee_bps or config.backtest.fee_bps,
        slippage_bps=definition.execution.slippage_bps or config.backtest.slippage_bps,
        share_precision=definition.execution.share_precision,
    )
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        result = run_strategy_backtest(frame, window.test_start, window.test_end, base_backtest)
        benchmark = run_dca_benchmark(data, window.test_start, window.test_end, base_backtest)
        rows.append(_row(result.metrics, benchmark.metrics, f"walk_forward_{index}"))
    holdout_start = research_end
    holdout = run_strategy_backtest(frame, holdout_start, len(data) - 1, base_backtest)
    holdout_benchmark = run_dca_benchmark(data, holdout_start, len(data) - 1, base_backtest)
    rows.append(_row(holdout.metrics, holdout_benchmark.metrics, "holdout"))
    stress_rows: list[dict[str, Any]] = []
    for multiplier in cost_multipliers or [1.0, 2.0, 3.0]:
        stressed = deepcopy(base_backtest)
        stressed.fee_bps *= multiplier
        stressed.slippage_bps *= multiplier
        result = run_strategy_backtest(frame, holdout_start, len(data) - 1, stressed)
        benchmark = run_dca_benchmark(data, holdout_start, len(data) - 1, stressed)
        stress_rows.append(_row(result.metrics, benchmark.metrics, "holdout_cost_stress", multiplier))
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)
    pd.DataFrame(rows).to_csv(run_dir / "validation_windows.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(run_dir / "cost_stress.csv", index=False)
    wf_rows = [row for row in rows if row["period"].startswith("walk_forward")]
    summary = {
        "strategy_id": definition.id,
        "logical_symbol": symbol.logical_symbol,
        "window_scheme": scheme,
        "holdout_bars": holdout_bars,
        "walk_forward_mean_excess_return_vs_dca": float(pd.DataFrame(wf_rows)["excess_return_vs_dca"].mean()) if wf_rows else 0.0,
        "holdout_excess_return_vs_dca": float(rows[-1]["excess_return_vs_dca"]),
        "holdout_max_drawdown": float(rows[-1]["max_drawdown"]),
        "cost_stress": stress_rows,
        "status": "pass" if rows[-1]["excess_return_vs_dca"] >= 0 else "review",
    }
    write_json(run_dir / "validation_summary.json", summary)
    write_json(run_dir / "data_quality.json", quality)
    write_json(run_dir / "manifest.json", {**_manifest(definition, symbol, run_id, symbol.local_path, quality, "validate"), "validation_status": summary["status"]})
    return run_dir
