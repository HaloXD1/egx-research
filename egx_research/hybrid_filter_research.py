from __future__ import annotations

import ast
import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.backtest import run_buy_hold_benchmark, run_dca_benchmark, run_strategy_backtest
from egx_research.config import AppConfig, load_config, save_config
from egx_research.data import load_price_data
from egx_research.optimization import aggregate_segment_summaries, score_segment
from egx_research.stock_rotation import load_stock_panel
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.strategies import build_blackcat_gate_signal, build_strategy_frame, normalize_params
from egx_research.utils import ensure_dir, write_json
from egx_research.validation import Window, build_walk_forward_windows, split_holdout


DEFAULT_PULLBACK_PARAMS: dict[str, Any] = {
    "kama_len": 26,
    "kama_fast": 3,
    "kama_slow": 52,
    "cci_len": 26,
    "buy_threshold": -45.0,
    "trend_buffer_atr": 1.7,
    "atr_len": 8,
}

DEFAULT_OVERLAY_PARAMS: dict[str, Any] = {
    "core_weight": 0.8,
    "kama_len": 35,
    "kama_fast": 5,
    "kama_slow": 51,
    "cci_len": 38,
    "cci_threshold": 20.0,
    "sleeve_ladder": "100_60_20",
    "atr_len": 7,
    "atr_stop": 4.4,
    "atr_trail": 2.6,
}

DEFAULT_GATE_PARAMS: dict[str, dict[str, Any]] = {
    "blackcat_ichimoku": {
        "tenkan_len": 7,
        "kijun_len": 34,
        "senkou_b_len": 72,
        "atr_stop": 3.2,
        "atr_trail": 2.3,
    },
    "blackcat_ravi": {
        "fast_len": 6,
        "slow_len": 44,
        "bias_len": 147,
        "ravi_entry": 1.2,
        "atr_stop": 2.0,
        "atr_trail": 2.6,
    },
    "blackcat_zlema_band": {
        "zlema_len": 43,
        "band_mult": 0.5,
        "trend_ma": 55,
        "atr_stop": 3.5,
        "atr_trail": 3.9,
    },
}


@dataclass
class HybridFilterRun:
    run_id: str
    run_dir: Path


def _load_summary_params(
    run_id: str | None,
    expected_family: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    if run_id is None:
        return dict(defaults)

    summary_path = Path("runs") / run_id / "report_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("top_family") != expected_family:
        raise ValueError(f"Run {run_id} is not a {expected_family} result.")
    return dict(summary["top_params"])


def _load_blackcat_gate_params(run_id: str | None) -> dict[str, dict[str, Any]]:
    params = copy.deepcopy(DEFAULT_GATE_PARAMS)
    candidates: list[Path]
    if run_id is not None:
        candidates = [Path("runs") / run_id / "etf_best_per_family.csv"]
    else:
        candidates = sorted(Path("runs").glob("blackcat*/etf_best_per_family.csv"))
    if not candidates:
        return params

    frame = pd.read_csv(candidates[-1])
    for family in params:
        subset = frame[frame["family"] == family]
        if subset.empty:
            continue
        params[family] = dict(ast.literal_eval(str(subset.iloc[0]["params"])))
    return params


def _benchmark_payload(
    data: pd.DataFrame,
    windows: list[Window],
    holdout_start: int,
    config: AppConfig,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    window_benchmarks: dict[int, dict[str, Any]] = {}
    for index, window in enumerate(windows):
        window_benchmarks[index] = {
            "dca": run_dca_benchmark(data, window.test_start, window.test_end, config.backtest)
        }
    holdout = {
        "dca": run_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest),
        "buy_hold": run_buy_hold_benchmark(data, holdout_start, len(data) - 1),
    }
    return window_benchmarks, holdout


def _score_windows(
    data: pd.DataFrame,
    config: AppConfig,
) -> tuple[list[Window], int, dict[int, dict[str, Any]], dict[str, Any], str]:
    research_end, _ = split_holdout(len(data), config.validation.holdout_ratio)
    holdout_start = research_end
    windows, window_scheme = build_walk_forward_windows(research_end, config.validation)
    window_benchmarks, holdout_benchmark = _benchmark_payload(data, windows, holdout_start, config)
    return windows, holdout_start, window_benchmarks, holdout_benchmark, window_scheme


def _variant_catalog(
    pullback_params: dict[str, Any],
    overlay_params: dict[str, Any],
    gate_params: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = [
        {
            "variant": "pullback_base",
            "base_family": "dca_pullback_only",
            "base_params": normalize_params("dca_pullback_only", pullback_params),
            "gate_family": None,
            "gate_params": None,
        },
        {
            "variant": "overlay_base",
            "base_family": "dca_tactical_overlay",
            "base_params": normalize_params("dca_tactical_overlay", overlay_params),
            "gate_family": None,
            "gate_params": None,
        },
    ]
    for gate_family, params in gate_params.items():
        variants.append(
            {
                "variant": f"pullback_gate_{gate_family.replace('blackcat_', '')}",
                "base_family": "dca_pullback_only",
                "base_params": normalize_params("dca_pullback_only", pullback_params),
                "gate_family": gate_family,
                "gate_params": normalize_params(gate_family, params),
            }
        )
        variants.append(
            {
                "variant": f"overlay_gate_{gate_family.replace('blackcat_', '')}",
                "base_family": "dca_tactical_overlay",
                "base_params": normalize_params("dca_tactical_overlay", overlay_params),
                "gate_family": gate_family,
                "gate_params": normalize_params(gate_family, params),
            }
        )
    return variants


def _build_variant_frame(data: pd.DataFrame, variant: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    frame = build_strategy_frame(data, variant["base_family"], variant["base_params"]).copy()
    gate_family = variant["gate_family"]
    if gate_family is None:
        return frame, 1.0

    gate_signal = build_blackcat_gate_signal(data, gate_family, variant["gate_params"]).astype(float)
    gate_on_ratio = float(gate_signal.mean()) if len(gate_signal) else 0.0

    if variant["base_family"] == "dca_pullback_only":
        frame["deploy_fraction"] = frame["deploy_fraction"].astype(float) * gate_signal
        frame["entry_signal"] = frame["deploy_fraction"] > 0.0
        return frame, gate_on_ratio

    if variant["base_family"] == "dca_tactical_overlay":
        floor = frame["floor_allocation"].astype(float)
        sleeve = frame["target_allocation"].astype(float) - floor
        frame["target_allocation"] = floor + sleeve * gate_signal
        frame["entry_signal"] = frame["target_allocation"] > 0.0
        return frame, gate_on_ratio

    raise ValueError(f"Unsupported base family: {variant['base_family']}")


def _evaluate_frame_windows(
    strategy_frame: pd.DataFrame,
    windows: list[Window],
    config: AppConfig,
    window_benchmarks: dict[int, dict[str, Any]],
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for index, window in enumerate(windows):
        result = run_strategy_backtest(strategy_frame, window.test_start, window.test_end, config.backtest)
        summary = score_segment(result.metrics, window_benchmarks[index]["dca"].metrics, config)
        rows.append(summary)
    return aggregate_segment_summaries(rows)


def _evaluate_frame_holdout(
    strategy_frame: pd.DataFrame,
    holdout_start: int,
    config: AppConfig,
    holdout_benchmark: dict[str, Any],
) -> dict[str, float]:
    result = run_strategy_backtest(strategy_frame, holdout_start, len(strategy_frame) - 1, config.backtest)
    return score_segment(result.metrics, holdout_benchmark["dca"].metrics, config)


def _evaluate_variant_on_data(
    data: pd.DataFrame,
    variant: dict[str, Any],
    config: AppConfig,
    *,
    windows: list[Window],
    holdout_start: int,
    window_benchmarks: dict[int, dict[str, Any]],
    holdout_benchmark: dict[str, Any],
) -> dict[str, Any]:
    frame, gate_on_ratio = _build_variant_frame(data, variant)
    wf_metrics = _evaluate_frame_windows(frame, windows, config, window_benchmarks)
    holdout_metrics = _evaluate_frame_holdout(frame, holdout_start, config, holdout_benchmark)
    return {
        "wf_metrics": wf_metrics,
        "holdout_metrics": holdout_metrics,
        "gate_on_ratio": gate_on_ratio,
    }


def _stock_summary(stock_rows: pd.DataFrame) -> pd.DataFrame:
    if stock_rows.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "symbols_evaluated",
                "stock_pass_rate",
                "stock_median_holdout_excess_return_vs_dca",
                "stock_mean_holdout_excess_return_vs_dca",
                "stock_median_holdout_cagr",
                "stock_median_holdout_max_drawdown",
                "stock_median_gate_on_ratio",
            ]
        )

    return (
        stock_rows.groupby("variant", sort=True)
        .agg(
            symbols_evaluated=("symbol", "nunique"),
            stock_pass_rate=("holdout_pass", "mean"),
            stock_median_holdout_excess_return_vs_dca=("holdout_excess_return_vs_dca", "median"),
            stock_mean_holdout_excess_return_vs_dca=("holdout_excess_return_vs_dca", "mean"),
            stock_median_holdout_cagr=("holdout_cagr", "median"),
            stock_median_holdout_max_drawdown=("holdout_max_drawdown", "median"),
            stock_median_gate_on_ratio=("gate_on_ratio", "median"),
        )
        .reset_index()
    )


def run_hybrid_filter_research(
    config_path: str | Path,
    stock_config_path: str | Path,
    *,
    run_id: str | None = None,
    pullback_run_id: str | None = None,
    overlay_run_id: str | None = None,
    blackcat_run_id: str | None = None,
    min_stock_bars: int | None = None,
) -> HybridFilterRun:
    config = load_config(config_path)
    stock_config = load_stock_rotation_config(stock_config_path)

    pullback_params = _load_summary_params(pullback_run_id, "dca_pullback_only", DEFAULT_PULLBACK_PARAMS)
    overlay_params = _load_summary_params(overlay_run_id, "dca_tactical_overlay", DEFAULT_OVERLAY_PARAMS)
    gate_params = _load_blackcat_gate_params(blackcat_run_id)
    variants = _variant_catalog(pullback_params, overlay_params, gate_params)

    etf_data = load_price_data(config.data.normalized_path)
    windows, holdout_start, window_benchmarks, holdout_benchmark, window_scheme = _score_windows(etf_data, config)
    min_bars = int(min_stock_bars or stock_config.validation.min_history_bars)

    run_id = run_id or datetime.now(UTC).strftime("hybrid-filter-%Y%m%dT%H%M%SZ")
    run_dir = ensure_dir(Path("runs") / run_id)
    save_config(run_dir / "config_snapshot.yaml", config)

    etf_rows: list[dict[str, Any]] = []
    for variant in variants:
        metrics = _evaluate_variant_on_data(
            etf_data,
            variant,
            config,
            windows=windows,
            holdout_start=holdout_start,
            window_benchmarks=window_benchmarks,
            holdout_benchmark=holdout_benchmark,
        )
        etf_rows.append(
            {
                "variant": variant["variant"],
                "base_family": variant["base_family"],
                "gate_family": variant["gate_family"] or "none",
                "window_scheme": window_scheme,
                "gate_on_ratio": metrics["gate_on_ratio"],
                "wf_excess_return_vs_dca": metrics["wf_metrics"]["excess_return_vs_dca"],
                "wf_max_drawdown": metrics["wf_metrics"]["max_drawdown"],
                "holdout_excess_return_vs_dca": metrics["holdout_metrics"]["excess_return_vs_dca"],
                "holdout_cagr": metrics["holdout_metrics"]["cagr"],
                "holdout_sharpe": metrics["holdout_metrics"]["sharpe"],
                "holdout_max_drawdown": metrics["holdout_metrics"]["max_drawdown"],
                "base_params": str(variant["base_params"]),
                "gate_params": str(variant["gate_params"] or {}),
            }
        )

    etf_summary = pd.DataFrame(etf_rows)
    etf_summary.to_csv(run_dir / "etf_variant_summary.csv", index=False)

    stock_panel = load_stock_panel(stock_config)
    stock_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for symbol, group in stock_panel.groupby("symbol", sort=True):
        data = (
            group.sort_values("date")[["date", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index(drop=True)
        )
        bars = len(data)
        if bars < min_bars:
            skipped_rows.append({"symbol": symbol, "bars": bars, "reason": "below_min_stock_bars"})
            continue

        try:
            stock_windows, stock_holdout_start, stock_window_benchmarks, stock_holdout_benchmark, stock_window_scheme = _score_windows(data, config)
        except ValueError as exc:
            skipped_rows.append({"symbol": symbol, "bars": bars, "reason": str(exc)})
            continue

        for variant in variants:
            metrics = _evaluate_variant_on_data(
                data,
                variant,
                config,
                windows=stock_windows,
                holdout_start=stock_holdout_start,
                window_benchmarks=stock_window_benchmarks,
                holdout_benchmark=stock_holdout_benchmark,
            )
            stock_rows.append(
                {
                    "symbol": symbol,
                    "variant": variant["variant"],
                    "base_family": variant["base_family"],
                    "gate_family": variant["gate_family"] or "none",
                    "bars": bars,
                    "window_scheme": stock_window_scheme,
                    "gate_on_ratio": metrics["gate_on_ratio"],
                    "holdout_excess_return_vs_dca": metrics["holdout_metrics"]["excess_return_vs_dca"],
                    "holdout_cagr": metrics["holdout_metrics"]["cagr"],
                    "holdout_sharpe": metrics["holdout_metrics"]["sharpe"],
                    "holdout_max_drawdown": metrics["holdout_metrics"]["max_drawdown"],
                    "wf_excess_return_vs_dca": metrics["wf_metrics"]["excess_return_vs_dca"],
                    "holdout_pass": float(metrics["holdout_metrics"]["excess_return_vs_dca"] > 0.0),
                }
            )

    stock_symbol_scores = pd.DataFrame(stock_rows)
    stock_symbol_scores.to_csv(run_dir / "stock_symbol_scores.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(run_dir / "stock_symbols_skipped.csv", index=False)

    stock_variant_summary = _stock_summary(stock_symbol_scores)
    stock_variant_summary.to_csv(run_dir / "stock_variant_summary.csv", index=False)

    leaderboard = etf_summary.merge(stock_variant_summary, on="variant", how="left")
    leaderboard["symbols_evaluated"] = leaderboard["symbols_evaluated"].fillna(0).astype(int)
    for column in [
        "stock_pass_rate",
        "stock_median_holdout_excess_return_vs_dca",
        "stock_mean_holdout_excess_return_vs_dca",
        "stock_median_holdout_cagr",
        "stock_median_holdout_max_drawdown",
        "stock_median_gate_on_ratio",
    ]:
        leaderboard[column] = leaderboard[column].fillna(0.0)

    leaderboard["overall_score"] = (
        0.40 * leaderboard["holdout_excess_return_vs_dca"]
        + 0.25 * leaderboard["stock_median_holdout_excess_return_vs_dca"]
        + 0.15 * leaderboard["stock_pass_rate"]
        + 0.10 * leaderboard["wf_excess_return_vs_dca"]
        + 0.05 * leaderboard["holdout_cagr"]
        - 0.10 * leaderboard["stock_median_holdout_max_drawdown"]
        - 0.05 * leaderboard["holdout_max_drawdown"]
    )
    leaderboard = leaderboard.sort_values(
        ["overall_score", "stock_pass_rate", "holdout_excess_return_vs_dca"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    top_variant = leaderboard.iloc[0]["variant"] if not leaderboard.empty else None
    top_row = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "etf_symbol": config.data.symbol,
            "stock_symbols_total": int(stock_panel["symbol"].nunique()),
            "stock_symbols_evaluated": int(stock_symbol_scores["symbol"].nunique()) if not stock_symbol_scores.empty else 0,
            "window_scheme": window_scheme,
            "pullback_params": pullback_params,
            "overlay_params": overlay_params,
            "gate_params": gate_params,
            "top_variant": top_variant,
            "top_variant_row": top_row,
            "ranking_formula": "0.40 ETF holdout excess + 0.25 stock median holdout excess + 0.15 stock pass rate + 0.10 ETF WF excess + 0.05 ETF holdout CAGR - 0.10 stock median holdout max DD - 0.05 ETF holdout max DD",
        },
    )

    return HybridFilterRun(run_id=run_id, run_dir=run_dir)
