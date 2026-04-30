from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.config import load_config, save_config
from egx_research.data import load_price_data
from egx_research.hybrid_filter_research import (
    DEFAULT_OVERLAY_PARAMS,
    DEFAULT_PULLBACK_PARAMS,
    _evaluate_frame_holdout,
    _evaluate_frame_windows,
    _load_blackcat_gate_params,
    _load_summary_params,
    _score_windows,
    _stock_summary,
)
from egx_research.stock_rotation import load_stock_panel
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.strategies import build_blackcat_gate_score, build_strategy_frame, normalize_params
from egx_research.utils import ensure_dir, write_json


SOFT_GATE_FAMILIES = ("blackcat_ichimoku", "blackcat_zlema_band")

PULLBACK_MIN_SCALE = 0.35
OVERLAY_MIN_SCALE = 0.25


@dataclass
class SoftFilterRun:
    run_id: str
    run_dir: Path


def _soft_variant_catalog(
    pullback_params: dict[str, Any],
    overlay_params: dict[str, Any],
    gate_params: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    variants = [
        {
            "variant": "pullback_base",
            "base_family": "dca_pullback_only",
            "base_params": normalize_params("dca_pullback_only", pullback_params),
            "score_family": None,
            "score_params": None,
            "score_mix": 1.0,
            "min_scale": 1.0,
        },
        {
            "variant": "overlay_base",
            "base_family": "dca_tactical_overlay",
            "base_params": normalize_params("dca_tactical_overlay", overlay_params),
            "score_family": None,
            "score_params": None,
            "score_mix": 1.0,
            "min_scale": 1.0,
        },
    ]

    for family in SOFT_GATE_FAMILIES:
        variants.append(
            {
                "variant": f"pullback_soft_{family.replace('blackcat_', '')}",
                "base_family": "dca_pullback_only",
                "base_params": normalize_params("dca_pullback_only", pullback_params),
                "score_family": family,
                "score_params": normalize_params(family, gate_params[family]),
                "score_mix": 1.0,
                "min_scale": PULLBACK_MIN_SCALE,
            }
        )
        variants.append(
            {
                "variant": f"overlay_soft_{family.replace('blackcat_', '')}",
                "base_family": "dca_tactical_overlay",
                "base_params": normalize_params("dca_tactical_overlay", overlay_params),
                "score_family": family,
                "score_params": normalize_params(family, gate_params[family]),
                "score_mix": 1.0,
                "min_scale": OVERLAY_MIN_SCALE,
            }
        )

    variants.append(
        {
            "variant": "pullback_soft_combo",
            "base_family": "dca_pullback_only",
            "base_params": normalize_params("dca_pullback_only", pullback_params),
            "score_family": "combo",
            "score_params": {family: normalize_params(family, gate_params[family]) for family in SOFT_GATE_FAMILIES},
            "score_mix": 1.0,
            "min_scale": PULLBACK_MIN_SCALE,
        }
    )
    variants.append(
        {
            "variant": "overlay_soft_combo",
            "base_family": "dca_tactical_overlay",
            "base_params": normalize_params("dca_tactical_overlay", overlay_params),
            "score_family": "combo",
            "score_params": {family: normalize_params(family, gate_params[family]) for family in SOFT_GATE_FAMILIES},
            "score_mix": 1.0,
            "min_scale": OVERLAY_MIN_SCALE,
        }
    )
    return variants


def _score_series(data: pd.DataFrame, variant: dict[str, Any]) -> pd.Series:
    if variant["score_family"] is None:
        return pd.Series(1.0, index=data.index, dtype=float)

    if variant["score_family"] == "combo":
        scores = [
            build_blackcat_gate_score(data, family, params)
            for family, params in variant["score_params"].items()
        ]
        return pd.concat(scores, axis=1).mean(axis=1).clip(0.0, 1.0)

    return build_blackcat_gate_score(data, variant["score_family"], variant["score_params"]).clip(0.0, 1.0)


def _build_soft_variant_frame(data: pd.DataFrame, variant: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    frame = build_strategy_frame(data, variant["base_family"], variant["base_params"]).copy()
    score = _score_series(data, variant).astype(float)
    scale = variant["min_scale"] + (1.0 - variant["min_scale"]) * score
    avg_scale = float(scale.mean()) if len(scale) else 0.0

    if variant["base_family"] == "dca_pullback_only":
        frame["deploy_fraction"] = frame["deploy_fraction"].astype(float) * scale
        frame["entry_signal"] = frame["deploy_fraction"] > 0.0
        return frame, avg_scale

    if variant["base_family"] == "dca_tactical_overlay":
        floor = frame["floor_allocation"].astype(float)
        sleeve = frame["target_allocation"].astype(float) - floor
        frame["target_allocation"] = floor + sleeve * scale
        frame["entry_signal"] = frame["target_allocation"] > 0.0
        return frame, avg_scale

    raise ValueError(f"Unsupported base family: {variant['base_family']}")


def run_soft_filter_research(
    config_path: str | Path,
    stock_config_path: str | Path,
    *,
    run_id: str | None = None,
    pullback_run_id: str | None = None,
    overlay_run_id: str | None = None,
    blackcat_run_id: str | None = None,
    min_stock_bars: int | None = None,
) -> SoftFilterRun:
    config = load_config(config_path)
    stock_config = load_stock_rotation_config(stock_config_path)

    pullback_params = _load_summary_params(pullback_run_id, "dca_pullback_only", DEFAULT_PULLBACK_PARAMS)
    overlay_params = _load_summary_params(overlay_run_id, "dca_tactical_overlay", DEFAULT_OVERLAY_PARAMS)
    gate_params = _load_blackcat_gate_params(blackcat_run_id)
    gate_params = {family: gate_params[family] for family in SOFT_GATE_FAMILIES}
    variants = _soft_variant_catalog(pullback_params, overlay_params, gate_params)

    etf_data = load_price_data(config.data.normalized_path)
    windows, holdout_start, window_benchmarks, holdout_benchmark, window_scheme = _score_windows(etf_data, config)
    min_bars = int(min_stock_bars or stock_config.validation.min_history_bars)

    run_id = run_id or datetime.now(UTC).strftime("soft-filter-%Y%m%dT%H%M%SZ")
    run_dir = ensure_dir(Path("runs") / run_id)
    save_config(run_dir / "config_snapshot.yaml", config)

    etf_rows: list[dict[str, Any]] = []
    for variant in variants:
        frame, avg_scale = _build_soft_variant_frame(etf_data, variant)
        wf_metrics = _evaluate_frame_windows(frame, windows, config, window_benchmarks)
        holdout_metrics = _evaluate_frame_holdout(frame, holdout_start, config, holdout_benchmark)
        etf_rows.append(
            {
                "variant": variant["variant"],
                "base_family": variant["base_family"],
                "score_family": variant["score_family"] or "none",
                "window_scheme": window_scheme,
                "avg_scale": avg_scale,
                "min_scale": variant["min_scale"],
                "wf_excess_return_vs_dca": wf_metrics["excess_return_vs_dca"],
                "wf_max_drawdown": wf_metrics["max_drawdown"],
                "holdout_excess_return_vs_dca": holdout_metrics["excess_return_vs_dca"],
                "holdout_cagr": holdout_metrics["cagr"],
                "holdout_sharpe": holdout_metrics["sharpe"],
                "holdout_max_drawdown": holdout_metrics["max_drawdown"],
                "base_params": str(variant["base_params"]),
                "score_params": str(variant["score_params"] or {}),
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
            frame, avg_scale = _build_soft_variant_frame(data, variant)
            wf_metrics = _evaluate_frame_windows(frame, stock_windows, config, stock_window_benchmarks)
            holdout_metrics = _evaluate_frame_holdout(frame, stock_holdout_start, config, stock_holdout_benchmark)
            stock_rows.append(
                {
                    "symbol": symbol,
                    "variant": variant["variant"],
                    "base_family": variant["base_family"],
                    "score_family": variant["score_family"] or "none",
                    "bars": bars,
                    "window_scheme": stock_window_scheme,
                    "gate_on_ratio": avg_scale,
                    "holdout_excess_return_vs_dca": holdout_metrics["excess_return_vs_dca"],
                    "holdout_cagr": holdout_metrics["cagr"],
                    "holdout_sharpe": holdout_metrics["sharpe"],
                    "holdout_max_drawdown": holdout_metrics["max_drawdown"],
                    "wf_excess_return_vs_dca": wf_metrics["excess_return_vs_dca"],
                    "holdout_pass": float(holdout_metrics["excess_return_vs_dca"] > 0.0),
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
            "pullback_min_scale": PULLBACK_MIN_SCALE,
            "overlay_min_scale": OVERLAY_MIN_SCALE,
            "top_variant": top_variant,
            "top_variant_row": top_row,
            "ranking_formula": "0.40 ETF holdout excess + 0.25 stock median holdout excess + 0.15 stock pass rate + 0.10 ETF WF excess + 0.05 ETF holdout CAGR - 0.10 stock median holdout max DD - 0.05 ETF holdout max DD",
        },
    )

    return SoftFilterRun(run_id=run_id, run_dir=run_dir)
