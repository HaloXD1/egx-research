from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler

from egx_research.backtest import run_buy_hold_benchmark, run_dca_benchmark, run_strategy_backtest
from egx_research.config import AppConfig, save_config
from egx_research.data import load_price_data
from egx_research.strategies import PARAMETER_SPACES, build_neighbors, build_strategy_frame, normalize_params, sample_params
from egx_research.utils import ensure_dir, to_native, write_json
from egx_research.validation import Window, build_walk_forward_windows, split_holdout


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def score_segment(strategy_metrics: dict[str, float], dca_metrics: dict[str, float], config: AppConfig) -> dict[str, float]:
    excess_return = float(strategy_metrics["twr_total_return"] - dca_metrics["twr_total_return"])
    score = (
        config.ranking.weights.return_dd * _clip(strategy_metrics["return_dd"], -5.0, 10.0)
        + config.ranking.weights.cagr * _clip(strategy_metrics["cagr"], -1.0, 1.5)
        + config.ranking.weights.profit_factor * _clip(strategy_metrics["profit_factor"], 0.0, 5.0)
        + config.ranking.weights.sharpe * _clip(strategy_metrics["sharpe"], -3.0, 5.0)
        + config.ranking.weights.excess_return_vs_dca * _clip(excess_return, -1.0, 1.5)
    )
    summary = {
        "score": float(score),
        "cagr": float(strategy_metrics["cagr"]),
        "sharpe": float(strategy_metrics["sharpe"]),
        "profit_factor": float(strategy_metrics["profit_factor"]),
        "max_drawdown": float(strategy_metrics["max_drawdown"]),
        "return_dd": float(strategy_metrics["return_dd"]),
        "closed_trades": float(strategy_metrics["closed_trades"]),
        "twr_total_return": float(strategy_metrics["twr_total_return"]),
        "excess_return_vs_dca": excess_return,
        "dca_twr_total_return": float(dca_metrics["twr_total_return"]),
        "final_equity": float(strategy_metrics["final_equity"]),
    }
    return summary


def aggregate_segment_summaries(summaries: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(summaries)
    aggregated = {
        "score": float(frame["score"].mean()),
        "cagr": float(frame["cagr"].mean()),
        "sharpe": float(frame["sharpe"].mean()),
        "profit_factor": float(frame["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(5.0).mean()),
        "max_drawdown": float(frame["max_drawdown"].max()),
        "return_dd": float(frame["return_dd"].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()),
        "closed_trades": float(frame["closed_trades"].sum()),
        "twr_total_return": float(frame["twr_total_return"].mean()),
        "excess_return_vs_dca": float(frame["excess_return_vs_dca"].mean()),
        "final_equity": float(frame["final_equity"].mean()),
    }
    return aggregated


def evaluate_windows(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    windows: list[Window],
    config: AppConfig,
    benchmarks: dict[int, dict[str, Any]],
) -> tuple[dict[str, float], pd.DataFrame]:
    strategy_frame = build_strategy_frame(data, family, params)
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        result = run_strategy_backtest(strategy_frame, window.test_start, window.test_end, config.backtest)
        dca_result = benchmarks[index]["dca"]
        summary = score_segment(result.metrics, dca_result.metrics, config)
        summary["window"] = index
        rows.append(summary)
    return aggregate_segment_summaries(rows), pd.DataFrame(rows)


def evaluate_holdout(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    holdout_start: int,
    config: AppConfig,
    benchmark: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    strategy_frame = build_strategy_frame(data, family, params)
    result = run_strategy_backtest(strategy_frame, holdout_start, len(data) - 1, config.backtest)
    summary = score_segment(result.metrics, benchmark["dca"].metrics, config)
    return summary, {
        "strategy": result,
        "dca": benchmark["dca"],
        "buy_hold": benchmark["buy_hold"],
    }


def evaluate_candidate(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    windows: list[Window],
    holdout_start: int,
    config: AppConfig,
    window_benchmarks: dict[int, dict[str, Any]],
    holdout_benchmark: dict[str, Any],
    quartile_threshold: float,
    objective_mode: str = "walk_forward_score",
    rank_value: float | None = None,
) -> dict[str, Any]:
    params = normalize_params(family, params)
    wf_metrics, wf_rows = evaluate_windows(data, family, params, windows, config, window_benchmarks)
    holdout_metrics, _ = evaluate_holdout(data, family, params, holdout_start, config, holdout_benchmark)

    neighbors = build_neighbors(family, params, config.search.robustness_neighbor_steps)
    neighbor_scores = []
    for neighbor in neighbors:
        if objective_mode == "holdout_excess_vs_dca":
            neighbor_metrics, _ = evaluate_holdout(data, family, neighbor, holdout_start, config, holdout_benchmark)
            neighbor_scores.append(float(neighbor_metrics["excess_return_vs_dca"]))
        else:
            neighbor_metrics, _ = evaluate_windows(data, family, neighbor, windows, config, window_benchmarks)
            neighbor_scores.append(float(neighbor_metrics["score"]))
    neighbor_pass_rate = (
        1.0
        if not neighbor_scores
        else float(np.mean([score >= quartile_threshold for score in neighbor_scores]))
    )

    filters = config.ranking.filters
    passed_filters = (
        wf_metrics["closed_trades"] >= filters.min_closed_trades
        and wf_metrics["max_drawdown"] <= filters.max_drawdown
        and wf_metrics["profit_factor"] >= filters.min_profit_factor
        and holdout_metrics["excess_return_vs_dca"] >= filters.min_holdout_excess_return
        and neighbor_pass_rate >= filters.min_neighbor_pass_rate
    )

    return {
        "family": family,
        "params": params,
        "wf_metrics": wf_metrics,
        "wf_rows": wf_rows.to_dict(orient="records"),
        "holdout_metrics": holdout_metrics,
        "neighbor_pass_rate": neighbor_pass_rate,
        "passed_filters": passed_filters,
        "rank_score": float(rank_value if rank_value is not None else wf_metrics["score"]),
    }


def _trial_to_row(trial: optuna.trial.FrozenTrial, family: str) -> dict[str, Any]:
    row: dict[str, Any] = {"family": family, "number": trial.number, "value": float(trial.value if trial.value is not None else 0.0)}
    for key, value in trial.params.items():
        row[f"param_{key}"] = value
    row.update({key: value for key, value in trial.user_attrs.items()})
    return row


def _benchmark_payload(data: pd.DataFrame, windows: list[Window], holdout_start: int, config: AppConfig) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    window_benchmarks: dict[int, dict[str, Any]] = {}
    for index, window in enumerate(windows):
        window_benchmarks[index] = {"dca": run_dca_benchmark(data, window.test_start, window.test_end, config.backtest)}
    holdout = {
        "dca": run_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest),
        "buy_hold": run_buy_hold_benchmark(data, holdout_start, len(data) - 1),
    }
    return window_benchmarks, holdout


def _family_importance(study: optuna.Study, family: str) -> pd.DataFrame:
    importance = get_param_importances(study)
    if not importance:
        return pd.DataFrame(columns=["family", "parameter", "importance"])
    return pd.DataFrame(
        [{"family": family, "parameter": key, "importance": float(value)} for key, value in importance.items()]
    )


def _top_family_candidates(candidates: list[dict[str, Any]], family: str, top_n: int) -> list[dict[str, Any]]:
    family_candidates = [candidate for candidate in candidates if candidate["family"] == family]
    family_candidates.sort(
        key=lambda item: (item["passed_filters"], item["rank_score"], item["holdout_metrics"]["excess_return_vs_dca"]),
        reverse=True,
    )
    return family_candidates[:top_n]


def optimize_run(
    config: AppConfig,
    config_path: str | Path,
    trials_override: int | None = None,
    family_override: str | None = None,
    run_id: str | None = None,
    objective_mode_override: str | None = None,
) -> str:
    normalized_path = Path(config.data.normalized_path)
    if not normalized_path.exists():
        raise FileNotFoundError(f"Normalized data missing: {normalized_path}")

    data = load_price_data(normalized_path)
    research_end, holdout_bars = split_holdout(len(data), config.validation.holdout_ratio)
    holdout_start = research_end
    research_length = research_end
    windows, window_scheme = build_walk_forward_windows(research_length, config.validation)
    window_benchmarks, holdout_benchmark = _benchmark_payload(data, windows, holdout_start, config)

    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_dir(Path("runs") / run_id)
    save_config(run_dir / "config_snapshot.yaml", config)
    objective_mode = objective_mode_override or config.search.objective_mode

    families = [family_override] if family_override else list(config.search.families)
    all_candidates: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    windows_frame = pd.DataFrame(
        [
            {
                **asdict(window),
                "train_start_date": str(data["date"].iloc[window.train_start].date()),
                "train_end_date": str(data["date"].iloc[window.train_end].date()),
                "test_start_date": str(data["date"].iloc[window.test_start].date()),
                "test_end_date": str(data["date"].iloc[window.test_end].date()),
            }
            for window in windows
        ]
    )
    windows_frame.to_csv(run_dir / "walk_forward_windows.csv", index=False)

    for family_index, family in enumerate(families):
        sampler = TPESampler(seed=config.search.random_seed + family_index)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            params = sample_params(trial, family)
            wf_metrics, _ = evaluate_windows(data, family, params, windows, config, window_benchmarks)
            holdout_metrics, _ = evaluate_holdout(data, family, params, holdout_start, config, holdout_benchmark)
            for key, value in wf_metrics.items():
                trial.set_user_attr(f"wf_{key}", float(value))
            for key, value in holdout_metrics.items():
                trial.set_user_attr(f"holdout_{key}", float(value))
            if objective_mode == "holdout_excess_vs_dca":
                return float(holdout_metrics["excess_return_vs_dca"])
            return float(wf_metrics["score"])

        study.optimize(objective, n_trials=trials_override or config.search.trials_per_family, show_progress_bar=False)

        trials_frame = pd.DataFrame([_trial_to_row(trial, family) for trial in study.trials])
        trials_frame.to_csv(run_dir / f"trials_{family}.csv", index=False)

        importance_frame = _family_importance(study, family)
        importance_frames.append(importance_frame)
        if not importance_frame.empty:
            importance_frame.to_csv(run_dir / f"importance_{family}.csv", index=False)

        quartile_threshold = float(trials_frame["value"].quantile(0.75)) if not trials_frame.empty else 0.0
        top_rows = trials_frame.sort_values("value", ascending=False).head(config.search.top_candidates_per_family)
        for _, row in top_rows.iterrows():
            params = {
                key.replace("param_", ""): row[key]
                for key in row.index
                if key.startswith("param_")
            }
            candidate = evaluate_candidate(
                data=data,
                family=family,
                params=params,
                windows=windows,
                holdout_start=holdout_start,
                config=config,
                window_benchmarks=window_benchmarks,
                holdout_benchmark=holdout_benchmark,
                quartile_threshold=quartile_threshold,
                objective_mode=objective_mode,
                rank_value=float(row["value"]),
            )
            all_candidates.append(candidate)

    importance_all = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    if not importance_all.empty:
        importance_all.to_csv(run_dir / "parameter_importance.csv", index=False)

    all_candidates.sort(
        key=lambda item: (item["passed_filters"], item["rank_score"], item["holdout_metrics"]["excess_return_vs_dca"]),
        reverse=True,
    )
    top10 = all_candidates[:10]
    top3_per_family = []
    for family in families:
        top3_per_family.extend(_top_family_candidates(all_candidates, family, 3))

    top10_frame = pd.DataFrame(
        [
            {
                "family": candidate["family"],
                "rank_score": candidate["rank_score"],
                "passed_filters": candidate["passed_filters"],
                "neighbor_pass_rate": candidate["neighbor_pass_rate"],
                "wf_cagr": candidate["wf_metrics"]["cagr"],
                "wf_profit_factor": candidate["wf_metrics"]["profit_factor"],
                "wf_max_drawdown": candidate["wf_metrics"]["max_drawdown"],
                "wf_excess_return_vs_dca": candidate["wf_metrics"]["excess_return_vs_dca"],
                "holdout_cagr": candidate["holdout_metrics"]["cagr"],
                "holdout_excess_return_vs_dca": candidate["holdout_metrics"]["excess_return_vs_dca"],
                "params": str(candidate["params"]),
            }
            for candidate in top10
        ]
    )
    top3_frame = pd.DataFrame(
        [
            {
                "family": candidate["family"],
                "rank_score": candidate["rank_score"],
                "passed_filters": candidate["passed_filters"],
                "neighbor_pass_rate": candidate["neighbor_pass_rate"],
                "wf_cagr": candidate["wf_metrics"]["cagr"],
                "holdout_excess_return_vs_dca": candidate["holdout_metrics"]["excess_return_vs_dca"],
                "params": str(candidate["params"]),
            }
            for candidate in top3_per_family
        ]
    )
    top10_frame.to_csv(run_dir / "top10_overall.csv", index=False)
    top3_frame.to_csv(run_dir / "top3_per_family.csv", index=False)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "normalized_path": str(normalized_path),
        "window_scheme": window_scheme,
        "holdout_bars": holdout_bars,
        "families": families,
        "trials_per_family": int(trials_override or config.search.trials_per_family),
        "objective_mode": objective_mode,
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "candidates.json", {"candidates": to_native(all_candidates)})
    return run_id
