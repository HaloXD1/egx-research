from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.samplers import TPESampler

from egx_research.backtest import (
    BacktestResult,
    _build_trade_frame,
    _compute_metrics,
    _round_shares,
    run_buy_hold_benchmark,
    run_dca_benchmark,
    run_strategy_backtest,
)
from egx_research.config import BacktestConfig
from egx_research.crypto_config import CryptoConfig, save_crypto_config
from egx_research.crypto_data import build_crypto_feature_panel
from egx_research.crypto_strategies import (
    CRYPTO_PARAMETER_SPACES,
    build_crypto_neighbors,
    build_crypto_strategy_frame,
    normalize_crypto_params,
    sample_crypto_params,
)
from egx_research.optimization import aggregate_segment_summaries, score_segment
from egx_research.reproducibility import build_run_provenance
from egx_research.utils import ensure_dir, to_native, write_json
from egx_research.validation import Window, build_walk_forward_windows, split_holdout


def load_crypto_feature_data(config: CryptoConfig) -> pd.DataFrame:
    path = Path(config.data.features_path)
    if not path.exists():
        return build_crypto_feature_panel(config)
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def _periodic_contribution_schedule(
    dates: pd.Series,
    start_idx: int,
    end_idx: int,
    initial_cash: float,
    contribution: float,
    frequency: str,
) -> pd.Series:
    schedule = pd.Series(0.0, index=range(len(dates)), dtype=float)
    schedule.iloc[start_idx] += float(initial_cash)
    seen: set[tuple[int, int] | tuple[int, int, int]] = set()
    for i in range(start_idx, end_idx + 1):
        date = pd.Timestamp(dates.iloc[i])
        if frequency == "W":
            iso = date.isocalendar()
            key: tuple[int, int] | tuple[int, int, int] = (int(iso.year), int(iso.week))
        else:
            key = (date.year, date.month, 0)
        if key in seen:
            continue
        seen.add(key)
        schedule.iloc[i] += float(contribution)
    return schedule


def run_weekly_dca_benchmark(data: pd.DataFrame, start_idx: int, end_idx: int, config: BacktestConfig) -> BacktestResult:
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000
    schedule = _periodic_contribution_schedule(
        data["date"],
        start_idx,
        end_idx,
        initial_cash=config.initial_cash,
        contribution=config.monthly_contribution / 4.0,
        frequency="W",
    )
    cash = 0.0
    shares = 0.0
    equity = pd.Series(np.nan, index=data.index, dtype=float)
    flows = pd.Series(0.0, index=data.index, dtype=float)
    purchases: list[dict[str, Any]] = []

    for i in range(start_idx, end_idx + 1):
        contribution = float(schedule.iloc[i])
        flows.iloc[i] += contribution
        cash += contribution
        if contribution > 0:
            fill_price = float(data["open"].iloc[i]) * (1.0 + slippage_rate)
            buyable = _round_shares(cash / (fill_price * (1.0 + fee_rate)), config.share_precision)
            if buyable > 0:
                gross = buyable * fill_price
                fee = gross * fee_rate
                cash -= gross + fee
                shares += buyable
                purchases.append(
                    {
                        "entry_date": data["date"].iloc[i],
                        "exit_date": data["date"].iloc[i],
                        "entry_price": fill_price,
                        "exit_price": fill_price,
                        "shares": buyable,
                        "bars_held": 0,
                        "pnl": 0.0,
                        "pnl_pct": 0.0,
                    }
                )
        equity.iloc[i] = cash + shares * float(data["close"].iloc[i])

    slice_equity = equity.iloc[start_idx : end_idx + 1].copy()
    slice_flows = flows.iloc[start_idx : end_idx + 1].copy()
    trades = _build_trade_frame(purchases)
    return BacktestResult(
        equity=slice_equity,
        flows=slice_flows,
        trades=trades,
        metrics=_compute_metrics(
            slice_equity,
            slice_flows,
            trades.iloc[0:0].copy(),
            config.annualization_periods,
        ),
    )


def _benchmark_payload(
    data: pd.DataFrame, windows: list[Window], holdout_start: int, config: CryptoConfig
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    window_benchmarks = {}
    for index, window in enumerate(windows):
        window_benchmarks[index] = {
            "monthly_dca": run_dca_benchmark(data, window.test_start, window.test_end, config.backtest),
            "weekly_dca": run_weekly_dca_benchmark(data, window.test_start, window.test_end, config.backtest),
        }
    holdout = {
        "monthly_dca": run_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest),
        "weekly_dca": run_weekly_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest),
        "buy_hold": run_buy_hold_benchmark(data, holdout_start, len(data) - 1),
    }
    return window_benchmarks, holdout


def evaluate_crypto_windows(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    windows: list[Window],
    config: CryptoConfig,
    benchmarks: dict[int, dict[str, Any]],
) -> tuple[dict[str, float], pd.DataFrame]:
    frame = build_crypto_strategy_frame(data, family, params)
    rows = []
    for index, window in enumerate(windows):
        result = run_strategy_backtest(frame, window.test_start, window.test_end, config.backtest)
        summary = score_segment(result.metrics, benchmarks[index]["monthly_dca"].metrics, config)
        summary["weekly_dca_twr_total_return"] = float(benchmarks[index]["weekly_dca"].metrics["twr_total_return"])
        summary["excess_return_vs_weekly_dca"] = float(
            result.metrics["twr_total_return"] - benchmarks[index]["weekly_dca"].metrics["twr_total_return"]
        )
        summary["window"] = index
        rows.append(summary)
    return aggregate_segment_summaries(rows), pd.DataFrame(rows)


def evaluate_crypto_holdout(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    holdout_start: int,
    config: CryptoConfig,
    benchmark: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    frame = build_crypto_strategy_frame(data, family, params)
    result = run_strategy_backtest(frame, holdout_start, len(data) - 1, config.backtest)
    summary = score_segment(result.metrics, benchmark["monthly_dca"].metrics, config)
    summary["weekly_dca_twr_total_return"] = float(benchmark["weekly_dca"].metrics["twr_total_return"])
    summary["excess_return_vs_weekly_dca"] = float(
        result.metrics["twr_total_return"] - benchmark["weekly_dca"].metrics["twr_total_return"]
    )
    return summary, {"strategy": result, **benchmark}


def evaluate_crypto_candidate(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    windows: list[Window],
    holdout_start: int,
    config: CryptoConfig,
    window_benchmarks: dict[int, dict[str, Any]],
    holdout_benchmark: dict[str, Any],
    quartile_threshold: float,
    objective_mode: str,
    rank_value: float | None = None,
) -> dict[str, Any]:
    params = normalize_crypto_params(family, params)
    wf_metrics, wf_rows = evaluate_crypto_windows(data, family, params, windows, config, window_benchmarks)
    holdout_metrics, _ = evaluate_crypto_holdout(data, family, params, holdout_start, config, holdout_benchmark)

    neighbor_scores = []
    for neighbor in build_crypto_neighbors(family, params, config.search.robustness_neighbor_steps):
        if objective_mode == "holdout_excess_vs_dca":
            neighbor_metrics, _ = evaluate_crypto_holdout(data, family, neighbor, holdout_start, config, holdout_benchmark)
            neighbor_scores.append(float(neighbor_metrics["excess_return_vs_dca"]))
        else:
            neighbor_metrics, _ = evaluate_crypto_windows(data, family, neighbor, windows, config, window_benchmarks)
            neighbor_scores.append(float(neighbor_metrics["score"]))
    neighbor_pass_rate = 1.0 if not neighbor_scores else float(np.mean([score >= quartile_threshold for score in neighbor_scores]))

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
    row: dict[str, Any] = {"family": family, "number": trial.number, "value": float(trial.value or 0.0)}
    for key, value in trial.params.items():
        row[f"param_{key}"] = value
    row.update({key: value for key, value in trial.user_attrs.items()})
    return row


def _family_importance(study: optuna.Study, family: str) -> pd.DataFrame:
    try:
        importance = get_param_importances(study)
    except Exception:
        importance = {}
    return pd.DataFrame(
        [{"family": family, "parameter": key, "importance": float(value)} for key, value in importance.items()]
    )


def _candidate_selection_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    holdout = candidate["holdout_metrics"]
    wf_metrics = candidate["wf_metrics"]
    holdout_excess = float(holdout["excess_return_vs_dca"])
    return (
        bool(candidate["passed_filters"]),
        holdout_excess > 0.0,
        float(holdout["score"]),
        holdout_excess,
        float(holdout["sharpe"]),
        -float(holdout["max_drawdown"]),
        float(candidate["neighbor_pass_rate"]),
        float(wf_metrics["score"]),
    )


def select_crypto_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No crypto candidates were produced.")
    return max(candidates, key=_candidate_selection_key)


def _top_family_candidates(candidates: list[dict[str, Any]], family: str, top_n: int) -> list[dict[str, Any]]:
    family_candidates = [candidate for candidate in candidates if candidate["family"] == family]
    family_candidates.sort(key=_candidate_selection_key, reverse=True)
    return family_candidates[:top_n]


def _candidate_summary_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": candidate["family"],
                "rank_score": candidate["rank_score"],
                "holdout_score": candidate["holdout_metrics"]["score"],
                "passed_filters": candidate["passed_filters"],
                "neighbor_pass_rate": candidate["neighbor_pass_rate"],
                "wf_cagr": candidate["wf_metrics"]["cagr"],
                "wf_sharpe": candidate["wf_metrics"]["sharpe"],
                "wf_max_drawdown": candidate["wf_metrics"]["max_drawdown"],
                "wf_excess_return_vs_dca": candidate["wf_metrics"]["excess_return_vs_dca"],
                "holdout_cagr": candidate["holdout_metrics"]["cagr"],
                "holdout_sharpe": candidate["holdout_metrics"]["sharpe"],
                "holdout_max_drawdown": candidate["holdout_metrics"]["max_drawdown"],
                "holdout_excess_return_vs_dca": candidate["holdout_metrics"]["excess_return_vs_dca"],
                "holdout_excess_return_vs_weekly_dca": candidate["holdout_metrics"]["excess_return_vs_weekly_dca"],
                "params": str(candidate["params"]),
            }
            for candidate in candidates
        ]
    )


def _date_index(data: pd.DataFrame, start_date: str, end_date: str) -> tuple[int, int] | None:
    dates = pd.to_datetime(data["date"])
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    if not mask.any():
        return None
    indices = np.flatnonzero(mask.to_numpy())
    return int(indices[0]), int(indices[-1])


def _write_holdout_equity(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    holdout_start: int,
    config: CryptoConfig,
    run_dir: Path,
) -> None:
    frame = build_crypto_strategy_frame(data, family, params)
    strategy = run_strategy_backtest(frame, holdout_start, len(data) - 1, config.backtest)
    monthly = run_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest)
    weekly = run_weekly_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest)
    buy_hold = run_buy_hold_benchmark(data, holdout_start, len(data) - 1) * float(strategy.flows.iloc[0])
    equity = pd.DataFrame(
        {
            "date": data["date"].iloc[holdout_start:].values,
            "strategy_equity": strategy.equity.values,
            "monthly_dca_equity": monthly.equity.values,
            "weekly_dca_equity": weekly.equity.values,
            "buy_hold_equity": buy_hold.values,
            "target_allocation": frame["target_allocation"].iloc[holdout_start:].values,
        }
    )
    equity.to_csv(run_dir / "crypto_equity_curve_holdout.csv", index=False)


def _write_ablation_summary(
    data: pd.DataFrame,
    candidate: dict[str, Any],
    holdout_start: int,
    config: CryptoConfig,
    holdout_benchmark: dict[str, Any],
    run_dir: Path,
) -> None:
    rows = []
    variants = {"full": candidate["params"]}
    if candidate["family"] == "crypto_ensemble_overlay":
        base = candidate["params"]
        variants = {
            "full": base,
            "price_only": {**base, "price_weight": 1.0, "onchain_weight": 0.0, "sentiment_weight": 0.0, "macro_weight": 0.0},
            "no_onchain": {**base, "onchain_weight": 0.0},
            "no_sentiment": {**base, "sentiment_weight": 0.0},
            "no_macro": {**base, "macro_weight": 0.0},
        }
    for name, params in variants.items():
        metrics, _ = evaluate_crypto_holdout(
            data,
            candidate["family"],
            normalize_crypto_params(candidate["family"], params),
            holdout_start,
            config,
            holdout_benchmark,
        )
        rows.append({"variant": name, **metrics})
    pd.DataFrame(rows).to_csv(run_dir / "crypto_ablation_summary.csv", index=False)


def _write_cost_stress(
    data: pd.DataFrame,
    candidate: dict[str, Any],
    holdout_start: int,
    config: CryptoConfig,
    run_dir: Path,
) -> None:
    rows = []
    for multiplier in (1.0, 2.0, 3.0):
        stressed = deepcopy(config)
        stressed.backtest.fee_bps = config.backtest.fee_bps * multiplier
        stressed.backtest.slippage_bps = config.backtest.slippage_bps * multiplier
        frame = build_crypto_strategy_frame(data, candidate["family"], candidate["params"])
        strategy = run_strategy_backtest(frame, holdout_start, len(data) - 1, stressed.backtest)
        monthly = run_dca_benchmark(data, holdout_start, len(data) - 1, stressed.backtest)
        rows.append(
            {
                "cost_multiplier": multiplier,
                "strategy_cagr": strategy.metrics["cagr"],
                "strategy_max_drawdown": strategy.metrics["max_drawdown"],
                "strategy_twr_total_return": strategy.metrics["twr_total_return"],
                "monthly_dca_twr_total_return": monthly.metrics["twr_total_return"],
                "excess_return_vs_dca": strategy.metrics["twr_total_return"] - monthly.metrics["twr_total_return"],
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "crypto_cost_stress.csv", index=False)


def _write_regime_summary(
    data: pd.DataFrame,
    candidate: dict[str, Any],
    config: CryptoConfig,
    run_dir: Path,
) -> None:
    latest_end = pd.Timestamp(data["date"].max())
    latest_start = latest_end - pd.Timedelta(days=365)
    periods = [
        ("2017_bull", "2017-01-01", "2017-12-31"),
        ("2018_bear", "2018-01-01", "2018-12-31"),
        ("2020_crash", "2020-02-15", "2020-04-30"),
        ("2021_cycle", "2021-01-01", "2021-12-31"),
        ("2022_bear", "2022-01-01", "2022-12-31"),
        ("2024_etf_halving", "2024-01-01", "2024-12-31"),
        ("latest_365d", str(latest_start.date()), str(latest_end.date())),
    ]
    rows = []
    frame = build_crypto_strategy_frame(data, candidate["family"], candidate["params"])
    for label, start, end in periods:
        indices = _date_index(data, start, end)
        if indices is None:
            continue
        start_idx, end_idx = indices
        if end_idx <= start_idx:
            continue
        strategy = run_strategy_backtest(frame, start_idx, end_idx, config.backtest)
        monthly = run_dca_benchmark(data, start_idx, end_idx, config.backtest)
        rows.append(
            {
                "regime": label,
                "start_date": str(pd.Timestamp(data["date"].iloc[start_idx]).date()),
                "end_date": str(pd.Timestamp(data["date"].iloc[end_idx]).date()),
                "strategy_cagr": strategy.metrics["cagr"],
                "strategy_max_drawdown": strategy.metrics["max_drawdown"],
                "strategy_twr_total_return": strategy.metrics["twr_total_return"],
                "monthly_dca_twr_total_return": monthly.metrics["twr_total_return"],
                "excess_return_vs_dca": strategy.metrics["twr_total_return"] - monthly.metrics["twr_total_return"],
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "crypto_regime_summary.csv", index=False)


def _copy_feature_coverage(config: CryptoConfig, run_dir: Path) -> None:
    path = Path(config.data.features_dir) / "BTCUSDT_feature_coverage.csv"
    if path.exists():
        pd.read_csv(path).to_csv(run_dir / "crypto_feature_coverage.csv", index=False)


def run_crypto_research(
    config: CryptoConfig,
    config_path: str | Path,
    trials_override: int | None = None,
    family_override: str | None = None,
    run_id: str | None = None,
    objective_mode_override: str | None = None,
) -> str:
    data = load_crypto_feature_data(config)
    if len(data) < config.validation.fallback_train_bars + config.validation.fallback_test_bars:
        raise ValueError("Not enough BTC feature history for crypto research.")

    research_end, holdout_bars = split_holdout(len(data), config.validation.holdout_ratio)
    holdout_start = research_end
    windows, window_scheme = build_walk_forward_windows(research_end, config.validation)
    window_benchmarks, holdout_benchmark = _benchmark_payload(data, windows, holdout_start, config)

    actual_run_id = run_id or f"crypto-btc-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)
    save_crypto_config(run_dir / "config_snapshot.yaml", config)
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
        if family not in CRYPTO_PARAMETER_SPACES:
            raise ValueError(f"Unknown crypto family: {family}")
        sampler = TPESampler(seed=config.search.random_seed + family_index)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            params = sample_crypto_params(trial, family)
            wf_metrics, _ = evaluate_crypto_windows(data, family, params, windows, config, window_benchmarks)
            holdout_metrics, _ = evaluate_crypto_holdout(data, family, params, holdout_start, config, holdout_benchmark)
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
            params = {key.replace("param_", ""): row[key] for key in row.index if key.startswith("param_")}
            all_candidates.append(
                evaluate_crypto_candidate(
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
            )

    if not all_candidates:
        raise ValueError("No crypto candidates were produced.")
    all_candidates.sort(key=_candidate_selection_key, reverse=True)
    top10 = all_candidates[:10]
    top3_per_family = []
    for family in families:
        top3_per_family.extend(_top_family_candidates(all_candidates, family, 3))

    _candidate_summary_frame(top10).to_csv(run_dir / "top10_overall.csv", index=False)
    _candidate_summary_frame(top3_per_family).to_csv(run_dir / "top3_per_family.csv", index=False)
    importance_all = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    if not importance_all.empty:
        importance_all.to_csv(run_dir / "parameter_importance.csv", index=False)

    top = select_crypto_candidate(all_candidates)
    _write_holdout_equity(data, top["family"], top["params"], holdout_start, config, run_dir)
    _write_ablation_summary(data, top, holdout_start, config, holdout_benchmark, run_dir)
    _write_cost_stress(data, top, holdout_start, config, run_dir)
    _write_regime_summary(data, top, config, run_dir)
    _copy_feature_coverage(config, run_dir)

    write_json(
        run_dir / "manifest.json",
        {
            "run_id": actual_run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "features_path": config.data.features_path,
            "normalized_path": config.data.normalized_path,
            "families": families,
            "holdout_bars": holdout_bars,
            "window_scheme": window_scheme,
            "objective_mode": objective_mode,
            "benchmark_primary": "monthly_dca",
            "benchmark_secondary": ["weekly_dca", "buy_hold"],
            "provenance": build_run_provenance(
                [
                    config_path,
                    config.data.features_path,
                    config.data.normalized_path,
                ]
            ),
        },
    )
    write_json(run_dir / "candidates.json", {"candidates": to_native(all_candidates)})
    write_json(
        run_dir / "crypto_research_summary.json",
        {
            "run_id": actual_run_id,
            "top_family": top["family"],
            "top_params": top["params"],
            "top_rank_score": top["rank_score"],
            "top_holdout_score": top["holdout_metrics"]["score"],
            "holdout_excess_return_vs_dca": top["holdout_metrics"]["excess_return_vs_dca"],
            "holdout_excess_return_vs_weekly_dca": top["holdout_metrics"]["excess_return_vs_weekly_dca"],
        },
    )
    return actual_run_id
