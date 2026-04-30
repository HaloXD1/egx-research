from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from egx_research.backtest import run_buy_hold_benchmark, run_dca_benchmark
from egx_research.config import AppConfig, load_config, save_config
from egx_research.data import load_price_data
from egx_research.optimization import evaluate_candidate, evaluate_holdout, evaluate_windows
from egx_research.stock_rotation import load_stock_panel
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.strategies import normalize_params, sample_params
from egx_research.utils import ensure_dir, write_json
from egx_research.validation import Window, build_walk_forward_windows, split_holdout


BLACKCAT_FAMILY_SET = [
    "blackcat_dynamic_momentum",
    "blackcat_multi_bbands",
    "blackcat_zlema_band",
    "blackcat_ichimoku",
    "blackcat_ravi",
    "blackcat_cci_rsi",
    "blackcat_superj",
]

BLACKCAT_SOURCES: dict[str, dict[str, str]] = {
    "blackcat_dynamic_momentum": {
        "title": "[blackcat] L1 Dynamic Momentum Indicator",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only adaptation of K/D momentum crossover plus EMA spread confirmation.",
    },
    "blackcat_multi_bbands": {
        "title": "[blackcat] L1 Dynamic Multi-Layer Bollinger Bands",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only pullback-reentry inside layered Bollinger bands.",
    },
    "blackcat_zlema_band": {
        "title": "[blackcat] L1 Zero-Lag EMA Band",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only breakout above zero-lag EMA band with trend filter.",
    },
    "blackcat_ichimoku": {
        "title": "[blackcat] L1 Ichimoku Cloud with Entry Signals",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only Tenkan/Kijun bullish cross above cloud.",
    },
    "blackcat_ravi": {
        "title": "[blackcat] L2 Range Action Verification Index (RAVI) with 3 SMA",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only RAVI trend confirmation above long bias SMA.",
    },
    "blackcat_cci_rsi": {
        "title": "[blackcat] L3 CCI-RSI Combo",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only CCI recovery plus RSI confirmation in an uptrend.",
    },
    "blackcat_superj": {
        "title": "[blackcat] L3 SuperJ",
        "url": "https://tr.tradingview.com/scripts/blackcat1402/",
        "adaptation": "Daily long-only smoothed J-line reversal with trigger cross and trend filter.",
    },
}

BLACKCAT_EXCLUDED = [
    {
        "title": "[blackcat] L1 MartinGale Scalping Strategy",
        "reason": "Excluded. Martingale/scalping conflicts with repo daily long-only rule.",
    }
]


@dataclass
class BlackcatResearchRun:
    run_id: str
    run_dir: Path


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


def _trial_to_row(trial: optuna.trial.FrozenTrial, family: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "number": trial.number,
        "value": float(trial.value if trial.value is not None else 0.0),
    }
    for key, value in trial.params.items():
        row[f"param_{key}"] = value
    return row


def _research_config(base_config: AppConfig) -> AppConfig:
    config = copy.deepcopy(base_config)
    config.search.families = list(BLACKCAT_FAMILY_SET)
    return config


def _score_windows(
    data: pd.DataFrame,
    config: AppConfig,
) -> tuple[list[Window], int, dict[int, dict[str, Any]], dict[str, Any], str]:
    research_end, _ = split_holdout(len(data), config.validation.holdout_ratio)
    holdout_start = research_end
    windows, window_scheme = build_walk_forward_windows(research_end, config.validation)
    window_benchmarks, holdout_benchmark = _benchmark_payload(data, windows, holdout_start, config)
    return windows, holdout_start, window_benchmarks, holdout_benchmark, window_scheme


def _fit_etf_family(
    family: str,
    data: pd.DataFrame,
    config: AppConfig,
    trials: int,
    run_dir: Path,
    family_index: int,
) -> dict[str, Any]:
    windows, holdout_start, window_benchmarks, holdout_benchmark, window_scheme = _score_windows(data, config)
    objective_mode = config.search.objective_mode

    sampler = TPESampler(seed=config.search.random_seed + family_index)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_params(trial, family)
        wf_metrics, _ = evaluate_windows(data, family, params, windows, config, window_benchmarks)
        holdout_metrics, _ = evaluate_holdout(data, family, params, holdout_start, config, holdout_benchmark)
        if objective_mode == "holdout_excess_vs_dca":
            return float(holdout_metrics["excess_return_vs_dca"])
        return float(wf_metrics["score"])

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    trials_frame = pd.DataFrame([_trial_to_row(trial, family) for trial in study.trials])
    trials_frame.to_csv(run_dir / f"etf_trials_{family}.csv", index=False)

    quartile_threshold = float(trials_frame["value"].quantile(0.75)) if not trials_frame.empty else 0.0
    top_rows = trials_frame.sort_values("value", ascending=False).head(config.search.top_candidates_per_family)
    candidates: list[dict[str, Any]] = []

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
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item["passed_filters"],
            item["rank_score"],
            item["holdout_metrics"]["excess_return_vs_dca"],
        ),
        reverse=True,
    )
    winner = candidates[0]
    candidate_frame = pd.DataFrame(
        [
            {
                "family": item["family"],
                "rank_score": item["rank_score"],
                "passed_filters": item["passed_filters"],
                "neighbor_pass_rate": item["neighbor_pass_rate"],
                "wf_excess_return_vs_dca": item["wf_metrics"]["excess_return_vs_dca"],
                "wf_max_drawdown": item["wf_metrics"]["max_drawdown"],
                "holdout_excess_return_vs_dca": item["holdout_metrics"]["excess_return_vs_dca"],
                "holdout_max_drawdown": item["holdout_metrics"]["max_drawdown"],
                "params": str(item["params"]),
            }
            for item in candidates
        ]
    )
    candidate_frame.to_csv(run_dir / f"etf_candidates_{family}.csv", index=False)

    return {
        "family": family,
        "title": BLACKCAT_SOURCES[family]["title"],
        "source_url": BLACKCAT_SOURCES[family]["url"],
        "adaptation": BLACKCAT_SOURCES[family]["adaptation"],
        "window_scheme": window_scheme,
        "params": normalize_params(family, winner["params"]),
        "passed_filters": winner["passed_filters"],
        "rank_score": float(winner["rank_score"]),
        "neighbor_pass_rate": float(winner["neighbor_pass_rate"]),
        "wf_metrics": winner["wf_metrics"],
        "holdout_metrics": winner["holdout_metrics"],
    }


def _evaluate_stock_symbols(
    panel: pd.DataFrame,
    family_winners: list[dict[str, Any]],
    config: AppConfig,
    min_stock_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for symbol, group in panel.groupby("symbol", sort=True):
        data = (
            group.sort_values("date")[["date", "open", "high", "low", "close", "volume"]]
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index(drop=True)
        )
        bars = len(data)
        if bars < min_stock_bars:
            skipped.append({"symbol": symbol, "bars": bars, "reason": "below_min_stock_bars"})
            continue

        try:
            windows, holdout_start, window_benchmarks, holdout_benchmark, window_scheme = _score_windows(data, config)
        except ValueError as exc:
            skipped.append({"symbol": symbol, "bars": bars, "reason": str(exc)})
            continue

        for family_result in family_winners:
            family = family_result["family"]
            params = family_result["params"]
            wf_metrics, _ = evaluate_windows(data, family, params, windows, config, window_benchmarks)
            holdout_metrics, _ = evaluate_holdout(data, family, params, holdout_start, config, holdout_benchmark)
            rows.append(
                {
                    "symbol": symbol,
                    "family": family,
                    "bars": bars,
                    "window_scheme": window_scheme,
                    "wf_score": wf_metrics["score"],
                    "wf_excess_return_vs_dca": wf_metrics["excess_return_vs_dca"],
                    "wf_max_drawdown": wf_metrics["max_drawdown"],
                    "holdout_excess_return_vs_dca": holdout_metrics["excess_return_vs_dca"],
                    "holdout_cagr": holdout_metrics["cagr"],
                    "holdout_sharpe": holdout_metrics["sharpe"],
                    "holdout_max_drawdown": holdout_metrics["max_drawdown"],
                    "holdout_final_equity": holdout_metrics["final_equity"],
                    "holdout_pass": float(holdout_metrics["excess_return_vs_dca"] > 0.0),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _family_summary(stock_rows: pd.DataFrame) -> pd.DataFrame:
    if stock_rows.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "symbols_evaluated",
                "stock_pass_rate",
                "stock_median_holdout_excess_return_vs_dca",
                "stock_mean_holdout_excess_return_vs_dca",
                "stock_median_holdout_cagr",
                "stock_median_holdout_max_drawdown",
                "stock_median_wf_excess_return_vs_dca",
            ]
        )

    summary = (
        stock_rows.groupby("family", sort=True)
        .agg(
            symbols_evaluated=("symbol", "nunique"),
            stock_pass_rate=("holdout_pass", "mean"),
            stock_median_holdout_excess_return_vs_dca=("holdout_excess_return_vs_dca", "median"),
            stock_mean_holdout_excess_return_vs_dca=("holdout_excess_return_vs_dca", "mean"),
            stock_median_holdout_cagr=("holdout_cagr", "median"),
            stock_median_holdout_max_drawdown=("holdout_max_drawdown", "median"),
            stock_median_wf_excess_return_vs_dca=("wf_excess_return_vs_dca", "median"),
        )
        .reset_index()
    )
    return summary


def run_blackcat_research(
    config_path: str | Path,
    stock_config_path: str | Path,
    trials_override: int | None = None,
    run_id: str | None = None,
    min_stock_bars: int | None = None,
) -> BlackcatResearchRun:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    base_config = _research_config(load_config(config_path))
    stock_config = load_stock_rotation_config(stock_config_path)

    etf_data = load_price_data(base_config.data.normalized_path)
    trials = int(trials_override or base_config.search.trials_per_family)
    min_bars = int(min_stock_bars or stock_config.validation.min_history_bars)

    run_id = run_id or datetime.now(UTC).strftime("blackcat-%Y%m%dT%H%M%SZ")
    run_dir = ensure_dir(Path("runs") / run_id)
    save_config(run_dir / "config_snapshot.yaml", base_config)

    family_winners: list[dict[str, Any]] = []
    for family_index, family in enumerate(BLACKCAT_FAMILY_SET):
        family_winners.append(
            _fit_etf_family(
                family=family,
                data=etf_data,
                config=base_config,
                trials=trials,
                run_dir=run_dir,
                family_index=family_index,
            )
        )

    etf_summary = pd.DataFrame(
        [
            {
                "family": item["family"],
                "title": item["title"],
                "source_url": item["source_url"],
                "passed_filters": item["passed_filters"],
                "rank_score": item["rank_score"],
                "neighbor_pass_rate": item["neighbor_pass_rate"],
                "wf_excess_return_vs_dca": item["wf_metrics"]["excess_return_vs_dca"],
                "wf_max_drawdown": item["wf_metrics"]["max_drawdown"],
                "holdout_excess_return_vs_dca": item["holdout_metrics"]["excess_return_vs_dca"],
                "holdout_cagr": item["holdout_metrics"]["cagr"],
                "holdout_sharpe": item["holdout_metrics"]["sharpe"],
                "holdout_max_drawdown": item["holdout_metrics"]["max_drawdown"],
                "params": str(item["params"]),
            }
            for item in family_winners
        ]
    )
    etf_summary = etf_summary.sort_values(
        ["passed_filters", "holdout_excess_return_vs_dca", "rank_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    etf_summary.to_csv(run_dir / "etf_best_per_family.csv", index=False)

    stock_panel = load_stock_panel(stock_config)
    stock_rows, skipped_rows = _evaluate_stock_symbols(
        panel=stock_panel,
        family_winners=family_winners,
        config=base_config,
        min_stock_bars=min_bars,
    )
    stock_rows.to_csv(run_dir / "stock_symbol_scores.csv", index=False)
    skipped_rows.to_csv(run_dir / "stock_symbols_skipped.csv", index=False)

    stock_summary = _family_summary(stock_rows)
    stock_summary.to_csv(run_dir / "stock_family_summary.csv", index=False)

    leaderboard = etf_summary.merge(stock_summary, on="family", how="left")
    leaderboard["symbols_evaluated"] = leaderboard["symbols_evaluated"].fillna(0).astype(int)
    for column in [
        "stock_pass_rate",
        "stock_median_holdout_excess_return_vs_dca",
        "stock_mean_holdout_excess_return_vs_dca",
        "stock_median_holdout_cagr",
        "stock_median_holdout_max_drawdown",
        "stock_median_wf_excess_return_vs_dca",
    ]:
        leaderboard[column] = leaderboard[column].fillna(0.0)

    leaderboard["overall_score"] = (
        0.30 * leaderboard["holdout_excess_return_vs_dca"]
        + 0.30 * leaderboard["stock_median_holdout_excess_return_vs_dca"]
        + 0.20 * leaderboard["stock_pass_rate"]
        + 0.10 * leaderboard["neighbor_pass_rate"]
        + 0.10 * leaderboard["wf_excess_return_vs_dca"]
        - 0.10 * leaderboard["stock_median_holdout_max_drawdown"]
        - 0.05 * leaderboard["holdout_max_drawdown"]
    )
    leaderboard = leaderboard.sort_values(
        [
            "overall_score",
            "stock_pass_rate",
            "stock_median_holdout_excess_return_vs_dca",
            "holdout_excess_return_vs_dca",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    top_family = leaderboard.iloc[0]["family"] if not leaderboard.empty else None
    top_params = next((item["params"] for item in family_winners if item["family"] == top_family), {})

    write_json(
        run_dir / "summary.json",
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "trials_per_family": trials,
            "families": BLACKCAT_FAMILY_SET,
            "min_stock_bars": min_bars,
            "etf_symbol": base_config.data.symbol,
            "stock_symbols_total": int(stock_panel["symbol"].nunique()),
            "stock_symbols_evaluated": int(stock_rows["symbol"].nunique()) if not stock_rows.empty else 0,
            "top_family": top_family,
            "top_params": top_params,
            "ranking_formula": "0.30 ETF holdout excess + 0.30 stock median holdout excess + 0.20 stock pass rate + 0.10 ETF neighbor pass + 0.10 ETF WF excess - 0.10 stock median holdout max DD - 0.05 ETF holdout max DD",
            "sources": BLACKCAT_SOURCES,
            "excluded": BLACKCAT_EXCLUDED,
        },
    )

    return BlackcatResearchRun(run_id=run_id, run_dir=run_dir)
