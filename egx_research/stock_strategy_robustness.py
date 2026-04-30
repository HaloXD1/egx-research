from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egx_research.backtest import run_dca_benchmark
from egx_research.data import load_price_data
from egx_research.stock_rotation import (
    first_trading_days,
    load_corporate_actions,
    load_dividend_actions,
    load_membership_snapshots,
    load_stock_fundamentals,
    load_stock_panel,
)
from egx_research.stock_rotation_config import StockRotationConfig, load_stock_rotation_config
from egx_research.stock_strategy_research import (
    StrategySimulation,
    build_strategy_feature_panel,
    load_disclosure_events,
    simulate_event_driven_strategy,
)
from egx_research.stock_strategy_validation import (
    REBOUND_MAX5_V1_PARAMS,
    REBOUND_MAX5_V2_PARAMS,
    REBOUND_MAX5_V3_PARAMS,
    _date_index,
    _period_benchmark_metrics,
    _slice_metrics,
    build_market_regime_filter,
    build_market_regime_filter_v3,
    build_rebound_v3_feature_panel,
)
from egx_research.utils import ensure_dir, write_json


@dataclass
class StockStrategyRobustnessRun:
    run_id: str
    run_dir: Path


@dataclass
class RobustnessData:
    config: StockRotationConfig
    panel: pd.DataFrame
    membership: pd.DataFrame
    disclosure_events: pd.DataFrame
    fundamentals: pd.DataFrame
    dividend_actions: pd.DataFrame
    corporate_actions: pd.DataFrame
    etf: pd.DataFrame
    index: pd.DataFrame
    calendar: pd.Series


@dataclass
class PreparedStrategy:
    name: str
    params: dict[str, Any]
    features: pd.DataFrame | None
    market_filter: pd.Series | None


def _load_robustness_data(config_path: str | Path) -> RobustnessData:
    config = load_stock_rotation_config(config_path)
    config.selection.method = "sector_multifactor"
    config.selection.use_total_return_features = True
    config.selection.require_long_term_trend = True
    config.selection.max_drawdown_252 = min(float(config.selection.max_drawdown_252), 0.60)

    etf = load_price_data(config.benchmark.etf_symbol_path)
    index = load_price_data(config.benchmark.index_symbol_path)
    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    panel = load_stock_panel(config)
    panel = panel[panel["date"] >= pd.Timestamp(calendar.iloc[0])].reset_index(drop=True)
    return RobustnessData(
        config=config,
        panel=panel,
        membership=load_membership_snapshots(config),
        disclosure_events=load_disclosure_events(config),
        fundamentals=load_stock_fundamentals(config),
        dividend_actions=load_dividend_actions(config),
        corporate_actions=load_corporate_actions(config),
        etf=etf,
        index=index,
        calendar=calendar,
    )


def _params_for_config(params: dict[str, Any], config: StockRotationConfig) -> dict[str, Any]:
    selected = dict(params)
    selected["fixed_sell_fee_egp"] = float(config.portfolio.fixed_buy_fee_egp)
    return selected


def _prepare_strategies(data: RobustnessData) -> dict[str, PreparedStrategy]:
    v1_params = _params_for_config(REBOUND_MAX5_V1_PARAMS, data.config)
    v2_params = _params_for_config(REBOUND_MAX5_V2_PARAMS, data.config)
    v3_params = _params_for_config(REBOUND_MAX5_V3_PARAMS, data.config)

    v1_features = build_strategy_feature_panel(
        data.panel,
        family="rebound",
        params=v1_params,
        disclosure_events=data.disclosure_events,
    )
    v2_features = build_strategy_feature_panel(
        data.panel,
        family="rebound",
        params=v2_params,
        disclosure_events=data.disclosure_events,
    )
    v2_filter = build_market_regime_filter(
        data.etf,
        data.calendar,
        ma_len=int(v2_params.get("market_filter_ma_len", 100)),
        slope_len=int(v2_params.get("market_filter_slope_len", 10)),
    )
    v3_regime = build_market_regime_filter_v3(
        etf=data.etf,
        index=data.index,
        panel=data.panel,
        membership=data.membership,
        calendar=data.calendar,
        params=v3_params,
    )
    v3_features, _ = build_rebound_v3_feature_panel(
        panel=data.panel,
        membership=data.membership,
        calendar=data.calendar,
        config=data.config,
        benchmark=data.index,
        params=v3_params,
        disclosure_events=data.disclosure_events,
        fundamentals=data.fundamentals,
        dividend_actions=data.dividend_actions,
        corporate_actions=data.corporate_actions,
    )

    return {
        "rebound_max5_v1": PreparedStrategy(
            "rebound_max5_v1",
            v1_params,
            v1_features,
            None,
        ),
        "rebound_max5_v2": PreparedStrategy(
            "rebound_max5_v2",
            v2_params,
            v2_features,
            v2_filter,
        ),
        "rebound_max5_v3": PreparedStrategy(
            "rebound_max5_v3",
            v3_params,
            v3_features,
            v3_regime.set_index("date")["market_regime_ok"],
        ),
    }


def _simulate_prepared(
    *,
    prepared: PreparedStrategy,
    data: RobustnessData,
    config: StockRotationConfig,
    start_idx: int = 0,
) -> StrategySimulation:
    if prepared.features is None:
        raise ValueError(f"Prepared strategy has no features: {prepared.name}")
    return simulate_event_driven_strategy(
        panel=data.panel,
        features=prepared.features,
        calendar=data.calendar,
        membership=data.membership,
        config=config,
        family="rebound",
        params=_params_for_config(prepared.params, config),
        start_idx=start_idx,
        market_filter=prepared.market_filter,
    )


def _row_from_metrics(
    *,
    strategy: str,
    period: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    metrics: dict[str, float],
    benchmark_metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "strategy": strategy,
        "period": period,
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "twr": float(metrics["twr_total_return"]),
        "cagr": float(metrics["cagr"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "sharpe": float(metrics["sharpe"]),
        "return_dd": float(metrics["return_dd"]),
        "money_multiple": float(metrics["money_multiple"]),
        "closed_trades": float(metrics.get("closed_trades", 0.0)),
        "excess_twr_vs_etf_dca": float(
            metrics["twr_total_return"] - benchmark_metrics["twr_total_return"]
        ),
        "excess_cagr_vs_etf_dca": float(metrics["cagr"] - benchmark_metrics["cagr"]),
    }
    if extra:
        row.update(extra)
    return row


def _sell_count(
    simulation: StrategySimulation,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    if simulation.actions.empty or "date" not in simulation.actions.columns:
        return 0
    dates = pd.to_datetime(simulation.actions["date"], errors="coerce")
    mask = (
        simulation.actions["action"].astype(str).eq("SELL")
        & (dates >= pd.Timestamp(start_date))
        & (dates <= pd.Timestamp(end_date))
    )
    return int(mask.sum())


def _period_metrics(
    *,
    simulation: StrategySimulation,
    data: RobustnessData,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    config: StockRotationConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    start_idx = _date_index(data.calendar, start_date, side="left")
    end_idx = _date_index(data.calendar, end_date, side="right")
    strategy_metrics = _slice_metrics(simulation, data.calendar, start_date, end_date)
    benchmark_metrics = _period_benchmark_metrics(
        etf=data.etf,
        start_idx=start_idx,
        end_idx=end_idx,
        config=config,
    )
    return strategy_metrics, benchmark_metrics


def _clone_config(
    config: StockRotationConfig,
    *,
    cost_multiplier: float = 1.0,
    contribution_day_offset: int = 0,
) -> StockRotationConfig:
    variant = copy.deepcopy(config)
    variant.backtest.fee_bps *= float(cost_multiplier)
    variant.backtest.slippage_bps *= float(cost_multiplier)
    variant.portfolio.fixed_buy_fee_egp *= float(cost_multiplier)
    variant.backtest.monthly_contribution_day_offset = int(contribution_day_offset)
    return variant


def _strategy_comparison(
    *,
    data: RobustnessData,
    prepared: dict[str, PreparedStrategy],
    train_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, StrategySimulation]]:
    simulations: dict[str, StrategySimulation] = {}
    rows: list[dict[str, Any]] = []
    end_date = pd.Timestamp(data.calendar.iloc[-1])
    test_start = pd.Timestamp(data.calendar[data.calendar > train_end].iloc[0])
    full_benchmark = run_dca_benchmark(
        data.etf,
        0,
        len(data.calendar) - 1,
        data.config.backtest,
    ).metrics
    test_benchmark = _period_benchmark_metrics(
        etf=data.etf,
        start_idx=_date_index(data.calendar, test_start, side="left"),
        end_idx=len(data.calendar) - 1,
        config=data.config,
    )

    rows.append(
        _row_from_metrics(
            strategy="etf_dca",
            period="full",
            start_date=pd.Timestamp(data.calendar.iloc[0]),
            end_date=end_date,
            metrics=full_benchmark,
            benchmark_metrics=full_benchmark,
        )
    )
    rows.append(
        _row_from_metrics(
            strategy="etf_dca",
            period="oos",
            start_date=test_start,
            end_date=end_date,
            metrics=test_benchmark,
            benchmark_metrics=test_benchmark,
        )
    )

    for name, strategy in prepared.items():
        simulation = _simulate_prepared(
            prepared=strategy,
            data=data,
            config=data.config,
        )
        simulations[name] = simulation
        rows.append(
            _row_from_metrics(
                strategy=name,
                period="full",
                start_date=pd.Timestamp(data.calendar.iloc[0]),
                end_date=end_date,
                metrics=simulation.metrics,
                benchmark_metrics=full_benchmark,
                extra={
                    "fee_to_contributions_ratio": float(
                        simulation.metrics.get("fee_to_contributions_ratio", 0.0)
                    ),
                    "mean_turnover_pct": float(
                        simulation.metrics.get("mean_turnover_pct", 0.0)
                    ),
                },
            )
        )
        test_metrics = _slice_metrics(simulation, data.calendar, test_start, end_date)
        rows.append(
            _row_from_metrics(
                strategy=name,
                period="oos",
                start_date=test_start,
                end_date=end_date,
                metrics=test_metrics,
                benchmark_metrics=test_benchmark,
                extra={
                    "closed_trades": float(_sell_count(simulation, test_start, end_date)),
                    "fee_to_contributions_ratio": float(
                        simulation.metrics.get("fee_to_contributions_ratio", 0.0)
                    ),
                    "mean_turnover_pct": float(
                        simulation.metrics.get("mean_turnover_pct", 0.0)
                    ),
                },
            )
        )
    return pd.DataFrame(rows), simulations


def _yearly_walk_forward(
    *,
    data: RobustnessData,
    simulations: dict[str, StrategySimulation],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in sorted(data.calendar.dt.year.unique()):
        mask = data.calendar.dt.year == int(year)
        if not bool(mask.any()):
            continue
        start = pd.Timestamp(data.calendar[mask].iloc[0])
        end = pd.Timestamp(data.calendar[mask].iloc[-1])
        benchmark = _period_benchmark_metrics(
            etf=data.etf,
            start_idx=_date_index(data.calendar, start, side="left"),
            end_idx=_date_index(data.calendar, end, side="right"),
            config=data.config,
        )
        rows.append(
            _row_from_metrics(
                strategy="etf_dca",
                period=str(int(year)),
                start_date=start,
                end_date=end,
                metrics=benchmark,
                benchmark_metrics=benchmark,
                extra={"year": int(year)},
            )
        )
        for name, simulation in simulations.items():
            metrics = _slice_metrics(simulation, data.calendar, start, end)
            rows.append(
                _row_from_metrics(
                    strategy=name,
                    period=str(int(year)),
                    start_date=start,
                    end_date=end,
                    metrics=metrics,
                    benchmark_metrics=benchmark,
                    extra={
                        "year": int(year),
                        "closed_trades": float(_sell_count(simulation, start, end)),
                    },
                )
            )
    return pd.DataFrame(rows)


def _cost_stress(
    *,
    data: RobustnessData,
    strategy: PreparedStrategy,
    train_end: pd.Timestamp,
    multipliers: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    end_date = pd.Timestamp(data.calendar.iloc[-1])
    test_start = pd.Timestamp(data.calendar[data.calendar > train_end].iloc[0])
    for multiplier in multipliers:
        config = _clone_config(data.config, cost_multiplier=multiplier)
        simulation = _simulate_prepared(
            prepared=strategy,
            data=data,
            config=config,
        )
        full_benchmark = run_dca_benchmark(
            data.etf,
            0,
            len(data.calendar) - 1,
            config.backtest,
        ).metrics
        test_benchmark = _period_benchmark_metrics(
            etf=data.etf,
            start_idx=_date_index(data.calendar, test_start, side="left"),
            end_idx=len(data.calendar) - 1,
            config=config,
        )
        rows.append(
            _row_from_metrics(
                strategy=strategy.name,
                period="full",
                start_date=pd.Timestamp(data.calendar.iloc[0]),
                end_date=end_date,
                metrics=simulation.metrics,
                benchmark_metrics=full_benchmark,
                extra={
                    "cost_multiplier": float(multiplier),
                    "fee_bps": float(config.backtest.fee_bps),
                    "slippage_bps": float(config.backtest.slippage_bps),
                    "fixed_fee_egp": float(config.portfolio.fixed_buy_fee_egp),
                    "fee_to_contributions_ratio": float(
                        simulation.metrics.get("fee_to_contributions_ratio", 0.0)
                    ),
                },
            )
        )
        test_metrics = _slice_metrics(simulation, data.calendar, test_start, end_date)
        rows.append(
            _row_from_metrics(
                strategy=strategy.name,
                period="oos",
                start_date=test_start,
                end_date=end_date,
                metrics=test_metrics,
                benchmark_metrics=test_benchmark,
                extra={
                    "closed_trades": float(_sell_count(simulation, test_start, end_date)),
                    "cost_multiplier": float(multiplier),
                    "fee_bps": float(config.backtest.fee_bps),
                    "slippage_bps": float(config.backtest.slippage_bps),
                    "fixed_fee_egp": float(config.portfolio.fixed_buy_fee_egp),
                    "fee_to_contributions_ratio": float(
                        simulation.metrics.get("fee_to_contributions_ratio", 0.0)
                    ),
                },
            )
        )
    return pd.DataFrame(rows)


def _sample_start_dates(
    calendar: pd.Series,
    *,
    train_end: pd.Timestamp,
    sample_count: int,
) -> list[pd.Timestamp]:
    monthly = [
        date
        for date in first_trading_days(calendar)
        if pd.Timestamp(date) <= train_end
    ]
    if not monthly:
        return [pd.Timestamp(calendar.iloc[0])]
    rng = np.random.default_rng(42)
    count = min(int(sample_count), len(monthly))
    picked = sorted(rng.choice(np.arange(len(monthly)), size=count, replace=False).tolist())
    dates = [pd.Timestamp(monthly[idx]) for idx in picked]
    start = pd.Timestamp(calendar.iloc[0])
    if start not in dates:
        dates.insert(0, start)
    return dates


def _start_date_sensitivity(
    *,
    data: RobustnessData,
    strategy: PreparedStrategy,
    train_end: pd.Timestamp,
    sample_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    end_date = pd.Timestamp(data.calendar.iloc[-1])
    for start in _sample_start_dates(
        data.calendar,
        train_end=train_end,
        sample_count=sample_count,
    ):
        start_idx = _date_index(data.calendar, start, side="left")
        simulation = _simulate_prepared(
            prepared=strategy,
            data=data,
            config=data.config,
            start_idx=start_idx,
        )
        benchmark = run_dca_benchmark(
            data.etf,
            start_idx,
            len(data.calendar) - 1,
            data.config.backtest,
        ).metrics
        rows.append(
            _row_from_metrics(
                strategy=strategy.name,
                period="start_date_sensitivity",
                start_date=start,
                end_date=end_date,
                metrics=simulation.metrics,
                benchmark_metrics=benchmark,
                extra={"start_idx": int(start_idx)},
            )
        )
    return pd.DataFrame(rows)


def _contribution_day_sensitivity(
    *,
    data: RobustnessData,
    strategy: PreparedStrategy,
    train_end: pd.Timestamp,
    offsets: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    end_date = pd.Timestamp(data.calendar.iloc[-1])
    test_start = pd.Timestamp(data.calendar[data.calendar > train_end].iloc[0])
    for offset in offsets:
        config = _clone_config(data.config, contribution_day_offset=int(offset))
        simulation = _simulate_prepared(
            prepared=strategy,
            data=data,
            config=config,
        )
        benchmark = run_dca_benchmark(
            data.etf,
            0,
            len(data.calendar) - 1,
            config.backtest,
        ).metrics
        rows.append(
            _row_from_metrics(
                strategy=strategy.name,
                period="full",
                start_date=pd.Timestamp(data.calendar.iloc[0]),
                end_date=end_date,
                metrics=simulation.metrics,
                benchmark_metrics=benchmark,
                extra={"monthly_contribution_day_offset": int(offset)},
            )
        )
        test_metrics = _slice_metrics(simulation, data.calendar, test_start, end_date)
        test_benchmark = _period_benchmark_metrics(
            etf=data.etf,
            start_idx=_date_index(data.calendar, test_start, side="left"),
            end_idx=len(data.calendar) - 1,
            config=config,
        )
        rows.append(
            _row_from_metrics(
                strategy=strategy.name,
                period="oos",
                start_date=test_start,
                end_date=end_date,
                metrics=test_metrics,
                benchmark_metrics=test_benchmark,
                extra={
                    "closed_trades": float(_sell_count(simulation, test_start, end_date)),
                    "monthly_contribution_day_offset": int(offset),
                },
            )
        )
    return pd.DataFrame(rows)


def _data_freshness_rows(data: RobustnessData, as_of: pd.Timestamp) -> list[dict[str, Any]]:
    freshness = [
        ("etf", data.etf, "date"),
        ("egx_index", data.index, "date"),
        ("stock_panel", data.panel, "date"),
    ]
    rows: list[dict[str, Any]] = []
    for name, frame, column in freshness:
        latest = pd.Timestamp(frame[column].max()) if not frame.empty else pd.NaT
        days_stale = (
            int((pd.Timestamp(as_of).normalize() - latest.normalize()).days)
            if pd.notna(latest)
            else None
        )
        rows.append(
            {
                "dataset": name,
                "rows": int(len(frame)),
                "start_date": (
                    str(pd.Timestamp(frame[column].min()).date())
                    if not frame.empty
                    else None
                ),
                "end_date": str(latest.date()) if pd.notna(latest) else None,
                "days_stale": days_stale,
                "status": (
                    "missing"
                    if days_stale is None
                    else "future_date"
                    if days_stale < 0
                    else "stale"
                    if days_stale > 7
                    else "fresh"
                ),
            }
        )
    if not data.fundamentals.empty:
        latest_period = pd.Timestamp(data.fundamentals["period_end"].max())
        latest_filing = pd.Timestamp(data.fundamentals["filing_date"].max())
        filing_days_stale = int(
            (pd.Timestamp(as_of).normalize() - latest_filing.normalize()).days
        )
        rows.append(
            {
                "dataset": "fundamentals",
                "rows": int(len(data.fundamentals)),
                "start_date": str(pd.Timestamp(data.fundamentals["period_end"].min()).date()),
                "end_date": str(latest_period.date()),
                "latest_filing_date": str(latest_filing.date()),
                "days_stale": filing_days_stale,
                "status": (
                    "future_filing_date"
                    if filing_days_stale < 0
                    else "future_period_end"
                    if latest_period > pd.Timestamp(as_of)
                    else "stale"
                    if filing_days_stale > 120
                    else "fresh"
                ),
            }
        )
    return rows


def _fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.astype(object).where(pd.notna(frame), "").itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _write_markdown_report(
    *,
    path: Path,
    comparison: pd.DataFrame,
    yearly: pd.DataFrame,
    cost: pd.DataFrame,
    start_sensitivity: pd.DataFrame,
    contribution_sensitivity: pd.DataFrame,
    freshness: pd.DataFrame,
) -> None:
    v3_oos = comparison[
        (comparison["strategy"] == "rebound_max5_v3") & (comparison["period"] == "oos")
    ].iloc[0]
    v2_oos = comparison[
        (comparison["strategy"] == "rebound_max5_v2") & (comparison["period"] == "oos")
    ].iloc[0]
    v3_yearly = yearly[yearly["strategy"] == "rebound_max5_v3"].copy()
    cost_oos = cost[cost["period"] == "oos"].copy()
    start_summary = start_sensitivity[["cagr", "max_drawdown", "sharpe"]].describe()
    contribution_oos = contribution_sensitivity[
        contribution_sensitivity["period"] == "oos"
    ].copy()

    lines = [
        "# Stock Strategy Robustness Pack",
        "",
        "## Headline",
        "",
        f"- v3 OOS TWR: `{_fmt_pct(v3_oos['twr'])}` vs v2 `{_fmt_pct(v2_oos['twr'])}`.",
        f"- v3 OOS CAGR: `{_fmt_pct(v3_oos['cagr'])}`.",
        f"- v3 OOS max DD: `{_fmt_pct(v3_oos['max_drawdown'])}`.",
        f"- v3 OOS Sharpe: `{v3_oos['sharpe']:.2f}`.",
        "",
        "## Stress Notes",
        "",
        f"- Worst v3 yearly excess TWR vs ETF DCA: `{_fmt_pct(v3_yearly['excess_twr_vs_etf_dca'].min())}`.",
        f"- x3 cost OOS CAGR: `{_fmt_pct(cost_oos.loc[cost_oos['cost_multiplier'].idxmax(), 'cagr'])}`.",
        f"- Start-date sensitivity median CAGR: `{_fmt_pct(start_summary.loc['50%', 'cagr'])}`.",
        f"- Worst contribution-day OOS CAGR: `{_fmt_pct(contribution_oos['cagr'].min())}`.",
        "",
        "## Files",
        "",
        "- `strategy_comparison.csv`",
        "- `yearly_walk_forward.csv`",
        "- `cost_stress.csv`",
        "- `start_date_sensitivity.csv`",
        "- `contribution_day_sensitivity.csv`",
        "- `data_freshness.csv`",
        "- `summary.json`",
        "",
        "## Data Freshness",
        "",
        _markdown_table(freshness),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stock_strategy_robustness(
    *,
    config_path: str | Path = Path("config/stock_rotation_multifactor.yaml"),
    run_id: str | None = None,
    train_end: str = "2023-12-31",
    start_samples: int = 12,
) -> StockStrategyRobustnessRun:
    data = _load_robustness_data(config_path)
    train_end_date = pd.Timestamp(train_end)
    actual_run_id = run_id or f"stock-strategy-robustness-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    prepared = _prepare_strategies(data)
    comparison, simulations = _strategy_comparison(
        data=data,
        prepared=prepared,
        train_end=train_end_date,
    )
    yearly = _yearly_walk_forward(data=data, simulations=simulations)
    cost = _cost_stress(
        data=data,
        strategy=prepared["rebound_max5_v3"],
        train_end=train_end_date,
        multipliers=[1.0, 2.0, 3.0],
    )
    start_sensitivity = _start_date_sensitivity(
        data=data,
        strategy=prepared["rebound_max5_v3"],
        train_end=train_end_date,
        sample_count=int(start_samples),
    )
    contribution_sensitivity = _contribution_day_sensitivity(
        data=data,
        strategy=prepared["rebound_max5_v3"],
        train_end=train_end_date,
        offsets=[0, 1, 2, 4, 9, 14],
    )
    freshness = pd.DataFrame(
        _data_freshness_rows(data, pd.Timestamp(datetime.now(UTC).date()))
    )

    comparison.to_csv(run_dir / "strategy_comparison.csv", index=False)
    yearly.to_csv(run_dir / "yearly_walk_forward.csv", index=False)
    cost.to_csv(run_dir / "cost_stress.csv", index=False)
    start_sensitivity.to_csv(run_dir / "start_date_sensitivity.csv", index=False)
    contribution_sensitivity.to_csv(
        run_dir / "contribution_day_sensitivity.csv",
        index=False,
    )
    freshness.to_csv(run_dir / "data_freshness.csv", index=False)

    v3_oos = comparison[
        (comparison["strategy"] == "rebound_max5_v3") & (comparison["period"] == "oos")
    ].iloc[0]
    summary = {
        "run_id": actual_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "train_end": str(train_end_date.date()),
        "checks": {
            "strategy_comparison_rows": int(len(comparison)),
            "yearly_rows": int(len(yearly)),
            "cost_stress_rows": int(len(cost)),
            "start_date_rows": int(len(start_sensitivity)),
            "contribution_day_rows": int(len(contribution_sensitivity)),
        },
        "v3_oos": json.loads(v3_oos.to_json()),
        "v3_cost_stress_oos": json.loads(
            cost[cost["period"] == "oos"].to_json(orient="records")
        ),
        "data_freshness": json.loads(freshness.to_json(orient="records")),
    }
    write_json(run_dir / "summary.json", summary)
    _write_markdown_report(
        path=run_dir / "robustness_report.md",
        comparison=comparison,
        yearly=yearly,
        cost=cost,
        start_sensitivity=start_sensitivity,
        contribution_sensitivity=contribution_sensitivity,
        freshness=freshness,
    )
    return StockStrategyRobustnessRun(run_id=actual_run_id, run_dir=run_dir)
