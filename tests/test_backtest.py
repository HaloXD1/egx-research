from __future__ import annotations

import pandas as pd

from egx_research.backtest import build_contribution_schedule, run_dca_benchmark, run_strategy_backtest
from egx_research.config import BacktestConfig


def test_contribution_schedule_uses_first_trading_day_in_slice() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-01-16", "2024-02-01", "2024-02-02"]))
    schedule = build_contribution_schedule(dates, 0, 3, initial_cash=100.0, monthly_contribution=50.0)
    assert schedule.iloc[0] == 150.0
    assert schedule.iloc[1] == 0.0
    assert schedule.iloc[2] == 50.0


def test_strategy_backtest_applies_next_open_fill_and_costs() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000, 1000, 1000],
            "entry_signal": [True, False, False],
            "exit_signal": [False, True, False],
            "atr": [1.0, 1.0, 1.0],
            "stop_mult": [10.0, 10.0, 10.0],
            "trail_mult": [0.0, 0.0, 0.0],
        }
    )
    config = BacktestConfig(initial_cash=1000.0, monthly_contribution=0.0, fee_bps=100.0, slippage_bps=100.0)
    result = run_strategy_backtest(frame, 0, 2, config)
    trade = result.trades.iloc[0]
    assert round(trade["entry_price"], 4) == 11.11
    assert round(trade["exit_price"], 4) == 11.88
    assert result.metrics["closed_trades"] == 1.0
    assert result.metrics["final_equity"] > 1000.0


def test_dca_benchmark_accumulates_shares_monthly() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"]),
            "open": [10.0, 10.0, 20.0, 20.0],
            "high": [10.0, 10.0, 20.0, 20.0],
            "low": [10.0, 10.0, 20.0, 20.0],
            "close": [10.0, 10.0, 20.0, 20.0],
            "volume": [1000, 1000, 1000, 1000],
        }
    )
    config = BacktestConfig(initial_cash=0.0, monthly_contribution=100.0, fee_bps=0.0, slippage_bps=0.0)
    result = run_dca_benchmark(frame, 0, 3, config)
    assert result.trades["shares"].sum() == 15.0
    assert result.metrics["final_equity"] == 300.0


def test_allocation_backtest_rebalances_partial_exposure() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [10.0, 10.0, 12.0, 12.0],
            "high": [10.0, 10.5, 12.5, 12.5],
            "low": [10.0, 9.5, 11.5, 11.5],
            "close": [10.0, 12.0, 12.0, 11.0],
            "volume": [1000, 1000, 1000, 1000],
            "target_allocation": [1.0, 0.5, 0.0, 0.0],
            "entry_signal": [True, True, False, False],
            "exit_signal": [False, False, True, True],
            "atr": [1.0, 1.0, 1.0, 1.0],
            "stop_mult": [2.0, 2.0, 2.0, 2.0],
            "trail_mult": [1.0, 1.0, 1.0, 1.0],
        }
    )
    config = BacktestConfig(initial_cash=1000.0, monthly_contribution=0.0, fee_bps=0.0, slippage_bps=0.0)
    result = run_strategy_backtest(frame, 0, 3, config)
    assert result.metrics["final_equity"] > 1000.0
    assert result.metrics["closed_trades"] >= 1.0


def test_allocation_backtest_respects_floor_allocation_on_stop() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5],
            "low": [9.5, 9.5, 8.0],
            "close": [10.0, 10.0, 8.5],
            "volume": [1000, 1000, 1000],
            "target_allocation": [1.0, 1.0, 1.0],
            "floor_allocation": [0.75, 0.75, 0.75],
            "entry_signal": [True, True, True],
            "exit_signal": [False, False, False],
            "atr": [1.0, 1.0, 1.0],
            "stop_mult": [1.0, 1.0, 1.0],
            "trail_mult": [0.0, 0.0, 0.0],
        }
    )
    config = BacktestConfig(initial_cash=1000.0, monthly_contribution=0.0, fee_bps=0.0, slippage_bps=0.0)
    result = run_strategy_backtest(frame, 0, 2, config)
    assert result.metrics["final_equity"] > 0.0
    assert result.metrics["closed_trades"] >= 1.0


def test_cash_deploy_backtest_buys_only_on_signal() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [10.0, 10.0, 8.0, 8.0],
            "high": [10.0, 10.0, 8.5, 8.5],
            "low": [10.0, 10.0, 7.5, 7.5],
            "close": [10.0, 10.0, 8.0, 8.0],
            "volume": [1000, 1000, 1000, 1000],
            "deploy_fraction": [0.0, 1.0, 0.0, 0.0],
            "entry_signal": [False, True, False, False],
            "exit_signal": [False, False, False, False],
            "atr": [0.0, 0.0, 0.0, 0.0],
            "stop_mult": [0.0, 0.0, 0.0, 0.0],
            "trail_mult": [0.0, 0.0, 0.0, 0.0],
        }
    )
    config = BacktestConfig(initial_cash=1000.0, monthly_contribution=0.0, fee_bps=0.0, slippage_bps=0.0)
    result = run_strategy_backtest(frame, 0, 3, config)
    assert result.trades["shares"].sum() == 125.0
    assert result.metrics["final_equity"] == 1000.0
