from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from egx_research.config import BacktestConfig


@dataclass
class BacktestResult:
    equity: pd.Series
    flows: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


def _round_shares(value: float, precision: int) -> float:
    if precision <= 0:
        return float(np.floor(value))
    factor = 10**precision
    return np.floor(value * factor) / factor


def build_contribution_schedule(
    dates: pd.Series,
    start_idx: int,
    end_idx: int,
    initial_cash: float,
    monthly_contribution: float,
    monthly_day_offset: int = 0,
) -> pd.Series:
    schedule = pd.Series(0.0, index=range(len(dates)), dtype=float)
    schedule.iloc[start_idx] += float(initial_cash)

    month_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(start_idx, end_idx + 1):
        date = pd.Timestamp(dates.iloc[i])
        key = (date.year, date.month)
        month_indices.setdefault(key, []).append(i)
    offset = max(0, int(monthly_day_offset))
    for indices in month_indices.values():
        schedule.iloc[indices[min(offset, len(indices) - 1)]] += float(
            monthly_contribution
        )
    return schedule


def _build_trade_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "shares",
                "bars_held",
                "pnl",
                "pnl_pct",
            ]
        )
    return pd.DataFrame(trades)


def _compute_metrics(
    equity: pd.Series,
    flows: pd.Series,
    trades: pd.DataFrame,
    annualization_periods: float = 252.0,
) -> dict[str, float]:
    if annualization_periods <= 0:
        raise ValueError("annualization_periods must be positive")
    if equity.empty:
        return {
            "final_equity": 0.0,
            "total_contributions": 0.0,
            "money_multiple": 0.0,
            "twr_total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "return_dd": 0.0,
            "profit_factor": 0.0,
            "closed_trades": 0.0,
        }

    returns = []
    for idx in range(len(equity)):
        if idx == 0:
            returns.append(0.0)
            continue
        prev_equity = float(equity.iloc[idx - 1])
        external_flow = float(flows.iloc[idx])
        if prev_equity <= 0:
            returns.append(0.0)
            continue
        returns.append((float(equity.iloc[idx]) - prev_equity - external_flow) / prev_equity)

    returns_series = pd.Series(returns, index=equity.index, dtype=float)
    wealth = (1.0 + returns_series).cumprod()
    wealth = wealth.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    max_drawdown = abs(float(drawdown.min()))
    twr_total_return = float(wealth.iloc[-1] - 1.0)

    periods = max(1, len(returns_series) - 1)
    if wealth.iloc[-1] <= 0:
        cagr = -1.0
    else:
        cagr = float(wealth.iloc[-1] ** (annualization_periods / periods) - 1.0)

    volatility = float(returns_series.iloc[1:].std(ddof=0))
    sharpe = (
        0.0
        if volatility == 0
        else float(
            np.sqrt(annualization_periods)
            * returns_series.iloc[1:].mean()
            / volatility
        )
    )
    return_dd = float(cagr / max_drawdown) if max_drawdown > 0 else float(cagr)

    if trades.empty:
        gross_profit = gross_loss = 0.0
    else:
        gross_profit = float(trades["pnl"].clip(lower=0).sum())
        gross_loss = float(trades["pnl"].clip(upper=0).sum())
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    total_contributions = float(flows.sum())
    final_equity = float(equity.iloc[-1])
    money_multiple = 0.0 if total_contributions <= 0 else float(final_equity / total_contributions)

    return {
        "final_equity": final_equity,
        "total_contributions": total_contributions,
        "money_multiple": money_multiple,
        "twr_total_return": twr_total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "return_dd": return_dd,
        "profit_factor": float(profit_factor),
        "closed_trades": float(len(trades)),
    }


def _run_allocation_backtest(
    strategy_frame: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    config: BacktestConfig,
) -> BacktestResult:
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000

    contributions = build_contribution_schedule(
        strategy_frame["date"],
        start_idx,
        end_idx,
        initial_cash=config.initial_cash,
        monthly_contribution=config.monthly_contribution,
        monthly_day_offset=config.monthly_contribution_day_offset,
    )
    equity = pd.Series(np.nan, index=strategy_frame.index, dtype=float)
    flows = pd.Series(0.0, index=strategy_frame.index, dtype=float)

    cash = 0.0
    shares = 0.0
    avg_cost_per_share = 0.0
    highest_close = 0.0
    active_stop: float | None = None
    pending_target: float | None = None
    trades: list[dict[str, Any]] = []
    trade_open: dict[str, Any] | None = None
    floor_series = (
        strategy_frame["floor_allocation"].astype(float)
        if "floor_allocation" in strategy_frame.columns
        else pd.Series(0.0, index=strategy_frame.index, dtype=float)
    )

    def _close_trade(exit_index: int, exit_price: float) -> None:
        nonlocal trade_open
        if trade_open is None:
            return
        pnl = float(trade_open["realized_pnl"])
        basis = float(trade_open["cost_basis"])
        trades.append(
            {
                "entry_date": trade_open["entry_date"],
                "exit_date": strategy_frame["date"].iloc[exit_index],
                "entry_price": trade_open["entry_price"],
                "exit_price": exit_price,
                "shares": trade_open["max_shares"],
                "bars_held": exit_index - trade_open["entry_index"],
                "pnl": pnl,
                "pnl_pct": 0.0 if basis == 0 else pnl / basis,
            }
        )
        trade_open = None

    for i in range(start_idx, end_idx + 1):
        flow = float(contributions.iloc[i])
        flows.iloc[i] += flow
        cash += flow

        open_price = float(strategy_frame["open"].iloc[i])
        close_price = float(strategy_frame["close"].iloc[i])
        atr_value = float(strategy_frame["atr"].iloc[i]) if pd.notna(strategy_frame["atr"].iloc[i]) else np.nan
        stop_mult = float(strategy_frame["stop_mult"].iloc[i])
        trail_mult = float(strategy_frame["trail_mult"].iloc[i])
        floor_allocation = float(floor_series.iloc[i])

        if pending_target is not None:
            total_equity = cash + shares * open_price
            target_value = total_equity * pending_target
            target_shares = _round_shares(target_value / open_price if open_price > 0 else 0.0, config.share_precision)

            if target_shares > shares:
                buy_shares = target_shares - shares
                fill_price = open_price * (1.0 + slippage_rate)
                gross = buy_shares * fill_price
                fee = gross * fee_rate
                affordable = _round_shares(cash / (fill_price * (1.0 + fee_rate)), config.share_precision)
                buy_shares = min(buy_shares, affordable)
                if buy_shares > 0:
                    gross = buy_shares * fill_price
                    fee = gross * fee_rate
                    previous_shares = shares
                    shares += buy_shares
                    cash -= gross + fee
                    avg_cost_per_share = 0.0 if shares <= 0 else ((avg_cost_per_share * previous_shares) + gross + fee) / shares
                    if trade_open is None:
                        trade_open = {
                            "entry_date": strategy_frame["date"].iloc[i],
                            "entry_index": i,
                            "entry_price": fill_price,
                            "cost_basis": gross + fee,
                            "realized_pnl": 0.0,
                            "max_shares": shares,
                        }
                    else:
                        trade_open["cost_basis"] += gross + fee
                        trade_open["max_shares"] = max(float(trade_open["max_shares"]), shares)
                    highest_close = close_price
                    if not np.isnan(atr_value):
                        active_stop = fill_price - atr_value * stop_mult

            elif target_shares < shares:
                sell_shares = shares - target_shares
                fill_price = open_price * (1.0 - slippage_rate)
                gross = sell_shares * fill_price
                fee = gross * fee_rate
                cash += gross - fee
                realized = gross - fee - avg_cost_per_share * sell_shares
                shares -= sell_shares
                if trade_open is not None:
                    trade_open["realized_pnl"] += realized
                if shares <= 0:
                    shares = 0.0
                    avg_cost_per_share = 0.0
                    highest_close = 0.0
                    active_stop = None
                    _close_trade(i, fill_price)

            pending_target = None

        if shares > 0:
            highest_close = max(highest_close, close_price)
            if not np.isnan(atr_value):
                base_stop = avg_cost_per_share - atr_value * stop_mult
                trail_stop = highest_close - atr_value * trail_mult if trail_mult > 0 else None
                levels = [level for level in (active_stop, base_stop, trail_stop) if level is not None]
                if levels:
                    active_stop = max(levels)

        next_target = float(strategy_frame["target_allocation"].iloc[i]) if i < end_idx else None
        if shares > 0 and active_stop is not None and close_price <= active_stop and i < end_idx:
            next_target = floor_allocation
        pending_target = next_target
        equity.iloc[i] = cash + shares * close_price

    slice_equity = equity.iloc[start_idx : end_idx + 1].copy()
    slice_flows = flows.iloc[start_idx : end_idx + 1].copy()

    if shares > 0 and trade_open is not None:
        final_price = float(strategy_frame["close"].iloc[end_idx])
        unrealized = shares * final_price - avg_cost_per_share * shares
        trade_open["realized_pnl"] += unrealized
        trades.append(
            {
                "entry_date": trade_open["entry_date"],
                "exit_date": strategy_frame["date"].iloc[end_idx],
                "entry_price": trade_open["entry_price"],
                "exit_price": final_price,
                "shares": trade_open["max_shares"],
                "bars_held": end_idx - trade_open["entry_index"],
                "pnl": float(trade_open["realized_pnl"]),
                "pnl_pct": 0.0 if trade_open["cost_basis"] == 0 else float(trade_open["realized_pnl"]) / float(trade_open["cost_basis"]),
            }
        )

    trade_frame = _build_trade_frame(trades)
    metrics = _compute_metrics(
        slice_equity,
        slice_flows,
        trade_frame,
        config.annualization_periods,
    )
    return BacktestResult(equity=slice_equity, flows=slice_flows, trades=trade_frame, metrics=metrics)


def _run_cash_deploy_backtest(
    strategy_frame: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    config: BacktestConfig,
) -> BacktestResult:
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000

    contributions = build_contribution_schedule(
        strategy_frame["date"],
        start_idx,
        end_idx,
        initial_cash=config.initial_cash,
        monthly_contribution=config.monthly_contribution,
        monthly_day_offset=config.monthly_contribution_day_offset,
    )
    equity = pd.Series(np.nan, index=strategy_frame.index, dtype=float)
    flows = pd.Series(0.0, index=strategy_frame.index, dtype=float)

    cash = 0.0
    shares = 0.0
    pending_fraction: float | None = None
    purchases: list[dict[str, Any]] = []

    for i in range(start_idx, end_idx + 1):
        flow = float(contributions.iloc[i])
        flows.iloc[i] += flow
        cash += flow

        open_price = float(strategy_frame["open"].iloc[i])
        close_price = float(strategy_frame["close"].iloc[i])

        if pending_fraction is not None and pending_fraction > 0:
            fill_price = open_price * (1.0 + slippage_rate)
            budget = cash * pending_fraction
            buyable = _round_shares(budget / (fill_price * (1.0 + fee_rate)), config.share_precision)
            if buyable > 0:
                gross = buyable * fill_price
                fee = gross * fee_rate
                cash -= gross + fee
                shares += buyable
                purchases.append(
                    {
                        "entry_date": strategy_frame["date"].iloc[i],
                        "exit_date": strategy_frame["date"].iloc[i],
                        "entry_price": fill_price,
                        "exit_price": fill_price,
                        "shares": buyable,
                        "bars_held": 0,
                        "pnl": 0.0,
                        "pnl_pct": 0.0,
                    }
                )
            pending_fraction = None

        next_fraction = float(strategy_frame["deploy_fraction"].iloc[i]) if i < end_idx else None
        pending_fraction = next_fraction if next_fraction is not None and next_fraction > 0 else None
        equity.iloc[i] = cash + shares * close_price

    slice_equity = equity.iloc[start_idx : end_idx + 1].copy()
    slice_flows = flows.iloc[start_idx : end_idx + 1].copy()
    trades = _build_trade_frame(purchases)
    metrics = _compute_metrics(
        slice_equity,
        slice_flows,
        trades.iloc[0:0].copy(),
        config.annualization_periods,
    )
    return BacktestResult(equity=slice_equity, flows=slice_flows, trades=trades, metrics=metrics)


def run_strategy_backtest(
    strategy_frame: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    config: BacktestConfig,
) -> BacktestResult:
    if "deploy_fraction" in strategy_frame.columns:
        return _run_cash_deploy_backtest(strategy_frame, start_idx, end_idx, config)
    if "target_allocation" in strategy_frame.columns:
        return _run_allocation_backtest(strategy_frame, start_idx, end_idx, config)

    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000

    contributions = build_contribution_schedule(
        strategy_frame["date"],
        start_idx,
        end_idx,
        initial_cash=config.initial_cash,
        monthly_contribution=config.monthly_contribution,
        monthly_day_offset=config.monthly_contribution_day_offset,
    )
    equity = pd.Series(np.nan, index=strategy_frame.index, dtype=float)
    flows = pd.Series(0.0, index=strategy_frame.index, dtype=float)

    cash = 0.0
    shares = 0.0
    pending_action: str | None = None
    pending_atr: float | None = None
    trade_open: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    highest_close = 0.0
    active_stop: float | None = None

    for i in range(start_idx, end_idx + 1):
        flows.iloc[i] += float(contributions.iloc[i])
        cash += float(contributions.iloc[i])

        open_price = float(strategy_frame["open"].iloc[i])
        close_price = float(strategy_frame["close"].iloc[i])
        atr_value = float(strategy_frame["atr"].iloc[i]) if pd.notna(strategy_frame["atr"].iloc[i]) else np.nan
        stop_mult = float(strategy_frame["stop_mult"].iloc[i])
        trail_mult = float(strategy_frame["trail_mult"].iloc[i])

        if pending_action == "exit" and shares > 0:
            fill_price = open_price * (1.0 - slippage_rate)
            gross = shares * fill_price
            fee = gross * fee_rate
            cash += gross - fee
            if trade_open is not None:
                total_entry_cost = trade_open["entry_value"] + trade_open["entry_fee"]
                pnl = gross - fee - total_entry_cost
                trades.append(
                    {
                        "entry_date": trade_open["entry_date"],
                        "exit_date": strategy_frame["date"].iloc[i],
                        "entry_price": trade_open["entry_price"],
                        "exit_price": fill_price,
                        "shares": shares,
                        "bars_held": i - trade_open["entry_index"],
                        "pnl": pnl,
                        "pnl_pct": 0.0 if total_entry_cost == 0 else pnl / total_entry_cost,
                    }
                )
            shares = 0.0
            pending_action = None
            trade_open = None
            highest_close = 0.0
            active_stop = None
            pending_atr = None

        elif pending_action == "entry" and shares == 0:
            fill_price = open_price * (1.0 + slippage_rate)
            max_shares = _round_shares(cash / (fill_price * (1.0 + fee_rate)), config.share_precision)
            if max_shares > 0:
                gross = max_shares * fill_price
                fee = gross * fee_rate
                cash -= gross + fee
                shares = max_shares
                trade_open = {
                    "entry_date": strategy_frame["date"].iloc[i],
                    "entry_index": i,
                    "entry_price": fill_price,
                    "entry_value": gross,
                    "entry_fee": fee,
                }
                highest_close = close_price
                if pending_atr is not None and not np.isnan(pending_atr):
                    active_stop = fill_price - pending_atr * stop_mult
            pending_action = None
            pending_atr = None

        if shares > 0:
            highest_close = max(highest_close, close_price)
            if not np.isnan(atr_value):
                base_stop = trade_open["entry_price"] - atr_value * stop_mult if trade_open is not None else None
                trail_stop = highest_close - atr_value * trail_mult if trail_mult > 0 else None
                candidates = [level for level in (active_stop, base_stop, trail_stop) if level is not None]
                if candidates:
                    active_stop = max(candidates)

        exit_now = False
        if shares > 0:
            if active_stop is not None and close_price <= active_stop:
                exit_now = True
            if bool(strategy_frame["exit_signal"].iloc[i]):
                exit_now = True

        entry_now = False
        if shares == 0 and i < end_idx and bool(strategy_frame["entry_signal"].iloc[i]):
            if pd.notna(strategy_frame["atr"].iloc[i]):
                entry_now = True

        if exit_now and i < end_idx:
            pending_action = "exit"
        elif entry_now and i < end_idx and pending_action is None:
            pending_action = "entry"
            pending_atr = float(strategy_frame["atr"].iloc[i])

        equity.iloc[i] = cash + shares * close_price

    slice_equity = equity.iloc[start_idx : end_idx + 1].copy()
    slice_flows = flows.iloc[start_idx : end_idx + 1].copy()
    trade_frame = _build_trade_frame(trades)
    metrics = _compute_metrics(
        slice_equity,
        slice_flows,
        trade_frame,
        config.annualization_periods,
    )
    return BacktestResult(equity=slice_equity, flows=slice_flows, trades=trade_frame, metrics=metrics)


def run_dca_benchmark(data: pd.DataFrame, start_idx: int, end_idx: int, config: BacktestConfig) -> BacktestResult:
    fee_rate = config.fee_bps / 10_000
    slippage_rate = config.slippage_bps / 10_000
    contributions = build_contribution_schedule(
        data["date"],
        start_idx,
        end_idx,
        initial_cash=config.initial_cash,
        monthly_contribution=config.monthly_contribution,
        monthly_day_offset=config.monthly_contribution_day_offset,
    )

    cash = 0.0
    shares = 0.0
    equity = pd.Series(np.nan, index=data.index, dtype=float)
    flows = pd.Series(0.0, index=data.index, dtype=float)
    purchase_records: list[dict[str, Any]] = []

    for i in range(start_idx, end_idx + 1):
        contribution = float(contributions.iloc[i])
        flows.iloc[i] += contribution
        cash += contribution
        if contribution > 0:
            open_price = float(data["open"].iloc[i]) * (1.0 + slippage_rate)
            buyable = _round_shares(cash / (open_price * (1.0 + fee_rate)), config.share_precision)
            if buyable > 0:
                gross = buyable * open_price
                fee = gross * fee_rate
                cash -= gross + fee
                shares += buyable
                purchase_records.append(
                    {
                        "entry_date": data["date"].iloc[i],
                        "exit_date": data["date"].iloc[i],
                        "entry_price": open_price,
                        "exit_price": open_price,
                        "shares": buyable,
                        "bars_held": 0,
                        "pnl": 0.0,
                        "pnl_pct": 0.0,
                    }
                )
        equity.iloc[i] = cash + shares * float(data["close"].iloc[i])

    slice_equity = equity.iloc[start_idx : end_idx + 1].copy()
    slice_flows = flows.iloc[start_idx : end_idx + 1].copy()
    trades = _build_trade_frame(purchase_records)
    metrics = _compute_metrics(
        slice_equity,
        slice_flows,
        trades.iloc[0:0].copy(),
        config.annualization_periods,
    )
    return BacktestResult(equity=slice_equity, flows=slice_flows, trades=trades, metrics=metrics)


def run_buy_hold_benchmark(data: pd.DataFrame, start_idx: int, end_idx: int) -> pd.Series:
    close = data["close"].iloc[start_idx : end_idx + 1].copy()
    return close / float(close.iloc[0])
