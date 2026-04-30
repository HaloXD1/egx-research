from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egx_research.backtest import (
    _build_trade_frame,
    _compute_metrics,
    _round_shares,
    build_contribution_schedule,
    run_dca_benchmark,
)
from egx_research.core_satellite import (
    build_core_satellite_selection_plan,
    score_core_satellite_snapshot,
    simulate_stock_dca,
)
from egx_research.data import load_price_data
from egx_research.relative_ic import _build_relative_history
from egx_research.stock_rotation import (
    _apply_selection_gates,
    _build_price_matrix,
    _position_value,
    _sell_positions,
    active_members_on_date,
    first_trading_days,
    load_membership_snapshots,
    load_stock_panel,
)
from egx_research.stock_rotation_config import (
    StockRotationConfig,
    load_stock_rotation_config,
)
from egx_research.utils import ensure_dir, write_json


@dataclass
class StockMomentumPyramidRun:
    run_id: str
    run_dir: Path


@dataclass
class PyramidSimulation:
    equity: pd.Series
    flows: pd.Series
    metrics: dict[str, float]
    actions: pd.DataFrame
    turnover: pd.DataFrame
    holdings: pd.DataFrame
    rankings: pd.DataFrame
    total_fees: float


def _holding_name_map(panel: pd.DataFrame) -> dict[str, str]:
    if "holding_name" not in panel.columns:
        return {}
    latest = panel.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    return dict(zip(latest["symbol"], latest["holding_name"], strict=False))


def _rank_snapshot(
    history: pd.DataFrame,
    membership: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    config: StockRotationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior = history[history["date"] < rebalance_date].copy()
    if prior.empty:
        return pd.DataFrame(), pd.DataFrame()
    snapshot = (
        prior.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False, group_keys=False)
        .tail(1)
        .copy()
    )
    active = active_members_on_date(membership, rebalance_date)
    snapshot = snapshot[snapshot["symbol"].isin(active)].copy()
    snapshot = _apply_selection_gates(snapshot, config)
    if snapshot.empty:
        return snapshot, pd.DataFrame()

    for column in [
        "rel_mom_126",
        "ratio_vs_sma50",
        "ratio_vs_sma200",
        "sma50_vs_sma200",
        "residual_strength_63_126b",
        "residual_strength_126_126b",
        "median_daily_value",
        "max_drawdown_252",
        "vol_63",
        "beta_252",
    ]:
        if column not in snapshot.columns:
            snapshot[column] = np.nan

    scored = score_core_satellite_snapshot(snapshot)
    if scored.empty:
        return snapshot, scored
    scored["rebalance_date"] = pd.Timestamp(rebalance_date)
    scored["selection_date"] = pd.to_datetime(scored["date"])
    scored["rank"] = range(1, len(scored) + 1)
    return snapshot, scored


def _sell_to_cash(
    *,
    date: pd.Timestamp,
    symbol: str,
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    before = len(actions)
    cash, traded, fees = _sell_positions(
        date=date,
        symbols=[symbol],
        positions=positions,
        cash=cash,
        open_row=open_row,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        reason=reason,
        actions=actions,
    )
    for action in actions[before:]:
        action["asset_type"] = "STOCK"
    return cash, traded, fees


def _buy_one(
    *,
    date: pd.Timestamp,
    symbol: str,
    positions: dict[str, float],
    cash: float,
    budget: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    fixed_buy_fee: float,
    share_precision: int,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    raw_open = open_row.get(symbol)
    if pd.isna(raw_open):
        return cash, 0.0, 0.0
    budget = min(float(budget), cash)
    fill_price = float(raw_open) * (1.0 + slippage_rate)
    if budget <= fixed_buy_fee or fill_price <= 0.0:
        return cash, 0.0, 0.0
    budget_shares = _round_shares(
        (budget - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)),
        share_precision,
    )
    cash_shares = _round_shares(
        (cash - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision
    )
    shares = min(budget_shares, cash_shares)
    if shares <= 0:
        return cash, 0.0, 0.0
    gross = shares * fill_price
    fee = gross * fee_rate + fixed_buy_fee
    cash -= gross + fee
    positions[symbol] = float(positions.get(symbol, 0.0)) + float(shares)
    actions.append(
        {
            "date": str(date.date()),
            "asset_type": "STOCK",
            "symbol": symbol,
            "action": "BUY",
            "shares": float(shares),
            "price": fill_price,
            "value": gross,
            "fee": fee,
            "reason": reason,
        }
    )
    return cash, gross, fee


def _trim_to_weight(
    *,
    date: pd.Timestamp,
    symbol: str,
    target_weight: float,
    total_equity: float,
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    share_precision: int,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    shares = float(positions.get(symbol, 0.0))
    raw_open = open_row.get(symbol)
    if shares <= 0.0 or pd.isna(raw_open):
        return cash, 0.0, 0.0
    current_value = shares * float(raw_open)
    target_value = max(0.0, total_equity * float(target_weight))
    if current_value <= target_value:
        return cash, 0.0, 0.0
    fill_price = float(raw_open) * (1.0 - slippage_rate)
    sell_shares = _round_shares((current_value - target_value) / fill_price, share_precision)
    sell_shares = min(sell_shares, shares)
    if sell_shares <= 0:
        return cash, 0.0, 0.0
    gross = sell_shares * fill_price
    fee = gross * fee_rate
    cash += gross - fee
    remaining = shares - sell_shares
    if remaining > 0:
        positions[symbol] = remaining
    else:
        positions.pop(symbol, None)
    actions.append(
        {
            "date": str(date.date()),
            "asset_type": "STOCK",
            "symbol": symbol,
            "action": "SELL",
            "shares": float(sell_shares),
            "price": fill_price,
            "value": gross,
            "fee": fee,
            "reason": reason,
        }
    )
    return cash, gross, fee


def _weighted_buy(
    *,
    date: pd.Timestamp,
    symbols: list[str],
    scores: dict[str, float],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    close_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    fixed_buy_fee: float,
    share_precision: int,
    max_weight: float,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    if not symbols or cash <= fixed_buy_fee:
        return cash, 0.0, 0.0
    total_equity = cash + _position_value(positions, open_row.fillna(close_row))
    room: dict[str, float] = {}
    raw_weights: dict[str, float] = {}
    for symbol in symbols:
        if pd.isna(open_row.get(symbol)):
            continue
        current_value = float(positions.get(symbol, 0.0)) * float(open_row.get(symbol))
        capacity = max(0.0, float(max_weight) * total_equity - current_value)
        if capacity <= fixed_buy_fee:
            continue
        room[symbol] = capacity
        raw_weights[symbol] = max(float(scores.get(symbol, 0.0)), 0.01)
    if not room:
        return cash, 0.0, 0.0

    weight_sum = sum(raw_weights.values())
    traded = 0.0
    fees = 0.0
    starting_cash = cash
    for symbol, raw_weight in sorted(
        raw_weights.items(), key=lambda item: item[1], reverse=True
    ):
        budget = min(starting_cash * raw_weight / weight_sum, room[symbol], cash)
        cash, gross, fee = _buy_one(
            date=date,
            symbol=symbol,
            positions=positions,
            cash=cash,
            budget=budget,
            open_row=open_row,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            fixed_buy_fee=fixed_buy_fee,
            share_precision=share_precision,
            reason=reason,
            actions=actions,
        )
        traded += gross
        fees += fee
    return cash, traded, fees


def simulate_momentum_pyramid(
    *,
    calendar: pd.Series,
    history: pd.DataFrame,
    membership: pd.DataFrame,
    panel: pd.DataFrame,
    config: StockRotationConfig,
    max_holdings: int,
    focus_n: int,
    max_weight: float,
    focus_fill_weight: float,
    exit_rank: int,
    exit_months: int,
) -> PyramidSimulation:
    open_px = _build_price_matrix(panel, calendar, "open", fill=False)
    close_px = _build_price_matrix(panel, calendar, "close", fill=True)
    fee_rate = config.backtest.fee_bps / 10_000
    slippage_rate = config.backtest.slippage_bps / 10_000
    fixed_buy_fee = float(config.portfolio.fixed_buy_fee_egp)
    share_precision = int(config.backtest.share_precision)
    name_map = _holding_name_map(panel)
    rebalance_dates = set(first_trading_days(calendar))
    contributions = build_contribution_schedule(
        calendar,
        0,
        len(calendar) - 1,
        config.backtest.initial_cash,
        config.backtest.monthly_contribution,
    )

    cash = 0.0
    positions: dict[str, float] = {}
    bad_rank_months: dict[str, int] = {}
    equity = pd.Series(index=range(len(calendar)), dtype=float)
    flows = pd.Series(0.0, index=range(len(calendar)), dtype=float)
    actions: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []
    current_scores: dict[str, float] = {}
    current_ranks: dict[str, int] = {}
    current_top: list[str] = []

    for i, value in enumerate(calendar):
        date = pd.Timestamp(value)
        contribution = float(contributions.iloc[i])
        cash += contribution
        flows.iloc[i] += contribution
        open_row = open_px.loc[date]
        close_row = close_px.loc[date]
        turnover_value = 0.0
        fees_paid = 0.0

        if date in rebalance_dates:
            raw_snapshot, scored = _rank_snapshot(history, membership, date, config)
            if not scored.empty:
                export = scored.copy()
                export["rebalance_date"] = str(date.date())
                export["selection_date"] = pd.to_datetime(
                    export["selection_date"]
                ).dt.date.astype(str)
                ranking_frames.append(export)
                current_top = scored.head(max_holdings)["symbol"].astype(str).tolist()
                current_scores = dict(
                    zip(scored["symbol"], scored["score"], strict=False)
                )
                current_ranks = dict(zip(scored["symbol"], scored["rank"], strict=False))
            else:
                current_top = []
                current_scores = {}
                current_ranks = {}

            raw_by_symbol = (
                raw_snapshot.set_index("symbol") if not raw_snapshot.empty else pd.DataFrame()
            )
            for symbol in list(positions):
                rank = current_ranks.get(symbol)
                if rank is None or int(rank) > int(exit_rank):
                    bad_rank_months[symbol] = bad_rank_months.get(symbol, 0) + 1
                else:
                    bad_rank_months[symbol] = 0

                if symbol in raw_by_symbol.index:
                    row = raw_by_symbol.loc[symbol]
                    trend_break = bool(
                        pd.notna(row.get("ratio_vs_sma200"))
                        and float(row.get("ratio_vs_sma200")) <= 0.0
                    )
                    residual_break = bool(
                        pd.notna(row.get("residual_strength_63_126b"))
                        and pd.notna(row.get("ratio_vs_sma50"))
                        and float(row.get("residual_strength_63_126b")) <= 0.0
                        and float(row.get("ratio_vs_sma50")) <= 0.0
                    )
                else:
                    trend_break = True
                    residual_break = True

                should_sell = (
                    trend_break
                    or residual_break
                    or bad_rank_months.get(symbol, 0) >= int(exit_months)
                )
                if should_sell:
                    cash, traded, fees = _sell_to_cash(
                        date=date,
                        symbol=symbol,
                        positions=positions,
                        cash=cash,
                        open_row=open_row,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        reason="pyramid_exit",
                        actions=actions,
                    )
                    turnover_value += traded
                    fees_paid += fees
                    bad_rank_months.pop(symbol, None)

            if len(positions) > max_holdings:
                ranked_held = sorted(
                    positions,
                    key=lambda item: int(current_ranks.get(item, 10_000)),
                    reverse=True,
                )
                for symbol in ranked_held[: len(positions) - max_holdings]:
                    cash, traded, fees = _sell_to_cash(
                        date=date,
                        symbol=symbol,
                        positions=positions,
                        cash=cash,
                        open_row=open_row,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        reason="pyramid_max_holdings_exit",
                        actions=actions,
                    )
                    turnover_value += traded
                    fees_paid += fees

            total_equity_open = cash + _position_value(
                positions, open_row.fillna(close_row)
            )
            for symbol in list(positions):
                current_weight = (
                    0.0
                    if total_equity_open <= 0.0 or pd.isna(open_row.get(symbol))
                    else float(positions[symbol]) * float(open_row.get(symbol))
                    / total_equity_open
                )
                if current_weight > float(max_weight) * 1.10:
                    cash, traded, fees = _trim_to_weight(
                        date=date,
                        symbol=symbol,
                        target_weight=max_weight,
                        total_equity=total_equity_open,
                        positions=positions,
                        cash=cash,
                        open_row=open_row,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        share_precision=share_precision,
                        reason="pyramid_cap_trim",
                        actions=actions,
                    )
                    turnover_value += traded
                    fees_paid += fees

        if cash > fixed_buy_fee and current_top:
            top_focus = current_top[:focus_n]
            lower = current_top[focus_n:max_holdings]
            total_equity_open = cash + _position_value(positions, open_row.fillna(close_row))
            focus_full = bool(top_focus) and all(
                (
                    float(positions.get(symbol, 0.0)) * float(open_row.get(symbol))
                    / total_equity_open
                    >= focus_fill_weight
                )
                if total_equity_open > 0.0 and pd.notna(open_row.get(symbol))
                else False
                for symbol in top_focus
            )
            buy_symbols = top_focus + (lower if focus_full else [])
            slots = max(0, max_holdings - len(positions))
            for symbol in current_top:
                if symbol not in positions and symbol not in buy_symbols and slots > 0:
                    buy_symbols.append(symbol)
                    slots -= 1
            buy_symbols = [symbol for symbol in buy_symbols if symbol in current_top]
            cash, traded, fees = _weighted_buy(
                date=date,
                symbols=buy_symbols,
                scores=current_scores,
                positions=positions,
                cash=cash,
                open_row=open_row,
                close_row=close_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                fixed_buy_fee=fixed_buy_fee,
                share_precision=share_precision,
                max_weight=max_weight,
                reason="pyramid_dca_buy",
                actions=actions,
            )
            turnover_value += traded
            fees_paid += fees

        total_equity_close = cash + _position_value(positions, close_row)
        equity.iloc[i] = total_equity_close

        if date in rebalance_dates:
            for symbol, shares in sorted(positions.items()):
                price = close_row.get(symbol)
                value_close = 0.0 if pd.isna(price) else float(shares) * float(price)
                holding_rows.append(
                    {
                        "date": str(date.date()),
                        "symbol": symbol,
                        "holding_name": name_map.get(symbol, symbol),
                        "shares": float(shares),
                        "close": None if pd.isna(price) else float(price),
                        "value": value_close,
                        "weight": 0.0
                        if total_equity_close <= 0.0
                        else value_close / total_equity_close,
                        "rank": current_ranks.get(symbol),
                        "score": current_scores.get(symbol),
                    }
                )
            turnover_rows.append(
                {
                    "date": str(date.date()),
                    "holdings": int(len(positions)),
                    "cash": cash,
                    "turnover_value": turnover_value,
                    "turnover_pct": 0.0
                    if total_equity_close <= 0.0
                    else turnover_value / total_equity_close,
                    "fees": fees_paid,
                }
            )

    actions_frame = pd.DataFrame(actions)
    if actions_frame.empty:
        actions_frame = pd.DataFrame(
            columns=[
                "date",
                "asset_type",
                "symbol",
                "action",
                "shares",
                "price",
                "value",
                "fee",
                "reason",
            ]
        )
    turnover = pd.DataFrame(turnover_rows)
    holdings = pd.DataFrame(holding_rows)
    rankings = (
        pd.concat(ranking_frames, ignore_index=True)
        if ranking_frames
        else pd.DataFrame()
    )
    total_fees = float(actions_frame["fee"].fillna(0.0).sum())
    return PyramidSimulation(
        equity=equity,
        flows=flows,
        metrics=_compute_metrics(equity.copy(), flows.copy(), _build_trade_frame([])),
        actions=actions_frame,
        turnover=turnover,
        holdings=holdings,
        rankings=rankings,
        total_fees=total_fees,
    )


def run_stock_momentum_pyramid_backtest(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_holdings: int = 10,
    focus_n: int = 5,
    max_weight: float = 0.25,
    focus_fill_weight: float = 0.20,
    exit_rank: int = 20,
    exit_months: int = 2,
) -> StockMomentumPyramidRun:
    config = load_stock_rotation_config(config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf_full = load_price_data(config.benchmark.etf_symbol_path)
    etf_history = etf_full.copy()
    if end_date is not None:
        etf_history = etf_history[
            etf_history["date"] <= pd.Timestamp(end_date)
        ].reset_index(drop=True)
    etf = etf_full.copy()
    if start_date is not None:
        etf = etf[etf["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date is not None:
        etf = etf[etf["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    if etf.empty:
        raise ValueError("No ETF dates remain after start/end filtering.")

    history = _build_relative_history(panel, etf_history, config, horizon_days=63)
    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    simulation = simulate_momentum_pyramid(
        calendar=calendar,
        history=history,
        membership=membership,
        panel=panel,
        config=config,
        max_holdings=max_holdings,
        focus_n=focus_n,
        max_weight=max_weight,
        focus_fill_weight=focus_fill_weight,
        exit_rank=exit_rank,
        exit_months=exit_months,
    )

    annual_selections, annual_map = build_core_satellite_selection_plan(
        history=history,
        membership=membership,
        calendar=calendar,
        config=config,
        top_n=max_holdings,
        rebalance_mode="annual",
    )
    first_rebalance = min(annual_map)
    benchmark_calendar = calendar[calendar >= first_rebalance].reset_index(drop=True)
    benchmark_etf = etf[etf["date"] >= first_rebalance].reset_index(drop=True)
    annual_map = {
        date: symbols for date, symbols in annual_map.items() if date >= first_rebalance
    }
    annual_top10 = simulate_stock_dca(
        calendar=benchmark_calendar,
        panel=panel,
        selection_map=annual_map,
        config=config,
        static=False,
    )
    etf_dca = run_dca_benchmark(
        benchmark_etf, 0, len(benchmark_etf) - 1, config.backtest
    )

    actual_run_id = run_id or f"stock-momentum-pyramid-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)
    simulation.rankings.to_csv(run_dir / "monthly_rankings.csv", index=False)
    simulation.holdings.to_csv(run_dir / "selected_holdings.csv", index=False)
    simulation.actions.to_csv(run_dir / "trade_actions.csv", index=False)
    simulation.turnover.to_csv(run_dir / "turnover.csv", index=False)
    annual_selections.to_csv(run_dir / "annual_top10_selected_holdings.csv", index=False)
    pd.DataFrame(
        {
            "date": calendar.values,
            "momentum_pyramid_equity": simulation.equity.values,
        }
    ).merge(
        pd.DataFrame(
            {
                "date": benchmark_calendar.values,
                "annual_top10_dca_equity": annual_top10.equity.values,
                "etf_dca_equity": etf_dca.equity.values,
            }
        ),
        on="date",
        how="left",
    ).to_csv(run_dir / "equity_curve.csv", index=False)

    latest_holdings = (
        simulation.holdings[
            simulation.holdings["date"] == simulation.holdings["date"].max()
        ]
        if not simulation.holdings.empty
        else pd.DataFrame()
    )
    summary = {
        "run_id": actual_run_id,
        "strategy": "stock_momentum_pyramid",
        "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
        "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "max_holdings": int(max_holdings),
        "focus_n": int(focus_n),
        "max_weight": float(max_weight),
        "focus_fill_weight": float(focus_fill_weight),
        "exit_rank": int(exit_rank),
        "exit_months": int(exit_months),
        "momentum_pyramid_metrics": simulation.metrics,
        "annual_top10_dca_metrics": annual_top10.metrics,
        "etf_dca_metrics": etf_dca.metrics,
        "momentum_pyramid_excess_vs_annual_top10": float(
            simulation.metrics["twr_total_return"]
            - annual_top10.metrics["twr_total_return"]
        ),
        "momentum_pyramid_excess_vs_etf_dca": float(
            simulation.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "total_fees": {
            "momentum_pyramid": float(simulation.total_fees),
            "annual_top10_dca": float(annual_top10.total_fees),
        },
        "trade_count": int(len(simulation.actions)),
        "latest_holdings": []
        if latest_holdings.empty
        else latest_holdings.sort_values("weight", ascending=False)[
            ["symbol", "weight", "rank", "score"]
        ].to_dict(orient="records"),
        "artifact_paths": {
            "monthly_rankings": str(run_dir / "monthly_rankings.csv"),
            "selected_holdings": str(run_dir / "selected_holdings.csv"),
            "equity_curve": str(run_dir / "equity_curve.csv"),
            "trade_actions": str(run_dir / "trade_actions.csv"),
            "turnover": str(run_dir / "turnover.csv"),
            "annual_top10_selected_holdings": str(
                run_dir / "annual_top10_selected_holdings.csv"
            ),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return StockMomentumPyramidRun(run_id=actual_run_id, run_dir=run_dir)
