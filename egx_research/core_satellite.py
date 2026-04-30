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
    run_buy_hold_benchmark,
    run_dca_benchmark,
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
class CoreSatelliteRun:
    run_id: str
    run_dir: Path


@dataclass
class Simulation:
    equity: pd.Series
    flows: pd.Series
    metrics: dict[str, float]
    actions: pd.DataFrame
    turnover: pd.DataFrame
    total_fees: float


ADAPTIVE_BALANCED_PROFILE = {
    "bull_core": 0.30,
    "neutral_core": 0.50,
    "stress_core": 0.80,
    "portfolio_drawdown_guard": 0.25,
    "sleeve_drawdown_guard": 0.30,
}
ADAPTIVE_AGGRESSIVE_PROFILE = {
    "bull_core": 0.20,
    "neutral_core": 0.50,
    "stress_core": 0.90,
    "portfolio_drawdown_guard": 0.25,
    "sleeve_drawdown_guard": 0.30,
}


def _holding_name_map(panel: pd.DataFrame) -> dict[str, str]:
    if "holding_name" not in panel.columns:
        return {}
    latest = panel.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    return dict(zip(latest["symbol"], latest["holding_name"], strict=False))


def _rank(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    ranks = pd.Series(0.5, index=series.index, dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna() & np.isfinite(values)
    if valid.sum() < 2 or values.loc[valid].nunique(dropna=True) <= 1:
        return ranks
    ranks.loc[valid] = values.loc[valid].rank(
        method="average", pct=True, ascending=higher_is_better
    )
    return ranks


def _rebalance_dates(calendar: pd.Series, mode: str) -> list[pd.Timestamp]:
    monthly = first_trading_days(calendar)
    if not monthly:
        return []
    if mode == "annual":
        dates = []
        seen_years: set[int] = set()
        for date in monthly:
            date = pd.Timestamp(date)
            if date.year not in seen_years:
                seen_years.add(date.year)
                dates.append(date)
        if pd.Timestamp(monthly[0]) not in dates:
            dates.insert(0, pd.Timestamp(monthly[0]))
        return dates
    if mode == "semiannual":
        dates = [pd.Timestamp(monthly[0])]
        for date in monthly:
            date = pd.Timestamp(date)
            if date.month in {1, 7} and date not in dates:
                dates.append(date)
        return sorted(dates)
    raise ValueError(f"Unsupported rebalance mode: {mode}")


def score_core_satellite_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    scored = snapshot.copy()
    scored["rank_rel_mom_126"] = _rank(scored["rel_mom_126"], higher_is_better=True)
    scored["rank_ratio_sma50"] = _rank(scored["ratio_vs_sma50"], higher_is_better=True)
    scored["rank_ratio_sma200"] = _rank(scored["ratio_vs_sma200"], higher_is_better=True)
    scored["rank_sma50_sma200"] = _rank(scored["sma50_vs_sma200"], higher_is_better=True)
    scored["rank_resid_63"] = _rank(
        scored["residual_strength_63_126b"], higher_is_better=True
    )
    scored["rank_resid_126"] = _rank(
        scored["residual_strength_126_126b"], higher_is_better=True
    )
    scored["rank_liquidity"] = _rank(scored["median_daily_value"], higher_is_better=True)
    scored["rank_drawdown"] = _rank(
        scored["max_drawdown_252"], higher_is_better=False
    )
    scored["rank_vol"] = _rank(scored["vol_63"], higher_is_better=False)
    scored["rank_beta"] = _rank(scored["beta_252"].abs(), higher_is_better=False)

    scored["score"] = (
        0.25 * scored["rank_rel_mom_126"]
        + 0.20
        * (
            scored["rank_ratio_sma50"]
            + scored["rank_ratio_sma200"]
            + scored["rank_sma50_sma200"]
        )
        / 3.0
        + 0.20 * (scored["rank_resid_63"] + scored["rank_resid_126"]) / 2.0
        + 0.10 * scored["rank_liquidity"]
        + 0.10 * scored["rank_drawdown"]
        + 0.10 * scored["rank_vol"]
        + 0.05 * scored["rank_beta"]
    )
    trend_ok = (scored["ratio_vs_sma200"] > 0.0) & (scored["sma50_vs_sma200"] > 0.0)
    scored = scored[trend_ok].copy()
    return scored.sort_values(
        ["score", "rel_mom_126", "residual_strength_126_126b", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_core_satellite_selection_plan(
    *,
    history: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    config: StockRotationConfig,
    top_n: int,
    rebalance_mode: str,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, list[str]]]:
    rows: list[dict[str, Any]] = []
    selection_map: dict[pd.Timestamp, list[str]] = {}
    name_map: dict[str, str] = {}
    if "holding_name" in history.columns:
        latest = history.sort_values(["symbol", "signal_date" if "signal_date" in history.columns else "date"])
        latest = latest.drop_duplicates(subset=["symbol"], keep="last")
        name_map = dict(zip(latest["symbol"], latest["holding_name"], strict=False))

    for rebalance_date in _rebalance_dates(calendar, rebalance_mode):
        prior = history[history["date"] < rebalance_date].copy()
        if prior.empty:
            continue
        snapshot = (
            prior.sort_values(["symbol", "date"])
            .groupby("symbol", as_index=False, group_keys=False)
            .tail(1)
            .copy()
        )
        active = active_members_on_date(membership, rebalance_date)
        snapshot = snapshot[snapshot["symbol"].isin(active)].copy()
        snapshot = _apply_selection_gates(snapshot, config)
        required = [
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
        ]
        for column in required:
            if column not in snapshot.columns:
                snapshot[column] = np.nan
        snapshot = snapshot.dropna(subset=["rel_mom_126", "ratio_vs_sma200"]).copy()
        if snapshot.empty:
            continue
        scored = score_core_satellite_snapshot(snapshot)
        if scored.empty:
            continue
        selected = scored.head(top_n).copy()
        symbols = selected["symbol"].astype(str).tolist()
        selection_map[pd.Timestamp(rebalance_date)] = symbols
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            rows.append(
                {
                    "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                    "selection_date": str(pd.Timestamp(row.date).date()),
                    "rank": rank,
                    "symbol": row.symbol,
                    "holding_name": name_map.get(row.symbol, row.symbol),
                    "score": float(row.score),
                    "rel_mom_126": float(row.rel_mom_126),
                    "ratio_vs_sma200": float(row.ratio_vs_sma200),
                    "residual_strength_126_126b": float(
                        row.residual_strength_126_126b
                    ),
                    "median_daily_value": float(row.median_daily_value),
                    "max_drawdown_252": float(row.max_drawdown_252),
                    "vol_63": float(row.vol_63),
                    "beta_252": float(row.beta_252),
                }
            )

    if not rows:
        raise ValueError("No core/satellite selections generated.")
    return pd.DataFrame(rows), selection_map


def _buy_etf(
    *,
    date: pd.Timestamp,
    cash: float,
    etf_shares: float,
    budget: float,
    open_price: float,
    fee_rate: float,
    slippage_rate: float,
    share_precision: int,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    budget = min(float(budget), cash)
    fill_price = float(open_price) * (1.0 + slippage_rate)
    if budget <= 0.0 or fill_price <= 0.0:
        return cash, etf_shares, 0.0, 0.0
    buy_shares = _round_shares(
        budget / (fill_price * (1.0 + fee_rate)), share_precision
    )
    if buy_shares <= 0:
        return cash, etf_shares, 0.0, 0.0
    gross = buy_shares * fill_price
    fee = gross * fee_rate
    cash -= gross + fee
    etf_shares += float(buy_shares)
    actions.append(
        {
            "date": str(date.date()),
            "asset_type": "ETF",
            "symbol": "EGX30_ETF",
            "action": "BUY",
            "shares": float(buy_shares),
            "price": fill_price,
            "value": gross,
            "fee": fee,
            "reason": reason,
        }
    )
    return cash, etf_shares, gross, fee


def _buy_stocks_equal(
    *,
    date: pd.Timestamp,
    symbols: list[str],
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
    tradable = [symbol for symbol in symbols if not pd.isna(open_row.get(symbol))]
    budget = min(float(budget), cash)
    if not tradable or budget <= fixed_buy_fee:
        return cash, 0.0, 0.0
    per_symbol_budget = budget / float(len(tradable))
    traded = 0.0
    fees = 0.0
    for symbol in tradable:
        fill_price = float(open_row.get(symbol)) * (1.0 + slippage_rate)
        max_budget = min(per_symbol_budget, cash)
        if max_budget <= fixed_buy_fee or fill_price <= 0.0:
            continue
        shares_budget = _round_shares(
            (max_budget - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)),
            share_precision,
        )
        shares_cash = _round_shares(
            (cash - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)),
            share_precision,
        )
        shares = min(shares_budget, shares_cash)
        if shares <= 0:
            continue
        gross = shares * fill_price
        fee = gross * fee_rate + fixed_buy_fee
        cash -= gross + fee
        positions[symbol] = float(positions.get(symbol, 0.0)) + float(shares)
        traded += gross
        fees += fee
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
    return cash, traded, fees


def _rebalance_stocks_to_target(
    *,
    date: pd.Timestamp,
    selected_symbols: list[str],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    close_row: pd.Series,
    target_stock_value: float,
    fee_rate: float,
    slippage_rate: float,
    fixed_buy_fee: float,
    share_precision: int,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    if not selected_symbols:
        return cash, 0.0, 0.0
    tradable = [symbol for symbol in selected_symbols if not pd.isna(open_row.get(symbol))]
    if not tradable:
        return cash, 0.0, 0.0
    target_value = max(0.0, target_stock_value) / float(len(selected_symbols))
    traded = 0.0
    fees = 0.0

    for symbol in tradable:
        shares = float(positions.get(symbol, 0.0))
        if shares <= 0:
            continue
        open_price = float(open_row.get(symbol))
        current_value = shares * open_price
        excess_value = current_value - target_value
        if excess_value <= 0:
            continue
        trade_price = open_price * (1.0 - slippage_rate)
        sell_shares = _round_shares(excess_value / trade_price, share_precision)
        sell_shares = min(shares, sell_shares)
        if sell_shares <= 0:
            continue
        gross = sell_shares * trade_price
        fee = gross * fee_rate
        cash += gross - fee
        traded += gross
        fees += fee
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
                "price": trade_price,
                "value": gross,
                "fee": fee,
                "reason": "stock_sleeve_trim",
            }
        )

    for symbol in tradable:
        open_price = float(open_row.get(symbol))
        current_value = float(positions.get(symbol, 0.0)) * open_price
        shortfall = target_value - current_value
        cash, gross, fee = _buy_stocks_equal(
            date=date,
            symbols=[symbol],
            positions=positions,
            cash=cash,
            budget=shortfall,
            open_row=open_row,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            fixed_buy_fee=fixed_buy_fee,
            share_precision=share_precision,
            reason="stock_sleeve_rebalance_buy",
            actions=actions,
        )
        traded += gross
        fees += fee
    return cash, traded, fees


def _simulation_result(
    equity: pd.Series,
    flows: pd.Series,
    actions: list[dict[str, Any]],
    turnover_rows: list[dict[str, Any]],
) -> Simulation:
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
    total_fees = float(actions_frame["fee"].fillna(0.0).sum())
    return Simulation(
        equity=equity,
        flows=flows,
        metrics=_compute_metrics(equity.copy(), flows.copy(), _build_trade_frame([])),
        actions=actions_frame,
        turnover=turnover,
        total_fees=total_fees,
    )


def simulate_stock_dca(
    *,
    calendar: pd.Series,
    panel: pd.DataFrame,
    selection_map: dict[pd.Timestamp, list[str]],
    config: StockRotationConfig,
    static: bool,
) -> Simulation:
    open_px = _build_price_matrix(panel, calendar, "open", fill=False)
    close_px = _build_price_matrix(panel, calendar, "close", fill=True)
    fee_rate = config.backtest.fee_bps / 10_000
    slippage_rate = config.backtest.slippage_bps / 10_000
    fixed_buy_fee = float(config.portfolio.fixed_buy_fee_egp)
    share_precision = int(config.backtest.share_precision)
    contributions = build_contribution_schedule(
        calendar,
        0,
        len(calendar) - 1,
        config.backtest.initial_cash,
        config.backtest.monthly_contribution,
    )
    positions: dict[str, float] = {}
    cash = 0.0
    current_selection: list[str] = []
    equity = pd.Series(index=range(len(calendar)), dtype=float)
    flows = pd.Series(0.0, index=range(len(calendar)), dtype=float)
    actions: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []

    for i, value in enumerate(calendar):
        date = pd.Timestamp(value)
        cash += float(contributions.iloc[i])
        flows.iloc[i] += float(contributions.iloc[i])
        open_row = open_px.loc[date]
        close_row = close_px.loc[date]
        turnover_value = 0.0
        fees = 0.0

        if date in selection_map and (not static or not current_selection):
            current_selection = selection_map[date]
            if not static:
                before = cash + _position_value(positions, open_row.fillna(close_row))
                unwanted = sorted(
                    symbol for symbol in positions if symbol not in set(current_selection)
                )
                cash, traded, paid = _sell_positions(
                    date=date,
                    symbols=unwanted,
                    positions=positions,
                    cash=cash,
                    open_row=open_row,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    reason="annual_selection_exit",
                    actions=actions,
                )
                turnover_value += traded
                fees += paid
                cash, traded, paid = _rebalance_stocks_to_target(
                    date=date,
                    selected_symbols=current_selection,
                    positions=positions,
                    cash=cash,
                    open_row=open_row,
                    close_row=close_row,
                    target_stock_value=cash + _position_value(positions, open_row.fillna(close_row)),
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    fixed_buy_fee=fixed_buy_fee,
                    share_precision=share_precision,
                    actions=actions,
                )
                turnover_value += traded
                fees += paid
                turnover_rows.append(
                    {
                        "date": str(date.date()),
                        "turnover_value": turnover_value,
                        "turnover_pct": 0.0 if before <= 0.0 else turnover_value / before,
                        "fees": fees,
                    }
                )

        if current_selection and cash > 0.0:
            cash, traded, paid = _buy_stocks_equal(
                date=date,
                symbols=current_selection,
                positions=positions,
                cash=cash,
                budget=cash,
                open_row=open_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                fixed_buy_fee=fixed_buy_fee,
                share_precision=share_precision,
                reason="static_top10_dca" if static else "annual_top10_dca",
                actions=actions,
            )
            turnover_value += traded
            fees += paid

        equity.iloc[i] = cash + _position_value(positions, close_row)

    return _simulation_result(equity, flows, actions, turnover_rows)


def simulate_core_satellite(
    *,
    calendar: pd.Series,
    panel: pd.DataFrame,
    etf: pd.DataFrame,
    selection_map: dict[pd.Timestamp, list[str]],
    config: StockRotationConfig,
    core_weight: float,
    drawdown_guard: float,
    adaptive_profile: dict[str, float] | None = None,
) -> Simulation:
    open_px = _build_price_matrix(panel, calendar, "open", fill=False)
    close_px = _build_price_matrix(panel, calendar, "close", fill=True)
    etf_frame = etf.set_index("date").reindex(pd.Index(pd.to_datetime(calendar)))
    fee_rate = config.backtest.fee_bps / 10_000
    slippage_rate = config.backtest.slippage_bps / 10_000
    fixed_buy_fee = float(config.portfolio.fixed_buy_fee_egp)
    share_precision = int(config.backtest.share_precision)
    satellite_weight = 1.0 - float(core_weight)
    contributions = build_contribution_schedule(
        calendar,
        0,
        len(calendar) - 1,
        config.backtest.initial_cash,
        config.backtest.monthly_contribution,
    )
    stock_positions: dict[str, float] = {}
    etf_shares = 0.0
    cash = 0.0
    current_selection: list[str] = []
    sleeve_peak = 0.0
    portfolio_peak = 0.0
    equity = pd.Series(index=range(len(calendar)), dtype=float)
    flows = pd.Series(0.0, index=range(len(calendar)), dtype=float)
    actions: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []

    def _basket_above_sma200(date: pd.Timestamp, symbols: list[str]) -> bool:
        available = [symbol for symbol in symbols if symbol in close_px.columns]
        if not available:
            return False
        history = close_px.loc[:date, available].dropna(how="all").tail(200)
        if len(history) < 200:
            return True
        current = history.iloc[-1].replace(0.0, np.nan)
        normalized = history.divide(current, axis=1)
        basket = normalized.mean(axis=1, skipna=True).dropna()
        return bool(not basket.empty and float(basket.iloc[-1]) > float(basket.mean()))

    def _regime_weights(
        date: pd.Timestamp,
        stock_value: float,
        total_equity_close: float,
    ) -> tuple[str, float, float, bool, float, float]:
        nonlocal sleeve_peak, portfolio_peak
        sleeve_peak = max(sleeve_peak, stock_value)
        portfolio_peak = max(portfolio_peak, total_equity_close)
        sleeve_drawdown = 0.0 if sleeve_peak <= 0.0 else stock_value / sleeve_peak - 1.0
        portfolio_drawdown = (
            0.0
            if portfolio_peak <= 0.0
            else total_equity_close / portfolio_peak - 1.0
        )
        fixed_guard = sleeve_drawdown <= -abs(float(drawdown_guard))
        if adaptive_profile is None:
            return (
                "fixed_guard" if fixed_guard else "fixed",
                float(core_weight),
                satellite_weight,
                fixed_guard,
                sleeve_drawdown,
                portfolio_drawdown,
            )

        row = etf_frame.loc[date]
        etf_close = float(row["close"])
        sma50 = float(row["sma50"]) if pd.notna(row.get("sma50")) else np.nan
        sma200 = float(row["sma200"]) if pd.notna(row.get("sma200")) else np.nan
        etf_stress = bool(pd.notna(sma200) and etf_close < sma200)
        stress = (
            etf_stress
            or portfolio_drawdown <= -abs(float(adaptive_profile["portfolio_drawdown_guard"]))
            or sleeve_drawdown <= -abs(float(adaptive_profile["sleeve_drawdown_guard"]))
        )
        basket_bull = _basket_above_sma200(date, current_selection)
        bull = bool(
            pd.notna(sma50)
            and pd.notna(sma200)
            and etf_close > sma200
            and sma50 > sma200
            and basket_bull
        )
        if stress:
            regime = "stress"
            active_core = float(adaptive_profile["stress_core"])
        elif bull:
            regime = "bull"
            active_core = float(adaptive_profile["bull_core"])
        else:
            regime = "neutral"
            active_core = float(adaptive_profile["neutral_core"])
        return (
            regime,
            active_core,
            1.0 - active_core,
            False,
            sleeve_drawdown,
            portfolio_drawdown,
        )

    for i, value in enumerate(calendar):
        date = pd.Timestamp(value)
        contribution = float(contributions.iloc[i])
        cash += contribution
        flows.iloc[i] += contribution
        open_row = open_px.loc[date]
        close_row = close_px.loc[date]
        etf_open = float(etf_frame.loc[date, "open"])
        etf_close = float(etf_frame.loc[date, "close"])
        stock_value_close = _position_value(stock_positions, close_row)
        total_equity_close = cash + etf_shares * etf_close + stock_value_close

        if date in selection_map:
            current_selection = selection_map[date]

        (
            regime,
            active_core_weight,
            active_satellite_weight,
            guard_active,
            sleeve_drawdown,
            portfolio_drawdown,
        ) = _regime_weights(
            date,
            stock_value_close,
            total_equity_close,
        )

        if date in selection_map:
            equity_before = (
                cash
                + etf_shares * etf_open
                + _position_value(stock_positions, open_row.fillna(close_row))
            )
            turnover_value = 0.0
            fees = 0.0
            unwanted = sorted(
                symbol for symbol in stock_positions if symbol not in set(current_selection)
            )
            cash, traded, paid = _sell_positions(
                date=date,
                symbols=unwanted,
                positions=stock_positions,
                cash=cash,
                open_row=open_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                reason="core_satellite_selection_exit",
                actions=actions,
            )
            turnover_value += traded
            fees += paid
            if adaptive_profile is not None or not guard_active:
                target_stock_value = equity_before * active_satellite_weight
                cash, traded, paid = _rebalance_stocks_to_target(
                    date=date,
                    selected_symbols=current_selection,
                    positions=stock_positions,
                    cash=cash,
                    open_row=open_row,
                    close_row=close_row,
                    target_stock_value=target_stock_value,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    fixed_buy_fee=fixed_buy_fee,
                    share_precision=share_precision,
                    actions=actions,
                )
                turnover_value += traded
                fees += paid
            cash, etf_shares, gross, paid = _buy_etf(
                date=date,
                cash=cash,
                etf_shares=etf_shares,
                budget=cash,
                open_price=etf_open,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                share_precision=share_precision,
                reason="core_rebalance_cash_sweep",
                actions=actions,
            )
            turnover_value += gross
            fees += paid
            turnover_rows.append(
                {
                    "date": str(date.date()),
                    "regime": regime,
                    "core_weight": active_core_weight,
                    "satellite_weight": active_satellite_weight,
                    "guard_active": bool(guard_active),
                    "sleeve_drawdown": sleeve_drawdown,
                    "portfolio_drawdown": portfolio_drawdown,
                    "turnover_value": turnover_value,
                    "turnover_pct": 0.0
                    if equity_before <= 0.0
                    else turnover_value / equity_before,
                    "fees": fees,
                }
            )
        elif contribution > 0.0:
            stock_budget = contribution * active_satellite_weight
            etf_budget = contribution * active_core_weight
            if guard_active or not current_selection:
                etf_budget += stock_budget
                stock_budget = 0.0
            cash, etf_shares, _, _ = _buy_etf(
                date=date,
                cash=cash,
                etf_shares=etf_shares,
                budget=etf_budget,
                open_price=etf_open,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                share_precision=share_precision,
                reason="monthly_core_dca"
                if stock_budget > 0
                else "monthly_core_plus_guard_redirect",
                actions=actions,
            )
            if stock_budget > 0.0:
                cash, _, _ = _buy_stocks_equal(
                    date=date,
                    symbols=current_selection,
                    positions=stock_positions,
                    cash=cash,
                    budget=stock_budget,
                    open_row=open_row,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    fixed_buy_fee=fixed_buy_fee,
                    share_precision=share_precision,
                    reason="monthly_satellite_dca",
                    actions=actions,
                )

        equity.iloc[i] = (
            cash + etf_shares * etf_close + _position_value(stock_positions, close_row)
        )

    return _simulation_result(equity, flows, actions, turnover_rows)


def run_core_satellite_backtest(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    core_weight: float = 0.70,
    rebalance_mode: str = "annual",
    drawdown_guard: float = 0.35,
) -> CoreSatelliteRun:
    if not 0.0 < core_weight < 1.0:
        raise ValueError("core_weight must be between 0 and 1.")
    config = load_stock_rotation_config(config_path)
    if top_n is not None:
        config.portfolio.top_n = int(top_n)

    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf_full = load_price_data(config.benchmark.etf_symbol_path)
    etf_full["sma50"] = etf_full["close"].rolling(50, min_periods=50).mean()
    etf_full["sma200"] = etf_full["close"].rolling(200, min_periods=200).mean()
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
    name_map = _holding_name_map(panel)
    if name_map:
        history["holding_name"] = history["symbol"].map(name_map).fillna(history["symbol"])
    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    selections, selection_map = build_core_satellite_selection_plan(
        history=history,
        membership=membership,
        calendar=calendar,
        config=config,
        top_n=int(config.portfolio.top_n),
        rebalance_mode=rebalance_mode,
    )
    first_rebalance = min(selection_map)
    calendar = calendar[calendar >= first_rebalance].reset_index(drop=True)
    etf = etf[etf["date"] >= first_rebalance].reset_index(drop=True)
    selection_map = {date: symbols for date, symbols in selection_map.items() if date >= first_rebalance}

    static_map = {first_rebalance: selection_map[first_rebalance]}
    static_top10 = simulate_stock_dca(
        calendar=calendar,
        panel=panel,
        selection_map=static_map,
        config=config,
        static=True,
    )
    annual_top10 = simulate_stock_dca(
        calendar=calendar,
        panel=panel,
        selection_map=selection_map,
        config=config,
        static=False,
    )
    core_satellite = simulate_core_satellite(
        calendar=calendar,
        panel=panel,
        etf=etf,
        selection_map=selection_map,
        config=config,
        core_weight=core_weight,
        drawdown_guard=drawdown_guard,
    )
    adaptive_balanced = simulate_core_satellite(
        calendar=calendar,
        panel=panel,
        etf=etf,
        selection_map=selection_map,
        config=config,
        core_weight=core_weight,
        drawdown_guard=drawdown_guard,
        adaptive_profile=ADAPTIVE_BALANCED_PROFILE,
    )
    adaptive_aggressive = simulate_core_satellite(
        calendar=calendar,
        panel=panel,
        etf=etf,
        selection_map=selection_map,
        config=config,
        core_weight=core_weight,
        drawdown_guard=drawdown_guard,
        adaptive_profile=ADAPTIVE_AGGRESSIVE_PROFILE,
    )
    etf_dca = run_dca_benchmark(etf, 0, len(etf) - 1, config.backtest)
    etf_buy_hold = run_buy_hold_benchmark(etf, 0, len(etf) - 1)

    actual_run_id = run_id or f"core-satellite-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)
    selections.to_csv(run_dir / "selected_holdings.csv", index=False)
    core_satellite.actions.to_csv(run_dir / "trade_actions.csv", index=False)
    core_satellite.turnover.to_csv(run_dir / "turnover.csv", index=False)
    adaptive_balanced.actions.to_csv(
        run_dir / "trade_actions_adaptive_balanced.csv", index=False
    )
    adaptive_balanced.turnover.to_csv(
        run_dir / "turnover_adaptive_balanced.csv", index=False
    )
    adaptive_aggressive.actions.to_csv(
        run_dir / "trade_actions_adaptive_aggressive.csv", index=False
    )
    adaptive_aggressive.turnover.to_csv(
        run_dir / "turnover_adaptive_aggressive.csv", index=False
    )
    pd.DataFrame(
        {
            "date": calendar.values,
            "core_satellite_equity": core_satellite.equity.values,
            "adaptive_balanced_equity": adaptive_balanced.equity.values,
            "adaptive_aggressive_equity": adaptive_aggressive.equity.values,
            "static_top10_dca_equity": static_top10.equity.values,
            "annual_top10_dca_equity": annual_top10.equity.values,
            "etf_dca_equity": etf_dca.equity.values,
            "etf_buy_hold_norm": etf_buy_hold.values,
        }
    ).to_csv(run_dir / "equity_curve.csv", index=False)

    latest_rebalance = max(selection_map)
    latest_selected = selections[
        pd.to_datetime(selections["rebalance_date"]) == latest_rebalance
    ].sort_values("rank")
    summary = {
        "run_id": actual_run_id,
        "strategy": "etf_core_stock_satellite_drawdown_guard",
        "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
        "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "top_n": int(config.portfolio.top_n),
        "core_weight": float(core_weight),
        "satellite_weight": float(1.0 - core_weight),
        "rebalance_mode": rebalance_mode,
        "drawdown_guard": float(drawdown_guard),
        "core_satellite_metrics": core_satellite.metrics,
        "adaptive_balanced_profile": ADAPTIVE_BALANCED_PROFILE,
        "adaptive_balanced_metrics": adaptive_balanced.metrics,
        "adaptive_aggressive_profile": ADAPTIVE_AGGRESSIVE_PROFILE,
        "adaptive_aggressive_metrics": adaptive_aggressive.metrics,
        "static_top10_dca_metrics": static_top10.metrics,
        "annual_top10_dca_metrics": annual_top10.metrics,
        "etf_dca_metrics": etf_dca.metrics,
        "etf_buy_hold_return": float(etf_buy_hold.iloc[-1] - 1.0),
        "core_satellite_excess_vs_etf_dca": float(
            core_satellite.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "adaptive_balanced_excess_vs_etf_dca": float(
            adaptive_balanced.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "adaptive_aggressive_excess_vs_etf_dca": float(
            adaptive_aggressive.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "static_top10_excess_vs_etf_dca": float(
            static_top10.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "annual_top10_excess_vs_etf_dca": float(
            annual_top10.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "total_fees": {
            "core_satellite": float(core_satellite.total_fees),
            "adaptive_balanced": float(adaptive_balanced.total_fees),
            "adaptive_aggressive": float(adaptive_aggressive.total_fees),
            "static_top10_dca": float(static_top10.total_fees),
            "annual_top10_dca": float(annual_top10.total_fees),
        },
        "number_of_rebalances": int(len(selection_map)),
        "latest_rebalance_date": str(latest_rebalance.date()),
        "latest_selected_stocks": latest_selected["symbol"].tolist(),
        "first_selected_stocks": static_map[first_rebalance],
        "costs": {
            "fee_bps": float(config.backtest.fee_bps),
            "slippage_bps": float(config.backtest.slippage_bps),
            "fixed_buy_fee_egp": float(config.portfolio.fixed_buy_fee_egp),
            "share_precision": int(config.backtest.share_precision),
        },
        "artifact_paths": {
            "selected_holdings": str(run_dir / "selected_holdings.csv"),
            "equity_curve": str(run_dir / "equity_curve.csv"),
            "trade_actions": str(run_dir / "trade_actions.csv"),
            "turnover": str(run_dir / "turnover.csv"),
            "trade_actions_adaptive_balanced": str(
                run_dir / "trade_actions_adaptive_balanced.csv"
            ),
            "turnover_adaptive_balanced": str(
                run_dir / "turnover_adaptive_balanced.csv"
            ),
            "trade_actions_adaptive_aggressive": str(
                run_dir / "trade_actions_adaptive_aggressive.csv"
            ),
            "turnover_adaptive_aggressive": str(
                run_dir / "turnover_adaptive_aggressive.csv"
            ),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return CoreSatelliteRun(run_id=actual_run_id, run_dir=run_dir)
