from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from egx_research.backtest import (
    _build_trade_frame,
    _compute_metrics,
    build_contribution_schedule,
    run_buy_hold_benchmark,
    run_dca_benchmark,
    run_strategy_backtest,
)
from egx_research.data import load_price_data
from egx_research.stock_rotation import active_members_on_date, load_membership_snapshots, load_stock_panel
from egx_research.stock_rotation_config import StockRotationConfig, load_stock_rotation_config
from egx_research.strategies import build_strategy_frame
from egx_research.utils import ensure_dir, write_json


DEFAULT_PULLBACK_PARAMS = {
    "kama_len": 26,
    "kama_fast": 3,
    "kama_slow": 52,
    "cci_len": 26,
    "buy_threshold": -45.0,
    "trend_buffer_atr": 1.7,
    "atr_len": 8,
}


@dataclass
class AnnualTop10Run:
    run_id: str
    run_dir: Path


def annual_rebalance_dates(dates: pd.Series) -> list[pd.Timestamp]:
    ordered = pd.Series(pd.to_datetime(dates)).sort_values().drop_duplicates().reset_index(drop=True)
    if ordered.empty:
        return []

    schedule = [pd.Timestamp(ordered.iloc[0])]
    for _, group in ordered.groupby(ordered.dt.year):
        first_date = pd.Timestamp(group.iloc[0])
        if first_date != schedule[-1]:
            schedule.append(first_date)
    return schedule


def _selection_date_before(index: pd.Index, rebalance_date: pd.Timestamp) -> pd.Timestamp:
    prior_dates = index[index < rebalance_date]
    if len(prior_dates) == 0:
        raise ValueError(f"No selection date before rebalance date {rebalance_date.date()}.")
    return pd.Timestamp(prior_dates[-1])


def _load_pullback_params(run_id: str | None) -> tuple[str | None, dict[str, Any]]:
    if run_id is not None:
        summary_path = Path("runs") / run_id / "report_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Pullback report summary missing: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        family = summary.get("top_family")
        if family != "dca_pullback_only":
            raise ValueError(f"Run {run_id} is not a dca_pullback_only result.")
        return run_id, summary["top_params"]

    candidates = sorted(Path("runs").glob("dca-pullback-only-*/report_summary.json"))
    if not candidates:
        return None, dict(DEFAULT_PULLBACK_PARAMS)

    summary_path = candidates[-1]
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return summary_path.parent.name, summary["top_params"]


def _price_matrix(panel: pd.DataFrame, dates: pd.Series, column: str, *, fill: bool) -> pd.DataFrame:
    matrix = panel.pivot(index="date", columns="symbol", values=column).sort_index()
    matrix = matrix.reindex(pd.Index(pd.to_datetime(dates)))
    return matrix.ffill() if fill else matrix


def _close_history(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(index="date", columns="symbol", values="close").sort_index()


def _holding_names(panel: pd.DataFrame) -> dict[str, str]:
    latest = panel.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol"], keep="last")
    return dict(zip(latest["symbol"], latest["holding_name"], strict=False))


def _annual_selections(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    calendar_dates: pd.Series,
    top_n: int,
    lookback_bars: int,
) -> pd.DataFrame:
    close_history = _close_history(panel)
    trailing_returns = close_history / close_history.shift(lookback_bars) - 1.0
    name_map = _holding_names(panel)

    rows: list[dict[str, Any]] = []
    for rebalance_date in annual_rebalance_dates(calendar_dates):
        selection_date = _selection_date_before(close_history.index, rebalance_date)
        active_members = active_members_on_date(membership, rebalance_date)
        if not active_members:
            continue

        ranked = trailing_returns.loc[selection_date]
        ranked = ranked[ranked.index.isin(active_members)].dropna().sort_values(ascending=False).head(top_n)
        for rank, item in enumerate(ranked.items(), start=1):
            symbol, trailing_return = item
            rows.append(
                {
                    "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                    "selection_date": str(pd.Timestamp(selection_date).date()),
                    "rank": rank,
                    "symbol": symbol,
                    "holding_name": name_map.get(symbol, symbol),
                    "trailing_return": float(trailing_return),
                }
            )

    if not rows:
        raise ValueError("No annual selections could be built from the available stock history.")
    return pd.DataFrame(rows)


def build_annual_top10_basket(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    benchmark_dates: pd.Series,
    *,
    top_n: int,
    lookback_bars: int,
    hold_cash_when_few: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = _annual_selections(panel, membership, benchmark_dates, top_n=top_n, lookback_bars=lookback_bars)
    rebalance_schedule = pd.to_datetime(selections["rebalance_date"].drop_duplicates().tolist())

    calendar = pd.Series(pd.to_datetime(benchmark_dates)).sort_values().drop_duplicates().reset_index(drop=True)
    open_px = _price_matrix(panel, calendar, "open", fill=False)
    high_px = _price_matrix(panel, calendar, "high", fill=False)
    low_px = _price_matrix(panel, calendar, "low", fill=False)
    close_px = _price_matrix(panel, calendar, "close", fill=True)
    volume_px = _price_matrix(panel, calendar, "volume", fill=False).fillna(0.0)

    open_px = open_px.fillna(close_px)
    high_px = high_px.fillna(pd.concat([open_px, close_px]).groupby(level=0).max())
    low_px = low_px.fillna(pd.concat([open_px, close_px]).groupby(level=0).min())

    basket_segments: list[pd.DataFrame] = []
    scale = 100.0
    rebalance_dates = list(pd.to_datetime(rebalance_schedule))

    for idx, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else None
        segment_mask = calendar >= rebalance_date
        if next_rebalance is not None:
            segment_mask &= calendar < next_rebalance
        segment_dates = calendar[segment_mask]
        if segment_dates.empty:
            continue

        segment_rows = selections[selections["rebalance_date"] == str(pd.Timestamp(rebalance_date).date())]
        symbols = segment_rows["symbol"].tolist()
        selection_date = pd.Timestamp(segment_rows["selection_date"].iloc[0])
        base_close = panel[(panel["date"] == selection_date) & (panel["symbol"].isin(symbols))].set_index("symbol")["close"]
        symbols = [symbol for symbol in symbols if symbol in base_close.index]
        if not symbols:
            continue

        cash_weight = 0.0
        if hold_cash_when_few and len(symbols) < top_n:
            cash_weight = 1.0 - (len(symbols) / float(top_n))

        segment_open = open_px.loc[segment_dates, symbols].divide(base_close[symbols], axis=1).mean(axis=1)
        segment_high = high_px.loc[segment_dates, symbols].divide(base_close[symbols], axis=1).mean(axis=1)
        segment_low = low_px.loc[segment_dates, symbols].divide(base_close[symbols], axis=1).mean(axis=1)
        segment_close = close_px.loc[segment_dates, symbols].divide(base_close[symbols], axis=1).mean(axis=1)
        segment_volume = volume_px.loc[segment_dates, symbols].sum(axis=1)

        if cash_weight > 0:
            invest_weight = 1.0 - cash_weight
            segment_open = cash_weight + invest_weight * segment_open
            segment_high = cash_weight + invest_weight * segment_high
            segment_low = cash_weight + invest_weight * segment_low
            segment_close = cash_weight + invest_weight * segment_close

        frame = pd.DataFrame(
            {
                "date": segment_dates.values,
                "open": segment_open.values * scale,
                "high": segment_high.values * scale,
                "low": segment_low.values * scale,
                "close": segment_close.values * scale,
                "volume": segment_volume.values,
                "selected_count": len(symbols),
                "cash_weight": cash_weight,
            }
        )
        frame["high"] = frame[["high", "open", "close"]].max(axis=1)
        frame["low"] = frame[["low", "open", "close"]].min(axis=1)
        basket_segments.append(frame)
        scale = float(frame["close"].iloc[-1])

    if not basket_segments:
        raise ValueError("Annual basket series is empty.")

    basket = pd.concat(basket_segments, ignore_index=True)
    basket = basket.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return basket, selections


def _summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        [
            "strategy",
            "final_equity",
            "money_multiple",
            "twr_total_return",
            "cagr",
            "sharpe",
            "max_drawdown",
        ]
    ]


def generate_annual_top10_report(run_id: str) -> Path:
    run_dir = Path("runs") / run_id
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    basket = pd.read_csv(run_dir / "basket_ohlcv.csv", parse_dates=["date"])
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    selections = pd.read_csv(run_dir / "annual_selections.csv")

    curve = go.Figure()
    curve.add_trace(go.Scatter(x=equity["date"], y=equity["basket_immediate_equity"], name="Annual Top10 Immediate"))
    curve.add_trace(go.Scatter(x=equity["date"], y=equity["basket_pullback_equity"], name="Annual Top10 Pullback"))
    curve.add_trace(go.Scatter(x=equity["date"], y=equity["etf_dca_equity"], name="ETF DCA"))
    curve.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Value")

    price = go.Figure()
    price.add_trace(go.Scatter(x=basket["date"], y=basket["close"], name="Annual Top10 Basket"))
    price.update_layout(title="Synthetic Basket Close", xaxis_title="Date", yaxis_title="Close")

    metrics_table = _summary_table(
        [
            {"strategy": "Annual Top10 Immediate", **summary["basket_immediate_metrics"]},
            {"strategy": "Annual Top10 Pullback", **summary["basket_pullback_metrics"]},
            {"strategy": "ETF DCA", **summary["etf_dca_metrics"]},
        ]
    ).to_html(index=False)

    html = [
        "<html><head><title>Annual Top10 Research Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>Run {run_id}</h1>",
        (
            f"<p>Selection: annual top {summary['top_n']} active EGX members by prior "
            f"{summary['lookback_bars']}-bar return. Pullback source run: {summary['pullback_source_run_id']}.</p>"
        ),
        "<h2>Summary</h2>",
        metrics_table,
        (
            f"<p>Basket pure buy & hold return: {summary['basket_buy_hold_return']:.4f} | "
            f"Pullback excess vs annual-top10 immediate deploy: {summary['basket_pullback_excess_vs_immediate']:.4f}</p>"
        ),
        "<h2>Equity Curve</h2>",
        curve.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Basket Price</h2>",
        price.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Annual Selections</h2>",
        selections.to_html(index=False),
        "</body></html>",
    ]

    report_path = run_dir / "annual_top10_report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


def run_annual_top10_backtest(
    stock_config_path: str | Path,
    *,
    run_id: str | None = None,
    top_n: int | None = None,
    lookback_bars: int = 252,
    pullback_run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> AnnualTop10Run:
    config: StockRotationConfig = load_stock_rotation_config(stock_config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)

    if start_date is not None:
        etf = etf[etf["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date is not None:
        etf = etf[etf["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    if etf.empty:
        raise ValueError("No ETF benchmark dates remain after start/end filtering.")

    actual_top_n = int(top_n or config.portfolio.top_n)
    basket, selections = build_annual_top10_basket(
        panel,
        membership,
        etf["date"],
        top_n=actual_top_n,
        lookback_bars=lookback_bars,
        hold_cash_when_few=config.portfolio.hold_cash_when_few,
    )

    pullback_source_run_id, pullback_params = _load_pullback_params(pullback_run_id)
    pullback_frame = build_strategy_frame(basket, "dca_pullback_only", pullback_params)
    basket_pullback = run_strategy_backtest(pullback_frame, 0, len(basket) - 1, config.backtest)
    basket_immediate = run_dca_benchmark(basket, 0, len(basket) - 1, config.backtest)
    basket_buy_hold_curve = run_buy_hold_benchmark(basket, 0, len(basket) - 1)

    etf_start_idx = int(etf.index[etf["date"] >= basket["date"].iloc[0]][0])
    etf_end_idx = int(etf.index[etf["date"] <= basket["date"].iloc[-1]][-1])
    etf_dca = run_dca_benchmark(etf, etf_start_idx, etf_end_idx, config.backtest)
    etf_buy_hold_curve = run_buy_hold_benchmark(etf, etf_start_idx, etf_end_idx)

    run_id = run_id or f"annual-top10-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / run_id)

    basket.to_csv(run_dir / "basket_ohlcv.csv", index=False)
    selections.to_csv(run_dir / "annual_selections.csv", index=False)

    equity = pd.DataFrame(
        {
            "date": basket["date"].values,
            "basket_immediate_equity": basket_immediate.equity.values,
            "basket_pullback_equity": basket_pullback.equity.values,
            "basket_buy_hold_norm": basket_buy_hold_curve.values,
            "etf_dca_equity": etf_dca.equity.values[: len(basket)],
            "etf_buy_hold_norm": etf_buy_hold_curve.values[: len(basket)],
        }
    )
    equity.to_csv(run_dir / "equity_curve.csv", index=False)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "stock_config_path": str(stock_config_path),
        "benchmark_path": config.benchmark.etf_symbol_path,
        "top_n": actual_top_n,
        "lookback_bars": lookback_bars,
        "pullback_source_run_id": pullback_source_run_id,
        "pullback_params": pullback_params,
        "start_date": str(pd.Timestamp(basket["date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(basket["date"].iloc[-1]).date()),
    }
    write_json(run_dir / "manifest.json", manifest)

    summary = {
        "start_date": str(pd.Timestamp(basket["date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(basket["date"].iloc[-1]).date()),
        "top_n": actual_top_n,
        "lookback_bars": lookback_bars,
        "selection_method": "annual_top_n_by_prior_return",
        "pullback_source_run_id": pullback_source_run_id,
        "pullback_params": pullback_params,
        "basket_immediate_metrics": basket_immediate.metrics,
        "basket_pullback_metrics": basket_pullback.metrics,
        "basket_pullback_excess_vs_immediate": (
            basket_pullback.metrics["twr_total_return"] - basket_immediate.metrics["twr_total_return"]
        ),
        "basket_buy_hold_return": float(basket_buy_hold_curve.iloc[-1] - 1.0),
        "etf_dca_metrics": etf_dca.metrics,
        "etf_buy_hold_return": float(etf_buy_hold_curve.iloc[-1] - 1.0),
        "selection_years": int(selections["rebalance_date"].nunique()),
    }
    write_json(run_dir / "summary.json", summary)

    generate_annual_top10_report(run_id)
    return AnnualTop10Run(run_id=run_id, run_dir=run_dir)


def audit_selection_coverage(panel: pd.DataFrame, selections: pd.DataFrame, benchmark_dates: pd.Series) -> pd.DataFrame:
    calendar = pd.Series(pd.to_datetime(benchmark_dates)).sort_values().drop_duplicates().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    rebalance_dates = sorted(pd.to_datetime(selections["rebalance_date"].drop_duplicates()))

    for index, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else calendar.iloc[-1] + pd.Timedelta(days=1)
        segment_dates = calendar[(calendar >= rebalance_date) & (calendar < next_rebalance)]
        segment_set = set(segment_dates)
        segment_rows = selections[selections["rebalance_date"] == str(pd.Timestamp(rebalance_date).date())]
        for row in segment_rows.itertuples(index=False):
            present_dates = set(panel.loc[(panel["symbol"] == row.symbol) & (panel["date"].isin(list(segment_set))), "date"])
            rows.append(
                {
                    "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                    "symbol": row.symbol,
                    "days": len(segment_set),
                    "present_days": len(present_dates),
                    "missing_days": len(segment_set - present_dates),
                    "coverage": 0.0 if not segment_set else float(len(present_dates) / len(segment_set)),
                }
            )

    return pd.DataFrame(rows)


def _round_shares(value: float, precision: int) -> float:
    if precision <= 0:
        return float(int(max(0.0, value)))
    factor = 10**precision
    return float(int(max(0.0, value) * factor) / factor)


def _signal_matrix(panel: pd.DataFrame, calendar: pd.Series, pullback_params: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        price = group[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        strategy = build_strategy_frame(price, "dca_pullback_only", pullback_params)
        frames.append(
            pd.DataFrame(
                {
                    "date": strategy["date"].values,
                    "symbol": symbol,
                    "deploy_fraction": strategy["deploy_fraction"].values,
                }
            )
        )

    signals = pd.concat(frames, ignore_index=True).pivot(index="date", columns="symbol", values="deploy_fraction").sort_index()
    return signals.reindex(pd.Index(pd.to_datetime(calendar))).ffill().fillna(0.0)


def _position_value(positions: dict[str, float], price_row: pd.Series) -> float:
    total = 0.0
    for symbol, shares in positions.items():
        price = price_row.get(symbol)
        if pd.isna(price):
            continue
        total += float(shares) * float(price)
    return total


def _sell_positions(
    date: pd.Timestamp,
    symbols: list[str],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    actions: list[dict[str, Any]],
    reason: str,
) -> float:
    for symbol in symbols:
        shares = float(positions.get(symbol, 0.0))
        if shares <= 0:
            continue
        raw_open = open_row.get(symbol)
        if pd.isna(raw_open):
            continue
        fill_price = float(raw_open) * (1.0 - slippage_rate)
        gross = shares * fill_price
        fee = gross * fee_rate
        cash += gross - fee
        actions.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "symbol": symbol,
                "action": "SELL",
                "shares": shares,
                "price": fill_price,
                "value": gross,
                "fee": fee,
                "reason": reason,
            }
        )
        positions.pop(symbol, None)
    return cash


def _buy_equal_split(
    date: pd.Timestamp,
    symbols: list[str],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    fixed_buy_fee: float,
    share_precision: int,
    actions: list[dict[str, Any]],
    reason: str,
) -> float:
    tradable = [symbol for symbol in symbols if not pd.isna(open_row.get(symbol))]
    if not tradable or cash <= 0:
        return cash

    starting_cash = cash
    per_symbol_budget = starting_cash / float(len(tradable))
    for symbol in tradable:
        raw_open = open_row.get(symbol)
        fill_price = float(raw_open) * (1.0 + slippage_rate)
        if fill_price <= 0:
            continue

        max_budget = min(per_symbol_budget, cash)
        if max_budget <= fixed_buy_fee:
            continue

        max_shares_budget = _round_shares((max_budget - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision)
        max_shares_cash = _round_shares((cash - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision)
        buy_shares = min(max_shares_budget, max_shares_cash)
        if buy_shares <= 0:
            continue

        gross = buy_shares * fill_price
        fee = gross * fee_rate + fixed_buy_fee
        cash -= gross + fee
        positions[symbol] = float(positions.get(symbol, 0.0)) + buy_shares
        actions.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "symbol": symbol,
                "action": "BUY",
                "shares": buy_shares,
                "price": fill_price,
                "value": gross,
                "fee": fee,
                "reason": reason,
            }
        )
    return cash


def _rebalance_equal_weight(
    date: pd.Timestamp,
    selected_symbols: list[str],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    close_row: pd.Series,
    top_n: int,
    fee_rate: float,
    slippage_rate: float,
    fixed_buy_fee: float,
    share_precision: int,
    actions: list[dict[str, Any]],
) -> float:
    if not selected_symbols or top_n <= 0:
        return cash

    tradable_selected = [symbol for symbol in selected_symbols if not pd.isna(open_row.get(symbol))]
    if not tradable_selected:
        return cash

    valuation_row = open_row.fillna(close_row)
    total_equity = cash + _position_value(positions, valuation_row)
    target_value = total_equity / float(top_n)

    for symbol in tradable_selected:
        shares = float(positions.get(symbol, 0.0))
        fill_price = float(open_row.get(symbol))
        current_value = shares * fill_price
        delta_value = current_value - target_value
        if delta_value <= 0:
            continue

        sell_shares = _round_shares(delta_value / (fill_price * (1.0 - slippage_rate)), share_precision)
        sell_shares = min(sell_shares, shares)
        if sell_shares <= 0:
            continue

        trade_price = fill_price * (1.0 - slippage_rate)
        gross = sell_shares * trade_price
        fee = gross * fee_rate
        cash += gross - fee
        remaining = shares - sell_shares
        if remaining > 0:
            positions[symbol] = remaining
        else:
            positions.pop(symbol, None)
        actions.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "symbol": symbol,
                "action": "SELL",
                "shares": sell_shares,
                "price": trade_price,
                "value": gross,
                "fee": fee,
                "reason": "annual_rebalance_trim",
            }
        )

    for symbol in tradable_selected:
        fill_price = float(open_row.get(symbol)) * (1.0 + slippage_rate)
        if fill_price <= 0:
            continue

        shares = float(positions.get(symbol, 0.0))
        current_value = shares * float(open_row.get(symbol))
        delta_value = target_value - current_value
        if delta_value <= fixed_buy_fee:
            continue

        affordable_target = _round_shares((delta_value - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision)
        affordable_cash = _round_shares((cash - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision)
        buy_shares = min(affordable_target, affordable_cash)
        if buy_shares <= 0:
            continue

        gross = buy_shares * fill_price
        fee = gross * fee_rate + fixed_buy_fee
        cash -= gross + fee
        positions[symbol] = shares + buy_shares
        actions.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "symbol": symbol,
                "action": "BUY",
                "shares": buy_shares,
                "price": fill_price,
                "value": gross,
                "fee": fee,
                "reason": "annual_rebalance_buy",
            }
        )

    return cash


def _simulate_stock_strategy(
    *,
    mode: str,
    calendar: pd.Series,
    selections: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    signals: pd.DataFrame,
    config: StockRotationConfig,
    top_n: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, float]]:
    fee_rate = config.backtest.fee_bps / 10_000
    slippage_rate = config.backtest.slippage_bps / 10_000
    fixed_buy_fee = config.portfolio.fixed_buy_fee_egp
    share_precision = config.backtest.share_precision

    selection_map = {
        pd.Timestamp(date): rows["symbol"].tolist()
        for date, rows in selections.groupby(pd.to_datetime(selections["rebalance_date"]))
    }
    rebalance_dates = set(selection_map)
    signal_prev = signals.shift(1).fillna(0.0)

    contributions = build_contribution_schedule(
        calendar,
        0,
        len(calendar) - 1,
        initial_cash=config.backtest.initial_cash,
        monthly_contribution=config.backtest.monthly_contribution,
    )
    equity = pd.Series(index=range(len(calendar)), dtype=float)
    flows = pd.Series(0.0, index=range(len(calendar)), dtype=float)
    actions: list[dict[str, Any]] = []

    cash = 0.0
    positions: dict[str, float] = {}
    current_selection: list[str] = []

    for i, date in enumerate(calendar):
        date = pd.Timestamp(date)
        cash += float(contributions.iloc[i])
        flows.iloc[i] += float(contributions.iloc[i])

        open_row = open_px.loc[date]
        close_row = close_px.loc[date]

        if date in rebalance_dates:
            current_selection = selection_map[date]

        if mode == "pullback" and date in rebalance_dates:
            undesired = sorted(list(positions))
        else:
            undesired = sorted([symbol for symbol in positions if symbol not in set(current_selection)])
        cash = _sell_positions(
            date,
            undesired,
            positions,
            cash,
            open_row,
            fee_rate,
            slippage_rate,
            actions,
            reason="annual_rebalance_exit",
        )

        if mode == "immediate" and date in rebalance_dates:
            cash = _rebalance_equal_weight(
                date,
                current_selection,
                positions,
                cash,
                open_row,
                close_row,
                top_n,
                fee_rate,
                slippage_rate,
                fixed_buy_fee,
                share_precision,
                actions,
            )

        if mode == "immediate":
            if date not in rebalance_dates:
                cash = _buy_equal_split(
                    date,
                    current_selection,
                    positions,
                    cash,
                    open_row,
                    fee_rate,
                    slippage_rate,
                    fixed_buy_fee,
                    share_precision,
                    actions,
                    reason="cash_deploy_current_basket",
                )
        elif mode == "pullback":
            active_signals = []
            if i > 0 and current_selection:
                signal_row = signal_prev.loc[date]
                active_signals = [
                    symbol
                    for symbol in current_selection
                    if symbol in signal_row.index and float(signal_row.get(symbol, 0.0)) > 0.0
                ]
            cash = _buy_equal_split(
                date,
                active_signals,
                positions,
                cash,
                open_row,
                fee_rate,
                slippage_rate,
                fixed_buy_fee,
                share_precision,
                actions,
                reason="pullback_signal_buy",
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        equity.iloc[i] = cash + _position_value(positions, close_row)

    trade_frame = _build_trade_frame([])
    metrics = _compute_metrics(equity.copy(), flows.copy(), trade_frame)
    return equity, flows, pd.DataFrame(actions), metrics


def generate_annual_top10_stock_report(run_id: str) -> Path:
    run_dir = Path("runs") / run_id
    with (run_dir / "summary_stock.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    equity = pd.read_csv(run_dir / "equity_curve_stock.csv", parse_dates=["date"])
    selections = pd.read_csv(run_dir / "annual_selections.csv")
    coverage = pd.read_csv(run_dir / "selection_coverage.csv")

    curve = go.Figure()
    curve.add_trace(go.Scatter(x=equity["date"], y=equity["stock_immediate_equity"], name="Stock Immediate"))
    curve.add_trace(go.Scatter(x=equity["date"], y=equity["stock_pullback_equity"], name="Stock Pullback"))
    curve.add_trace(go.Scatter(x=equity["date"], y=equity["etf_dca_equity"], name="ETF DCA"))
    curve.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Value")

    metrics_table = _summary_table(
        [
            {"strategy": "Stock Immediate", **summary["stock_immediate_metrics"]},
            {"strategy": "Stock Pullback", **summary["stock_pullback_metrics"]},
            {"strategy": "ETF DCA", **summary["etf_dca_metrics"]},
        ]
    ).to_html(index=False)

    coverage_table = coverage.sort_values(["coverage", "missing_days", "rebalance_date", "symbol"]).head(20).to_html(index=False)

    html = [
        "<html><head><title>Annual Top10 Stock Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>Run {run_id}</h1>",
        (
            f"<p>Selection: annual top {summary['top_n']} active EGX members by prior "
            f"{summary['lookback_bars']}-bar return. Membership last verified date: {summary['membership_last_effective_date']}.</p>"
        ),
        (
            f"<p>Worst selected-leg coverage: {summary['coverage_min']:.4f} | "
            f"Selected legs below 95% coverage: {summary['coverage_under_95_count']}.</p>"
        ),
        "<h2>Summary</h2>",
        metrics_table,
        "<h2>Equity Curve</h2>",
        curve.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Coverage Audit (worst 20)</h2>",
        coverage_table,
        "<h2>Annual Selections</h2>",
        selections.to_html(index=False),
        "</body></html>",
    ]

    report_path = run_dir / "annual_top10_stock_report.html"
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


def run_annual_top10_stock_backtest(
    stock_config_path: str | Path,
    *,
    run_id: str | None = None,
    top_n: int | None = None,
    lookback_bars: int = 252,
    pullback_run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> AnnualTop10Run:
    config: StockRotationConfig = load_stock_rotation_config(stock_config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)

    if start_date is not None:
        etf = etf[etf["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date is not None:
        etf = etf[etf["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    if etf.empty:
        raise ValueError("No ETF benchmark dates remain after start/end filtering.")

    actual_top_n = int(top_n or config.portfolio.top_n)
    calendar = pd.Series(pd.to_datetime(etf["date"])).sort_values().drop_duplicates().reset_index(drop=True)
    selections = _annual_selections(panel, membership, calendar, top_n=actual_top_n, lookback_bars=lookback_bars)
    coverage = audit_selection_coverage(panel, selections, calendar)
    pullback_source_run_id, pullback_params = _load_pullback_params(pullback_run_id)

    open_px = _price_matrix(panel, calendar, "open", fill=False)
    close_px = _price_matrix(panel, calendar, "close", fill=True)
    signals = _signal_matrix(panel, calendar, pullback_params)

    stock_immediate_equity, _, stock_immediate_actions, stock_immediate_metrics = _simulate_stock_strategy(
        mode="immediate",
        calendar=calendar,
        selections=selections,
        open_px=open_px,
        close_px=close_px,
        signals=signals,
        config=config,
        top_n=actual_top_n,
    )
    stock_pullback_equity, _, stock_pullback_actions, stock_pullback_metrics = _simulate_stock_strategy(
        mode="pullback",
        calendar=calendar,
        selections=selections,
        open_px=open_px,
        close_px=close_px,
        signals=signals,
        config=config,
        top_n=actual_top_n,
    )

    etf_dca = run_dca_benchmark(etf, 0, len(etf) - 1, config.backtest)
    etf_buy_hold_curve = run_buy_hold_benchmark(etf, 0, len(etf) - 1)

    run_id = run_id or f"annual-top10-stock-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / run_id)

    selections.to_csv(run_dir / "annual_selections.csv", index=False)
    coverage.to_csv(run_dir / "selection_coverage.csv", index=False)
    stock_immediate_actions.to_csv(run_dir / "stock_immediate_actions.csv", index=False)
    stock_pullback_actions.to_csv(run_dir / "stock_pullback_actions.csv", index=False)

    equity = pd.DataFrame(
        {
            "date": calendar.values,
            "stock_immediate_equity": stock_immediate_equity.values,
            "stock_pullback_equity": stock_pullback_equity.values,
            "etf_dca_equity": etf_dca.equity.values,
            "etf_buy_hold_norm": etf_buy_hold_curve.values,
        }
    )
    equity.to_csv(run_dir / "equity_curve_stock.csv", index=False)

    summary = {
        "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
        "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "top_n": actual_top_n,
        "lookback_bars": lookback_bars,
        "selection_method": "annual_top_n_by_prior_return",
        "pullback_source_run_id": pullback_source_run_id,
        "pullback_params": pullback_params,
        "stock_immediate_metrics": stock_immediate_metrics,
        "stock_pullback_metrics": stock_pullback_metrics,
        "stock_pullback_excess_vs_immediate": (
            stock_pullback_metrics["twr_total_return"] - stock_immediate_metrics["twr_total_return"]
        ),
        "etf_dca_metrics": etf_dca.metrics,
        "etf_buy_hold_return": float(etf_buy_hold_curve.iloc[-1] - 1.0),
        "selection_years": int(selections["rebalance_date"].nunique()),
        "coverage_min": float(coverage["coverage"].min()),
        "coverage_under_95_count": int((coverage["coverage"] < 0.95).sum()),
        "membership_last_effective_date": str(pd.Timestamp(membership["effective_date"].max()).date()),
    }
    write_json(run_dir / "summary_stock.json", summary)

    generate_annual_top10_stock_report(run_id)
    return AnnualTop10Run(run_id=run_id, run_dir=run_dir)
