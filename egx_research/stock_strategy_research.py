from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from egx_research.backtest import (
    _build_trade_frame,
    _compute_metrics,
    _round_shares,
    build_contribution_schedule,
    run_dca_benchmark,
)
from egx_research.data import load_price_data
from egx_research.indicators import atr, bollinger_bands, cci, kama, rsi, sma
from egx_research.stock_rotation import (
    _build_price_matrix,
    active_members_on_date,
    load_membership_snapshots,
    load_stock_panel,
)
from egx_research.stock_rotation_config import (
    StockRotationConfig,
    load_stock_rotation_config,
)
from egx_research.stock_rotation_reporting import generate_stock_strategy_report
from egx_research.utils import ensure_dir, write_json


STOCK_STRATEGY_FAMILIES = ["breakout", "rebound", "news_event_rules"]


@dataclass
class StockStrategyResearchRun:
    run_id: str
    run_dir: Path


@dataclass
class StrategySimulation:
    equity: pd.Series
    flows: pd.Series
    metrics: dict[str, float]
    actions: pd.DataFrame
    positions: pd.DataFrame
    turnover: pd.DataFrame
    latest_setups: pd.DataFrame
    total_fees: float


def load_disclosure_events(config: StockRotationConfig) -> pd.DataFrame:
    path = Path(config.storage.root_dir) / config.storage.disclosure_events_filename
    if not path.exists():
        return pd.DataFrame(
            columns=["symbol", "event_date", "event_class", "title", "summary"]
        )
    frame = pd.read_csv(path, parse_dates=["event_date"])
    if frame.empty:
        return pd.DataFrame(
            columns=["symbol", "event_date", "event_class", "title", "summary"]
        )
    frame["symbol"] = frame["symbol"].astype(str)
    frame["event_class"] = frame["event_class"].fillna("other").astype(str)
    return frame.sort_values(["symbol", "event_date"]).reset_index(drop=True)


def _scan_dates(calendar: pd.Series, mode: str) -> set[pd.Timestamp]:
    ordered = pd.Series(pd.to_datetime(calendar)).sort_values().drop_duplicates()
    if mode == "weekly":
        return set(ordered.groupby([ordered.dt.year, ordered.dt.isocalendar().week]).first())
    if mode == "monthly":
        return set(ordered.groupby([ordered.dt.year, ordered.dt.month]).first())
    raise ValueError(f"Unsupported scan mode: {mode}")


def _sample_params(trial: optuna.trial.Trial, family: str) -> dict[str, Any]:
    common = {
        "scan_mode": trial.suggest_categorical("scan_mode", ["weekly", "monthly"]),
        "top_n": trial.suggest_int("top_n", 4, 10),
        "max_positions": trial.suggest_int("max_positions", 3, 8),
        "atr_len": trial.suggest_int("atr_len", 7, 21),
        "target_atr": trial.suggest_float("target_atr", 2.0, 6.0),
        "stop_atr": trial.suggest_float("stop_atr", 1.0, 3.5),
        "trail_atr": trial.suggest_float("trail_atr", 1.5, 5.0),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 20, 126),
    }
    if family == "breakout":
        common.update(
            {
                "lookback": trial.suggest_int("lookback", 20, 90),
                "trend_len": trial.suggest_int("trend_len", 50, 160),
                "volume_short": trial.suggest_int("volume_short", 10, 30),
                "volume_long": trial.suggest_int("volume_long", 40, 90),
                "volume_ratio": trial.suggest_float("volume_ratio", 1.05, 2.0),
            }
        )
    elif family == "rebound":
        common.update(
            {
                "trend_len": trial.suggest_int("trend_len", 50, 160),
                "rsi_len": trial.suggest_int("rsi_len", 5, 21),
                "rsi_entry": trial.suggest_float("rsi_entry", 25.0, 45.0),
                "rsi_reclaim": trial.suggest_float("rsi_reclaim", 35.0, 55.0),
                "bb_len": trial.suggest_int("bb_len", 15, 40),
                "bb_std": trial.suggest_float("bb_std", 1.5, 2.5),
                "cci_len": trial.suggest_int("cci_len", 10, 40),
                "cci_reclaim": trial.suggest_float("cci_reclaim", -120.0, -20.0),
            }
        )
    elif family == "news_event_rules":
        common.update(
            {
                "event_window_days": trial.suggest_int("event_window_days", 7, 60),
                "confirm_lookback": trial.suggest_int("confirm_lookback", 10, 45),
                "trend_len": trial.suggest_int("trend_len", 40, 140),
                "min_event_score": trial.suggest_float("min_event_score", 0.5, 2.5),
                "volume_short": trial.suggest_int("volume_short", 10, 30),
                "volume_long": trial.suggest_int("volume_long", 40, 90),
            }
        )
    else:
        raise ValueError(f"Unsupported family: {family}")
    common["max_positions"] = min(int(common["max_positions"]), int(common["top_n"]))
    return common


def _default_params(family: str) -> dict[str, Any]:
    base = {
        "scan_mode": "weekly",
        "top_n": 8,
        "max_positions": 5,
        "atr_len": 14,
        "target_atr": 3.5,
        "stop_atr": 2.0,
        "trail_atr": 3.0,
        "max_hold_bars": 63,
    }
    if family == "breakout":
        return {
            **base,
            "lookback": 55,
            "trend_len": 100,
            "volume_short": 20,
            "volume_long": 63,
            "volume_ratio": 1.2,
        }
    if family == "rebound":
        return {
            **base,
            "trend_len": 100,
            "rsi_len": 14,
            "rsi_entry": 35.0,
            "rsi_reclaim": 45.0,
            "bb_len": 20,
            "bb_std": 2.0,
            "cci_len": 20,
            "cci_reclaim": -80.0,
        }
    if family == "news_event_rules":
        return {
            **base,
            "event_window_days": 30,
            "confirm_lookback": 20,
            "trend_len": 80,
            "min_event_score": 1.0,
            "volume_short": 20,
            "volume_long": 63,
        }
    raise ValueError(f"Unsupported family: {family}")


def _event_features(
    dates: pd.Series, symbol_events: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date")
    weights = {
        "dividend": 1.5,
        "earnings": 1.0,
        "capital_action": -1.0,
        "governance": 0.25,
        "other": 0.0,
    }
    if symbol_events.empty:
        base["event_score"] = 0.0
        base["event_count"] = 0.0
        return base

    events = symbol_events.copy()
    events["event_weight"] = events["event_class"].map(weights).fillna(0.0)
    daily = (
        events.groupby("event_date", as_index=False)
        .agg(event_count=("event_weight", "size"), event_score=("event_weight", "sum"))
        .sort_values("event_date")
    )
    daily["cum_count"] = daily["event_count"].cumsum()
    daily["cum_score"] = daily["event_score"].cumsum()

    latest = pd.merge_asof(
        base,
        daily[["event_date", "cum_count", "cum_score"]],
        left_on="date",
        right_on="event_date",
        direction="backward",
    )
    cutoff = pd.DataFrame({"cutoff": base["date"] - pd.Timedelta(days=int(window_days))})
    prior = pd.merge_asof(
        cutoff,
        daily[["event_date", "cum_count", "cum_score"]],
        left_on="cutoff",
        right_on="event_date",
        direction="backward",
    )
    base["event_count"] = latest["cum_count"].fillna(0.0) - prior["cum_count"].fillna(0.0)
    base["event_score"] = latest["cum_score"].fillna(0.0) - prior["cum_score"].fillna(0.0)
    return base


def build_strategy_feature_panel(
    panel: pd.DataFrame,
    *,
    family: str,
    params: dict[str, Any],
    disclosure_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        frame = group.sort_values("date").copy().reset_index(drop=True)
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = frame["volume"]
        frame["atr"] = atr(high, low, close, int(params["atr_len"]))
        frame["ret_21"] = close / close.shift(21) - 1.0
        frame["ret_63"] = close / close.shift(63) - 1.0
        frame["daily_value"] = close * volume
        frame["median_daily_value_63"] = frame["daily_value"].rolling(
            63, min_periods=21
        ).median()
        frame["median_daily_volume_63"] = volume.rolling(63, min_periods=21).median()
        frame["kama"] = kama(close, 26, 3, 52)
        frame["trend_ma"] = sma(close, int(params.get("trend_len", 100)))
        frame["trend_ok"] = (close > frame["trend_ma"]) & (frame["kama"] >= frame["kama"].shift(1))
        frame["signal_fail"] = close < frame["trend_ma"]

        if family == "breakout":
            lookback = int(params["lookback"])
            prior_high = high.shift(1).rolling(lookback, min_periods=lookback).max()
            volume_ratio = (
                volume.rolling(int(params["volume_short"]), min_periods=int(params["volume_short"])).mean()
                / volume.rolling(int(params["volume_long"]), min_periods=int(params["volume_long"])).mean().replace(0.0, np.nan)
            )
            frame["entry_signal"] = (
                (close > prior_high)
                & frame["trend_ok"]
                & (volume_ratio >= float(params["volume_ratio"]))
            )
            frame["rank_score"] = (
                (close / prior_high.replace(0.0, np.nan) - 1.0).fillna(0.0)
                + frame["ret_63"].fillna(0.0)
                + volume_ratio.fillna(0.0) * 0.1
            )
        elif family == "rebound":
            lower, mid, _ = bollinger_bands(
                close, int(params["bb_len"]), float(params["bb_std"])
            )
            rsi_line = rsi(close, int(params["rsi_len"]))
            cci_line = cci(high, low, close, int(params["cci_len"]))
            was_oversold = (
                (rsi_line.shift(1).rolling(5, min_periods=1).min() <= float(params["rsi_entry"]))
                | (close.shift(1).rolling(5, min_periods=1).min() <= lower.shift(1))
            )
            frame["entry_signal"] = (
                was_oversold
                & frame["trend_ok"]
                & (rsi_line >= float(params["rsi_reclaim"]))
                & (cci_line >= float(params["cci_reclaim"]))
                & (close > mid)
            )
            frame["rank_score"] = (
                (100.0 - rsi_line).fillna(0.0) / 100.0
                + (close / frame["trend_ma"].replace(0.0, np.nan) - 1.0).fillna(0.0)
                + frame["ret_63"].fillna(0.0)
            )
        elif family == "news_event_rules":
            events = (
                pd.DataFrame()
                if disclosure_events is None or disclosure_events.empty
                else disclosure_events[disclosure_events["symbol"] == str(symbol)]
            )
            event_frame = _event_features(
                frame["date"], events, int(params["event_window_days"])
            )
            frame = frame.merge(event_frame, on="date", how="left")
            prior_high = high.shift(1).rolling(
                int(params["confirm_lookback"]),
                min_periods=int(params["confirm_lookback"]),
            ).max()
            volume_ratio = (
                volume.rolling(int(params["volume_short"]), min_periods=int(params["volume_short"])).mean()
                / volume.rolling(int(params["volume_long"]), min_periods=int(params["volume_long"])).mean().replace(0.0, np.nan)
            )
            frame["entry_signal"] = (
                (frame["event_score"] >= float(params["min_event_score"]))
                & (close > prior_high)
                & (close > frame["trend_ma"])
                & (volume_ratio >= 1.0)
            )
            frame["rank_score"] = (
                frame["event_score"].fillna(0.0)
                + frame["ret_21"].fillna(0.0)
                + volume_ratio.fillna(0.0) * 0.05
            )
            frame["signal_fail"] = (frame["event_score"].fillna(0.0) <= 0.0) | (
                close < frame["trend_ma"]
            )
        else:
            raise ValueError(f"Unsupported family: {family}")

        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _latest_before(features: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    prior = features[features["date"] < pd.Timestamp(date)].copy()
    if prior.empty:
        return prior
    return (
        prior.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["symbol"], keep="last")
        .reset_index(drop=True)
    )


def _sell(
    *,
    date: pd.Timestamp,
    symbol: str,
    position: dict[str, Any],
    cash: float,
    open_px: pd.DataFrame,
    fee_rate: float,
    slippage_rate: float,
    fixed_sell_fee: float,
    reason: str,
    actions: list[dict[str, Any]],
    closed_trades: list[dict[str, Any]],
    shares_to_sell: float | None = None,
) -> tuple[float, float, float, bool]:
    raw_open = open_px.at[date, symbol] if symbol in open_px.columns else np.nan
    if pd.isna(raw_open):
        return cash, 0.0, 0.0, False
    original_shares = float(position["shares"])
    shares = original_shares if shares_to_sell is None else min(original_shares, float(shares_to_sell))
    if shares <= 0:
        return cash, 0.0, 0.0, False
    fill = float(raw_open) * (1.0 - slippage_rate)
    gross = shares * fill
    fee = gross * fee_rate + float(fixed_sell_fee)
    cash += gross - fee
    sold_fraction = shares / max(original_shares, 1e-9)
    entry_value = float(position["entry_value"]) * sold_fraction
    entry_fee = float(position["entry_fee"]) * sold_fraction
    pnl = gross - fee - entry_value - entry_fee
    actions.append(
        {
            "date": str(date.date()),
            "symbol": symbol,
            "action": "SELL",
            "shares": shares,
            "price": fill,
            "value": gross,
            "fee": fee,
            "reason": reason,
        }
    )
    closed_trades.append(
        {
            "entry_date": position["entry_date"],
            "exit_date": str(date.date()),
            "entry_price": float(position["entry_price"]),
            "exit_price": fill,
            "shares": shares,
            "bars_held": int(position["bars_held"]),
            "pnl": pnl,
            "pnl_pct": pnl / max(entry_value + entry_fee, 1e-9),
        }
    )
    sold_all = shares >= original_shares - 1e-9
    if not sold_all:
        position["shares"] = original_shares - shares
        position["entry_value"] = float(position["entry_value"]) - entry_value
        position["entry_fee"] = float(position["entry_fee"]) - entry_fee
    return cash, gross, fee, sold_all


def simulate_event_driven_strategy(
    *,
    panel: pd.DataFrame,
    features: pd.DataFrame,
    calendar: pd.Series,
    membership: pd.DataFrame,
    config: StockRotationConfig,
    family: str,
    params: dict[str, Any],
    start_idx: int = 0,
    market_filter: pd.Series | None = None,
) -> StrategySimulation:
    dates = pd.Series(pd.to_datetime(calendar)).reset_index(drop=True)
    open_px = _build_price_matrix(panel, dates, "open", fill=False)
    close_px = _build_price_matrix(panel, dates, "close", fill=True)
    feature_by_date = {
        pd.Timestamp(date): rows.set_index("symbol")
        for date, rows in features.groupby("date", sort=True)
    }

    fee_rate = float(config.backtest.fee_bps) / 10_000
    slippage_rate = float(config.backtest.slippage_bps) / 10_000
    fixed_fee = float(config.portfolio.fixed_buy_fee_egp)
    fixed_sell_fee = float(params.get("fixed_sell_fee_egp", 0.0) or 0.0)
    share_precision = int(config.backtest.share_precision)
    max_positions = int(params["max_positions"])
    top_n = int(params["top_n"])
    scan_dates = _scan_dates(dates, str(params.get("scan_mode", "weekly")))
    min_median_value = float(params.get("min_median_daily_value_egp", 0.0) or 0.0)
    min_median_volume = float(params.get("min_median_daily_volume", 0.0) or 0.0)
    max_position_weight = float(params.get("max_position_weight", 1.0) or 1.0)
    risk_per_trade_pct = float(params.get("risk_per_trade_pct", 0.0) or 0.0)
    partial_exit_fraction = float(params.get("partial_exit_fraction", 0.0) or 0.0)
    partial_target_atr = float(params.get("partial_target_atr", 0.0) or 0.0)
    cooldown_bars_after_stop = int(params.get("cooldown_bars_after_stop", 0) or 0)
    portfolio_drawdown_pause_pct = float(
        params.get("portfolio_drawdown_pause_pct", 0.0) or 0.0
    )
    max_sector_positions = int(params.get("max_sector_positions", 0) or 0)
    exit_on_market_filter_fail = bool(params.get("exit_on_market_filter_fail", False))
    market_filter_fail_min_hold_bars = int(
        params.get("market_filter_fail_min_hold_bars", 1) or 1
    )
    market_fail_trail_atr = float(params.get("market_fail_trail_atr", 0.0) or 0.0)
    min_position_size_mult = float(params.get("min_position_size_mult", 0.0) or 0.0)
    max_position_size_mult = float(params.get("max_position_size_mult", 2.0) or 2.0)
    move_stop_to_entry_after_partial = bool(
        params.get("move_stop_to_entry_after_partial", False)
    )
    if market_filter is not None:
        aligned_market_filter = pd.Series(market_filter).copy()
        if len(aligned_market_filter) == len(dates) and not isinstance(
            aligned_market_filter.index, pd.DatetimeIndex
        ):
            aligned_market_filter.index = pd.to_datetime(dates)
        else:
            aligned_market_filter.index = pd.to_datetime(aligned_market_filter.index)
        market_ok_by_date = {
            pd.Timestamp(date): bool(value)
            for date, value in aligned_market_filter.dropna().items()
        }
    else:
        market_ok_by_date = None
    contributions = build_contribution_schedule(
        dates,
        start_idx,
        len(dates) - 1,
        config.backtest.initial_cash,
        config.backtest.monthly_contribution,
        monthly_day_offset=config.backtest.monthly_contribution_day_offset,
    )

    cash = 0.0
    positions: dict[str, dict[str, Any]] = {}
    active_rank: list[str] = []
    equity = pd.Series(0.0, index=range(len(dates)), dtype=float)
    flows = pd.Series(0.0, index=range(len(dates)), dtype=float)
    actions: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []
    latest_setups = pd.DataFrame()
    total_fees = 0.0
    cooldown_until: dict[str, int] = {}
    peak_equity_so_far = 0.0

    for i, date in enumerate(dates):
        if i < start_idx:
            continue
        cash += float(contributions.iloc[i])
        flows.iloc[i] += float(contributions.iloc[i])
        traded = 0.0
        day_fees = 0.0

        prev_date = dates.iloc[i - 1] if i > 0 else None
        if i > 0:
            peak_equity_so_far = max(peak_equity_so_far, float(equity.iloc[i - 1]))
        prev_rows = (
            feature_by_date.get(pd.Timestamp(prev_date), pd.DataFrame())
            if prev_date is not None
            else pd.DataFrame()
        )

        for symbol in list(positions):
            position = positions[symbol]
            if symbol in prev_rows.index:
                prev = prev_rows.loc[symbol]
                position["bars_held"] = int(position["bars_held"]) + 1
                position["peak_close"] = max(
                    float(position["peak_close"]), float(prev["close"])
                )
                prev_market_ok = (
                    True
                    if market_ok_by_date is None
                    else bool(market_ok_by_date.get(pd.Timestamp(prev_date), True))
                )
                trail_atr = float(params["trail_atr"])
                if not prev_market_ok and market_fail_trail_atr > 0.0:
                    trail_atr = min(trail_atr, market_fail_trail_atr)
                trail_stop = float(position["peak_close"]) - trail_atr * float(prev["atr"])
                hard_stop = float(position["stop_price"])
                stop_price = max(hard_stop, trail_stop)
                reason = None
                if float(prev["high"]) >= float(position["target_price"]):
                    reason = "target"
                elif float(prev["low"]) <= stop_price:
                    reason = "stop"
                elif (
                    exit_on_market_filter_fail
                    and not prev_market_ok
                    and int(position["bars_held"]) >= market_filter_fail_min_hold_bars
                ):
                    reason = "market_filter_fail"
                elif bool(prev.get("signal_fail", False)):
                    reason = "signal_fail"
                elif int(position["bars_held"]) >= int(params["max_hold_bars"]):
                    reason = "max_hold"
                if reason is not None:
                    cash, value, fees, sold_all = _sell(
                        date=date,
                        symbol=symbol,
                        position=position,
                        cash=cash,
                        open_px=open_px,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        fixed_sell_fee=fixed_sell_fee,
                        reason=reason,
                        actions=actions,
                        closed_trades=closed_trades,
                    )
                    if value > 0:
                        traded += value
                        day_fees += fees
                        total_fees += fees
                        if sold_all:
                            positions.pop(symbol, None)
                            if reason == "stop" and cooldown_bars_after_stop > 0:
                                cooldown_until[symbol] = i + cooldown_bars_after_stop
                elif (
                    partial_exit_fraction > 0.0
                    and partial_target_atr > 0.0
                    and not bool(position.get("partial_taken", False))
                    and float(prev["high"])
                    >= float(position.get("partial_target_price", float("inf")))
                ):
                    shares_to_sell = float(position["shares"]) * min(
                        1.0, max(0.0, partial_exit_fraction)
                    )
                    cash, value, fees, sold_all = _sell(
                        date=date,
                        symbol=symbol,
                        position=position,
                        cash=cash,
                        open_px=open_px,
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        fixed_sell_fee=fixed_sell_fee,
                        reason="partial_target",
                        actions=actions,
                        closed_trades=closed_trades,
                        shares_to_sell=shares_to_sell,
                    )
                    if value > 0:
                        traded += value
                        day_fees += fees
                        total_fees += fees
                        if sold_all:
                            positions.pop(symbol, None)
                        else:
                            position["partial_taken"] = True
                            if move_stop_to_entry_after_partial:
                                position["stop_price"] = max(
                                    float(position["stop_price"]),
                                    float(position["entry_price"]),
                                )

        if pd.Timestamp(date) in scan_dates:
            snapshot = _latest_before(features, pd.Timestamp(date))
            if not snapshot.empty:
                active = active_members_on_date(membership, pd.Timestamp(date))
                snapshot = snapshot[
                    snapshot["symbol"].isin(active)
                    & snapshot["rank_score"].notna()
                    & snapshot["atr"].notna()
                ].copy()
                if min_median_value > 0.0:
                    snapshot = snapshot[
                        snapshot["median_daily_value_63"].fillna(0.0)
                        >= min_median_value
                    ]
                if min_median_volume > 0.0:
                    snapshot = snapshot[
                        snapshot["median_daily_volume_63"].fillna(0.0)
                        >= min_median_volume
                    ]
                snapshot = snapshot.sort_values("rank_score", ascending=False).head(top_n)
                active_rank = snapshot["symbol"].astype(str).tolist()
                latest_setups = snapshot.copy()

        market_allows_entries = (
            True
            if market_ok_by_date is None
            else bool(market_ok_by_date.get(pd.Timestamp(date), True))
        )
        previous_equity = float(equity.iloc[i - 1]) if i > 0 else float(cash)
        previous_drawdown = (
            0.0
            if peak_equity_so_far <= 0.0
            else previous_equity / peak_equity_so_far - 1.0
        )
        drawdown_allows_entries = (
            portfolio_drawdown_pause_pct <= 0.0
            or previous_drawdown >= -portfolio_drawdown_pause_pct
        )
        if (
            not prev_rows.empty
            and cash > 0
            and len(positions) < max_positions
            and market_allows_entries
            and drawdown_allows_entries
        ):
            candidates = [
                symbol
                for symbol in active_rank
                if symbol not in positions
                and i >= cooldown_until.get(symbol, -1)
                and symbol in prev_rows.index
                and bool(prev_rows.loc[symbol].get("entry_signal", False))
                and symbol in open_px.columns
                and pd.notna(open_px.at[date, symbol])
            ]
            prev_close_row = (
                close_px.loc[prev_date]
                if prev_date is not None and prev_date in close_px.index
                else pd.Series(dtype=float)
            )
            slots = max(0, max_positions - len(positions))
            for symbol in candidates[:slots]:
                row = prev_rows.loc[symbol]
                raw_open = open_px.at[date, symbol]
                if pd.isna(raw_open) or pd.isna(row.get("atr")) or float(row["atr"]) <= 0:
                    continue
                if max_sector_positions > 0:
                    sector = str(row.get("sector", "unknown"))
                    current_sector_positions = sum(
                        1
                        for position in positions.values()
                        if str(position.get("sector", "unknown")) == sector
                    )
                    if current_sector_positions >= max_sector_positions:
                        continue
                slots_left = max(1, max_positions - len(positions))
                fill = float(raw_open) * (1.0 + slippage_rate)
                marked_position_value = 0.0
                for held_symbol, position in positions.items():
                    price = prev_close_row.get(held_symbol)
                    if pd.notna(price):
                        marked_position_value += float(position["shares"]) * float(price)
                current_equity = max(cash + marked_position_value, cash)
                position_size_mult = float(row.get("position_size_mult", 1.0) or 1.0)
                position_size_mult = min(
                    max(position_size_mult, min_position_size_mult),
                    max_position_size_mult,
                )
                slot_budget = cash / float(slots_left) * position_size_mult
                cap_budget = (
                    max_position_weight * current_equity * position_size_mult
                    if max_position_weight > 0.0
                    else slot_budget
                )
                atr_value = float(row["atr"])
                stop_distance = max(float(params["stop_atr"]) * atr_value, fill * 0.01)
                if risk_per_trade_pct > 0.0:
                    risk_budget = risk_per_trade_pct * current_equity * position_size_mult
                    risk_budget_notional = risk_budget * fill / stop_distance
                else:
                    risk_budget_notional = slot_budget
                budget = min(slot_budget, cap_budget, risk_budget_notional)
                shares = _round_shares(
                    max(0.0, budget - fixed_fee) / (fill * (1.0 + fee_rate)),
                    share_precision,
                )
                cost = shares * fill
                fee = cost * fee_rate + fixed_fee
                if shares <= 0 or cost + fee > cash:
                    continue
                cash -= cost + fee
                traded += cost
                day_fees += fee
                total_fees += fee
                positions[symbol] = {
                    "entry_date": str(date.date()),
                    "entry_price": fill,
                    "entry_value": cost,
                    "entry_fee": fee,
                    "shares": shares,
                    "bars_held": 0,
                    "peak_close": float(row["close"]),
                    "target_price": fill + float(params["target_atr"]) * atr_value,
                    "stop_price": fill - float(params["stop_atr"]) * atr_value,
                    "partial_target_price": fill
                    + float(partial_target_atr) * atr_value,
                    "partial_taken": False,
                    "sector": str(row.get("sector", "unknown")),
                }
                actions.append(
                    {
                        "date": str(date.date()),
                        "symbol": symbol,
                        "action": "BUY",
                        "shares": shares,
                        "price": fill,
                        "value": cost,
                        "fee": fee,
                        "reason": f"{family}_entry",
                    }
                )

        close_row = close_px.loc[date] if date in close_px.index else pd.Series(dtype=float)
        position_value = 0.0
        for symbol, position in positions.items():
            price = close_row.get(symbol)
            if pd.notna(price):
                position_value += float(position["shares"]) * float(price)
                position_rows.append(
                    {
                        "date": str(date.date()),
                        "symbol": symbol,
                        "shares": float(position["shares"]),
                        "close": float(price),
                        "market_value": float(position["shares"]) * float(price),
                        "entry_date": position["entry_date"],
                        "bars_held": int(position["bars_held"]),
                    }
                )
        equity.iloc[i] = cash + position_value
        peak_equity_so_far = max(peak_equity_so_far, float(equity.iloc[i]))
        turnover_rows.append(
            {
                "date": str(date.date()),
                "traded_value": traded,
                "fees": day_fees,
                "turnover_pct": 0.0 if equity.iloc[i] <= 0 else traded / equity.iloc[i],
                "open_positions": len(positions),
                "cash": cash,
            }
        )

    trades = _build_trade_frame(closed_trades)
    metrics = _compute_metrics(equity.iloc[start_idx:].reset_index(drop=True), flows.iloc[start_idx:].reset_index(drop=True), trades)
    metrics["total_fees"] = float(total_fees)
    metrics["fee_to_contributions_ratio"] = (
        0.0
        if metrics["total_contributions"] <= 0
        else float(total_fees / metrics["total_contributions"])
    )
    metrics["mean_turnover_pct"] = (
        0.0
        if not turnover_rows
        else float(pd.DataFrame(turnover_rows)["turnover_pct"].fillna(0.0).mean())
    )
    return StrategySimulation(
        equity=equity,
        flows=flows,
        metrics=metrics,
        actions=pd.DataFrame(actions),
        positions=pd.DataFrame(position_rows),
        turnover=pd.DataFrame(turnover_rows),
        latest_setups=latest_setups,
        total_fees=float(total_fees),
    )


def _holdout_metrics(
    simulation: StrategySimulation, holdout_start: int
) -> dict[str, float]:
    trades = simulation.actions[
        (simulation.actions.get("action") == "SELL")
        if not simulation.actions.empty
        else pd.Series(dtype=bool)
    ]
    return _compute_metrics(
        simulation.equity.iloc[holdout_start:].reset_index(drop=True),
        simulation.flows.iloc[holdout_start:].reset_index(drop=True),
        _build_trade_frame([]) if trades.empty else _build_trade_frame([]),
    )


def _candidate_row(
    *,
    family: str,
    params: dict[str, Any],
    simulation: StrategySimulation,
    holdout: dict[str, float],
    etf_holdout: dict[str, float],
    passed: bool,
) -> dict[str, Any]:
    return {
        "family": family,
        "passed_constraints": bool(passed),
        "holdout_excess_vs_etf_dca": float(
            holdout["twr_total_return"] - etf_holdout["twr_total_return"]
        ),
        "holdout_cagr": float(holdout["cagr"]),
        "holdout_max_drawdown": float(holdout["max_drawdown"]),
        "twr_total_return": float(simulation.metrics["twr_total_return"]),
        "cagr": float(simulation.metrics["cagr"]),
        "max_drawdown": float(simulation.metrics["max_drawdown"]),
        "closed_trades": float(simulation.metrics["closed_trades"]),
        "mean_turnover_pct": float(simulation.metrics["mean_turnover_pct"]),
        "fee_to_contributions_ratio": float(
            simulation.metrics["fee_to_contributions_ratio"]
        ),
        "params": json.dumps(params, sort_keys=True),
    }


def run_stock_strategy_research(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    families: list[str] | None = None,
    trials_override: int | None = None,
    run_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fee_bps_override: float | None = None,
    slippage_bps_override: float | None = None,
    fixed_fee_egp_override: float | None = None,
    fixed_fee_on_sell: bool = False,
) -> StockStrategyResearchRun:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = load_stock_rotation_config(config_path)
    if fee_bps_override is not None:
        config.backtest.fee_bps = float(fee_bps_override)
    if slippage_bps_override is not None:
        config.backtest.slippage_bps = float(slippage_bps_override)
    if fixed_fee_egp_override is not None:
        config.portfolio.fixed_buy_fee_egp = float(fixed_fee_egp_override)
    selected_families = families or list(STOCK_STRATEGY_FAMILIES)
    unknown = sorted(set(selected_families).difference(STOCK_STRATEGY_FAMILIES))
    if unknown:
        raise ValueError(f"Unsupported families: {unknown}")

    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    disclosure_events = load_disclosure_events(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
    if start_date is not None:
        etf = etf[etf["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date is not None:
        etf = etf[etf["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    if etf.empty:
        raise ValueError("No ETF benchmark dates remain after start/end filtering.")

    calendar = pd.Series(pd.to_datetime(etf["date"])).sort_values().drop_duplicates().reset_index(drop=True)
    min_date = pd.Timestamp(calendar.iloc[0])
    panel = panel[panel["date"] >= min_date].reset_index(drop=True)
    if len(calendar) < 60:
        raise ValueError("Not enough benchmark bars for stock strategy research.")

    holdout_bars = max(20, int(len(calendar) * float(config.model_selection.holdout_ratio)))
    holdout_start = max(0, len(calendar) - holdout_bars)
    etf_holdout = run_dca_benchmark(etf, holdout_start, len(etf) - 1, config.backtest).metrics
    etf_full = run_dca_benchmark(etf, 0, len(etf) - 1, config.backtest)

    trials = int(trials_override or 25)
    actual_run_id = run_id or f"stock-strategy-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    leaderboard_rows: list[dict[str, Any]] = []
    family_winners: dict[str, Any] = {}

    for family_index, family in enumerate(selected_families):
        trial_rows: list[dict[str, Any]] = []
        best_score = -float("inf")
        best_params = _default_params(family)
        best_simulation: StrategySimulation | None = None
        best_holdout: dict[str, float] | None = None

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=1000 + family_index),
        )

        def objective(trial: optuna.trial.Trial) -> float:
            nonlocal best_score, best_params, best_simulation, best_holdout
            params = _sample_params(trial, family)
            if fixed_fee_on_sell:
                params["fixed_sell_fee_egp"] = float(config.portfolio.fixed_buy_fee_egp)
            features = build_strategy_feature_panel(
                panel,
                family=family,
                params=params,
                disclosure_events=disclosure_events,
            )
            simulation = simulate_event_driven_strategy(
                panel=panel,
                features=features,
                calendar=calendar,
                membership=membership,
                config=config,
                family=family,
                params=params,
            )
            holdout = _holdout_metrics(simulation, holdout_start)
            excess = float(holdout["twr_total_return"] - etf_holdout["twr_total_return"])
            penalty = max(0.0, float(holdout["max_drawdown"]) - float(config.model_selection.max_drawdown))
            score = excess - penalty
            trial_rows.append(
                {
                    "family": family,
                    "number": trial.number,
                    "value": score,
                    "holdout_excess_vs_etf_dca": excess,
                    "holdout_max_drawdown": float(holdout["max_drawdown"]),
                    "closed_trades": float(simulation.metrics["closed_trades"]),
                    "params": json.dumps(params, sort_keys=True),
                }
            )
            if score > best_score:
                best_score = score
                best_params = params
                best_simulation = simulation
                best_holdout = holdout
            return score

        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        pd.DataFrame(trial_rows).to_csv(run_dir / f"trials_{family}.csv", index=False)

        if best_simulation is None or best_holdout is None:
            features = build_strategy_feature_panel(
                panel,
                family=family,
                params=best_params,
                disclosure_events=disclosure_events,
            )
            if fixed_fee_on_sell:
                best_params = dict(best_params)
                best_params["fixed_sell_fee_egp"] = float(config.portfolio.fixed_buy_fee_egp)
            best_simulation = simulate_event_driven_strategy(
                panel=panel,
                features=features,
                calendar=calendar,
                membership=membership,
                config=config,
                family=family,
                params=best_params,
            )
            best_holdout = _holdout_metrics(best_simulation, holdout_start)

        constraints = config.model_selection
        passed = (
            best_holdout["twr_total_return"] - etf_holdout["twr_total_return"]
            >= constraints.min_holdout_excess_return
            and best_holdout["max_drawdown"] <= constraints.max_drawdown
            and best_simulation.metrics["mean_turnover_pct"]
            <= constraints.max_mean_rebalance_turnover_pct
            and best_simulation.metrics["fee_to_contributions_ratio"]
            <= constraints.max_fee_to_contributions_ratio
        )
        leaderboard_rows.append(
            _candidate_row(
                family=family,
                params=best_params,
                simulation=best_simulation,
                holdout=best_holdout,
                etf_holdout=etf_holdout,
                passed=passed,
            )
        )
        family_winners[family] = {
            "params": best_params,
            "metrics": best_simulation.metrics,
            "holdout_metrics": best_holdout,
            "passed_constraints": bool(passed),
        }

        best_simulation.actions.to_csv(run_dir / f"actions_{family}.csv", index=False)
        best_simulation.positions.to_csv(run_dir / f"positions_{family}.csv", index=False)
        best_simulation.turnover.to_csv(run_dir / f"turnover_{family}.csv", index=False)
        latest = best_simulation.latest_setups.copy()
        if not latest.empty:
            latest.to_csv(run_dir / f"latest_setups_{family}.csv", index=False)
        else:
            pd.DataFrame().to_csv(run_dir / f"latest_setups_{family}.csv", index=False)
        pd.DataFrame(
            {
                "date": calendar.values,
                "strategy_equity": best_simulation.equity.values,
                "etf_dca_equity": etf_full.equity.values,
            }
        ).to_csv(run_dir / f"equity_curve_{family}.csv", index=False)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["passed_constraints", "holdout_excess_vs_etf_dca", "holdout_cagr"],
        ascending=[False, False, False],
    )
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
    best_family = None if leaderboard.empty else str(leaderboard.iloc[0]["family"])
    summary = {
        "run_id": actual_run_id,
        "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
        "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "families": selected_families,
        "best_family": best_family,
        "holdout_start_date": str(pd.Timestamp(calendar.iloc[holdout_start]).date()),
        "trials_per_family": trials,
        "etf_dca_metrics": etf_full.metrics,
        "etf_holdout_metrics": etf_holdout,
        "family_winners": family_winners,
    }
    write_json(run_dir / "summary.json", summary)
    generate_stock_strategy_report(actual_run_id)
    return StockStrategyResearchRun(run_id=actual_run_id, run_dir=run_dir)
