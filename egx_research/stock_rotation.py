from __future__ import annotations

import copy
import json
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
from egx_research.fundamentals import (
    DERIVED_FACTOR_COLUMNS,
    load_fundamentals,
    merge_fundamentals_asof,
)
from egx_research.indicators import kama
from egx_research.stock_rotation_config import (
    StockRotationConfig,
    load_stock_rotation_config,
)
from egx_research.strategies import build_strategy_frame
from egx_research.utils import ensure_dir, write_json


DEFAULT_PULLBACK_PARAMS: dict[str, Any] = {
    "kama_len": 26,
    "kama_fast": 3,
    "kama_slow": 52,
    "cci_len": 26,
    "buy_threshold": -45.0,
    "trend_buffer_atr": 1.7,
    "atr_len": 8,
}


SECTOR_BY_SYMBOL = {
    "ADIB": "banks",
    "COMI": "banks",
    "HRHO": "financials",
    "BFTH": "financials",
    "BTFH": "financials",
    "CCAP": "financials",
    "TMGH": "real_estate",
    "PHDC": "real_estate",
    "EMFD": "real_estate",
    "ORHD": "real_estate",
    "HELI": "real_estate",
    "EFIH": "technology",
    "FWRY": "technology",
    "ETEL": "telecom",
    "ORAS": "industrials",
    "GBCO": "industrials",
    "RAYA": "industrials",
    "EGAL": "materials",
    "ABUK": "materials",
    "AMOC": "energy",
    "EAST": "consumer",
    "EFID": "consumer",
    "JUFO": "consumer",
    "ISPH": "healthcare",
    "RMDA": "healthcare",
    "MCQE": "materials",
    "ARCC": "materials",
    "EGCH": "materials",
    "ORWE": "consumer",
    "OIH": "financials",
    "VLMR": "holdco",
    "VLMRA": "holdco",
}


def infer_stock_sector(symbol: str, holding_name: str | None = None) -> str:
    symbol = str(symbol).strip().upper()
    if symbol in SECTOR_BY_SYMBOL:
        return SECTOR_BY_SYMBOL[symbol]
    name = str(holding_name or "").lower()
    if any(token in name for token in ["bank", "cib", "adib"]):
        return "banks"
    if any(token in name for token in ["real", "housing", "development", "holding"]):
        return "real_estate"
    if any(token in name for token in ["technology", "digital", "electronic", "fawry"]):
        return "technology"
    if any(token in name for token in ["cement", "aluminum", "fertilizer", "chemical"]):
        return "materials"
    if any(token in name for token in ["pharma", "medical", "health"]):
        return "healthcare"
    if any(token in name for token in ["food", "tobacco", "eastern"]):
        return "consumer"
    return "industrials"


@dataclass
class StockRotationRun:
    run_id: str
    run_dir: Path


@dataclass
class StockSelectionRun:
    run_id: str
    run_dir: Path


@dataclass
class EqualWeightBenchmark:
    equity: pd.Series
    flows: pd.Series
    metrics: dict[str, float]
    actions: list[dict[str, Any]]
    monthly_rows: list[dict[str, Any]]


@dataclass
class DeploymentSimulation:
    equity: pd.Series
    flows: pd.Series
    metrics: dict[str, float]
    actions: pd.DataFrame
    rebalance_turnover: pd.DataFrame
    total_fees: float


def load_membership_snapshots(config: StockRotationConfig) -> pd.DataFrame:
    root = Path(config.storage.root_dir)
    preferred = root / "membership_verified_partial.csv"
    path = (
        preferred if preferred.exists() else root / config.storage.membership_filename
    )
    if not path.exists():
        raise FileNotFoundError(f"Membership snapshots missing: {path}")
    frame = pd.read_csv(path, parse_dates=["effective_date"])
    return frame.sort_values(["symbol", "effective_date"]).reset_index(drop=True)


def load_stock_panel(config: StockRotationConfig) -> pd.DataFrame:
    path = Path(config.storage.root_dir) / config.storage.panel_filename
    if not path.exists():
        raise FileNotFoundError(f"Stock panel missing: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = (
        frame.sort_values(["symbol", "date"])
        .drop_duplicates(subset=["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return frame


def _empty_dividend_actions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "event_date", "cash_amount", "currency", "source"]
    )


def _empty_corporate_actions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["symbol", "event_date", "event_type", "is_dilutive", "source"]
    )


def load_dividend_actions(config: StockRotationConfig) -> pd.DataFrame:
    path = Path(config.storage.root_dir) / config.storage.dividend_actions_filename
    if not path.exists():
        return _empty_dividend_actions()

    frame = pd.read_csv(path, parse_dates=["event_date"])
    if frame.empty:
        return _empty_dividend_actions()

    required = {"symbol", "event_date", "cash_amount"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dividend actions missing columns: {sorted(missing)}")

    frame["symbol"] = frame["symbol"].astype(str)
    frame["cash_amount"] = pd.to_numeric(frame["cash_amount"], errors="coerce")
    return frame.sort_values(["symbol", "event_date"]).reset_index(drop=True)


def load_corporate_actions(config: StockRotationConfig) -> pd.DataFrame:
    path = Path(config.storage.root_dir) / config.storage.corporate_actions_filename
    if not path.exists():
        return _empty_corporate_actions()

    frame = pd.read_csv(path, parse_dates=["event_date"])
    if frame.empty:
        return _empty_corporate_actions()

    required = {"symbol", "event_date", "event_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Corporate actions missing columns: {sorted(missing)}")

    frame["symbol"] = frame["symbol"].astype(str)
    frame["event_type"] = frame["event_type"].astype(str).str.lower()
    if "is_dilutive" in frame.columns:
        frame["is_dilutive"] = frame["is_dilutive"].fillna(True).astype(bool)
    else:
        frame["is_dilutive"] = True
    return frame.sort_values(["symbol", "event_date"]).reset_index(drop=True)


def load_stock_fundamentals(config: StockRotationConfig) -> pd.DataFrame:
    path = Path(config.storage.root_dir) / config.storage.fundamentals_filename
    return load_fundamentals(path)


def _trailing_event_window(
    dates: pd.Series,
    events: pd.DataFrame,
    *,
    window_days: int,
    value_col: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    base = pd.DataFrame({"date": pd.to_datetime(dates)}).sort_values("date").reset_index(
        drop=True
    )
    if events.empty:
        zeros = pd.Series(0.0, index=base.index, dtype=float)
        return zeros, zeros.copy()

    ordered = events.sort_values("event_date").copy()
    ordered["event_value"] = (
        pd.to_numeric(ordered[value_col], errors="coerce").fillna(0.0)
        if value_col is not None and value_col in ordered.columns
        else 0.0
    )
    ordered["cum_count"] = np.arange(1, len(ordered) + 1, dtype=float)
    ordered["cum_value"] = ordered["event_value"].cumsum()

    latest = pd.merge_asof(
        base,
        ordered[["event_date", "cum_count", "cum_value"]],
        left_on="date",
        right_on="event_date",
        direction="backward",
    )
    cutoff = pd.DataFrame({"cutoff": base["date"] - pd.Timedelta(days=window_days)})
    prior = pd.merge_asof(
        cutoff,
        ordered[["event_date", "cum_count", "cum_value"]],
        left_on="cutoff",
        right_on="event_date",
        direction="backward",
    )

    count = latest["cum_count"].fillna(0.0) - prior["cum_count"].fillna(0.0)
    value = latest["cum_value"].fillna(0.0) - prior["cum_value"].fillna(0.0)
    return count.astype(float), value.astype(float)


def build_stock_features(
    panel: pd.DataFrame,
    config: StockRotationConfig,
    *,
    benchmark: pd.DataFrame | None = None,
    dividend_actions: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features: list[pd.DataFrame] = []
    w3 = config.selection.rel_strength_window_3m
    w6 = config.selection.rel_strength_window_6m
    liquidity_window = max(1, int(config.selection.liquidity_window_bars))
    coverage_window = max(1, int(config.validation.coverage_lookback_bars))
    benchmark_close = None
    if benchmark is not None and {"date", "close"}.issubset(set(benchmark.columns)):
        benchmark_close = (
            benchmark[["date", "close"]]
            .rename(columns={"close": "benchmark_close"})
            .sort_values("date")
        )
    dividend_actions = (
        dividend_actions if dividend_actions is not None else _empty_dividend_actions()
    )
    corporate_actions = (
        corporate_actions
        if corporate_actions is not None
        else _empty_corporate_actions()
    )

    for _, group in panel.groupby("symbol", sort=True):
        frame = group.sort_values("date").copy()
        symbol = str(frame["symbol"].iloc[0])
        holding_name = (
            str(frame["holding_name"].dropna().iloc[-1])
            if "holding_name" in frame.columns and frame["holding_name"].notna().any()
            else symbol
        )
        frame["sector"] = infer_stock_sector(symbol, holding_name)
        if benchmark_close is not None:
            frame = frame.merge(benchmark_close, on="date", how="left")
        frame["kama"] = kama(frame["close"], 26, 3, 52)
        frame["kama_rising"] = frame["kama"] > frame["kama"].shift(1)

        symbol_dividends = dividend_actions[dividend_actions["symbol"] == symbol]
        dividend_by_date = (
            symbol_dividends.groupby(pd.to_datetime(symbol_dividends["event_date"]).dt.normalize())[
                "cash_amount"
            ]
            .sum()
            .astype(float)
            if not symbol_dividends.empty
            else pd.Series(dtype=float)
        )
        frame_dates = pd.to_datetime(frame["date"]).dt.normalize()
        frame["cash_dividend"] = frame_dates.map(dividend_by_date).fillna(0.0)
        frame["cum_dividend"] = frame["cash_dividend"].cumsum()

        if config.selection.use_total_return_features:
            frame["ret_3m"] = (
                frame["close"] + frame["cum_dividend"] - frame["cum_dividend"].shift(w3)
            ) / frame["close"].shift(w3) - 1.0
            frame["ret_6m"] = (
                frame["close"] + frame["cum_dividend"] - frame["cum_dividend"].shift(w6)
            ) / frame["close"].shift(w6) - 1.0
            frame["ret_1m"] = (
                frame["close"] + frame["cum_dividend"] - frame["cum_dividend"].shift(21)
            ) / frame["close"].shift(21) - 1.0
            frame["ret_6_1"] = (
                frame["close"].shift(21)
                + frame["cum_dividend"].shift(21)
                - frame["cum_dividend"].shift(126)
            ) / frame["close"].shift(126) - 1.0
            frame["ret_12_1"] = (
                frame["close"].shift(21)
                + frame["cum_dividend"].shift(21)
                - frame["cum_dividend"].shift(252)
            ) / frame["close"].shift(252) - 1.0
            returns = (frame["close"] + frame["cash_dividend"]) / frame["close"].shift(1) - 1.0
        else:
            frame["ret_3m"] = frame["close"] / frame["close"].shift(w3) - 1.0
            frame["ret_6m"] = frame["close"] / frame["close"].shift(w6) - 1.0
            frame["ret_1m"] = frame["close"] / frame["close"].shift(21) - 1.0
            frame["ret_6_1"] = frame["close"].shift(21) / frame["close"].shift(126) - 1.0
            frame["ret_12_1"] = frame["close"].shift(21) / frame["close"].shift(252) - 1.0
            returns = frame["close"].pct_change()

        frame["sma200"] = frame["close"].rolling(200, min_periods=150).mean()
        frame["close_vs_sma200"] = frame["close"] / frame["sma200"] - 1.0
        frame["vol_63"] = returns.rolling(63, min_periods=63).std(ddof=0) * np.sqrt(252.0)
        frame["downside_vol_126"] = (
            returns.where(returns < 0.0, 0.0)
            .rolling(126, min_periods=126)
            .std(ddof=0)
            * np.sqrt(252.0)
        )
        rolling_peak = frame["close"].rolling(252, min_periods=126).max()
        drawdown = frame["close"] / rolling_peak - 1.0
        frame["max_drawdown_252"] = (
            drawdown.rolling(252, min_periods=126).min().abs()
        )

        if "benchmark_close" in frame.columns:
            benchmark_returns = frame["benchmark_close"].pct_change()
            cov = returns.rolling(252, min_periods=126).cov(benchmark_returns)
            var = benchmark_returns.rolling(252, min_periods=126).var(ddof=0)
            frame["beta_252"] = cov / var.replace(0.0, np.nan)
        else:
            frame["beta_252"] = np.nan

        frame["daily_value"] = frame["close"] * frame["volume"]
        frame["median_daily_value"] = (
            frame["daily_value"]
            .rolling(liquidity_window, min_periods=liquidity_window)
            .median()
        )
        frame["median_daily_volume"] = (
            frame["volume"]
            .rolling(liquidity_window, min_periods=liquidity_window)
            .median()
        )
        frame["coverage_ratio"] = (
            frame["close"]
            .notna()
            .astype(float)
            .rolling(coverage_window, min_periods=coverage_window)
            .mean()
        )

        div_count_3y, div_cash_1y = _trailing_event_window(
            frame["date"],
            symbol_dividends,
            window_days=1095,
            value_col="cash_amount",
        )
        _, div_cash_1y = _trailing_event_window(
            frame["date"],
            symbol_dividends,
            window_days=366,
            value_col="cash_amount",
        )
        frame["dividend_event_count_3y"] = (
            div_count_3y if not symbol_dividends.empty else np.nan
        )
        frame["dividend_yield_1y"] = (
            div_cash_1y / frame["close"].replace(0.0, np.nan)
            if not symbol_dividends.empty
            else np.nan
        )

        symbol_actions = corporate_actions[
            corporate_actions["symbol"] == str(frame["symbol"].iloc[0])
        ]
        if not symbol_actions.empty:
            dilutive = symbol_actions[symbol_actions["is_dilutive"].fillna(True)].copy()
            dilutive_count_3y, _ = _trailing_event_window(
                frame["date"], dilutive, window_days=1095
            )
            frame["dilutive_event_count_3y"] = dilutive_count_3y
        else:
            frame["dilutive_event_count_3y"] = np.nan

        features.append(frame)

    result = pd.concat(features, ignore_index=True)
    return merge_fundamentals_asof(
        result,
        fundamentals if fundamentals is not None else pd.DataFrame(),
    )


def build_etf_features(etf: pd.DataFrame, config: StockRotationConfig) -> pd.DataFrame:
    w3 = config.selection.rel_strength_window_3m
    frame = etf.sort_values("date").copy()
    frame["ret_3m"] = frame["close"] / frame["close"].shift(w3) - 1.0
    return frame


def _rank_series(
    series: pd.Series, *, higher_is_better: bool = True
) -> tuple[pd.Series, bool]:
    ranked = pd.Series(0.5, index=series.index, dtype=float)
    valid = series.notna() & np.isfinite(series)
    if not valid.any():
        return ranked, False
    ranked.loc[valid] = series.loc[valid].rank(
        method="average", pct=True, ascending=higher_is_better
    )
    return ranked, True


def _combine_component_ranks(
    index: pd.Index, components: list[tuple[pd.Series, float]], *, fallback: float = 0.5
) -> pd.Series:
    if not components:
        return pd.Series(fallback, index=index, dtype=float)
    total = pd.Series(0.0, index=index, dtype=float)
    weight_sum = 0.0
    for series, weight in components:
        total += series.astype(float) * float(weight)
        weight_sum += float(weight)
    if weight_sum <= 0:
        return pd.Series(fallback, index=index, dtype=float)
    return total / weight_sum


def _neutral_rank_component(
    frame: pd.DataFrame, column: str, *, higher_is_better: bool = True
) -> pd.Series:
    series = frame.get(column, pd.Series(np.nan, index=frame.index))
    ranked, _ = _rank_series(series, higher_is_better=higher_is_better)
    return ranked


def _rank_blend(
    frame: pd.DataFrame, specs: list[tuple[str, bool, float]]
) -> pd.Series:
    return _combine_component_ranks(
        frame.index,
        [
            (_neutral_rank_component(frame, column, higher_is_better=higher), weight)
            for column, higher, weight in specs
        ],
    )


def _sector_value_score(eligible: pd.DataFrame) -> pd.Series:
    result = pd.Series(0.5, index=eligible.index, dtype=float)
    for sector, group in eligible.groupby("sector", sort=False):
        if sector in {"banks", "financials"}:
            specs = [
                ("book_to_price", True, 0.45),
                ("earnings_yield", True, 0.20),
                ("dividend_yield_1y", True, 0.20),
                ("leverage_inverse", True, 0.15),
            ]
        elif sector == "real_estate":
            specs = [
                ("book_to_price", True, 0.40),
                ("earnings_yield", True, 0.20),
                ("cash_conversion", True, 0.20),
                ("leverage_inverse", True, 0.20),
            ]
        elif sector == "technology":
            specs = [
                ("earnings_yield", True, 0.30),
                ("net_income_growth", True, 0.30),
                ("revenue_growth", True, 0.25),
                ("book_to_price", True, 0.15),
            ]
        else:
            specs = [
                ("earnings_yield", True, 0.35),
                ("book_to_price", True, 0.25),
                ("dividend_yield_1y", True, 0.20),
                ("leverage_inverse", True, 0.20),
            ]
        result.loc[group.index] = _rank_blend(group, specs)
    return result


def _sector_quality_score(eligible: pd.DataFrame) -> pd.Series:
    result = pd.Series(0.5, index=eligible.index, dtype=float)
    for sector, group in eligible.groupby("sector", sort=False):
        if sector in {"banks", "financials"}:
            specs = [
                ("roe", True, 0.45),
                ("roa", True, 0.20),
                ("positive_earnings_count_4p", True, 0.20),
                ("margin_stability_4p", True, 0.15),
            ]
        elif sector == "real_estate":
            specs = [
                ("cash_conversion", True, 0.35),
                ("leverage_inverse", True, 0.25),
                ("roe", True, 0.20),
                ("positive_earnings_count_4p", True, 0.20),
            ]
        elif sector == "technology":
            specs = [
                ("roe", True, 0.25),
                ("revenue_growth", True, 0.25),
                ("net_income_growth", True, 0.25),
                ("cash_conversion", True, 0.25),
            ]
        else:
            specs = [
                ("roe", True, 0.25),
                ("roa", True, 0.20),
                ("cash_conversion", True, 0.20),
                ("leverage_inverse", True, 0.20),
                ("positive_earnings_count_4p", True, 0.15),
            ]
        result.loc[group.index] = _rank_blend(group, specs)
    return result


def _apply_multifactor_risk_filters(
    eligible: pd.DataFrame, config: StockRotationConfig
) -> pd.DataFrame:
    filtered = eligible.copy()
    if (
        config.selection.require_long_term_trend
        and "close_vs_sma200" in filtered.columns
    ):
        filtered = filtered[filtered["close_vs_sma200"].fillna(-1.0) > 0.0].copy()
    if "max_drawdown_252" in filtered.columns:
        filtered = filtered[
            filtered["max_drawdown_252"].fillna(1.0)
            <= float(config.selection.max_drawdown_252)
        ].copy()
    return filtered


def _score_multifactor_snapshot(
    snapshot: pd.DataFrame, config: StockRotationConfig
) -> pd.DataFrame:
    eligible = snapshot[
        snapshot["close"].notna()
        & snapshot["median_daily_value"].notna()
        & snapshot["median_daily_volume"].notna()
        & snapshot["coverage_ratio"].notna()
        & snapshot["ret_12_1"].notna()
        & snapshot["ret_6_1"].notna()
        & snapshot["ret_1m"].notna()
        & snapshot["vol_63"].notna()
        & snapshot["beta_252"].notna()
    ].copy()
    eligible["sector"] = eligible.apply(
        lambda row: infer_stock_sector(
            row.get("symbol", ""), row.get("holding_name", "")
        )
        if pd.isna(row.get("sector", np.nan))
        else row.get("sector"),
        axis=1,
    )
    eligible = _apply_multifactor_risk_filters(eligible, config)
    if eligible.empty:
        return eligible

    factor_weights = config.selection.factor_weights
    sector_aware = getattr(config.selection, "method", "") == "sector_multifactor"

    mom_12_rank, _ = _rank_series(eligible["ret_12_1"], higher_is_better=True)
    mom_6_rank, _ = _rank_series(eligible["ret_6_1"], higher_is_better=True)
    rev_rank, _ = _rank_series(eligible["ret_1m"], higher_is_better=False)
    eligible["score_momentum"] = (
        0.20 * mom_12_rank + 0.10 * mom_6_rank + 0.05 * rev_rank
    ) / 0.35

    if sector_aware:
        eligible["score_value"] = _sector_value_score(eligible)
    else:
        eligible["score_value"] = _combine_component_ranks(
            eligible.index,
            [
                (
                    _neutral_rank_component(
                        eligible, "dividend_yield_1y", higher_is_better=True
                    ),
                    0.10,
                ),
                (
                    _neutral_rank_component(
                        eligible, "earnings_yield", higher_is_better=True
                    ),
                    0.05,
                ),
                (
                    _neutral_rank_component(
                        eligible, "book_to_price", higher_is_better=True
                    ),
                    0.05,
                ),
            ],
        )

    quality_components: list[tuple[pd.Series, float]] = [
        (
            _neutral_rank_component(
                eligible, "dividend_event_count_3y", higher_is_better=True
            ),
            0.03,
        ),
        (
            _neutral_rank_component(
                eligible, "dilutive_event_count_3y", higher_is_better=False
            ),
            0.03,
        ),
        (_neutral_rank_component(eligible, "roe", higher_is_better=True), 0.035),
        (_neutral_rank_component(eligible, "roa", higher_is_better=True), 0.025),
        (
            _neutral_rank_component(
                eligible, "cash_conversion", higher_is_better=True
            ),
            0.025,
        ),
        (
            _neutral_rank_component(
                eligible, "leverage_inverse", higher_is_better=True
            ),
            0.025,
        ),
        (
            _neutral_rank_component(
                eligible, "positive_earnings_count_4p", higher_is_better=True
            ),
            0.015,
        ),
        (
            _neutral_rank_component(
                eligible, "margin_stability_4p", higher_is_better=True
            ),
            0.015,
        ),
    ]
    eligible["score_quality"] = (
        _sector_quality_score(eligible)
        if sector_aware
        else _combine_component_ranks(eligible.index, quality_components)
    )
    eligible["score_growth"] = _rank_blend(
        eligible,
        [
            ("revenue_growth", True, 0.30),
            ("net_income_growth", True, 0.35),
            ("ret_6_1", True, 0.20),
            ("ret_12_1", True, 0.15),
        ],
    )

    beta_rank, _ = _rank_series(eligible["beta_252"].abs(), higher_is_better=False)
    vol_rank, _ = _rank_series(eligible["vol_63"], higher_is_better=False)
    eligible["score_low_risk"] = (
        0.10 * beta_rank + 0.05 * vol_rank
    ) / 0.15

    liquidity_rank, _ = _rank_series(
        eligible["median_daily_value"], higher_is_better=True
    )
    eligible["score_liquidity"] = liquidity_rank

    eligible["score"] = (
        factor_weights.momentum * eligible["score_momentum"]
        + factor_weights.value * eligible["score_value"]
        + factor_weights.quality * eligible["score_quality"]
        + factor_weights.growth * eligible["score_growth"]
        + factor_weights.low_risk * eligible["score_low_risk"]
        + factor_weights.liquidity * eligible["score_liquidity"]
    )
    return eligible.sort_values(
        ["score", "score_momentum", "score_low_risk", "median_daily_value"],
        ascending=False,
    ).reset_index(drop=True)


def first_trading_days(dates: pd.Series) -> list[pd.Timestamp]:
    ordered = pd.Series(pd.to_datetime(dates)).sort_values().drop_duplicates()
    return list(ordered.groupby([ordered.dt.year, ordered.dt.month]).first())


def annual_first_trading_days(dates: pd.Series) -> list[pd.Timestamp]:
    ordered = pd.Series(pd.to_datetime(dates)).sort_values().drop_duplicates()
    return list(ordered.groupby(ordered.dt.year).first())


def score_snapshot(
    snapshot: pd.DataFrame, etf_ret_3m: float, config: StockRotationConfig
) -> pd.DataFrame:
    if getattr(config.selection, "method", "relative_strength") in {
        "multi_factor_core",
        "sector_multifactor",
    }:
        return _score_multifactor_snapshot(snapshot, config)

    eligible = snapshot[
        snapshot["close"].notna()
        & snapshot["kama"].notna()
        & snapshot["ret_3m"].notna()
        & snapshot["ret_6m"].notna()
        & (snapshot["close"] > snapshot["kama"])
        & snapshot["kama_rising"]
        & (snapshot["ret_3m"] > etf_ret_3m)
    ].copy()
    if eligible.empty:
        return eligible

    eligible["rs_spread_3m"] = eligible["ret_3m"] - etf_ret_3m
    eligible["rank_3m"] = eligible["ret_3m"].rank(method="average", pct=True)
    eligible["rank_6m"] = eligible["ret_6m"].rank(method="average", pct=True)
    eligible["rank_rs"] = eligible["rs_spread_3m"].rank(method="average", pct=True)
    weights = config.selection.score_weights
    eligible["score"] = (
        weights.return_3m_rank * eligible["rank_3m"]
        + weights.return_6m_rank * eligible["rank_6m"]
        + weights.rel_strength_spread_3m * eligible["rank_rs"]
    )
    return eligible.sort_values(
        ["score", "ret_3m", "ret_6m"], ascending=False
    ).reset_index(drop=True)


def active_members_on_date(membership: pd.DataFrame, date: pd.Timestamp) -> set[str]:
    eligible = membership[membership["effective_date"] <= pd.Timestamp(date)]
    if eligible.empty:
        return set()
    latest = eligible.sort_values(["symbol", "effective_date"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    return set(latest[latest["is_member"]]["symbol"].tolist())


def _apply_selection_gates(
    snapshot: pd.DataFrame, config: StockRotationConfig
) -> pd.DataFrame:
    required_cols = {"median_daily_value", "median_daily_volume", "coverage_ratio"}
    if snapshot.empty or not required_cols.issubset(set(snapshot.columns)):
        return snapshot
    return snapshot[
        (
            snapshot["median_daily_value"]
            >= float(config.selection.min_median_daily_value_egp)
        )
        & (
            snapshot["median_daily_volume"]
            >= float(config.selection.min_median_daily_volume)
        )
        & (snapshot["coverage_ratio"] >= float(config.validation.min_coverage_ratio))
    ].copy()


def _select_with_turnover_buffer(
    scored: pd.DataFrame,
    *,
    top_n: int,
    previous_selection: set[str],
    turnover_buffer_score: float,
) -> pd.DataFrame:
    ranked = scored.sort_values(
        ["score", "ret_3m", "ret_6m"], ascending=False
    ).reset_index(drop=True)
    if ranked.empty or top_n <= 0:
        return ranked.head(0)

    base = ranked.head(top_n).copy()
    if not previous_selection or len(ranked) <= top_n:
        return base

    cutoff_score = float(base["score"].iloc[-1])
    retained = ranked[
        ranked["symbol"].isin(previous_selection)
        & (ranked["score"] >= cutoff_score - float(turnover_buffer_score))
    ].copy()

    kept_symbols = set(base["symbol"].tolist()) | set(retained["symbol"].tolist())
    kept = ranked[ranked["symbol"].isin(kept_symbols)].copy()
    if len(kept) < top_n:
        fill = ranked[~ranked["symbol"].isin(kept_symbols)].head(top_n - len(kept))
        kept = pd.concat([kept, fill], ignore_index=True)

    kept = (
        kept.sort_values(["score", "ret_3m", "ret_6m"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return kept


def _select_sector_capped(scored: pd.DataFrame, *, top_n: int, max_sector_weight: float) -> pd.DataFrame:
    if scored.empty or top_n <= 0 or "sector" not in scored.columns:
        return scored.head(max(top_n, 0)).copy()
    cap_count = max(1, int(np.floor(float(max_sector_weight) * float(top_n))))
    ranked = scored.sort_values(
        ["score", "ret_3m", "ret_6m"], ascending=False
    ).reset_index(drop=True)
    selected_rows = []
    sector_counts: dict[str, int] = {}
    for row in ranked.itertuples(index=False):
        sector = str(getattr(row, "sector", "unknown"))
        if sector_counts.get(sector, 0) >= cap_count:
            continue
        selected_rows.append(row._asdict())
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_rows) >= top_n:
            break
    if len(selected_rows) < top_n:
        selected_symbols = {row["symbol"] for row in selected_rows}
        for row in ranked.itertuples(index=False):
            if row.symbol in selected_symbols:
                continue
            selected_rows.append(row._asdict())
            selected_symbols.add(row.symbol)
            if len(selected_rows) >= top_n:
                break
    return pd.DataFrame(selected_rows, columns=ranked.columns).reset_index(drop=True)


def select_rebalance_portfolio(
    snapshot: pd.DataFrame,
    etf_ret_3m: float,
    config: StockRotationConfig,
    active_members: set[str],
    *,
    previous_selection: set[str] | None = None,
    turnover_buffer_score: float | None = None,
    apply_gates: bool = False,
) -> pd.DataFrame:
    filtered = snapshot.copy()
    if active_members:
        filtered = filtered[filtered["symbol"].isin(active_members)].copy()
    if apply_gates:
        filtered = _apply_selection_gates(filtered, config)

    scored = score_snapshot(filtered, etf_ret_3m, config)
    if "score" not in scored.columns:
        scored["score"] = pd.Series(dtype=float)

    top_n = int(config.portfolio.top_n)
    max_sector_weight = float(getattr(config.selection, "max_sector_weight", 1.0))
    if max_sector_weight < 1.0 and "sector" in scored.columns:
        selected = _select_sector_capped(
            scored, top_n=top_n, max_sector_weight=max_sector_weight
        )
    elif previous_selection is None:
        selected = scored.head(top_n).copy()
    else:
        selected = _select_with_turnover_buffer(
            scored,
            top_n=top_n,
            previous_selection=set(previous_selection),
            turnover_buffer_score=float(
                config.portfolio.turnover_buffer_score
                if turnover_buffer_score is None
                else turnover_buffer_score
            ),
        )

    if selected.empty:
        selected["target_weight"] = pd.Series(dtype=float)
        return selected

    selected["target_weight"] = min(
        1.0 / float(top_n), float(getattr(config.selection, "max_position_weight", 1.0))
    )
    return selected.reset_index(drop=True)


def find_overlap_start_date(
    features: pd.DataFrame,
    etf_features: pd.DataFrame,
    membership: pd.DataFrame,
    config: StockRotationConfig,
) -> pd.Timestamp:
    dates = first_trading_days(etf_features["date"])
    required_symbols = config.portfolio.top_n
    remaining_bars_required = config.validation.min_history_bars

    for date in dates:
        prev_dates = etf_features.loc[etf_features["date"] < date, "date"]
        if prev_dates.empty:
            continue
        prev_date = pd.Timestamp(prev_dates.iloc[-1])
        snapshot = features[features["date"] == prev_date]
        snapshot = snapshot[
            snapshot["symbol"].isin(active_members_on_date(membership, date))
        ]
        valid = snapshot[
            snapshot["close"].notna()
            & snapshot["kama"].notna()
            & snapshot["ret_3m"].notna()
            & snapshot["ret_6m"].notna()
        ]
        remaining = int((etf_features["date"] >= date).sum())
        if (
            valid["symbol"].nunique() >= required_symbols
            and remaining >= remaining_bars_required
        ):
            return pd.Timestamp(date)
    raise ValueError(
        "Could not find an overlapping start date with enough stock history and benchmark bars."
    )


def _build_price_matrix(
    panel: pd.DataFrame, dates: pd.Series, column: str, fill: bool = True
) -> pd.DataFrame:
    matrix = panel.pivot(index="date", columns="symbol", values=column).sort_index()
    matrix = matrix.reindex(pd.Index(pd.to_datetime(dates)))
    return matrix.ffill() if fill else matrix


def _holding_name_map(panel: pd.DataFrame) -> dict[str, str]:
    latest = panel.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    return dict(zip(latest["symbol"], latest["holding_name"], strict=False))


def run_equal_weight_benchmark(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    dates: pd.Series,
    config: StockRotationConfig,
) -> EqualWeightBenchmark:
    fee_rate = config.backtest.fee_bps / 10_000
    slippage_rate = config.backtest.slippage_bps / 10_000
    fixed_fee = float(config.portfolio.fixed_buy_fee_egp)
    share_precision = int(config.backtest.share_precision)
    name_map = _holding_name_map(panel)
    open_px = _build_price_matrix(panel, dates, "open", fill=False)
    close_px = _build_price_matrix(panel, dates, "close", fill=True)

    contributions = build_contribution_schedule(
        dates,
        0,
        len(dates) - 1,
        config.backtest.initial_cash,
        config.backtest.monthly_contribution,
    )
    cash = 0.0
    positions: dict[str, float] = {}
    equity = pd.Series(index=dates.index, dtype=float)
    flows = pd.Series(0.0, index=dates.index, dtype=float)
    actions: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    for i, date in enumerate(dates):
        contribution = float(contributions.iloc[i])
        cash += contribution
        flows.iloc[i] += contribution

        if contribution > 0:
            active_members = sorted(active_members_on_date(membership, date))
            tradable_symbols = [
                symbol
                for symbol in active_members
                if symbol in open_px.columns and pd.notna(open_px.loc[date, symbol])
            ]
            starting_cash = cash
            deployed_cash = 0.0
            purchased_symbols = 0
            total_buy_fees = 0.0
            per_symbol_budget = (
                0.0
                if not tradable_symbols
                else starting_cash / float(len(tradable_symbols))
            )

            for symbol in tradable_symbols:
                fill_price = float(open_px.loc[date, symbol]) * (1.0 + slippage_rate)
                max_shares = _round_shares(
                    max(0.0, per_symbol_budget - fixed_fee)
                    / (fill_price * (1.0 + fee_rate)),
                    share_precision,
                )
                affordable = _round_shares(
                    max(0.0, cash - fixed_fee) / (fill_price * (1.0 + fee_rate)),
                    share_precision,
                )
                buy_shares = min(max_shares, affordable)
                if buy_shares <= 0:
                    continue

                gross = buy_shares * fill_price
                bps_fee = gross * fee_rate
                total_cost = gross + bps_fee + fixed_fee
                cash -= total_cost
                deployed_cash += total_cost
                total_buy_fees += bps_fee + fixed_fee
                purchased_symbols += 1
                positions[symbol] = float(positions.get(symbol, 0.0)) + float(
                    buy_shares
                )
                actions.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "symbol": symbol,
                        "holding_name": name_map.get(symbol, symbol),
                        "action": "BUY",
                        "shares": float(buy_shares),
                        "price": fill_price,
                        "value": gross,
                        "fee": bps_fee + fixed_fee,
                        "fixed_fee": fixed_fee,
                        "bps_fee": bps_fee,
                    }
                )

            monthly_rows.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "contribution": contribution,
                    "active_members": len(active_members),
                    "tradable_members": len(tradable_symbols),
                    "purchased_symbols": purchased_symbols,
                    "per_symbol_budget": per_symbol_budget,
                    "deployed_cash": deployed_cash,
                    "buy_fees": total_buy_fees,
                    "cash_balance": cash,
                }
            )

        total_equity_close = cash
        for symbol, shares in positions.items():
            if pd.notna(close_px.loc[date, symbol]):
                total_equity_close += shares * float(close_px.loc[date, symbol])
        equity.iloc[i] = total_equity_close

    metrics = _compute_metrics(equity, flows, _build_trade_frame([]))
    return EqualWeightBenchmark(
        equity=equity,
        flows=flows,
        metrics=metrics,
        actions=actions,
        monthly_rows=monthly_rows,
    )


def run_stock_rotation_backtest(
    config_path: str | Path, run_id: str | None = None
) -> StockRotationRun:
    config = load_stock_rotation_config(config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf = (
        pd.read_csv(config.benchmark.etf_symbol_path, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    universe = pd.read_csv(
        Path(config.storage.root_dir) / config.storage.universe_filename
    )
    dividend_actions = load_dividend_actions(config)
    corporate_actions = load_corporate_actions(config)
    fundamentals = load_stock_fundamentals(config)

    features = build_stock_features(
        panel,
        config,
        benchmark=etf,
        dividend_actions=dividend_actions,
        corporate_actions=corporate_actions,
        fundamentals=fundamentals,
    )
    etf_features = build_etf_features(etf, config)
    start_date = find_overlap_start_date(features, etf_features, membership, config)

    etf = etf[etf["date"] >= start_date].reset_index(drop=True)
    dates = etf["date"]
    rebalance_dates = set(first_trading_days(dates))
    close_px = _build_price_matrix(panel, dates, "close", fill=True)
    open_px = _build_price_matrix(panel, dates, "open", fill=False)

    contributions = build_contribution_schedule(
        dates,
        0,
        len(dates) - 1,
        config.backtest.initial_cash,
        config.backtest.monthly_contribution,
    )
    cash = 0.0
    positions: dict[str, float] = {}
    equity = pd.Series(index=dates.index, dtype=float)
    flows = pd.Series(0.0, index=dates.index, dtype=float)
    actions: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []

    for i, date in enumerate(dates):
        contribution = float(contributions.iloc[i])
        cash += contribution
        flows.iloc[i] += contribution

        if pd.Timestamp(date) in rebalance_dates:
            prev_rows = etf_features.loc[
                etf_features["date"] < date, ["date", "ret_3m"]
            ]
            if not prev_rows.empty:
                prev_date = pd.Timestamp(prev_rows.iloc[-1]["date"])
                etf_ret_3m = float(prev_rows.iloc[-1]["ret_3m"])
                snapshot = features[features["date"] == prev_date]
                active_members = active_members_on_date(membership, date)
                selected = select_rebalance_portfolio(
                    snapshot, etf_ret_3m, config, active_members
                )

                total_equity = cash
                for symbol, shares in positions.items():
                    px = (
                        float(open_px.loc[date, symbol])
                        if pd.notna(open_px.loc[date, symbol])
                        else float(close_px.loc[date, symbol])
                    )
                    total_equity += shares * px

                desired_symbols = set(selected["symbol"])
                turnover_value = 0.0

                for symbol, shares in list(positions.items()):
                    if symbol not in desired_symbols and pd.notna(
                        open_px.loc[date, symbol]
                    ):
                        px = float(open_px.loc[date, symbol])
                        value = shares * px
                        cash += value
                        turnover_value += abs(value)
                        actions.append(
                            {
                                "date": str(pd.Timestamp(date).date()),
                                "symbol": symbol,
                                "action": "SELL",
                                "shares": shares,
                                "price": px,
                                "value": value,
                            }
                        )
                        positions.pop(symbol, None)

                target_weights = {
                    row.symbol: float(row.target_weight)
                    for row in selected.itertuples(index=False)
                }
                for symbol in desired_symbols:
                    if pd.isna(open_px.loc[date, symbol]):
                        continue
                    px = float(open_px.loc[date, symbol])
                    current_shares = float(positions.get(symbol, 0.0))
                    current_value = current_shares * px
                    target_value = total_equity * target_weights[symbol]
                    delta_value = target_value - current_value
                    if abs(delta_value) < 1e-9:
                        continue
                    delta_shares = int(delta_value / px)
                    if delta_shares > 0:
                        fixed_fee = config.portfolio.fixed_buy_fee_egp
                        affordable = int(max(0.0, cash - fixed_fee) / px)
                        buy_shares = float(min(delta_shares, affordable))
                        if buy_shares > 0:
                            value = buy_shares * px
                            cash -= value + fixed_fee
                            positions[symbol] = current_shares + buy_shares
                            turnover_value += abs(value) + fixed_fee
                            actions.append(
                                {
                                    "date": str(pd.Timestamp(date).date()),
                                    "symbol": symbol,
                                    "action": "BUY",
                                    "shares": buy_shares,
                                    "price": px,
                                    "value": value,
                                    "fee": fixed_fee,
                                }
                            )
                    elif delta_shares < 0:
                        sell_shares = float(min(abs(delta_shares), current_shares))
                        if sell_shares > 0:
                            value = sell_shares * px
                            cash += value
                            positions[symbol] = current_shares - sell_shares
                            if positions[symbol] <= 0:
                                positions.pop(symbol, None)
                            turnover_value += abs(value)
                            actions.append(
                                {
                                    "date": str(pd.Timestamp(date).date()),
                                    "symbol": symbol,
                                    "action": "SELL",
                                    "shares": sell_shares,
                                    "price": px,
                                    "value": value,
                                    "fee": 0.0,
                                }
                            )

                selected_records = selected[
                    ["symbol", "holding_name", "score", "target_weight"]
                ].copy()
                if selected_records.empty:
                    monthly_rows.append(
                        {
                            "date": str(pd.Timestamp(date).date()),
                            "symbol": None,
                            "holding_name": None,
                            "score": None,
                            "target_weight": None,
                            "cash_weight": 1.0,
                        }
                    )
                else:
                    cash_weight = 1.0 - float(selected_records["target_weight"].sum())
                    for row in selected_records.itertuples(index=False):
                        monthly_rows.append(
                            {
                                "date": str(pd.Timestamp(date).date()),
                                "symbol": row.symbol,
                                "holding_name": row.holding_name,
                                "score": row.score,
                                "target_weight": row.target_weight,
                                "cash_weight": cash_weight,
                            }
                        )
                turnover_rows.append(
                    {
                        "date": str(pd.Timestamp(date).date()),
                        "turnover_value": turnover_value,
                        "turnover_pct": 0.0
                        if total_equity <= 0
                        else turnover_value / total_equity,
                        "cash_balance": cash,
                    }
                )

        total_equity_close = cash
        for symbol, shares in positions.items():
            if pd.notna(close_px.loc[date, symbol]):
                total_equity_close += shares * float(close_px.loc[date, symbol])
        equity.iloc[i] = total_equity_close

    summary_metrics = _compute_metrics(equity, flows, _build_trade_frame([]))
    equal_weight = run_equal_weight_benchmark(panel, membership, dates, config)
    etf_dca = run_dca_benchmark(etf, 0, len(etf) - 1, config.backtest)
    etf_buy_hold = run_buy_hold_benchmark(etf, 0, len(etf) - 1)

    run_id = run_id or f"stock-rotation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / run_id)

    pd.DataFrame(monthly_rows).to_csv(
        run_dir / "selected_holdings_monthly.csv", index=False
    )
    pd.DataFrame(turnover_rows).to_csv(run_dir / "turnover_monthly.csv", index=False)
    pd.DataFrame(actions).to_csv(run_dir / "trade_actions.csv", index=False)
    pd.DataFrame(equal_weight.monthly_rows).to_csv(
        run_dir / "equal_weight_allocations_monthly.csv", index=False
    )
    pd.DataFrame(equal_weight.actions).to_csv(
        run_dir / "equal_weight_trade_actions.csv", index=False
    )
    pd.DataFrame(
        {
            "date": dates,
            "strategy_equity": equity.values,
            "equal_weight_equity": equal_weight.equity.values,
            "etf_dca_equity": etf_dca.equity.values,
            "etf_buy_hold_norm": etf_buy_hold.values,
        }
    ).to_csv(run_dir / "equity_curve.csv", index=False)

    summary = {
        "strategy_metrics": summary_metrics,
        "equal_weight_metrics": equal_weight.metrics,
        "etf_dca_metrics": etf_dca.metrics,
        "etf_buy_hold_return": float(etf_buy_hold.iloc[-1] - 1.0),
        "excess_vs_etf_dca": float(
            summary_metrics["twr_total_return"] - etf_dca.metrics["twr_total_return"]
        ),
        "equal_weight_excess_vs_etf_dca": float(
            equal_weight.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "start_date": str(pd.Timestamp(dates.iloc[0]).date()),
        "end_date": str(pd.Timestamp(dates.iloc[-1]).date()),
        "universe_size": int(universe["ticker"].nunique()),
        "membership_snapshots": int(membership["effective_date"].nunique()),
        "fixed_buy_fee_egp": float(config.portfolio.fixed_buy_fee_egp),
        "fundamental_rows_loaded": int(len(fundamentals)),
        "equal_weight_buy_count": int(len(equal_weight.actions)),
        "equal_weight_method": "buy_only_monthly_dca_equal_split_across_tradable_active_members",
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "panel_path": str(
                Path(config.storage.root_dir) / config.storage.panel_filename
            ),
            "fundamentals_path": str(
                Path(config.storage.root_dir) / config.storage.fundamentals_filename
            ),
            "benchmark_path": str(config.benchmark.etf_symbol_path),
            "start_date": summary["start_date"],
            "end_date": summary["end_date"],
        },
    )
    return StockRotationRun(run_id=run_id, run_dir=run_dir)


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


def _build_pullback_signal_matrix(
    panel: pd.DataFrame, calendar: pd.Series, pullback_params: dict[str, Any]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        price = (
            group[["date", "open", "high", "low", "close", "volume"]]
            .sort_values("date")
            .reset_index(drop=True)
        )
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

    signals = (
        pd.concat(frames, ignore_index=True)
        .pivot(index="date", columns="symbol", values="deploy_fraction")
        .sort_index()
    )
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
    *,
    date: pd.Timestamp,
    symbols: list[str],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    traded_value = 0.0
    fees_paid = 0.0
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
        traded_value += gross
        fees_paid += fee
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
    return cash, traded_value, fees_paid


def _buy_equal_split(
    *,
    date: pd.Timestamp,
    symbols: list[str],
    positions: dict[str, float],
    cash: float,
    open_row: pd.Series,
    fee_rate: float,
    slippage_rate: float,
    fixed_buy_fee: float,
    share_precision: int,
    reason: str,
    actions: list[dict[str, Any]],
) -> tuple[float, float, float]:
    tradable = [symbol for symbol in symbols if not pd.isna(open_row.get(symbol))]
    if not tradable or cash <= 0:
        return cash, 0.0, 0.0

    starting_cash = cash
    per_symbol_budget = starting_cash / float(len(tradable))
    traded_value = 0.0
    fees_paid = 0.0

    for symbol in tradable:
        raw_open = open_row.get(symbol)
        fill_price = float(raw_open) * (1.0 + slippage_rate)
        if fill_price <= 0:
            continue

        max_budget = min(per_symbol_budget, cash)
        if max_budget <= fixed_buy_fee:
            continue

        max_shares_budget = _round_shares(
            (max_budget - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)),
            share_precision,
        )
        max_shares_cash = _round_shares(
            (cash - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision
        )
        buy_shares = min(max_shares_budget, max_shares_cash)
        if buy_shares <= 0:
            continue

        gross = buy_shares * fill_price
        fee = gross * fee_rate + fixed_buy_fee
        cash -= gross + fee
        traded_value += gross
        fees_paid += fee
        positions[symbol] = float(positions.get(symbol, 0.0)) + float(buy_shares)
        actions.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "symbol": symbol,
                "action": "BUY",
                "shares": float(buy_shares),
                "price": fill_price,
                "value": gross,
                "fee": fee,
                "reason": reason,
            }
        )
    return cash, traded_value, fees_paid


def _rebalance_equal_weight(
    *,
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
) -> tuple[float, float, float]:
    if not selected_symbols or top_n <= 0:
        return cash, 0.0, 0.0

    tradable_selected = [
        symbol for symbol in selected_symbols if not pd.isna(open_row.get(symbol))
    ]
    if not tradable_selected:
        return cash, 0.0, 0.0

    valuation_row = open_row.fillna(close_row)
    total_equity = cash + _position_value(positions, valuation_row)
    target_value = total_equity / float(top_n)

    traded_value = 0.0
    fees_paid = 0.0

    for symbol in tradable_selected:
        shares = float(positions.get(symbol, 0.0))
        fill_price = float(open_row.get(symbol))
        current_value = shares * fill_price
        delta_value = current_value - target_value
        if delta_value <= 0:
            continue

        sell_shares = _round_shares(
            delta_value / (fill_price * (1.0 - slippage_rate)), share_precision
        )
        sell_shares = min(sell_shares, shares)
        if sell_shares <= 0:
            continue

        trade_price = fill_price * (1.0 - slippage_rate)
        gross = sell_shares * trade_price
        fee = gross * fee_rate
        cash += gross - fee
        traded_value += gross
        fees_paid += fee
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
                "reason": "rebalance_trim",
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

        affordable_target = _round_shares(
            (delta_value - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)),
            share_precision,
        )
        affordable_cash = _round_shares(
            (cash - fixed_buy_fee) / (fill_price * (1.0 + fee_rate)), share_precision
        )
        buy_shares = min(affordable_target, affordable_cash)
        if buy_shares <= 0:
            continue

        gross = buy_shares * fill_price
        fee = gross * fee_rate + fixed_buy_fee
        cash -= gross + fee
        traded_value += gross
        fees_paid += fee
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
                "reason": "rebalance_buy",
            }
        )

    return cash, traded_value, fees_paid


def _build_rebalance_dates(calendar: pd.Series, mode: str) -> list[pd.Timestamp]:
    if mode == "monthly":
        return first_trading_days(calendar)
    if mode == "annual":
        return annual_first_trading_days(calendar)
    raise ValueError(f"Unsupported rebalance mode: {mode}")


def _maybe_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def _maybe_date(value: Any) -> str | None:
    return None if pd.isna(value) else str(pd.Timestamp(value).date())


def _factor_coverage(snapshot: pd.DataFrame, column: str) -> float | None:
    if snapshot.empty or column not in snapshot.columns:
        return None
    return float(snapshot[column].notna().mean())


def _build_factor_coverage_report(
    *,
    features: pd.DataFrame,
    etf_features: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    config: StockRotationConfig,
    rebalance_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rebalance_date in _build_rebalance_dates(calendar, rebalance_mode):
        prev_rows = etf_features.loc[etf_features["date"] < rebalance_date, ["date"]]
        if prev_rows.empty:
            continue

        prev_date = pd.Timestamp(prev_rows.iloc[-1]["date"])
        snapshot = features[features["date"] == prev_date].copy()
        active_members = active_members_on_date(membership, rebalance_date)
        if active_members:
            snapshot = snapshot[snapshot["symbol"].isin(active_members)].copy()
        gated = _apply_selection_gates(snapshot, config)

        has_fundamentals = (
            gated["has_fundamentals"].astype(bool)
            if "has_fundamentals" in gated.columns and not gated.empty
            else pd.Series(dtype=bool)
        )
        row: dict[str, Any] = {
            "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
            "selection_date": str(prev_date.date()),
            "candidates_after_gates": int(gated["symbol"].nunique())
            if "symbol" in gated.columns
            else 0,
            "fundamental_rows_available": int(has_fundamentals.sum()),
            "fundamental_coverage_ratio": None
            if gated.empty
            else float(has_fundamentals.mean()),
        }
        for column in DERIVED_FACTOR_COLUMNS:
            if column == "fundamental_row_age_days":
                continue
            row[f"{column}_coverage"] = _factor_coverage(gated, column)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_missing_fundamental_warnings(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty or "has_fundamentals" not in selections.columns:
        return pd.DataFrame(columns=["rebalance_date", "warning", "symbols"])

    missing = selections[~selections["has_fundamentals"].fillna(False).astype(bool)]
    rows = []
    for rebalance_date, group in missing.groupby("rebalance_date", sort=True):
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "warning": "selected_without_fundamentals",
                "symbols": ",".join(group["symbol"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows, columns=["rebalance_date", "warning", "symbols"])


def _prepare_selection_plan(
    *,
    features: pd.DataFrame,
    etf_features: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    config: StockRotationConfig,
    rebalance_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, list[str]]]:
    rebalance_dates = _build_rebalance_dates(calendar, rebalance_mode)
    selection_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    previous_selection: set[str] = set()
    selection_map: dict[pd.Timestamp, list[str]] = {}

    for rebalance_date in rebalance_dates:
        prev_rows = etf_features.loc[
            etf_features["date"] < rebalance_date, ["date", "ret_3m"]
        ]
        if prev_rows.empty:
            continue

        prev_date = pd.Timestamp(prev_rows.iloc[-1]["date"])
        etf_ret_3m = float(prev_rows.iloc[-1]["ret_3m"])
        snapshot = features[features["date"] == prev_date].copy()
        active_members = active_members_on_date(membership, rebalance_date)
        if active_members:
            snapshot = snapshot[snapshot["symbol"].isin(active_members)].copy()

        before_count = int(snapshot["symbol"].nunique())

        if snapshot.empty:
            selected = snapshot
            coverage_fail = 0
            liquidity_fail = 0
            gated_count = 0
        else:
            if {"median_daily_value", "median_daily_volume", "coverage_ratio"}.issubset(
                set(snapshot.columns)
            ):
                liquidity_pass = (
                    snapshot["median_daily_value"]
                    >= float(config.selection.min_median_daily_value_egp)
                ) & (
                    snapshot["median_daily_volume"]
                    >= float(config.selection.min_median_daily_volume)
                )
                coverage_pass = snapshot["coverage_ratio"] >= float(
                    config.validation.min_coverage_ratio
                )
                liquidity_fail = int((~liquidity_pass).sum())
                coverage_fail = int((~coverage_pass).sum())
                gated = snapshot[liquidity_pass & coverage_pass].copy()
            else:
                gated = snapshot.copy()
                liquidity_fail = 0
                coverage_fail = 0

            gated_count = int(gated["symbol"].nunique())
            selected = select_rebalance_portfolio(
                gated,
                etf_ret_3m,
                config,
                active_members,
                previous_selection=previous_selection,
                turnover_buffer_score=config.portfolio.turnover_buffer_score,
                apply_gates=False,
            )

        selected_symbols = selected["symbol"].tolist() if not selected.empty else []
        selection_map[pd.Timestamp(rebalance_date)] = selected_symbols

        retained_count = int(len(set(selected_symbols) & previous_selection))
        cutoff_score = (
            float(selected["score"].iloc[-1])
            if (not selected.empty and "score" in selected.columns)
            else None
        )
        fundamental_coverage_ratio = (
            None
            if gated.empty or "has_fundamentals" not in gated.columns
            else float(gated["has_fundamentals"].fillna(False).astype(bool).mean())
        )
        selected_missing_fundamentals = (
            0
            if selected.empty or "has_fundamentals" not in selected.columns
            else int((~selected["has_fundamentals"].fillna(False).astype(bool)).sum())
        )
        diagnostics_rows.append(
            {
                "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                "selection_date": str(pd.Timestamp(prev_date).date()),
                "active_members": int(len(active_members)),
                "candidates_before_gates": before_count,
                "candidates_after_gates": gated_count,
                "coverage_fail_count": coverage_fail,
                "liquidity_fail_count": liquidity_fail,
                "selected_count": int(len(selected_symbols)),
                "retained_from_previous": retained_count,
                "cutoff_score": cutoff_score,
                "fundamental_coverage_ratio": fundamental_coverage_ratio,
                "selected_missing_fundamentals": selected_missing_fundamentals,
            }
        )

        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            values = row._asdict()
            selection_rows.append(
                {
                    "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                    "selection_date": str(pd.Timestamp(prev_date).date()),
                    "rank": rank,
                    "symbol": row.symbol,
                    "holding_name": row.holding_name,
                    "sector": values.get("sector"),
                    "score": float(row.score),
                    "ret_3m": float(row.ret_3m),
                    "ret_6m": float(row.ret_6m),
                    "median_daily_value": float(row.median_daily_value)
                    if "median_daily_value" in selected.columns
                    else None,
                    "median_daily_volume": float(row.median_daily_volume)
                    if "median_daily_volume" in selected.columns
                    else None,
                    "coverage_ratio": float(row.coverage_ratio)
                    if "coverage_ratio" in selected.columns
                    else None,
                    "score_momentum": float(row.score_momentum)
                    if "score_momentum" in selected.columns
                    else None,
                    "score_value": float(row.score_value)
                    if "score_value" in selected.columns
                    else None,
                    "score_quality": float(row.score_quality)
                    if "score_quality" in selected.columns
                    else None,
                    "score_growth": float(row.score_growth)
                    if "score_growth" in selected.columns
                    else None,
                    "score_low_risk": float(row.score_low_risk)
                    if "score_low_risk" in selected.columns
                    else None,
                    "score_liquidity": float(row.score_liquidity)
                    if "score_liquidity" in selected.columns
                    else None,
                    "has_fundamentals": bool(values.get("has_fundamentals", False)),
                    "fundamental_period_end": _maybe_date(
                        values.get("fundamental_period_end")
                    ),
                    "fundamental_filing_date": _maybe_date(
                        values.get("fundamental_filing_date")
                    ),
                    "net_income_ttm": _maybe_float(values.get("net_income_ttm")),
                    "earnings_yield": _maybe_float(values.get("earnings_yield")),
                    "pe_ratio": _maybe_float(values.get("pe_ratio")),
                    "book_to_price": _maybe_float(values.get("book_to_price")),
                    "pb_ratio": _maybe_float(values.get("pb_ratio")),
                    "roe": _maybe_float(values.get("roe")),
                    "roa": _maybe_float(values.get("roa")),
                    "cash_conversion": _maybe_float(values.get("cash_conversion")),
                    "leverage_inverse": _maybe_float(
                        values.get("leverage_inverse")
                    ),
                    "revenue_growth": _maybe_float(values.get("revenue_growth")),
                    "net_income_growth": _maybe_float(
                        values.get("net_income_growth")
                    ),
                    "positive_earnings_count_4p": _maybe_float(
                        values.get("positive_earnings_count_4p")
                    ),
                    "margin_stability_4p": _maybe_float(
                        values.get("margin_stability_4p")
                    ),
                    "fundamental_row_age_days": _maybe_float(
                        values.get("fundamental_row_age_days")
                    ),
                    "target_weight": float(row.target_weight),
                }
            )

        previous_selection = set(selected_symbols)

    if not diagnostics_rows:
        raise ValueError(
            "No rebalance diagnostics could be generated for stock-selection backtest."
        )

    selections_frame = pd.DataFrame(selection_rows)
    diagnostics_frame = pd.DataFrame(diagnostics_rows)
    return selections_frame, diagnostics_frame, selection_map


def _simulate_deployment(
    *,
    mode: str,
    calendar: pd.Series,
    selection_map: dict[pd.Timestamp, list[str]],
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    signals: pd.DataFrame,
    config: StockRotationConfig,
    top_n: int,
) -> DeploymentSimulation:
    fee_rate = config.backtest.fee_bps / 10_000
    slippage_rate = config.backtest.slippage_bps / 10_000
    fixed_buy_fee = config.portfolio.fixed_buy_fee_egp
    share_precision = config.backtest.share_precision

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
    rebalance_rows: list[dict[str, Any]] = []

    cash = 0.0
    positions: dict[str, float] = {}
    current_selection: list[str] = []

    for i, date in enumerate(calendar):
        date = pd.Timestamp(date)
        contribution = float(contributions.iloc[i])
        cash += contribution
        flows.iloc[i] += contribution

        open_row = open_px.loc[date]
        close_row = close_px.loc[date]

        if date in rebalance_dates:
            current_selection = selection_map[date]

        if date in rebalance_dates:
            equity_before_rebalance = cash + _position_value(
                positions, open_row.fillna(close_row)
            )
            turnover_value = 0.0
            rebalance_fees = 0.0

            undesired = sorted(
                [symbol for symbol in positions if symbol not in set(current_selection)]
            )
            cash, traded, fees = _sell_positions(
                date=date,
                symbols=undesired,
                positions=positions,
                cash=cash,
                open_row=open_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                reason="rebalance_exit",
                actions=actions,
            )
            turnover_value += traded
            rebalance_fees += fees

            if mode == "immediate":
                cash, traded, fees = _rebalance_equal_weight(
                    date=date,
                    selected_symbols=current_selection,
                    positions=positions,
                    cash=cash,
                    open_row=open_row,
                    close_row=close_row,
                    top_n=top_n,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    fixed_buy_fee=fixed_buy_fee,
                    share_precision=share_precision,
                    actions=actions,
                )
                turnover_value += traded
                rebalance_fees += fees

            rebalance_rows.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "turnover_value": turnover_value,
                    "turnover_pct": 0.0
                    if equity_before_rebalance <= 0
                    else turnover_value / equity_before_rebalance,
                    "rebalance_fees": rebalance_fees,
                    "cash_balance": cash,
                }
            )

        if mode == "immediate":
            cash, _, _ = _buy_equal_split(
                date=date,
                symbols=current_selection,
                positions=positions,
                cash=cash,
                open_row=open_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                fixed_buy_fee=fixed_buy_fee,
                share_precision=share_precision,
                reason="cash_deploy_current_basket",
                actions=actions,
            )
        elif mode == "pullback":
            active_signals: list[str] = []
            if i > 0 and current_selection:
                signal_row = signal_prev.loc[date]
                active_signals = [
                    symbol
                    for symbol in current_selection
                    if symbol in signal_row.index
                    and float(signal_row.get(symbol, 0.0)) > 0.0
                ]
            cash, _, _ = _buy_equal_split(
                date=date,
                symbols=active_signals,
                positions=positions,
                cash=cash,
                open_row=open_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                fixed_buy_fee=fixed_buy_fee,
                share_precision=share_precision,
                reason="pullback_signal_buy",
                actions=actions,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        equity.iloc[i] = cash + _position_value(positions, close_row)

    actions_frame = pd.DataFrame(actions)
    rebalance_frame = pd.DataFrame(rebalance_rows)
    total_fees = (
        0.0
        if actions_frame.empty
        else float(actions_frame.get("fee", pd.Series(dtype=float)).fillna(0.0).sum())
    )
    metrics = _compute_metrics(equity.copy(), flows.copy(), _build_trade_frame([]))
    return DeploymentSimulation(
        equity=equity,
        flows=flows,
        metrics=metrics,
        actions=actions_frame,
        rebalance_turnover=rebalance_frame,
        total_fees=total_fees,
    )


def _slice_holdout_metrics(
    equity: pd.Series, flows: pd.Series, holdout_start_idx: int
) -> dict[str, float]:
    if holdout_start_idx < 0:
        holdout_start_idx = 0
    if holdout_start_idx >= len(equity):
        holdout_start_idx = max(0, len(equity) - 1)
    holdout_equity = equity.iloc[holdout_start_idx:].reset_index(drop=True)
    holdout_flows = flows.iloc[holdout_start_idx:].reset_index(drop=True)
    return _compute_metrics(holdout_equity, holdout_flows, _build_trade_frame([]))


def _neighbor_configs(config: StockRotationConfig) -> list[StockRotationConfig]:
    variants: list[StockRotationConfig] = []

    for delta in (-0.01, 0.01):
        variant = copy.deepcopy(config)
        variant.portfolio.turnover_buffer_score = max(
            0.0, variant.portfolio.turnover_buffer_score + delta
        )
        variants.append(variant)

    for delta in (-0.02, 0.02):
        variant = copy.deepcopy(config)
        variant.validation.min_coverage_ratio = min(
            1.0, max(0.5, variant.validation.min_coverage_ratio + delta)
        )
        variants.append(variant)

    unique: list[StockRotationConfig] = []
    seen: set[tuple[float, float]] = set()
    for variant in variants:
        key = (
            float(variant.portfolio.turnover_buffer_score),
            float(variant.validation.min_coverage_ratio),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    return unique


def _build_candidate_summary(
    *,
    style: str,
    simulation: DeploymentSimulation,
    etf_holdout_metrics: dict[str, float],
    holdout_start_idx: int,
    neighbor_pass_rate: float,
    config: StockRotationConfig,
) -> dict[str, Any]:
    holdout_metrics = _slice_holdout_metrics(
        simulation.equity, simulation.flows, holdout_start_idx
    )
    holdout_excess = float(
        holdout_metrics["twr_total_return"] - etf_holdout_metrics["twr_total_return"]
    )
    mean_turnover = (
        0.0
        if simulation.rebalance_turnover.empty
        else float(simulation.rebalance_turnover["turnover_pct"].fillna(0.0).mean())
    )
    total_contributions = float(simulation.metrics["total_contributions"])
    fee_to_contributions_ratio = (
        0.0
        if total_contributions <= 0
        else float(simulation.total_fees / total_contributions)
    )

    constraints = config.model_selection
    passes_constraints = (
        holdout_excess >= constraints.min_holdout_excess_return
        and float(holdout_metrics["max_drawdown"]) <= constraints.max_drawdown
        and float(neighbor_pass_rate) >= constraints.min_neighbor_pass_rate
        and mean_turnover <= constraints.max_mean_rebalance_turnover_pct
        and fee_to_contributions_ratio <= constraints.max_fee_to_contributions_ratio
    )

    return {
        "style": style,
        "full_metrics": simulation.metrics,
        "holdout_metrics": holdout_metrics,
        "holdout_excess_vs_etf_dca": holdout_excess,
        "neighbor_pass_rate": float(neighbor_pass_rate),
        "mean_rebalance_turnover_pct": mean_turnover,
        "total_fees": float(simulation.total_fees),
        "fee_to_contributions_ratio": fee_to_contributions_ratio,
        "passes_constraints": bool(passes_constraints),
    }


def run_stock_selection_backtest(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    pullback_run_id: str | None = None,
    rebalance_mode: str = "monthly",
    start_date: str | None = None,
    end_date: str | None = None,
) -> StockSelectionRun:
    config = load_stock_rotation_config(config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
    dividend_actions = load_dividend_actions(config)
    corporate_actions = load_corporate_actions(config)
    fundamentals = load_stock_fundamentals(config)

    if start_date is not None:
        etf = etf[etf["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    if end_date is not None:
        etf = etf[etf["date"] <= pd.Timestamp(end_date)].reset_index(drop=True)
    if etf.empty:
        raise ValueError("No ETF benchmark dates remain after start/end filtering.")

    features = build_stock_features(
        panel,
        config,
        benchmark=etf,
        dividend_actions=dividend_actions,
        corporate_actions=corporate_actions,
        fundamentals=fundamentals,
    )
    etf_features = build_etf_features(etf, config)

    overlap_start = find_overlap_start_date(features, etf_features, membership, config)
    effective_start = max(
        pd.Timestamp(overlap_start), pd.Timestamp(etf["date"].iloc[0])
    )
    etf = etf[etf["date"] >= effective_start].reset_index(drop=True)
    if etf.empty:
        raise ValueError("No ETF benchmark dates remain after overlap alignment.")

    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(calendar) < 20:
        raise ValueError(
            "Not enough aligned bars to run enhanced stock-selection backtest."
        )

    selections, diagnostics, selection_map = _prepare_selection_plan(
        features=features,
        etf_features=etf_features,
        membership=membership,
        calendar=calendar,
        config=config,
        rebalance_mode=rebalance_mode,
    )
    factor_coverage = _build_factor_coverage_report(
        features=features,
        etf_features=etf_features,
        membership=membership,
        calendar=calendar,
        config=config,
        rebalance_mode=rebalance_mode,
    )
    missing_fundamental_warnings = _build_missing_fundamental_warnings(selections)

    open_px = _build_price_matrix(panel, calendar, "open", fill=False)
    close_px = _build_price_matrix(panel, calendar, "close", fill=True)
    pullback_source_run_id, pullback_params = _load_pullback_params(pullback_run_id)
    signals = _build_pullback_signal_matrix(panel, calendar, pullback_params)

    top_n = int(config.portfolio.top_n)
    immediate = _simulate_deployment(
        mode="immediate",
        calendar=calendar,
        selection_map=selection_map,
        open_px=open_px,
        close_px=close_px,
        signals=signals,
        config=config,
        top_n=top_n,
    )
    pullback = _simulate_deployment(
        mode="pullback",
        calendar=calendar,
        selection_map=selection_map,
        open_px=open_px,
        close_px=close_px,
        signals=signals,
        config=config,
        top_n=top_n,
    )

    etf_dca = run_dca_benchmark(etf, 0, len(etf) - 1, config.backtest)
    etf_buy_hold = run_buy_hold_benchmark(etf, 0, len(etf) - 1)

    holdout_bars = max(
        1, int(len(calendar) * float(config.model_selection.holdout_ratio))
    )
    holdout_start_idx = max(0, len(calendar) - holdout_bars)
    etf_holdout_metrics = _slice_holdout_metrics(
        etf_dca.equity, etf_dca.flows, holdout_start_idx
    )

    neighbor_pass_rates: dict[str, float] = {"immediate": 1.0, "pullback": 1.0}
    for style in ["immediate", "pullback"]:
        flags: list[float] = []
        for neighbor in _neighbor_configs(config):
            _, _, neighbor_selection_map = _prepare_selection_plan(
                features=features,
                etf_features=etf_features,
                membership=membership,
                calendar=calendar,
                config=neighbor,
                rebalance_mode=rebalance_mode,
            )
            neighbor_sim = _simulate_deployment(
                mode=style,
                calendar=calendar,
                selection_map=neighbor_selection_map,
                open_px=open_px,
                close_px=close_px,
                signals=signals,
                config=neighbor,
                top_n=int(neighbor.portfolio.top_n),
            )
            neighbor_holdout = _slice_holdout_metrics(
                neighbor_sim.equity, neighbor_sim.flows, holdout_start_idx
            )
            neighbor_excess = float(
                neighbor_holdout["twr_total_return"]
                - etf_holdout_metrics["twr_total_return"]
            )
            neighbor_turnover = (
                0.0
                if neighbor_sim.rebalance_turnover.empty
                else float(
                    neighbor_sim.rebalance_turnover["turnover_pct"].fillna(0.0).mean()
                )
            )
            neighbor_contrib = float(neighbor_sim.metrics["total_contributions"])
            neighbor_fee_ratio = (
                0.0
                if neighbor_contrib <= 0
                else float(neighbor_sim.total_fees / neighbor_contrib)
            )

            constraints = config.model_selection
            neighbor_pass = (
                neighbor_excess >= constraints.min_holdout_excess_return
                and float(neighbor_holdout["max_drawdown"]) <= constraints.max_drawdown
                and neighbor_turnover <= constraints.max_mean_rebalance_turnover_pct
                and neighbor_fee_ratio <= constraints.max_fee_to_contributions_ratio
            )
            flags.append(1.0 if neighbor_pass else 0.0)
        neighbor_pass_rates[style] = (
            1.0 if not flags else float(sum(flags) / len(flags))
        )

    candidates = [
        _build_candidate_summary(
            style="immediate",
            simulation=immediate,
            etf_holdout_metrics=etf_holdout_metrics,
            holdout_start_idx=holdout_start_idx,
            neighbor_pass_rate=neighbor_pass_rates["immediate"],
            config=config,
        ),
        _build_candidate_summary(
            style="pullback",
            simulation=pullback,
            etf_holdout_metrics=etf_holdout_metrics,
            holdout_start_idx=holdout_start_idx,
            neighbor_pass_rate=neighbor_pass_rates["pullback"],
            config=config,
        ),
    ]

    passed = [candidate for candidate in candidates if candidate["passes_constraints"]]
    if passed:
        passed.sort(
            key=lambda item: (
                item["holdout_excess_vs_etf_dca"],
                item["holdout_metrics"]["cagr"],
                -item["holdout_metrics"]["max_drawdown"],
            ),
            reverse=True,
        )
        selected_style = passed[0]["style"]
        selection_reason = "Best holdout excess among constraint-passing styles."
    else:
        selected_style = None
        selection_reason = "No style passed all model-selection constraints."

    best_holdout = max(candidates, key=lambda item: item["holdout_excess_vs_etf_dca"])[
        "style"
    ]

    run_id = (
        run_id
        or f"stock-select-{rebalance_mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir = ensure_dir(Path("runs") / run_id)

    selections.to_csv(run_dir / "selected_holdings_rebalance.csv", index=False)
    diagnostics.to_csv(run_dir / "selection_diagnostics.csv", index=False)
    factor_coverage.to_csv(run_dir / "factor_coverage_rebalance.csv", index=False)
    missing_fundamental_warnings.to_csv(
        run_dir / "missing_fundamental_warnings.csv", index=False
    )
    immediate.actions.to_csv(run_dir / "actions_immediate.csv", index=False)
    pullback.actions.to_csv(run_dir / "actions_pullback.csv", index=False)
    immediate.rebalance_turnover.to_csv(run_dir / "turnover_immediate.csv", index=False)
    pullback.rebalance_turnover.to_csv(run_dir / "turnover_pullback.csv", index=False)

    pd.DataFrame(
        {
            "date": calendar.values,
            "immediate_equity": immediate.equity.values,
            "pullback_equity": pullback.equity.values,
            "etf_dca_equity": etf_dca.equity.values,
            "etf_buy_hold_norm": etf_buy_hold.values,
        }
    ).to_csv(run_dir / "equity_curve_selection.csv", index=False)

    summary = {
        "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
        "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "rebalance_mode": rebalance_mode,
        "top_n": top_n,
        "selection_method": (
            "sector_aware_multi_factor_core_v1"
            if config.selection.method == "sector_multifactor"
            else "institutional_multi_factor_core_v0"
            if config.selection.method == "multi_factor_core"
            else "kama_plus_3m6mrs_with_liquidity_coverage_gates"
        ),
        "turnover_buffer_score": float(config.portfolio.turnover_buffer_score),
        "pullback_source_run_id": pullback_source_run_id,
        "pullback_params": pullback_params,
        "dividend_actions_loaded": int(len(dividend_actions)),
        "corporate_actions_loaded": int(len(corporate_actions)),
        "fundamental_rows_loaded": int(len(fundamentals)),
        "immediate_metrics": immediate.metrics,
        "pullback_metrics": pullback.metrics,
        "etf_dca_metrics": etf_dca.metrics,
        "etf_buy_hold_return": float(etf_buy_hold.iloc[-1] - 1.0),
        "holdout_ratio": float(config.model_selection.holdout_ratio),
        "holdout_start_date": str(
            pd.Timestamp(calendar.iloc[holdout_start_idx]).date()
        ),
        "etf_holdout_metrics": etf_holdout_metrics,
        "candidate_evaluation": candidates,
        "selected_style": selected_style,
        "best_holdout_style": best_holdout,
        "selection_reason": selection_reason,
        "constraints": {
            "min_holdout_excess_return": float(
                config.model_selection.min_holdout_excess_return
            ),
            "max_drawdown": float(config.model_selection.max_drawdown),
            "min_neighbor_pass_rate": float(
                config.model_selection.min_neighbor_pass_rate
            ),
            "max_mean_rebalance_turnover_pct": float(
                config.model_selection.max_mean_rebalance_turnover_pct
            ),
            "max_fee_to_contributions_ratio": float(
                config.model_selection.max_fee_to_contributions_ratio
            ),
        },
        "coverage_min_selected": (
            None
            if selections.empty
            else float(selections["coverage_ratio"].dropna().min())
        ),
        "selection_rebalance_count": int(len(diagnostics)),
        "missing_fundamental_warning_count": int(len(missing_fundamental_warnings)),
    }
    write_json(run_dir / "summary_selection.json", summary)
    write_json(
        run_dir / "manifest_selection.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "benchmark_path": str(config.benchmark.etf_symbol_path),
            "panel_path": str(
                Path(config.storage.root_dir) / config.storage.panel_filename
            ),
            "fundamentals_path": str(
                Path(config.storage.root_dir) / config.storage.fundamentals_filename
            ),
            "rebalance_mode": rebalance_mode,
            "start_date": summary["start_date"],
            "end_date": summary["end_date"],
        },
    )

    return StockSelectionRun(run_id=run_id, run_dir=run_dir)
