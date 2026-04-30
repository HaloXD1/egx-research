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

from egx_research.backtest import _compute_metrics, run_dca_benchmark
from egx_research.data import load_price_data
from egx_research.stock_rotation import (
    active_members_on_date,
    build_stock_features,
    first_trading_days,
    load_corporate_actions,
    load_dividend_actions,
    load_membership_snapshots,
    load_stock_fundamentals,
    load_stock_panel,
    score_snapshot,
)
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.stock_strategy_research import (
    StrategySimulation,
    build_strategy_feature_panel,
    load_disclosure_events,
    simulate_event_driven_strategy,
)
from egx_research.utils import ensure_dir, write_json


@dataclass
class StockStrategyValidationRun:
    run_id: str
    run_dir: Path


CURRENT_REBOUND_PARAMS: dict[str, Any] = {
    "scan_mode": "monthly",
    "top_n": 6,
    "max_positions": 3,
    "atr_len": 15,
    "target_atr": 4.482664295015792,
    "stop_atr": 1.056478065634162,
    "trail_atr": 4.811367627216187,
    "max_hold_bars": 38,
    "trend_len": 100,
    "rsi_len": 5,
    "rsi_entry": 44.80609886093903,
    "rsi_reclaim": 54.98431843581888,
    "bb_len": 40,
    "bb_std": 1.8495062824542354,
    "cci_len": 27,
    "cci_reclaim": -118.02475902638353,
}


REBOUND_MAX5_V1_PARAMS: dict[str, Any] = {
    "scan_mode": "monthly",
    "top_n": 11,
    "max_positions": 5,
    "atr_len": 17,
    "target_atr": 4.315735805005512,
    "stop_atr": 1.8288119451257165,
    "trail_atr": 2.613699668526334,
    "max_hold_bars": 21,
    "trend_len": 118,
    "rsi_len": 8,
    "rsi_entry": 44.42461952950172,
    "rsi_reclaim": 51.81425005623103,
    "bb_len": 35,
    "bb_std": 1.601592838006187,
    "cci_len": 14,
    "cci_reclaim": -59.18150878256561,
    "fixed_sell_fee_egp": 5.0,
}


REBOUND_MAX5_V2_PARAMS: dict[str, Any] = {
    **REBOUND_MAX5_V1_PARAMS,
    "max_position_weight": 0.25,
    "market_filter_ma_len": 100,
    "market_filter_slope_len": 10,
}


REBOUND_MAX5_V3_PARAMS: dict[str, Any] = {
    **REBOUND_MAX5_V2_PARAMS,
    "factor_score_weight": 0.08,
    "rebound_score_weight": 0.80,
    "momentum_score_weight": 0.07,
    "quality_score_weight": 0.03,
    "low_risk_score_weight": 0.02,
    "min_factor_score": 0.0,
    "min_momentum_score": 0.0,
    "min_quality_score": 0.0,
    "max_factor_score_age_days": 75,
    "min_median_daily_value_egp": 100_000.0,
    "min_median_daily_volume": 1_000.0,
    "market_filter_ma_len": 100,
    "market_filter_slope_len": 10,
    "market_filter_breadth_sma_len": 100,
    "market_filter_breadth_min": 0.45,
    "market_filter_breadth_soft_min": 0.35,
    "market_filter_breadth_slope_len": 20,
}


def build_market_regime_filter(
    etf: pd.DataFrame,
    calendar: pd.Series,
    *,
    ma_len: int = 180,
    slope_len: int = 20,
) -> pd.Series:
    frame = (
        etf[["date", "close"]]
        .assign(date=lambda data: pd.to_datetime(data["date"]))
        .sort_values("date")
        .drop_duplicates("date")
    )
    aligned = pd.DataFrame({"date": pd.to_datetime(calendar)}).merge(
        frame, on="date", how="left"
    )
    close = aligned["close"].ffill()
    long_ma = close.rolling(int(ma_len), min_periods=max(20, int(ma_len // 3))).mean()
    medium_ma = close.rolling(80, min_periods=30).mean()
    improving = long_ma > long_ma.shift(int(slope_len))
    market_ok = (close > long_ma) | ((close > medium_ma) & improving)
    return pd.Series(market_ok.fillna(False).to_numpy(), index=pd.to_datetime(calendar))


def _aligned_market_regime(
    prices: pd.DataFrame,
    calendar: pd.Series,
    *,
    ma_len: int,
    slope_len: int,
) -> pd.Series:
    frame = (
        prices[["date", "close"]]
        .assign(date=lambda data: pd.to_datetime(data["date"]))
        .sort_values("date")
        .drop_duplicates("date")
    )
    aligned = pd.DataFrame({"date": pd.to_datetime(calendar)}).merge(
        frame, on="date", how="left"
    )
    close = aligned["close"].ffill()
    long_ma = close.rolling(int(ma_len), min_periods=max(20, int(ma_len // 3))).mean()
    medium_ma = close.rolling(80, min_periods=30).mean()
    improving = long_ma > long_ma.shift(int(slope_len))
    return pd.Series(
        ((close > long_ma) | ((close > medium_ma) & improving)).fillna(False).to_numpy(),
        index=pd.to_datetime(calendar),
    )


def _breadth_regime_frame(
    *,
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    sma_len: int,
    min_breadth: float,
    soft_min_breadth: float,
    slope_len: int,
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(calendar)).sort_values().drop_duplicates()
    frames: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        frame = group.sort_values("date").copy()
        frame["stock_sma"] = frame["close"].rolling(
            int(sma_len), min_periods=max(20, int(sma_len // 2))
        ).mean()
        frame["above_sma"] = frame["close"] > frame["stock_sma"]
        frames.append(frame[["date", "symbol", "above_sma"]])
    if not frames:
        return pd.DataFrame(
            {
                "date": dates,
                "breadth": 0.0,
                "breadth_regime_ok": False,
            }
        )

    breadth_source = pd.concat(frames, ignore_index=True)
    matrix = (
        breadth_source.pivot_table(
            index="date", columns="symbol", values="above_sma", aggfunc="last"
        )
        .sort_index()
        .reindex(dates)
        .ffill()
    )
    breadth_values: list[float] = []
    for date in dates:
        active = active_members_on_date(membership, pd.Timestamp(date))
        active_cols = [symbol for symbol in active if symbol in matrix.columns]
        if not active_cols:
            breadth_values.append(np.nan)
            continue
        breadth_values.append(float(matrix.loc[date, active_cols].astype(float).mean()))

    breadth = pd.Series(breadth_values, index=dates, dtype=float)
    rising = breadth > breadth.shift(int(slope_len))
    ok = (breadth >= float(min_breadth)) | (
        (breadth >= float(soft_min_breadth)) & rising
    )
    return pd.DataFrame(
        {
            "date": dates,
            "breadth": breadth.fillna(0.0).to_numpy(),
            "breadth_regime_ok": ok.fillna(False).to_numpy(),
        }
    )


def build_market_regime_filter_v3(
    *,
    etf: pd.DataFrame,
    index: pd.DataFrame,
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    params: dict[str, Any],
) -> pd.DataFrame:
    ma_len = int(params.get("market_filter_ma_len", 100))
    slope_len = int(params.get("market_filter_slope_len", 10))
    dates = pd.Series(pd.to_datetime(calendar)).sort_values().drop_duplicates()
    etf_ok = _aligned_market_regime(
        etf,
        dates,
        ma_len=ma_len,
        slope_len=slope_len,
    )
    index_ok = _aligned_market_regime(
        index,
        dates,
        ma_len=ma_len,
        slope_len=slope_len,
    )
    breadth = _breadth_regime_frame(
        panel=panel,
        membership=membership,
        calendar=dates,
        sma_len=int(params.get("market_filter_breadth_sma_len", 100)),
        min_breadth=float(params.get("market_filter_breadth_min", 0.45)),
        soft_min_breadth=float(params.get("market_filter_breadth_soft_min", 0.35)),
        slope_len=int(params.get("market_filter_breadth_slope_len", 20)),
    ).set_index("date")
    votes = (
        etf_ok.astype(int)
        + index_ok.astype(int)
        + breadth["breadth_regime_ok"].reindex(dates).fillna(False).astype(int)
    )
    result = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "etf_regime_ok": etf_ok.to_numpy(),
            "index_regime_ok": index_ok.to_numpy(),
            "breadth": breadth["breadth"].reindex(dates).fillna(0.0).to_numpy(),
            "breadth_regime_ok": breadth["breadth_regime_ok"]
            .reindex(dates)
            .fillna(False)
            .to_numpy(),
            "market_regime_ok": (votes >= 2).to_numpy(),
        }
    )
    regime_cols = [
        "etf_regime_ok",
        "index_regime_ok",
        "breadth_regime_ok",
        "market_regime_ok",
    ]
    result[regime_cols] = result[regime_cols].shift(1).fillna(False).astype(bool)
    result["breadth"] = result["breadth"].shift(1).fillna(0.0)
    return result


def _latest_snapshot_before(features: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    prior = features[features["date"] < pd.Timestamp(date)].copy()
    if prior.empty:
        return prior
    return (
        prior.sort_values(["symbol", "date"])
        .drop_duplicates("symbol", keep="last")
        .reset_index(drop=True)
    )


def _build_monthly_factor_scores(
    *,
    stock_features: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    config: Any,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    output_columns = [
        "symbol",
        "date",
        "factor_date",
        "sector",
        "factor_score",
        "score_momentum",
        "score_value",
        "score_quality",
        "score_growth",
        "score_low_risk",
        "score_liquidity",
        "has_fundamentals",
        "fundamental_filing_date",
        "fundamental_row_age_days",
    ]
    for rebalance_date in first_trading_days(calendar):
        snapshot = _latest_snapshot_before(stock_features, pd.Timestamp(rebalance_date))
        if snapshot.empty:
            continue
        active = active_members_on_date(membership, pd.Timestamp(rebalance_date))
        snapshot = snapshot[snapshot["symbol"].isin(active)].copy()
        if snapshot.empty:
            continue
        scored = score_snapshot(snapshot, etf_ret_3m=0.0, config=config)
        if scored.empty or "score" not in scored.columns:
            continue
        scored = scored.copy()
        scored["factor_date"] = pd.Timestamp(rebalance_date)
        rows.append(scored.rename(columns={"score": "factor_score"}).reindex(columns=output_columns))
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])


def _merge_factor_scores_asof(
    features: pd.DataFrame, factor_scores: pd.DataFrame
) -> pd.DataFrame:
    factor_columns = [
        "date",
        "factor_date",
        "sector",
        "factor_score",
        "score_momentum",
        "score_value",
        "score_quality",
        "score_growth",
        "score_low_risk",
        "score_liquidity",
        "has_fundamentals",
        "fundamental_filing_date",
        "fundamental_row_age_days",
    ]
    if factor_scores.empty:
        result = features.copy()
        for column in factor_columns:
            if column != "date":
                result[column] = np.nan
        return result

    merged_frames: list[pd.DataFrame] = []
    for symbol, group in features.groupby("symbol", sort=False):
        left = group.sort_values("date").copy()
        right = factor_scores[factor_scores["symbol"] == str(symbol)].sort_values("date")
        if right.empty:
            left["factor_score"] = np.nan
            merged_frames.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right[factor_columns],
            on="date",
            direction="backward",
            suffixes=("", "_factor"),
        )
        merged_frames.append(merged)
    return pd.concat(merged_frames, ignore_index=True)


def build_rebound_v3_feature_panel(
    *,
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    config: Any,
    benchmark: pd.DataFrame,
    params: dict[str, Any],
    disclosure_events: pd.DataFrame,
    fundamentals: pd.DataFrame,
    dividend_actions: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_strategy_feature_panel(
        panel,
        family="rebound",
        params=params,
        disclosure_events=disclosure_events,
    )
    stock_features = build_stock_features(
        panel,
        config,
        benchmark=benchmark,
        dividend_actions=dividend_actions,
        corporate_actions=corporate_actions,
        fundamentals=fundamentals,
    )
    factor_scores = _build_monthly_factor_scores(
        stock_features=stock_features,
        membership=membership,
        calendar=calendar,
        config=config,
    )
    features = _merge_factor_scores_asof(features, factor_scores)
    features["raw_rebound_rank_score"] = features["rank_score"]
    features["rebound_rank_pct"] = features.groupby("date")[
        "raw_rebound_rank_score"
    ].rank(method="average", pct=True)

    fill_half = [
        "factor_score",
        "score_momentum",
        "score_quality",
        "score_low_risk",
    ]
    for column in fill_half:
        if column not in features.columns:
            features[column] = np.nan
    factor_age = (
        pd.to_datetime(features["date"]) - pd.to_datetime(features.get("factor_date"))
    ).dt.days
    factor_is_fresh = factor_age <= float(params.get("max_factor_score_age_days", 50))
    factor_gate = (
        features["factor_score"].fillna(0.0) >= float(params.get("min_factor_score", 0.0))
    )
    momentum_gate = (
        features["score_momentum"].fillna(0.0)
        >= float(params.get("min_momentum_score", 0.0))
    )
    quality_gate = (
        features["score_quality"].fillna(0.0)
        >= float(params.get("min_quality_score", 0.0))
    )
    features["entry_signal"] = (
        features["entry_signal"].fillna(False)
        & factor_is_fresh.fillna(False)
        & factor_gate
        & momentum_gate
        & quality_gate
    )
    weights = {
        "rebound_rank_pct": float(params.get("rebound_score_weight", 0.45)),
        "factor_score": float(params.get("factor_score_weight", 0.25)),
        "score_momentum": float(params.get("momentum_score_weight", 0.15)),
        "score_quality": float(params.get("quality_score_weight", 0.10)),
        "score_low_risk": float(params.get("low_risk_score_weight", 0.05)),
    }
    total_weight = max(sum(weights.values()), 1e-9)
    features["rank_score"] = sum(
        features[column].fillna(0.5).astype(float) * weight
        for column, weight in weights.items()
    ) / total_weight
    return features, factor_scores


def _slice_metrics(
    simulation: StrategySimulation,
    dates: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, float]:
    mask = (dates >= start_date) & (dates <= end_date)
    if not bool(mask.any()):
        return _compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame())
    return _compute_metrics(
        simulation.equity.loc[mask].reset_index(drop=True),
        simulation.flows.loc[mask].reset_index(drop=True),
        pd.DataFrame(),
    )


def _sample_rebound_params(
    trial: optuna.trial.Trial,
    *,
    max_positions: int,
    fixed_fee_on_sell: bool,
    fixed_sell_fee: float,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "scan_mode": trial.suggest_categorical("scan_mode", ["weekly", "monthly"]),
        "top_n": trial.suggest_int("top_n", max_positions, 12),
        "max_positions": max_positions,
        "atr_len": trial.suggest_int("atr_len", 7, 24),
        "target_atr": trial.suggest_float("target_atr", 2.0, 6.5),
        "stop_atr": trial.suggest_float("stop_atr", 0.8, 3.2),
        "trail_atr": trial.suggest_float("trail_atr", 1.5, 5.5),
        "max_hold_bars": trial.suggest_int("max_hold_bars", 15, 126),
        "trend_len": trial.suggest_int("trend_len", 45, 180),
        "rsi_len": trial.suggest_int("rsi_len", 5, 24),
        "rsi_entry": trial.suggest_float("rsi_entry", 20.0, 48.0),
        "rsi_reclaim": trial.suggest_float("rsi_reclaim", 35.0, 60.0),
        "bb_len": trial.suggest_int("bb_len", 14, 45),
        "bb_std": trial.suggest_float("bb_std", 1.4, 2.7),
        "cci_len": trial.suggest_int("cci_len", 8, 45),
        "cci_reclaim": trial.suggest_float("cci_reclaim", -140.0, -15.0),
    }
    if fixed_fee_on_sell:
        params["fixed_sell_fee_egp"] = float(fixed_sell_fee)
    return params


def _normalise_seed_params(
    params: dict[str, Any], *, max_positions: int, fixed_fee_on_sell: bool, fixed_sell_fee: float
) -> dict[str, Any]:
    seeded = dict(params)
    seeded["max_positions"] = max_positions
    seeded["top_n"] = max(max_positions, int(seeded.get("top_n", max_positions)))
    if fixed_fee_on_sell:
        seeded["fixed_sell_fee_egp"] = float(fixed_sell_fee)
    else:
        seeded.pop("fixed_sell_fee_egp", None)
    return seeded


def _score_candidate(
    *,
    metrics: dict[str, float],
    benchmark: dict[str, float],
    min_cagr_floor: float,
) -> float:
    cagr = float(metrics["cagr"])
    max_drawdown = float(metrics["max_drawdown"])
    sharpe = float(metrics["sharpe"])
    return_dd = float(metrics["return_dd"])
    excess_cagr = cagr - float(benchmark["cagr"])
    excess_return = float(metrics["twr_total_return"]) - float(benchmark["twr_total_return"])
    drawdown_saving = float(benchmark["max_drawdown"]) - max_drawdown
    score = (
        1.15 * excess_cagr
        + 0.20 * excess_return
        + 0.22 * sharpe
        + 0.20 * return_dd
        + 0.55 * drawdown_saving
    )
    if cagr < min_cagr_floor:
        score -= (min_cagr_floor - cagr) * 2.0
    if max_drawdown > 0.45:
        score -= (max_drawdown - 0.45) * 1.5
    return float(score)


def _simulate(
    *,
    panel: pd.DataFrame,
    calendar: pd.Series,
    membership: pd.DataFrame,
    config: Any,
    disclosure_events: pd.DataFrame,
    params: dict[str, Any],
    market_filter: pd.Series | None = None,
) -> StrategySimulation:
    features = build_strategy_feature_panel(
        panel,
        family="rebound",
        params=params,
        disclosure_events=disclosure_events,
    )
    return simulate_event_driven_strategy(
        panel=panel,
        features=features,
        calendar=calendar,
        membership=membership,
        config=config,
        family="rebound",
        params=params,
        market_filter=market_filter,
    )


def _optimise_for_train_window(
    *,
    panel: pd.DataFrame,
    calendar: pd.Series,
    membership: pd.DataFrame,
    config: Any,
    disclosure_events: pd.DataFrame,
    train_end: pd.Timestamp,
    benchmark_metrics: dict[str, float],
    max_positions: int,
    trials: int,
    seed: int,
    fixed_fee_on_sell: bool,
) -> tuple[dict[str, Any], StrategySimulation, dict[str, float], list[dict[str, Any]]]:
    train_start = pd.Timestamp(calendar.iloc[0])
    best_score = -float("inf")
    best_params = _normalise_seed_params(
        CURRENT_REBOUND_PARAMS,
        max_positions=max_positions,
        fixed_fee_on_sell=fixed_fee_on_sell,
        fixed_sell_fee=float(config.portfolio.fixed_buy_fee_egp),
    )
    best_simulation = _simulate(
        panel=panel,
        calendar=calendar,
        membership=membership,
        config=config,
        disclosure_events=disclosure_events,
        params=best_params,
    )
    best_train = _slice_metrics(best_simulation, calendar, train_start, train_end)
    min_cagr_floor = max(0.0, float(benchmark_metrics["cagr"]) * 0.85)
    best_score = _score_candidate(
        metrics=best_train,
        benchmark=benchmark_metrics,
        min_cagr_floor=min_cagr_floor,
    )
    rows = [
        {
            "number": -1,
            "score": best_score,
            "max_positions": max_positions,
            "params": json.dumps(best_params, sort_keys=True),
            **{f"train_{key}": value for key, value in best_train.items()},
        }
    ]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )
    study.enqueue_trial(
        {
            key: value
            for key, value in best_params.items()
            if key != "fixed_sell_fee_egp"
        }
    )

    def objective(trial: optuna.trial.Trial) -> float:
        nonlocal best_score, best_params, best_simulation, best_train
        params = _sample_rebound_params(
            trial,
            max_positions=max_positions,
            fixed_fee_on_sell=fixed_fee_on_sell,
            fixed_sell_fee=float(config.portfolio.fixed_buy_fee_egp),
        )
        simulation = _simulate(
            panel=panel,
            calendar=calendar,
            membership=membership,
            config=config,
            disclosure_events=disclosure_events,
            params=params,
        )
        train_metrics = _slice_metrics(simulation, calendar, train_start, train_end)
        score = _score_candidate(
            metrics=train_metrics,
            benchmark=benchmark_metrics,
            min_cagr_floor=min_cagr_floor,
        )
        rows.append(
            {
                "number": trial.number,
                "score": score,
                "max_positions": max_positions,
                "params": json.dumps(params, sort_keys=True),
                **{f"train_{key}": value for key, value in train_metrics.items()},
            }
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_simulation = simulation
            best_train = train_metrics
        return score

    study.optimize(objective, n_trials=int(trials), show_progress_bar=False)
    return best_params, best_simulation, best_train, rows


def _period_benchmark_metrics(
    *,
    etf: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    config: Any,
) -> dict[str, float]:
    return run_dca_benchmark(etf, start_idx, end_idx, config.backtest).metrics


def _yearly_rows(
    *,
    simulation: StrategySimulation,
    etf: pd.DataFrame,
    calendar: pd.Series,
    config: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in sorted(calendar.dt.year.unique()):
        mask = calendar.dt.year == int(year)
        if not bool(mask.any()):
            continue
        start = pd.Timestamp(calendar[mask].iloc[0])
        end = pd.Timestamp(calendar[mask].iloc[-1])
        start_idx = _date_index(calendar, start, side="left")
        end_idx = _date_index(calendar, end, side="right")
        strategy_metrics = _slice_metrics(simulation, calendar, start, end)
        etf_metrics = _period_benchmark_metrics(
            etf=etf,
            start_idx=start_idx,
            end_idx=end_idx,
            config=config,
        )
        rows.append(
            {
                "year": int(year),
                **{f"strategy_{key}": value for key, value in strategy_metrics.items()},
                **{f"etf_{key}": value for key, value in etf_metrics.items()},
                "excess_twr_vs_etf_dca": float(
                    strategy_metrics["twr_total_return"]
                    - etf_metrics["twr_total_return"]
                ),
            }
        )
    return rows


def run_rebound_max5_v2(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    run_id: str | None = None,
    params: dict[str, Any] | None = None,
    train_end: str = "2023-12-31",
    use_market_filter: bool = True,
) -> StockStrategyValidationRun:
    config = load_stock_rotation_config(config_path)
    selected_params = dict(params or REBOUND_MAX5_V2_PARAMS)
    selected_params["fixed_sell_fee_egp"] = float(config.portfolio.fixed_buy_fee_egp)

    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    disclosure_events = load_disclosure_events(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    panel = panel[panel["date"] >= pd.Timestamp(calendar.iloc[0])].reset_index(drop=True)
    actual_run_id = run_id or f"rebound-max5-v2-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    market_filter = (
        build_market_regime_filter(
            etf,
            calendar,
            ma_len=int(selected_params.get("market_filter_ma_len", 180)),
            slope_len=int(selected_params.get("market_filter_slope_len", 20)),
        )
        if use_market_filter
        else None
    )
    simulation = _simulate(
        panel=panel,
        calendar=calendar,
        membership=membership,
        config=config,
        disclosure_events=disclosure_events,
        params=selected_params,
        market_filter=market_filter,
    )
    etf_full = run_dca_benchmark(etf, 0, len(calendar) - 1, config.backtest)

    train_end_date = pd.Timestamp(train_end)
    test_start_date = pd.Timestamp(calendar[calendar > train_end_date].iloc[0])
    train_metrics = _slice_metrics(
        simulation, calendar, pd.Timestamp(calendar.iloc[0]), train_end_date
    )
    test_metrics = _slice_metrics(
        simulation, calendar, test_start_date, pd.Timestamp(calendar.iloc[-1])
    )
    test_start_idx = _date_index(calendar, test_start_date, side="left")
    etf_test_metrics = _period_benchmark_metrics(
        etf=etf,
        start_idx=test_start_idx,
        end_idx=len(calendar) - 1,
        config=config,
    )

    simulation.actions.to_csv(run_dir / "actions.csv", index=False)
    simulation.positions.to_csv(run_dir / "positions.csv", index=False)
    simulation.turnover.to_csv(run_dir / "turnover.csv", index=False)
    pd.DataFrame(
        {
            "date": calendar.values,
            "strategy_equity": simulation.equity.values,
            "etf_dca_equity": etf_full.equity.values,
            "market_regime_ok": (
                market_filter.reindex(pd.to_datetime(calendar)).fillna(False).to_numpy()
                if market_filter is not None
                else True
            ),
        }
    ).to_csv(run_dir / "equity_curve.csv", index=False)
    pd.DataFrame(
        _yearly_rows(
            simulation=simulation,
            etf=etf,
            calendar=calendar,
            config=config,
        )
    ).to_csv(run_dir / "yearly_performance.csv", index=False)

    summary = {
        "run_id": actual_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "strategy": "rebound_max5_v2",
        "params": selected_params,
        "use_market_filter": bool(use_market_filter),
        "train_start": str(pd.Timestamp(calendar.iloc[0]).date()),
        "train_end": str(train_end_date.date()),
        "test_start": str(test_start_date.date()),
        "test_end": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "metrics": simulation.metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "etf_dca_metrics": etf_full.metrics,
        "etf_test_metrics": etf_test_metrics,
        "test_excess_twr_vs_etf_dca": float(
            test_metrics["twr_total_return"] - etf_test_metrics["twr_total_return"]
        ),
        "test_excess_cagr_vs_etf_dca": float(
            test_metrics["cagr"] - etf_test_metrics["cagr"]
        ),
    }
    write_json(run_dir / "summary.json", summary)
    return StockStrategyValidationRun(run_id=actual_run_id, run_dir=run_dir)


def run_rebound_max5_v3(
    *,
    config_path: str | Path = Path("config/stock_rotation_multifactor.yaml"),
    run_id: str | None = None,
    params: dict[str, Any] | None = None,
    train_end: str = "2023-12-31",
    use_market_filter: bool = True,
) -> StockStrategyValidationRun:
    config = load_stock_rotation_config(config_path)
    config.selection.method = "sector_multifactor"
    config.selection.use_total_return_features = True
    config.selection.require_long_term_trend = True
    config.selection.max_drawdown_252 = min(float(config.selection.max_drawdown_252), 0.60)
    selected_params = dict(params or REBOUND_MAX5_V3_PARAMS)
    selected_params["fixed_sell_fee_egp"] = float(config.portfolio.fixed_buy_fee_egp)

    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    disclosure_events = load_disclosure_events(config)
    fundamentals = load_stock_fundamentals(config)
    dividend_actions = load_dividend_actions(config)
    corporate_actions = load_corporate_actions(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
    index = load_price_data(config.benchmark.index_symbol_path)
    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    panel = panel[panel["date"] >= pd.Timestamp(calendar.iloc[0])].reset_index(drop=True)
    actual_run_id = run_id or f"rebound-max5-v3-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    regime_frame = (
        build_market_regime_filter_v3(
            etf=etf,
            index=index,
            panel=panel,
            membership=membership,
            calendar=calendar,
            params=selected_params,
        )
        if use_market_filter
        else pd.DataFrame(
            {
                "date": calendar.values,
                "etf_regime_ok": True,
                "index_regime_ok": True,
                "breadth": np.nan,
                "breadth_regime_ok": True,
                "market_regime_ok": True,
            }
        )
    )
    market_filter = regime_frame.set_index("date")["market_regime_ok"]
    features, factor_scores = build_rebound_v3_feature_panel(
        panel=panel,
        membership=membership,
        calendar=calendar,
        config=config,
        benchmark=index,
        params=selected_params,
        disclosure_events=disclosure_events,
        fundamentals=fundamentals,
        dividend_actions=dividend_actions,
        corporate_actions=corporate_actions,
    )
    simulation = simulate_event_driven_strategy(
        panel=panel,
        features=features,
        calendar=calendar,
        membership=membership,
        config=config,
        family="rebound",
        params=selected_params,
        market_filter=market_filter if use_market_filter else None,
    )
    etf_full = run_dca_benchmark(etf, 0, len(calendar) - 1, config.backtest)

    train_end_date = pd.Timestamp(train_end)
    test_start_date = pd.Timestamp(calendar[calendar > train_end_date].iloc[0])
    train_metrics = _slice_metrics(
        simulation, calendar, pd.Timestamp(calendar.iloc[0]), train_end_date
    )
    test_metrics = _slice_metrics(
        simulation, calendar, test_start_date, pd.Timestamp(calendar.iloc[-1])
    )
    test_start_idx = _date_index(calendar, test_start_date, side="left")
    etf_test_metrics = _period_benchmark_metrics(
        etf=etf,
        start_idx=test_start_idx,
        end_idx=len(calendar) - 1,
        config=config,
    )

    simulation.actions.to_csv(run_dir / "actions.csv", index=False)
    simulation.positions.to_csv(run_dir / "positions.csv", index=False)
    simulation.turnover.to_csv(run_dir / "turnover.csv", index=False)
    regime_frame.to_csv(run_dir / "market_regime.csv", index=False)
    factor_scores.to_csv(run_dir / "factor_scores.csv", index=False)
    latest = simulation.latest_setups.copy()
    latest.to_csv(run_dir / "latest_setups.csv", index=False)
    equity_curve = pd.DataFrame(
        {
            "date": calendar.values,
            "strategy_equity": simulation.equity.values,
            "etf_dca_equity": etf_full.equity.values,
        }
    ).merge(regime_frame, on="date", how="left")
    equity_curve.to_csv(run_dir / "equity_curve.csv", index=False)
    pd.DataFrame(
        _yearly_rows(
            simulation=simulation,
            etf=etf,
            calendar=calendar,
            config=config,
        )
    ).to_csv(run_dir / "yearly_performance.csv", index=False)

    factor_summary = {
        "rows": int(len(factor_scores)),
        "symbols": int(factor_scores["symbol"].nunique()) if not factor_scores.empty else 0,
        "first_factor_date": (
            str(pd.Timestamp(factor_scores["date"].min()).date())
            if not factor_scores.empty
            else None
        ),
        "last_factor_date": (
            str(pd.Timestamp(factor_scores["date"].max()).date())
            if not factor_scores.empty
            else None
        ),
        "fundamental_rows": int(len(fundamentals)),
        "feature_rows_with_factor_score": int(features["factor_score"].notna().sum()),
    }
    summary = {
        "run_id": actual_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "strategy": "rebound_max5_v3",
        "params": selected_params,
        "use_market_filter": bool(use_market_filter),
        "data_coverage": {
            "etf_start": str(pd.Timestamp(etf["date"].min()).date()),
            "etf_end": str(pd.Timestamp(etf["date"].max()).date()),
            "index_start": str(pd.Timestamp(index["date"].min()).date()),
            "index_end": str(pd.Timestamp(index["date"].max()).date()),
            "panel_start": str(pd.Timestamp(panel["date"].min()).date()),
            "panel_end": str(pd.Timestamp(panel["date"].max()).date()),
        },
        "factor_summary": factor_summary,
        "train_start": str(pd.Timestamp(calendar.iloc[0]).date()),
        "train_end": str(train_end_date.date()),
        "test_start": str(test_start_date.date()),
        "test_end": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "metrics": simulation.metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "etf_dca_metrics": etf_full.metrics,
        "etf_test_metrics": etf_test_metrics,
        "test_excess_twr_vs_etf_dca": float(
            test_metrics["twr_total_return"] - etf_test_metrics["twr_total_return"]
        ),
        "test_excess_cagr_vs_etf_dca": float(
            test_metrics["cagr"] - etf_test_metrics["cagr"]
        ),
    }
    write_json(run_dir / "summary.json", summary)
    return StockStrategyValidationRun(run_id=actual_run_id, run_dir=run_dir)


def _date_index(dates: pd.Series, date: pd.Timestamp, *, side: str) -> int:
    if side == "left":
        matches = np.flatnonzero(dates >= date)
    else:
        matches = np.flatnonzero(dates <= date)
    if len(matches) == 0:
        return 0 if side == "left" else len(dates) - 1
    return int(matches[0] if side == "left" else matches[-1])


def run_stock_strategy_validation(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    run_id: str | None = None,
    trials: int = 80,
    max_positions: list[int] | None = None,
    train_end: str = "2023-12-31",
    fixed_fee_on_sell: bool = True,
) -> StockStrategyValidationRun:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    config = load_stock_rotation_config(config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    disclosure_events = load_disclosure_events(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
    calendar = (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    min_date = pd.Timestamp(calendar.iloc[0])
    panel = panel[panel["date"] >= min_date].reset_index(drop=True)
    position_counts = max_positions or [3, 4, 5, 6, 8]
    actual_run_id = run_id or f"stock-strategy-validation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    train_end_date = pd.Timestamp(train_end)
    test_start_date = pd.Timestamp(calendar[calendar > train_end_date].iloc[0])
    end_date = pd.Timestamp(calendar.iloc[-1])
    train_end_idx = _date_index(calendar, train_end_date, side="right")
    test_start_idx = _date_index(calendar, test_start_date, side="left")

    train_benchmark = _period_benchmark_metrics(
        etf=etf,
        start_idx=0,
        end_idx=train_end_idx,
        config=config,
    )
    test_benchmark = _period_benchmark_metrics(
        etf=etf,
        start_idx=test_start_idx,
        end_idx=len(calendar) - 1,
        config=config,
    )
    full_benchmark = _period_benchmark_metrics(
        etf=etf,
        start_idx=0,
        end_idx=len(calendar) - 1,
        config=config,
    )

    static_rows: list[dict[str, Any]] = []
    all_trial_rows: list[dict[str, Any]] = []
    best_static: dict[str, Any] | None = None
    best_static_simulation: StrategySimulation | None = None

    for idx, count in enumerate(position_counts):
        params, simulation, train_metrics, trial_rows = _optimise_for_train_window(
            panel=panel,
            calendar=calendar,
            membership=membership,
            config=config,
            disclosure_events=disclosure_events,
            train_end=train_end_date,
            benchmark_metrics=train_benchmark,
            max_positions=int(count),
            trials=int(trials),
            seed=2300 + idx,
            fixed_fee_on_sell=fixed_fee_on_sell,
        )
        test_metrics = _slice_metrics(simulation, calendar, test_start_date, end_date)
        full_metrics = simulation.metrics
        row = {
            "validation": "static_freeze",
            "max_positions": int(count),
            "train_start": str(pd.Timestamp(calendar.iloc[0]).date()),
            "train_end": str(train_end_date.date()),
            "test_start": str(test_start_date.date()),
            "test_end": str(end_date.date()),
            "params": json.dumps(params, sort_keys=True),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{f"full_{key}": value for key, value in full_metrics.items()},
        }
        row["test_excess_twr_vs_etf_dca"] = (
            float(test_metrics["twr_total_return"])
            - float(test_benchmark["twr_total_return"])
        )
        row["test_excess_cagr_vs_etf_dca"] = (
            float(test_metrics["cagr"]) - float(test_benchmark["cagr"])
        )
        row["test_drawdown_saving_vs_etf_dca"] = (
            float(test_benchmark["max_drawdown"])
            - float(test_metrics["max_drawdown"])
        )
        row["oos_balanced_score"] = (
            float(test_metrics["cagr"])
            + 0.20 * float(test_metrics["sharpe"])
            + 0.20 * float(test_metrics["return_dd"])
            - 0.90 * float(test_metrics["max_drawdown"])
        )
        static_rows.append(row)
        for trial_row in trial_rows:
            all_trial_rows.append({"validation": "static_freeze", **trial_row})
        if best_static is None or row["oos_balanced_score"] > best_static["oos_balanced_score"]:
            best_static = row
            best_static_simulation = simulation

    annual_rows: list[dict[str, Any]] = []
    annual_trial_rows: list[dict[str, Any]] = []
    years = sorted(calendar.dt.year.unique())
    for test_year in [year for year in years if 2020 <= int(year) <= int(years[-1])]:
        year_mask = calendar.dt.year == int(test_year)
        if not bool(year_mask.any()):
            continue
        fold_train_end = pd.Timestamp(f"{int(test_year) - 1}-12-31")
        if fold_train_end < pd.Timestamp("2018-12-31") or fold_train_end <= min_date:
            continue
        fold_test_start = pd.Timestamp(calendar[year_mask].iloc[0])
        fold_test_end = pd.Timestamp(calendar[year_mask].iloc[-1])
        fold_train_end_idx = _date_index(calendar, fold_train_end, side="right")
        fold_test_start_idx = _date_index(calendar, fold_test_start, side="left")
        fold_test_end_idx = _date_index(calendar, fold_test_end, side="right")
        fold_train_benchmark = _period_benchmark_metrics(
            etf=etf,
            start_idx=0,
            end_idx=fold_train_end_idx,
            config=config,
        )
        fold_test_benchmark = _period_benchmark_metrics(
            etf=etf,
            start_idx=fold_test_start_idx,
            end_idx=fold_test_end_idx,
            config=config,
        )
        for idx, count in enumerate(position_counts):
            params, simulation, train_metrics, trial_rows = _optimise_for_train_window(
                panel=panel,
                calendar=calendar,
                membership=membership,
                config=config,
                disclosure_events=disclosure_events,
                train_end=fold_train_end,
                benchmark_metrics=fold_train_benchmark,
                max_positions=int(count),
                trials=max(8, int(trials // 3)),
                seed=3100 + int(test_year) * 10 + idx,
                fixed_fee_on_sell=fixed_fee_on_sell,
            )
            test_metrics = _slice_metrics(
                simulation, calendar, fold_test_start, fold_test_end
            )
            row = {
                "validation": "annual_walk_forward",
                "test_year": int(test_year),
                "max_positions": int(count),
                "train_start": str(pd.Timestamp(calendar.iloc[0]).date()),
                "train_end": str(fold_train_end.date()),
                "test_start": str(fold_test_start.date()),
                "test_end": str(fold_test_end.date()),
                "params": json.dumps(params, sort_keys=True),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            row["test_excess_twr_vs_etf_dca"] = (
                float(test_metrics["twr_total_return"])
                - float(fold_test_benchmark["twr_total_return"])
            )
            row["test_excess_cagr_vs_etf_dca"] = (
                float(test_metrics["cagr"]) - float(fold_test_benchmark["cagr"])
            )
            row["test_drawdown_saving_vs_etf_dca"] = (
                float(fold_test_benchmark["max_drawdown"])
                - float(test_metrics["max_drawdown"])
            )
            row["oos_balanced_score"] = (
                float(test_metrics["cagr"])
                + 0.20 * float(test_metrics["sharpe"])
                + 0.20 * float(test_metrics["return_dd"])
                - 0.90 * float(test_metrics["max_drawdown"])
            )
            annual_rows.append(row)
            for trial_row in trial_rows:
                annual_trial_rows.append(
                    {
                        "validation": "annual_walk_forward",
                        "test_year": int(test_year),
                        **trial_row,
                    }
                )

    static = pd.DataFrame(static_rows).sort_values("oos_balanced_score", ascending=False)
    annual = pd.DataFrame(annual_rows)
    static.to_csv(run_dir / "static_freeze_summary.csv", index=False)
    annual.to_csv(run_dir / "annual_walk_forward_summary.csv", index=False)
    pd.DataFrame(all_trial_rows).to_csv(run_dir / "static_freeze_trials.csv", index=False)
    pd.DataFrame(annual_trial_rows).to_csv(
        run_dir / "annual_walk_forward_trials.csv", index=False
    )
    if best_static_simulation is not None:
        best_static_simulation.actions.to_csv(run_dir / "best_actions.csv", index=False)
        best_static_simulation.positions.to_csv(run_dir / "best_positions.csv", index=False)
        best_static_simulation.turnover.to_csv(run_dir / "best_turnover.csv", index=False)
        pd.DataFrame(
            {
                "date": calendar.values,
                "strategy_equity": best_static_simulation.equity.values,
            }
        ).to_csv(run_dir / "best_equity_curve.csv", index=False)

    position_summary = (
        annual.groupby("max_positions", as_index=False)
        .agg(
            folds=("test_year", "count"),
            avg_oos_cagr=("test_cagr", "mean"),
            median_oos_cagr=("test_cagr", "median"),
            avg_oos_twr=("test_twr_total_return", "mean"),
            avg_oos_max_drawdown=("test_max_drawdown", "mean"),
            worst_oos_max_drawdown=("test_max_drawdown", "max"),
            avg_oos_sharpe=("test_sharpe", "mean"),
            avg_excess_twr_vs_etf=("test_excess_twr_vs_etf_dca", "mean"),
            positive_excess_rate=(
                "test_excess_twr_vs_etf_dca",
                lambda values: float((values > 0).mean()),
            ),
        )
        .sort_values("avg_oos_max_drawdown")
        if not annual.empty
        else pd.DataFrame()
    )
    position_summary.to_csv(run_dir / "position_count_summary.csv", index=False)

    summary = {
        "run_id": actual_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "family": "rebound",
        "position_counts": position_counts,
        "trials_per_static_position_count": int(trials),
        "trials_per_walk_forward_fold": max(8, int(trials // 3)),
        "fixed_fee_on_sell": bool(fixed_fee_on_sell),
        "train_end": str(train_end_date.date()),
        "test_start": str(test_start_date.date()),
        "test_end": str(end_date.date()),
        "etf_train_metrics": train_benchmark,
        "etf_test_metrics": test_benchmark,
        "etf_full_metrics": full_benchmark,
        "best_static": None if best_static is None else best_static,
    }
    write_json(run_dir / "summary.json", summary)
    return StockStrategyValidationRun(run_id=actual_run_id, run_dir=run_dir)
