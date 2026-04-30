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
    StockRotationConfig,
    active_members_on_date,
    build_etf_features,
    build_stock_features,
    first_trading_days,
    load_corporate_actions,
    load_dividend_actions,
    load_membership_snapshots,
    load_stock_fundamentals,
    load_stock_panel,
    score_snapshot,
    _apply_selection_gates,
    _build_price_matrix,
    _build_pullback_signal_matrix,
    _load_pullback_params,
    _simulate_deployment,
    _slice_holdout_metrics,
)
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.utils import ensure_dir, write_json


FACTOR_COLUMNS = [
    "score_momentum",
    "score_value",
    "score_quality",
    "score_growth",
    "score_low_risk",
    "score_liquidity",
]

ENSEMBLE_MODELS = {
    "momentum_quality": {
        "score_momentum": 0.50,
        "score_quality": 0.30,
        "score_growth": 0.10,
        "score_liquidity": 0.10,
    },
    "value_quality": {
        "score_value": 0.45,
        "score_quality": 0.30,
        "score_momentum": 0.15,
        "score_liquidity": 0.10,
    },
    "lowrisk_income": {
        "score_low_risk": 0.35,
        "score_value": 0.30,
        "score_liquidity": 0.20,
        "score_quality": 0.15,
    },
}


@dataclass
class StockFactorResearchRun:
    run_id: str
    run_dir: Path


def _calendar_from_etf(etf: pd.DataFrame) -> pd.Series:
    return (
        pd.Series(pd.to_datetime(etf["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _future_total_return(
    row: pd.Series,
    *,
    close_px: pd.DataFrame,
    cum_div: pd.DataFrame,
    selection_date: pd.Timestamp,
    horizon_bars: int,
) -> float | None:
    symbol = str(row["symbol"])
    if symbol not in close_px.columns or selection_date not in close_px.index:
        return None
    start_pos = close_px.index.get_loc(selection_date)
    if isinstance(start_pos, slice):
        start_pos = start_pos.start
    end_pos = int(start_pos) + int(horizon_bars)
    if end_pos >= len(close_px.index):
        return None
    end_date = close_px.index[end_pos]
    start_close = close_px.at[selection_date, symbol]
    end_close = close_px.at[end_date, symbol]
    if pd.isna(start_close) or pd.isna(end_close) or float(start_close) <= 0.0:
        return None
    start_div = cum_div.at[selection_date, symbol] if symbol in cum_div.columns else 0.0
    end_div = cum_div.at[end_date, symbol] if symbol in cum_div.columns else 0.0
    return float((end_close + end_div - start_div) / start_close - 1.0)


def _score_snapshot_for_research(
    snapshot: pd.DataFrame, etf_ret_3m: float, config: StockRotationConfig
) -> pd.DataFrame:
    variant = copy.deepcopy(config)
    variant.selection.method = "sector_multifactor"
    scored = score_snapshot(snapshot, etf_ret_3m, variant)
    for column in FACTOR_COLUMNS:
        if column not in scored.columns:
            scored[column] = 0.5
    return scored


def _rebalance_snapshots(
    *,
    features: pd.DataFrame,
    etf_features: pd.DataFrame,
    membership: pd.DataFrame,
    calendar: pd.Series,
    config: StockRotationConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, float]]:
    rows: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, float]] = []
    for rebalance_date in first_trading_days(calendar):
        prev_rows = etf_features.loc[
            etf_features["date"] < rebalance_date, ["date", "ret_3m"]
        ]
        if prev_rows.empty:
            continue
        selection_date = pd.Timestamp(prev_rows.iloc[-1]["date"])
        etf_ret_3m = float(prev_rows.iloc[-1]["ret_3m"])
        snapshot = features[features["date"] == selection_date].copy()
        active = active_members_on_date(membership, rebalance_date)
        if active:
            snapshot = snapshot[snapshot["symbol"].isin(active)].copy()
        snapshot = _apply_selection_gates(snapshot, config)
        scored = _score_snapshot_for_research(snapshot, etf_ret_3m, config)
        if scored.empty:
            continue
        rows.append((pd.Timestamp(rebalance_date), selection_date, scored, etf_ret_3m))
    return rows


def _factor_ic_tables(
    snapshots: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, float]],
    *,
    close_px: pd.DataFrame,
    cum_div: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, Any]] = []
    horizons = {"1m": 21, "3m": 63, "6m": 126}
    for rebalance_date, selection_date, scored, _ in snapshots:
        frame = scored.copy()
        for label, bars in horizons.items():
            frame[f"future_return_{label}"] = frame.apply(
                _future_total_return,
                axis=1,
                close_px=close_px,
                cum_div=cum_div,
                selection_date=selection_date,
                horizon_bars=bars,
            )
            returns = frame[f"future_return_{label}"]
            for factor in FACTOR_COLUMNS:
                valid = returns.notna() & frame[factor].notna()
                monthly_rows.append(
                    {
                        "rebalance_date": str(rebalance_date.date()),
                        "selection_date": str(selection_date.date()),
                        "horizon": label,
                        "factor": factor,
                        "ic": None
                        if int(valid.sum()) < 4
                        else float(frame.loc[valid, factor].corr(returns.loc[valid], method="spearman")),
                        "observations": int(valid.sum()),
                    }
                )
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        summary = pd.DataFrame(columns=["factor", "horizon", "mean_ic", "hit_rate", "observations"])
    else:
        summary = (
            monthly.dropna(subset=["ic"])
            .groupby(["factor", "horizon"], as_index=False)
            .agg(
                mean_ic=("ic", "mean"),
                median_ic=("ic", "median"),
                hit_rate=("ic", lambda s: float((s > 0.0).mean())),
                months=("ic", "count"),
                avg_observations=("observations", "mean"),
            )
            .sort_values(["horizon", "mean_ic"], ascending=[True, False])
            .reset_index(drop=True)
        )
    return monthly, summary


def _training_weights(ic_history: pd.DataFrame) -> dict[str, float]:
    defaults = {
        "score_momentum": 0.30,
        "score_value": 0.20,
        "score_quality": 0.20,
        "score_growth": 0.10,
        "score_low_risk": 0.10,
        "score_liquidity": 0.10,
    }
    if ic_history.empty or ic_history["rebalance_date"].nunique() < 24:
        return defaults
    train = ic_history[
        (ic_history["horizon"].isin(["3m", "6m"])) & ic_history["ic"].notna()
    ].copy()
    if train.empty:
        return defaults
    means = train.groupby("factor")["ic"].mean().clip(lower=0.0)
    if float(means.sum()) <= 0.0:
        return defaults
    weights = (means / means.sum()).to_dict()
    return {factor: float(weights.get(factor, 0.0)) for factor in FACTOR_COLUMNS}


def _apply_sector_cap(scored: pd.DataFrame, top_n: int, max_sector_weight: float) -> pd.DataFrame:
    cap_count = max(1, int(np.floor(float(max_sector_weight) * float(top_n))))
    rows = []
    counts: dict[str, int] = {}
    ranked = scored.sort_values(["score", "score_momentum"], ascending=False).reset_index(drop=True)
    for row in ranked.itertuples(index=False):
        sector = str(getattr(row, "sector", "unknown"))
        if counts.get(sector, 0) >= cap_count:
            continue
        rows.append(row._asdict())
        counts[sector] = counts.get(sector, 0) + 1
        if len(rows) >= top_n:
            break
    if len(rows) < top_n:
        symbols = {row["symbol"] for row in rows}
        for row in ranked.itertuples(index=False):
            if row.symbol in symbols:
                continue
            rows.append(row._asdict())
            if len(rows) >= top_n:
                break
    return pd.DataFrame(rows, columns=ranked.columns)


def _ensemble_score(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    frame = scored.copy()
    vote = pd.Series(0.0, index=frame.index)
    rank_score = pd.Series(0.0, index=frame.index)
    for weights in ENSEMBLE_MODELS.values():
        model_score = pd.Series(0.0, index=frame.index)
        for factor, weight in weights.items():
            model_score += frame[factor].fillna(0.5) * float(weight)
        ranks = model_score.rank(ascending=False, method="average")
        vote += (ranks <= top_n).astype(float)
        rank_score += 1.0 / ranks.clip(lower=1.0)
    frame["ensemble_votes"] = vote
    frame["ensemble_rank_score"] = rank_score
    frame["score"] = vote + rank_score
    return frame


def _selection_rows(
    *,
    snapshots: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, float]],
    ic_monthly: pd.DataFrame,
    config: StockRotationConfig,
    mode: str,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, list[str]], pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    selection_map: dict[pd.Timestamp, list[str]] = {}
    top_n = int(config.portfolio.top_n)
    for rebalance_date, selection_date, scored, _ in snapshots:
        frame = scored.copy()
        if mode == "walk_forward_ic":
            train = ic_monthly[pd.to_datetime(ic_monthly["rebalance_date"]) < rebalance_date]
            weights = _training_weights(train)
            frame["score"] = sum(frame[factor].fillna(0.5) * weight for factor, weight in weights.items())
        elif mode == "ensemble":
            weights = {}
            frame = _ensemble_score(frame, top_n)
        else:
            raise ValueError(f"Unsupported factor mode: {mode}")

        selected = _apply_sector_cap(
            frame, top_n=top_n, max_sector_weight=float(config.selection.max_sector_weight)
        ).head(top_n)
        selected = selected.copy()
        selected["rebalance_date"] = str(rebalance_date.date())
        selected["selection_date"] = str(selection_date.date())
        selected["rank"] = range(1, len(selected) + 1)
        selected["target_weight"] = min(
            1.0 / float(top_n), float(config.selection.max_position_weight)
        )
        selections.append(selected)
        selection_map[rebalance_date] = selected["symbol"].astype(str).tolist()
        diagnostics.append(
            {
                "rebalance_date": str(rebalance_date.date()),
                "selection_date": str(selection_date.date()),
                "mode": mode,
                "selected_count": int(len(selected)),
                "weights": json.dumps(weights, sort_keys=True),
            }
        )
    return (
        pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(),
        selection_map,
        pd.DataFrame(diagnostics),
    )


def _data_quality_flags(features: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    latest_date = pd.Timestamp(calendar.iloc[-1])
    snapshot = features[features["date"] <= latest_date].sort_values(["symbol", "date"]).drop_duplicates("symbol", keep="last")
    rows: list[dict[str, Any]] = []
    for row in snapshot.itertuples(index=False):
        values = row._asdict()
        symbol = str(values.get("symbol"))
        def add(flag: str, value: Any) -> None:
            rows.append({"symbol": symbol, "flag": flag, "value": value})

        if not bool(values.get("has_fundamentals", False)):
            add("missing_fundamentals", "")
        age = values.get("fundamental_row_age_days")
        if pd.notna(age) and float(age) > 300:
            add("stale_fundamentals_days", float(age))
        pe = values.get("pe_ratio")
        if pd.notna(pe) and (float(pe) <= 0.0 or float(pe) > 80.0):
            add("pe_outlier", float(pe))
        pb = values.get("pb_ratio")
        if pd.notna(pb) and (float(pb) <= 0.0 or float(pb) > 20.0):
            add("pb_outlier", float(pb))
        roe = values.get("roe")
        if pd.notna(roe) and (float(roe) < -1.0 or float(roe) > 1.5):
            add("roe_outlier", float(roe))
        currency = str(values.get("currency", ""))
        if currency and currency != "EGP" and currency != "nan":
            add("non_egp_statement_currency", currency)
    return pd.DataFrame(rows, columns=["symbol", "flag", "value"])


def _balanced_score(metrics: dict[str, float], holdout: dict[str, float]) -> float:
    return (
        0.35 * float(metrics["cagr"])
        + 0.35 * float(holdout["cagr"])
        - 0.20 * float(metrics["max_drawdown"])
        - 0.10 * float(holdout["max_drawdown"])
    )


def run_stock_factor_research(
    config_path: str | Path = Path("config/stock_rotation_multifactor.yaml"),
    *,
    run_id: str | None = None,
    rebalance_mode: str = "monthly",
) -> StockFactorResearchRun:
    if rebalance_mode != "monthly":
        raise ValueError("stock-factor-research currently supports monthly rebalances.")
    config = load_stock_rotation_config(config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
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
    calendar = _calendar_from_etf(etf)
    snapshots = _rebalance_snapshots(
        features=features,
        etf_features=etf_features,
        membership=membership,
        calendar=calendar,
        config=config,
    )
    if not snapshots:
        raise ValueError("No factor snapshots generated.")

    close_px = _build_price_matrix(features, calendar, "close", fill=True)
    cum_div = _build_price_matrix(features, calendar, "cum_dividend", fill=True).fillna(0.0)
    ic_monthly, ic_summary = _factor_ic_tables(snapshots, close_px=close_px, cum_div=cum_div)
    qa_flags = _data_quality_flags(features, calendar)

    run_id = run_id or f"stock-factor-research-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / run_id)

    pullback_source_run_id, pullback_params = _load_pullback_params(None)
    signals = _build_pullback_signal_matrix(panel, calendar, pullback_params)
    open_px = _build_price_matrix(panel, calendar, "open", fill=False)
    close_trade_px = _build_price_matrix(panel, calendar, "close", fill=True)
    etf_dca = run_dca_benchmark(etf, 0, len(etf) - 1, config.backtest)
    holdout_bars = max(1, int(len(calendar) * float(config.model_selection.holdout_ratio)))
    holdout_start_idx = max(0, len(calendar) - holdout_bars)
    etf_holdout = _slice_holdout_metrics(etf_dca.equity, etf_dca.flows, holdout_start_idx)

    candidates: list[dict[str, Any]] = []
    equity_payload: dict[str, Any] = {"date": calendar.values, "etf_dca_equity": etf_dca.equity.values}
    for mode in ["ensemble", "walk_forward_ic"]:
        selections, selection_map, diagnostics = _selection_rows(
            snapshots=snapshots,
            ic_monthly=ic_monthly,
            config=config,
            mode=mode,
        )
        selections.to_csv(run_dir / f"{mode}_selected_holdings.csv", index=False)
        diagnostics.to_csv(run_dir / f"{mode}_diagnostics.csv", index=False)
        for deploy_mode in ["immediate", "pullback"]:
            sim = _simulate_deployment(
                mode=deploy_mode,
                calendar=calendar,
                selection_map=selection_map,
                open_px=open_px,
                close_px=close_trade_px,
                signals=signals,
                config=config,
                top_n=int(config.portfolio.top_n),
            )
            holdout = _slice_holdout_metrics(sim.equity, sim.flows, holdout_start_idx)
            name = f"{mode}_{deploy_mode}"
            equity_payload[f"{name}_equity"] = sim.equity.values
            candidates.append(
                {
                    "name": name,
                    "selection_mode": mode,
                    "deployment_mode": deploy_mode,
                    "metrics": sim.metrics,
                    "holdout_metrics": holdout,
                    "holdout_excess_vs_etf_dca": float(
                        holdout["twr_total_return"] - etf_holdout["twr_total_return"]
                    ),
                    "balanced_score": _balanced_score(sim.metrics, holdout),
                    "total_fees": float(sim.total_fees),
                }
            )

    candidates = sorted(candidates, key=lambda item: item["balanced_score"], reverse=True)
    ic_monthly.to_csv(run_dir / "factor_ic_monthly.csv", index=False)
    ic_summary.to_csv(run_dir / "factor_ic_summary.csv", index=False)
    qa_flags.to_csv(run_dir / "data_quality_flags.csv", index=False)
    pd.DataFrame(equity_payload).to_csv(run_dir / "equity_curve_factor_research.csv", index=False)
    write_json(
        run_dir / "summary_factor_research.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
            "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
            "fundamental_rows_loaded": int(len(fundamentals)),
            "qa_flag_count": int(len(qa_flags)),
            "pullback_source_run_id": pullback_source_run_id,
            "best_candidate": candidates[0] if candidates else None,
            "candidates": candidates,
            "etf_dca_metrics": etf_dca.metrics,
            "etf_holdout_metrics": etf_holdout,
        },
    )
    return StockFactorResearchRun(run_id=run_id, run_dir=run_dir)
