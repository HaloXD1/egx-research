from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from egx_research.backtest import (
    _build_trade_frame,
    _compute_metrics,
    build_contribution_schedule,
    run_buy_hold_benchmark,
    run_dca_benchmark,
)
from egx_research.data import load_price_data
from egx_research.relative_signal import build_relative_signal_frame
from egx_research.stock_rotation import (
    _apply_selection_gates,
    _build_price_matrix,
    _position_value,
    _rebalance_equal_weight,
    _sell_positions,
    active_members_on_date,
    build_stock_features,
    first_trading_days,
    load_membership_snapshots,
    load_stock_panel,
)
from egx_research.stock_rotation_config import (
    StockRotationConfig,
    load_stock_rotation_config,
)
from egx_research.utils import ensure_dir, write_json


IC_FEATURE_COLUMNS = [
    "rel_mom_21",
    "rel_mom_63",
    "rel_mom_126",
    "ratio_vs_sma50",
    "ratio_vs_sma200",
    "sma50_vs_sma200",
    "vol_adj_rel_mom_63",
    "volume_ratio_20_63",
    "mom63_volume_score",
    "down_day_excess_63",
    "residual_strength_21_126b",
    "residual_strength_63_126b",
    "residual_strength_126_126b",
]
DIAGNOSTIC_COLUMNS = ["beta_252", "corr_252"]
TARGET_COLUMN = "future_63d_excess"
TARGET_END_COLUMN = "target_end_date_63"


@dataclass
class RelativeICBacktestRun:
    run_id: str
    run_dir: Path


@dataclass
class RelativeICSimulation:
    equity: pd.Series
    flows: pd.Series
    metrics: dict[str, float]
    actions: pd.DataFrame
    turnover: pd.DataFrame
    total_fees: float


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _hand_signal_score(row: pd.Series) -> float:
    score = 0.0
    score += 25.0 * float(row.get("rel_mom_63", np.nan) > 0.0)
    score += 20.0 * float(row.get("ratio_vs_sma50", np.nan) > 0.0)
    score += 20.0 * float(row.get("residual_strength_63_126b", np.nan) > 0.0)
    score += 15.0 * float(
        (row.get("rel_mom_63", np.nan) > 0.0)
        and (row.get("volume_ratio_20_63", np.nan) > 1.1)
    )
    score += 10.0 * float(row.get("ratio_vs_sma200", np.nan) > 0.0)
    score += 10.0 * float(row.get("rel_mom_126", np.nan) > 0.0)
    return score


def _finite_mask(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask &= values.notna() & np.isfinite(values)
    return mask


def _monthly_ic(
    frame: pd.DataFrame, feature: str, target_column: str
) -> float | None:
    if feature not in frame.columns or target_column not in frame.columns:
        return None
    valid = frame[_finite_mask(frame, [feature, target_column])]
    if len(valid) < 2:
        return None
    if valid[feature].nunique(dropna=True) <= 1:
        return None
    if valid[target_column].nunique(dropna=True) <= 1:
        return None
    ic = valid[feature].corr(valid[target_column], method="spearman")
    if pd.isna(ic) or not np.isfinite(ic):
        return None
    return float(ic)


def learn_ic_weights(
    monthly_panel: pd.DataFrame,
    rebalance_date: str | pd.Timestamp,
    *,
    feature_columns: Sequence[str] = IC_FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
    target_end_column: str = TARGET_END_COLUMN,
    warmup_months: int = 36,
) -> tuple[pd.Series, pd.DataFrame]:
    current_date = pd.Timestamp(rebalance_date)
    weights = pd.Series(0.0, index=list(feature_columns), dtype=float)
    rows: list[dict[str, Any]] = []

    if monthly_panel.empty or target_end_column not in monthly_panel.columns:
        diagnostics = pd.DataFrame(
            [
                {
                    "rebalance_date": str(current_date.date()),
                    "feature": feature,
                    "training_months": 0,
                    "training_rows": 0,
                    "monthly_ic_count": 0,
                    "raw_ic_mean": 0.0,
                    "weight": 0.0,
                    "warmup_complete": False,
                }
                for feature in feature_columns
            ]
        )
        return weights, diagnostics

    history = monthly_panel.copy()
    history[target_end_column] = pd.to_datetime(history[target_end_column])
    known = history[
        (history[target_end_column] < current_date) & history[target_column].notna()
    ].copy()

    if "rebalance_date" in known.columns:
        month_key = pd.to_datetime(known["rebalance_date"]).dt.to_period("M")
    elif "signal_date" in known.columns:
        month_key = pd.to_datetime(known["signal_date"]).dt.to_period("M")
    else:
        month_key = pd.Series([], dtype="period[M]")

    training_months = int(month_key.nunique()) if len(known) else 0
    training_rows = int(len(known))
    warmup_complete = training_months >= int(warmup_months)

    raw_means: dict[str, float] = {}
    for feature in feature_columns:
        monthly_values: list[float] = []
        if warmup_complete and feature in known.columns:
            grouped = known.groupby(month_key, sort=True)
            for _, group in grouped:
                ic = _monthly_ic(group, feature, target_column)
                if ic is not None:
                    monthly_values.append(ic)

        raw_mean = float(np.mean(monthly_values)) if monthly_values else 0.0
        raw_means[feature] = raw_mean
        rows.append(
            {
                "rebalance_date": str(current_date.date()),
                "feature": feature,
                "training_months": training_months,
                "training_rows": training_rows,
                "monthly_ic_count": int(len(monthly_values)),
                "raw_ic_mean": raw_mean,
                "weight": 0.0,
                "warmup_complete": bool(warmup_complete),
            }
        )

    abs_sum = float(sum(abs(value) for value in raw_means.values()))
    if warmup_complete and abs_sum > 0.0:
        for feature, value in raw_means.items():
            weights.loc[feature] = value / abs_sum

    diagnostics = pd.DataFrame(rows)
    diagnostics["weight"] = diagnostics["feature"].map(weights.to_dict()).fillna(0.0)
    return weights, diagnostics


def _percentile_ranks(series: pd.Series) -> pd.Series:
    ranks = pd.Series(0.5, index=series.index, dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna() & np.isfinite(values)
    if valid.sum() < 2:
        return ranks
    if values.loc[valid].nunique(dropna=True) <= 1:
        return ranks
    ranks.loc[valid] = values.loc[valid].rank(
        method="average", pct=True, ascending=True
    )
    return ranks


def score_ic_snapshot(
    snapshot: pd.DataFrame,
    weights: pd.Series,
    *,
    feature_columns: Sequence[str] = IC_FEATURE_COLUMNS,
) -> pd.DataFrame:
    scored = snapshot.copy()
    score = pd.Series(0.0, index=scored.index, dtype=float)
    for feature in feature_columns:
        ranks = _percentile_ranks(
            scored[feature] if feature in scored.columns else pd.Series(np.nan, index=scored.index)
        )
        scored[f"{feature}_rank"] = ranks
        score += ranks * float(weights.get(feature, 0.0))

    scored["ic_score"] = score
    scored["hand_signal_score"] = scored.apply(_hand_signal_score, axis=1)
    tie_cols = [
        column
        for column in ["ic_score", "rel_mom_63", "residual_strength_63_126b", "symbol"]
        if column in scored.columns
    ]
    ascending = [False, False, False, True][: len(tie_cols)]
    scored = scored.sort_values(tie_cols, ascending=ascending).reset_index(drop=True)
    scored["rank"] = range(1, len(scored) + 1)
    return scored


def _holding_name_map(panel: pd.DataFrame) -> dict[str, str]:
    if "holding_name" not in panel.columns:
        return {}
    latest = panel.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    return dict(zip(latest["symbol"], latest["holding_name"], strict=False))


def _build_relative_history(
    panel: pd.DataFrame,
    etf: pd.DataFrame,
    config: StockRotationConfig,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        stock = group[["date", "close", "volume"]].sort_values("date").copy()
        if len(stock) <= horizon_days:
            continue
        signal = build_relative_signal_frame(stock, etf, symbol=str(symbol))
        if signal.empty:
            continue
        source_target = f"excess_fwd_{horizon_days}"
        if source_target not in signal.columns:
            raise ValueError(f"Unsupported relative IC horizon: {horizon_days}")
        signal[TARGET_COLUMN] = signal[source_target]
        signal[TARGET_END_COLUMN] = signal["date"].shift(-horizon_days)
        frames.append(signal)

    if not frames:
        raise ValueError("No relative signal history could be built.")

    history = pd.concat(frames, ignore_index=True)
    stock_features = build_stock_features(panel, config, benchmark=etf)
    gate_columns = [
        column
        for column in [
            "date",
            "symbol",
            "open",
            "close",
            "median_daily_value",
            "median_daily_volume",
            "coverage_ratio",
            "ret_6_1",
            "ret_12_1",
            "vol_63",
            "max_drawdown_252",
        ]
        if column in stock_features.columns
    ]
    history = history.merge(
        stock_features[gate_columns],
        on=["date", "symbol"],
        how="left",
        suffixes=("", "_gate"),
    )
    return history.sort_values(["symbol", "date"]).reset_index(drop=True)


def _monthly_snapshot(
    history: pd.DataFrame,
    membership: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    config: StockRotationConfig,
) -> pd.DataFrame:
    prior = history[history["date"] < rebalance_date].copy()
    if prior.empty:
        return prior
    snapshot = (
        prior.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False, group_keys=False)
        .tail(1)
        .copy()
    )
    active_members = active_members_on_date(membership, rebalance_date)
    snapshot = snapshot[snapshot["symbol"].isin(active_members)].copy()
    if snapshot.empty:
        return snapshot

    valid_overlap = (
        snapshot["stock_close"].notna()
        & snapshot["benchmark_close"].notna()
        & np.isfinite(pd.to_numeric(snapshot["stock_close"], errors="coerce"))
        & np.isfinite(pd.to_numeric(snapshot["benchmark_close"], errors="coerce"))
    )
    snapshot = snapshot[valid_overlap].copy()
    if snapshot.empty:
        return snapshot

    snapshot = _apply_selection_gates(snapshot, config)
    if snapshot.empty:
        return snapshot

    snapshot["rebalance_date"] = pd.Timestamp(rebalance_date)
    snapshot["signal_date"] = pd.to_datetime(snapshot["date"])
    return snapshot.drop(columns=["date"]).reset_index(drop=True)


def _build_monthly_panel(
    history: pd.DataFrame,
    membership: pd.DataFrame,
    etf_dates: pd.Series,
    config: StockRotationConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rebalance_date in first_trading_days(etf_dates):
        snapshot = _monthly_snapshot(history, membership, pd.Timestamp(rebalance_date), config)
        if not snapshot.empty:
            rows.append(snapshot)
    if not rows:
        raise ValueError("No eligible monthly ranking panel rows after gates.")
    return pd.concat(rows, ignore_index=True).sort_values(
        ["rebalance_date", "symbol"]
    )


def _target_weight(config: StockRotationConfig, selected_count: int) -> float:
    if selected_count <= 0:
        return 0.0
    if bool(config.portfolio.hold_cash_when_few):
        return 1.0 / float(max(1, int(config.portfolio.top_n)))
    return 1.0 / float(selected_count)


def _build_rankings_and_selections(
    monthly_panel: pd.DataFrame,
    config: StockRotationConfig,
    *,
    warmup_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, list[str]]]:
    rankings: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    selection_map: dict[pd.Timestamp, list[str]] = {}
    top_n = int(config.portfolio.top_n)

    for rebalance_date in sorted(pd.to_datetime(monthly_panel["rebalance_date"]).unique()):
        current_date = pd.Timestamp(rebalance_date)
        snapshot = monthly_panel[
            pd.to_datetime(monthly_panel["rebalance_date"]) == current_date
        ].copy()
        weights, diagnostics = learn_ic_weights(
            monthly_panel,
            current_date,
            feature_columns=IC_FEATURE_COLUMNS,
            warmup_months=warmup_months,
        )
        weight_frames.append(diagnostics)

        warmup_complete = bool(diagnostics["warmup_complete"].iloc[0])
        if snapshot.empty or not warmup_complete:
            continue

        scored = score_ic_snapshot(snapshot, weights)
        selected_symbols = scored.head(top_n)["symbol"].astype(str).tolist()
        selected_count = len(selected_symbols)
        target_weight = _target_weight(config, selected_count)
        selected_set = set(selected_symbols)
        scored["selected"] = scored["symbol"].isin(selected_set)
        scored["target_weight"] = np.where(scored["selected"], target_weight, 0.0)
        scored["training_months"] = int(diagnostics["training_months"].iloc[0])
        scored["weight_abs_sum"] = float(weights.abs().sum())
        rankings.append(scored)

        if selected_symbols:
            selection_map[current_date] = selected_symbols
        for row in scored.head(top_n).itertuples(index=False):
            selections.append(
                {
                    "rebalance_date": str(current_date.date()),
                    "selection_date": str(pd.Timestamp(row.signal_date).date()),
                    "rank": int(row.rank),
                    "symbol": row.symbol,
                    "holding_name": getattr(row, "holding_name", row.symbol),
                    "ic_score": float(row.ic_score),
                    "hand_signal_score": float(row.hand_signal_score),
                    "target_weight": target_weight,
                    "selected_count": selected_count,
                }
            )

    rankings_frame = (
        pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    )
    selected_frame = pd.DataFrame(selections)
    weights_frame = (
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    )
    return rankings_frame, selected_frame, weights_frame, selection_map


def _simulate_relative_ic_portfolio(
    *,
    calendar: pd.Series,
    panel: pd.DataFrame,
    selection_map: dict[pd.Timestamp, list[str]],
    config: StockRotationConfig,
) -> RelativeICSimulation:
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
        initial_cash=config.backtest.initial_cash,
        monthly_contribution=config.backtest.monthly_contribution,
    )
    equity = pd.Series(index=range(len(calendar)), dtype=float)
    flows = pd.Series(0.0, index=range(len(calendar)), dtype=float)
    actions: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    positions: dict[str, float] = {}
    cash = 0.0

    for i, date_value in enumerate(calendar):
        date = pd.Timestamp(date_value)
        contribution = float(contributions.iloc[i])
        cash += contribution
        flows.iloc[i] += contribution

        open_row = open_px.loc[date]
        close_row = close_px.loc[date]

        if date in selection_map:
            selected_symbols = selection_map[date]
            equity_before = cash + _position_value(positions, open_row.fillna(close_row))
            turnover_value = 0.0
            rebalance_fees = 0.0

            undesired = sorted(
                symbol for symbol in positions if symbol not in set(selected_symbols)
            )
            cash, traded, fees = _sell_positions(
                date=date,
                symbols=undesired,
                positions=positions,
                cash=cash,
                open_row=open_row,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                reason="relative_ic_rebalance_exit",
                actions=actions,
            )
            turnover_value += traded
            rebalance_fees += fees

            target_denominator = (
                int(config.portfolio.top_n)
                if bool(config.portfolio.hold_cash_when_few)
                else max(1, len(selected_symbols))
            )
            cash, traded, fees = _rebalance_equal_weight(
                date=date,
                selected_symbols=selected_symbols,
                positions=positions,
                cash=cash,
                open_row=open_row,
                close_row=close_row,
                top_n=target_denominator,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                fixed_buy_fee=fixed_buy_fee,
                share_precision=share_precision,
                actions=actions,
            )
            turnover_value += traded
            rebalance_fees += fees

            turnover_rows.append(
                {
                    "date": str(date.date()),
                    "selected_count": int(len(selected_symbols)),
                    "target_denominator": int(target_denominator),
                    "turnover_value": turnover_value,
                    "turnover_pct": 0.0
                    if equity_before <= 0.0
                    else turnover_value / equity_before,
                    "rebalance_fees": rebalance_fees,
                    "cash_balance": cash,
                }
            )

        equity.iloc[i] = cash + _position_value(positions, close_row)

    actions_frame = pd.DataFrame(actions)
    if actions_frame.empty:
        actions_frame = pd.DataFrame(
            columns=["date", "symbol", "action", "shares", "price", "value", "fee", "reason"]
        )
    turnover_frame = pd.DataFrame(turnover_rows)
    total_fees = float(actions_frame["fee"].fillna(0.0).sum())
    metrics = _compute_metrics(equity.copy(), flows.copy(), _build_trade_frame([]))
    return RelativeICSimulation(
        equity=equity,
        flows=flows,
        metrics=metrics,
        actions=actions_frame,
        turnover=turnover_frame,
        total_fees=total_fees,
    )


def _json_records(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in frame.head(limit).to_dict(orient="records"):
        records.append({key: _safe_float(value) if isinstance(value, float) else value for key, value in row.items()})
    return records


def run_relative_ic_backtest(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    run_id: str | None = None,
    warmup_months: int = 36,
    horizon_days: int = 63,
) -> RelativeICBacktestRun:
    if horizon_days != 63:
        raise ValueError("relative-ic-backtest currently supports horizon_days=63.")

    config = load_stock_rotation_config(config_path)
    panel = load_stock_panel(config)
    membership = load_membership_snapshots(config)
    etf = load_price_data(config.benchmark.etf_symbol_path)
    name_map = _holding_name_map(panel)

    history = _build_relative_history(panel, etf, config, horizon_days=horizon_days)
    if name_map:
        history["holding_name"] = history["symbol"].map(name_map).fillna(history["symbol"])
    monthly_panel = _build_monthly_panel(history, membership, etf["date"], config)
    rankings, selected, weights, selection_map = _build_rankings_and_selections(
        monthly_panel,
        config,
        warmup_months=warmup_months,
    )
    if not selection_map:
        raise ValueError("No relative IC selections generated after warmup.")

    first_rebalance = min(selection_map)
    etf_run = etf[etf["date"] >= first_rebalance].reset_index(drop=True)
    calendar = (
        pd.Series(pd.to_datetime(etf_run["date"]))
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    selection_map = {
        date: symbols for date, symbols in selection_map.items() if date >= first_rebalance
    }
    simulation = _simulate_relative_ic_portfolio(
        calendar=calendar,
        panel=panel,
        selection_map=selection_map,
        config=config,
    )
    etf_dca = run_dca_benchmark(etf_run, 0, len(etf_run) - 1, config.backtest)
    etf_buy_hold = run_buy_hold_benchmark(etf_run, 0, len(etf_run) - 1)

    actual_run_id = run_id or f"relative-ic-etf-{datetime.now(UTC).strftime('%Y%m%d')}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    weights.to_csv(run_dir / "ic_weights.csv", index=False)
    rankings.to_csv(run_dir / "monthly_rankings.csv", index=False)
    selected.to_csv(run_dir / "selected_holdings.csv", index=False)
    simulation.actions.to_csv(run_dir / "trade_actions.csv", index=False)
    simulation.turnover.to_csv(run_dir / "turnover.csv", index=False)
    pd.DataFrame(
        {
            "date": calendar.values,
            "relative_ic_equity": simulation.equity.values,
            "etf_dca_equity": etf_dca.equity.values,
            "etf_buy_hold_norm": etf_buy_hold.values,
        }
    ).to_csv(run_dir / "equity_curve.csv", index=False)

    latest_rebalance = max(selection_map)
    latest_selected = selected[
        pd.to_datetime(selected["rebalance_date"]) == latest_rebalance
    ].sort_values("rank")
    latest_rankings = rankings[
        pd.to_datetime(rankings["rebalance_date"]) == latest_rebalance
    ].sort_values("rank")
    mean_turnover = (
        0.0
        if simulation.turnover.empty
        else float(simulation.turnover["turnover_pct"].fillna(0.0).mean())
    )
    average_selected_count = (
        0.0
        if selected.empty
        else float(selected.groupby("rebalance_date")["symbol"].count().mean())
    )

    summary = {
        "run_id": actual_run_id,
        "strategy": "ic_weighted_walk_forward_relative_strength",
        "target": TARGET_COLUMN,
        "benchmark": "EGX30 ETF",
        "benchmark_path": str(config.benchmark.etf_symbol_path),
        "panel_path": str(Path(config.storage.root_dir) / config.storage.panel_filename),
        "membership_source": "membership_verified_partial.csv",
        "start_date": str(pd.Timestamp(calendar.iloc[0]).date()),
        "end_date": str(pd.Timestamp(calendar.iloc[-1]).date()),
        "warmup_months": int(warmup_months),
        "target_horizon_days": int(horizon_days),
        "top_n": int(config.portfolio.top_n),
        "final_equity": float(simulation.metrics["final_equity"]),
        "twr_total_return": float(simulation.metrics["twr_total_return"]),
        "cagr": float(simulation.metrics["cagr"]),
        "sharpe": float(simulation.metrics["sharpe"]),
        "max_drawdown": float(simulation.metrics["max_drawdown"]),
        "excess_vs_etf_dca": float(
            simulation.metrics["twr_total_return"]
            - etf_dca.metrics["twr_total_return"]
        ),
        "total_fees": float(simulation.total_fees),
        "mean_turnover": mean_turnover,
        "number_of_rebalances": int(len(simulation.turnover)),
        "average_selected_count": average_selected_count,
        "latest_rebalance_date": str(latest_rebalance.date()),
        "latest_selected_stocks": latest_selected["symbol"].tolist(),
        "latest_top_ranked": _json_records(
            latest_rankings[["rank", "symbol", "ic_score", "hand_signal_score"]], 10
        ),
        "strategy_metrics": simulation.metrics,
        "etf_dca_metrics": etf_dca.metrics,
        "etf_buy_hold_return": float(etf_buy_hold.iloc[-1] - 1.0),
        "costs": {
            "fee_bps": float(config.backtest.fee_bps),
            "slippage_bps": float(config.backtest.slippage_bps),
            "fixed_buy_fee_egp": float(config.portfolio.fixed_buy_fee_egp),
            "share_precision": int(config.backtest.share_precision),
        },
        "artifact_paths": {
            "ic_weights": str(run_dir / "ic_weights.csv"),
            "monthly_rankings": str(run_dir / "monthly_rankings.csv"),
            "selected_holdings": str(run_dir / "selected_holdings.csv"),
            "equity_curve": str(run_dir / "equity_curve.csv"),
            "trade_actions": str(run_dir / "trade_actions.csv"),
            "turnover": str(run_dir / "turnover.csv"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return RelativeICBacktestRun(run_id=actual_run_id, run_dir=run_dir)
