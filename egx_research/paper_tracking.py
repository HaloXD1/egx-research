from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egx_research.backtest import run_dca_benchmark, run_strategy_backtest
from egx_research.config import AppConfig, load_config
from egx_research.data import load_price_data
from egx_research.stock_rotation import (
    load_corporate_actions,
    load_dividend_actions,
    load_membership_snapshots,
    load_stock_fundamentals,
    load_stock_panel,
)
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.stock_strategy_research import (
    load_disclosure_events,
    simulate_event_driven_strategy,
)
from egx_research.stock_strategy_validation import (
    REBOUND_MAX5_V3_PARAMS,
    REBOUND_MAX5_V4_PARAMS,
    REBOUND_MAX5_V5_PARAMS,
    _clean_future_fundamentals_for_run,
    build_market_regime_filter_v3,
    build_rebound_v3_feature_panel,
    build_rebound_v4_feature_panel,
)
from egx_research.strategies import build_strategy_frame
from egx_research.utils import ensure_dir, to_native, write_json


def _best_candidate(run_id: str) -> dict[str, Any]:
    path = Path("runs") / run_id / "candidates.json"
    with path.open("r", encoding="utf-8") as handle:
        candidates = json.load(handle)["candidates"]
    return max(candidates, key=lambda item: item["rank_score"])


def _latest_state(strategy_frame: pd.DataFrame) -> dict[str, Any]:
    last = strategy_frame.iloc[-1]
    payload: dict[str, Any] = {"date": str(pd.Timestamp(last["date"]).date())}
    if "deploy_fraction" in strategy_frame.columns:
        payload["signal"] = "BUY_ZONE" if float(last["deploy_fraction"]) > 0 else "WAIT"
        payload["deploy_fraction"] = float(last["deploy_fraction"])
    if "target_allocation" in strategy_frame.columns:
        payload["target_allocation"] = float(last["target_allocation"])
    return payload


def paper_track_run(
    model_run_id: str,
    start_date: str,
    config_path: str | Path | None = None,
    out_run_id: str | None = None,
) -> Path:
    config: AppConfig = load_config(config_path or "config/default.yaml")
    data = load_price_data(config.data.normalized_path)
    start_ts = pd.Timestamp(start_date)
    last_date = pd.Timestamp(data["date"].iloc[-1])
    candidate = _best_candidate(model_run_id)

    out_run_id = out_run_id or f"paper-track-{model_run_id}-{start_ts.date()}"
    run_dir = ensure_dir(Path("runs") / out_run_id)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_run_id": model_run_id,
        "family": candidate["family"],
        "params": candidate["params"],
        "requested_start_date": str(start_ts.date()),
        "latest_available_date": str(last_date.date()),
        "normalized_path": str(config.data.normalized_path),
    }

    if start_ts > last_date:
        manifest["status"] = "waiting_for_future_data"
        write_json(run_dir / "manifest.json", manifest)
        write_json(
            run_dir / "paper_track_summary.json",
            {
                "status": "waiting_for_future_data",
                "message": f"No data yet after {start_ts.date()}. Ingest new ETF data and rerun paper-track.",
                "latest_state": _latest_state(build_strategy_frame(data, candidate["family"], candidate["params"])),
            },
        )
        return run_dir

    start_idx = int(data.index[data["date"] >= start_ts][0])
    strategy_frame = build_strategy_frame(data, candidate["family"], candidate["params"])
    strategy = run_strategy_backtest(strategy_frame, start_idx, len(data) - 1, config.backtest)
    dca = run_dca_benchmark(data, start_idx, len(data) - 1, config.backtest)

    daily = pd.DataFrame(
        {
            "date": data["date"].iloc[start_idx:].values,
            "close": data["close"].iloc[start_idx:].values,
            "strategy_equity": strategy.equity.values,
            "dca_equity": dca.equity.values,
            "strategy_minus_dca": strategy.equity.values - dca.equity.values,
        }
    )
    if "deploy_fraction" in strategy_frame.columns:
        daily["deploy_fraction"] = strategy_frame["deploy_fraction"].iloc[start_idx:].values
    if "target_allocation" in strategy_frame.columns:
        daily["target_allocation"] = strategy_frame["target_allocation"].iloc[start_idx:].values

    daily.to_csv(run_dir / "paper_track_daily.csv", index=False)

    manifest["status"] = "tracked"
    manifest["actual_start_date"] = str(pd.Timestamp(data["date"].iloc[start_idx]).date())
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "paper_track_summary.json",
        {
            "status": "tracked",
            "strategy_twr_total_return": strategy.metrics["twr_total_return"],
            "dca_twr_total_return": dca.metrics["twr_total_return"],
            "excess_vs_dca": strategy.metrics["twr_total_return"] - dca.metrics["twr_total_return"],
            "strategy_final_equity": strategy.metrics["final_equity"],
            "dca_final_equity": dca.metrics["final_equity"],
            "latest_state": _latest_state(strategy_frame),
        },
    )
    write_json(run_dir / "model_candidate.json", to_native(candidate))
    return run_dir


def _stock_strategy_params(strategy: str, fixed_sell_fee: float) -> dict[str, Any]:
    if strategy == "rebound_max5_v3":
        params = dict(REBOUND_MAX5_V3_PARAMS)
    elif strategy == "rebound_max5_v4":
        params = dict(REBOUND_MAX5_V4_PARAMS)
    elif strategy == "rebound_max5_v5":
        params = dict(REBOUND_MAX5_V5_PARAMS)
    else:
        raise ValueError(
            "Unsupported stock paper strategy: "
            f"{strategy}. Expected rebound_max5_v3, rebound_max5_v4, or rebound_max5_v5."
        )
    params["fixed_sell_fee_egp"] = float(fixed_sell_fee)
    return params


def _data_freshness_status(
    *,
    latest_date: pd.Timestamp,
    as_of: pd.Timestamp,
    max_stale_days: int,
) -> dict[str, Any]:
    days_stale = int((pd.Timestamp(as_of).normalize() - latest_date.normalize()).days)
    return {
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "latest_available_date": str(latest_date.date()),
        "days_stale": days_stale,
        "max_stale_days": int(max_stale_days),
        "status": "future_date"
        if days_stale < 0
        else "stale"
        if days_stale > int(max_stale_days)
        else "fresh",
    }


def paper_track_stock_strategy(
    *,
    strategy: str = "rebound_max5_v5",
    start_date: str,
    config_path: str | Path = "config/stock_rotation_multifactor.yaml",
    out_run_id: str | None = None,
    max_data_stale_days: int = 7,
) -> Path:
    config = load_stock_rotation_config(config_path)
    config.selection.method = "sector_multifactor"
    config.selection.use_total_return_features = True
    config.selection.require_long_term_trend = True
    config.selection.max_drawdown_252 = min(float(config.selection.max_drawdown_252), 0.60)

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
    latest_date = pd.Timestamp(calendar.iloc[-1])
    start_ts = pd.Timestamp(start_date)
    fundamentals, data_quality = _clean_future_fundamentals_for_run(
        fundamentals,
        latest_date,
    )
    params = _stock_strategy_params(strategy, config.portfolio.fixed_buy_fee_egp)

    out_run_id = out_run_id or f"paper-track-{strategy}-{start_ts.date()}"
    run_dir = ensure_dir(Path("runs") / out_run_id)
    freshness = _data_freshness_status(
        latest_date=latest_date,
        as_of=pd.Timestamp(datetime.now(UTC).date()),
        max_stale_days=max_data_stale_days,
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "strategy": strategy,
        "params": params,
        "requested_start_date": str(start_ts.date()),
        "config_path": str(config_path),
        "latest_available_date": str(latest_date.date()),
        "data_freshness": freshness,
        "data_quality": data_quality,
    }

    if start_ts > latest_date:
        manifest["status"] = "waiting_for_future_data"
        write_json(run_dir / "manifest.json", manifest)
        write_json(
            run_dir / "paper_track_summary.json",
            {
                "status": "waiting_for_future_data",
                "message": f"No data yet after {start_ts.date()}. Refresh data and rerun paper-track.",
                "data_freshness": freshness,
            },
        )
        return run_dir

    regime_frame = build_market_regime_filter_v3(
        etf=etf,
        index=index,
        panel=panel,
        membership=membership,
        calendar=calendar,
        params=params,
    )
    if strategy == "rebound_max5_v3":
        features, factor_scores = build_rebound_v3_feature_panel(
            panel=panel,
            membership=membership,
            calendar=calendar,
            config=config,
            benchmark=index,
            params=params,
            disclosure_events=disclosure_events,
            fundamentals=fundamentals,
            dividend_actions=dividend_actions,
            corporate_actions=corporate_actions,
        )
    else:
        features, factor_scores = build_rebound_v4_feature_panel(
            panel=panel,
            membership=membership,
            calendar=calendar,
            config=config,
            benchmark=index,
            params=params,
            disclosure_events=disclosure_events,
            fundamentals=fundamentals,
            dividend_actions=dividend_actions,
            corporate_actions=corporate_actions,
        )

    start_idx = int(calendar[calendar >= start_ts].index[0])
    model_state_start_idx = 0
    simulation = simulate_event_driven_strategy(
        panel=panel,
        features=features,
        calendar=calendar,
        membership=membership,
        config=config,
        family="rebound",
        params=params,
        start_idx=model_state_start_idx,
        market_filter=regime_frame.set_index("date")["market_regime_ok"],
    )
    latest_equity = float(simulation.equity.iloc[-1])
    latest_positions = (
        simulation.positions[
            pd.to_datetime(simulation.positions["date"], errors="coerce") == latest_date
        ].copy()
        if not simulation.positions.empty
        else pd.DataFrame()
    )
    if not latest_positions.empty:
        latest_positions["target_weight"] = (
            latest_positions["market_value"].astype(float) / max(latest_equity, 1e-9)
        )
        latest_positions = latest_positions.sort_values(
            "target_weight", ascending=False
        ).reset_index(drop=True)
    else:
        latest_positions = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "shares",
                "close",
                "market_value",
                "entry_date",
                "bars_held",
                "target_weight",
            ]
        )

    latest_setups = simulation.latest_setups.copy()
    if not latest_setups.empty:
        latest_setups = latest_setups.sort_values("rank_score", ascending=False)
        latest_setups["candidate_action"] = np.where(
            latest_setups["entry_signal"].fillna(False).astype(bool),
            "watch_buy_signal",
            "watch_no_entry",
        )
    else:
        latest_setups = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "holding_name",
                "entry_signal",
                "rank_score",
                "expected_net_edge_pct",
                "edge_cost_ratio",
                "position_size_mult",
                "candidate_action",
            ]
        )

    latest_positions.to_csv(run_dir / "target_holdings.csv", index=False)
    latest_setups.to_csv(run_dir / "latest_setups.csv", index=False)
    regime_frame.tail(252).to_csv(run_dir / "market_regime_recent.csv", index=False)
    simulation.turnover.tail(252).to_csv(run_dir / "paper_turnover_recent.csv", index=False)
    factor_scores.tail(1000).to_csv(run_dir / "factor_scores_recent.csv", index=False)

    market_ok = bool(regime_frame.iloc[-1]["market_regime_ok"])
    trade_allowed = freshness["status"] == "fresh" and market_ok
    manifest["status"] = "tracked"
    manifest["actual_start_date"] = str(pd.Timestamp(calendar.iloc[start_idx]).date())
    manifest["model_state_start_date"] = str(
        pd.Timestamp(calendar.iloc[model_state_start_idx]).date()
    )
    write_json(run_dir / "manifest.json", manifest)
    write_json(
        run_dir / "paper_track_summary.json",
        {
            "status": "tracked",
            "strategy": strategy,
            "latest_date": str(latest_date.date()),
            "paper_accounting_start_date": str(
                pd.Timestamp(calendar.iloc[start_idx]).date()
            ),
            "model_state_start_date": str(
                pd.Timestamp(calendar.iloc[model_state_start_idx]).date()
            ),
            "trade_allowed": bool(trade_allowed),
            "block_reasons": [
                reason
                for reason, blocked in [
                    ("data_not_fresh", freshness["status"] != "fresh"),
                    ("market_regime_filter_off", not market_ok),
                ]
                if blocked
            ],
            "latest_equity": latest_equity,
            "cash": float(
                simulation.turnover.iloc[-1]["cash"]
                if not simulation.turnover.empty and "cash" in simulation.turnover
                else 0.0
            ),
            "open_positions": int(len(latest_positions)),
            "latest_market_regime_ok": market_ok,
            "data_freshness": freshness,
            "data_quality": data_quality,
            "target_holdings": to_native(
                latest_positions.head(20).to_dict(orient="records")
            ),
            "top_setups": to_native(latest_setups.head(20).to_dict(orient="records")),
        },
    )
    return run_dir
