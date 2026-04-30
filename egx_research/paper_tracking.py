from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from egx_research.backtest import run_dca_benchmark, run_strategy_backtest
from egx_research.config import AppConfig, load_config
from egx_research.data import load_price_data
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
