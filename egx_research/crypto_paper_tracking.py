from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egx_research.backtest import run_buy_hold_benchmark, run_dca_benchmark, run_strategy_backtest
from egx_research.crypto_config import CryptoConfig, load_crypto_config
from egx_research.crypto_research import load_crypto_feature_data, run_weekly_dca_benchmark, select_crypto_candidate
from egx_research.crypto_strategies import build_crypto_strategy_frame
from egx_research.indicators import ema, kama, sma
from egx_research.utils import ensure_dir, to_native, write_json


def _best_crypto_candidate(model_run_id: str) -> dict[str, Any]:
    path = Path("runs") / model_run_id / "candidates.json"
    with path.open("r", encoding="utf-8") as handle:
        candidates = json.load(handle)["candidates"]
    return select_crypto_candidate(candidates)


def _data_freshness_status(
    *,
    latest_date: pd.Timestamp,
    as_of: pd.Timestamp,
    max_stale_days: int,
) -> dict[str, Any]:
    days_stale = int((pd.Timestamp(as_of).normalize() - latest_date.normalize()).days)
    if days_stale < 0:
        status = "future_date"
    elif days_stale > int(max_stale_days):
        status = "stale"
    else:
        status = "fresh"
    return {
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "latest_available_date": str(latest_date.date()),
        "days_stale": days_stale,
        "max_stale_days": int(max_stale_days),
        "status": status,
    }


def _family_diagnostics(frame: pd.DataFrame, family: str, params: dict[str, Any]) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce")
    latest = frame.iloc[-1]
    diagnostics: dict[str, Any] = {
        "close": float(latest["close"]),
        "atr": float(latest["atr"]) if pd.notna(latest.get("atr")) else None,
    }

    if "return_1d" in frame:
        diagnostics["return_1d"] = float(latest["return_1d"]) if pd.notna(latest["return_1d"]) else None
    if "return_7d" in frame:
        diagnostics["return_7d"] = float(latest["return_7d"]) if pd.notna(latest["return_7d"]) else None
    if "realized_vol_30d" in frame:
        diagnostics["realized_vol_30d"] = (
            float(latest["realized_vol_30d"]) if pd.notna(latest["realized_vol_30d"]) else None
        )
    if "CapMVRVCur" in frame:
        diagnostics["mvrv"] = float(latest["CapMVRVCur"]) if pd.notna(latest["CapMVRVCur"]) else None
    if "fear_greed_value" in frame:
        diagnostics["fear_greed_value"] = (
            float(latest["fear_greed_value"]) if pd.notna(latest["fear_greed_value"]) else None
        )
        diagnostics["fear_greed_classification"] = latest.get("fear_greed_classification")
    if "funding_rate_mean" in frame:
        diagnostics["funding_rate_mean"] = (
            float(latest["funding_rate_mean"]) if pd.notna(latest["funding_rate_mean"]) else None
        )

    if family == "crypto_price_trend":
        fast = ema(close, int(params["fast_ema"]))
        slow = ema(close, int(params["slow_ema"]))
        trend_line = sma(close, int(params["trend_ma"]))
        diagnostics.update(
            {
                "fast_ema": float(fast.iloc[-1]),
                "slow_ema": float(slow.iloc[-1]),
                "trend_ma": float(trend_line.iloc[-1]),
                "fast_above_slow": bool(fast.iloc[-1] > slow.iloc[-1]),
                "close_above_trend_ma": bool(close.iloc[-1] > trend_line.iloc[-1]),
            }
        )
    elif family == "crypto_dca_overlay":
        kama_line = kama(
            close,
            int(params["kama_len"]),
            int(params["kama_fast"]),
            int(params["kama_slow"]),
        )
        diagnostics.update(
            {
                "kama": float(kama_line.iloc[-1]),
                "close_above_kama": bool(close.iloc[-1] > kama_line.iloc[-1]),
            }
        )
    elif family == "crypto_macro_overlay":
        btc_trend = sma(close, int(params["btc_trend_ma"]))
        diagnostics.update(
            {
                "btc_trend_ma": float(btc_trend.iloc[-1]),
                "close_above_btc_trend_ma": bool(close.iloc[-1] > btc_trend.iloc[-1]),
            }
        )

    for column in ("macro_nasdaq", "macro_us10y", "macro_dollar", "macro_fed_liquidity", "macro_vix"):
        if column in frame:
            value = latest[column]
            diagnostics[column] = float(value) if pd.notna(value) else None

    return to_native(diagnostics)


def build_current_crypto_signal(frame: pd.DataFrame, family: str, params: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot build a signal from an empty crypto frame.")

    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else latest
    target = float(latest["target_allocation"])
    previous_target = float(previous["target_allocation"])
    floor = float(latest["floor_allocation"]) if "floor_allocation" in frame else 0.0
    delta = target - previous_target

    if target <= floor + 1e-9:
        regime = "defensive"
    elif target >= 0.95:
        regime = "risk_on"
    else:
        regime = "partial_risk"

    if delta > 0.05:
        action = "increase_to_target"
    elif delta < -0.05:
        action = "reduce_to_target"
    elif regime == "risk_on":
        action = "hold_full_btc"
    elif regime == "defensive":
        action = "hold_cash_or_floor"
    else:
        action = "hold_partial_btc"

    reasons: list[str] = []
    diagnostics = _family_diagnostics(frame, family, params)
    if family == "crypto_price_trend":
        if diagnostics.get("fast_above_slow") and diagnostics.get("close_above_trend_ma"):
            reasons.append("fast EMA is above slow EMA and BTC closed above trend MA")
        else:
            if not diagnostics.get("fast_above_slow"):
                reasons.append("fast EMA is not above slow EMA")
            if not diagnostics.get("close_above_trend_ma"):
                reasons.append("BTC closed below trend MA")
    else:
        if target > floor:
            reasons.append("model target is above defensive floor")
        else:
            reasons.append("model target is at defensive floor")
    if abs(delta) <= 0.05:
        reasons.append("target allocation unchanged vs prior daily signal")

    return {
        "date": str(pd.Timestamp(latest["date"]).date()),
        "family": family,
        "signal": regime,
        "action": action,
        "execution_timing": "next_daily_open_after_latest_close",
        "target_allocation": target,
        "previous_target_allocation": previous_target,
        "target_allocation_change": delta,
        "floor_allocation": floor,
        "close": float(latest["close"]),
        "reasons": reasons,
        "diagnostics": diagnostics,
    }


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_float(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _write_current_signal_report(
    *,
    run_dir: Path,
    model_run_id: str,
    signal: dict[str, Any],
    summary: dict[str, Any],
    recent: pd.DataFrame,
) -> Path:
    rows = []
    for row in recent.tail(12).to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(pd.Timestamp(row['date']).date()))}</td>"
            f"<td>{_format_float(row.get('close'))}</td>"
            f"<td>{_format_pct(row.get('target_allocation'))}</td>"
            f"<td>{html.escape(str(row.get('signal', '')))}</td>"
            "</tr>"
        )
    diagnostics = signal.get("diagnostics", {})
    diagnostic_rows = [
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in diagnostics.items()
    ]
    reason_items = "".join(f"<li>{html.escape(reason)}</li>" for reason in signal.get("reasons", []))
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BTC Current Signal</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #172033; margin: 32px; line-height: 1.45; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d8dee9; border-radius: 6px; padding: 12px; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; }}
    th {{ background: #f5f7fb; }}
  </style>
</head>
<body>
  <h1>BTC Current Signal</h1>
  <p>Model run: {html.escape(model_run_id)}. Signal date: {html.escape(signal["date"])}. Execution: next daily open.</p>
  <div class="grid">
    <div class="metric"><div class="label">Signal</div><div class="value">{html.escape(signal["signal"])}</div></div>
    <div class="metric"><div class="label">Action</div><div class="value">{html.escape(signal["action"])}</div></div>
    <div class="metric"><div class="label">Target</div><div class="value">{_format_pct(signal["target_allocation"])}</div></div>
    <div class="metric"><div class="label">Close</div><div class="value">{_format_float(signal["close"])}</div></div>
  </div>
  <h2>Paper Track</h2>
  <table>
    <tr><th>Strategy TWR</th><td>{_format_pct(summary.get("strategy_twr_total_return"))}</td></tr>
    <tr><th>Monthly DCA TWR</th><td>{_format_pct(summary.get("monthly_dca_twr_total_return"))}</td></tr>
    <tr><th>Excess vs DCA</th><td>{_format_pct(summary.get("excess_vs_monthly_dca"))}</td></tr>
    <tr><th>Data status</th><td>{html.escape(str(summary.get("data_freshness", {}).get("status", "n/a")))}</td></tr>
  </table>
  <h2>Why</h2>
  <ul>{reason_items}</ul>
  <h2>Diagnostics</h2>
  <table>{''.join(diagnostic_rows)}</table>
  <h2>Recent Signals</h2>
  <table>
    <tr><th>Date</th><th>Close</th><th>Target</th><th>Signal</th></tr>
    {''.join(rows)}
  </table>
</body>
</html>
"""
    path = run_dir / "current_signal_report.html"
    path.write_text(report, encoding="utf-8")
    return path


def _recent_signal_frame(frame: pd.DataFrame, lookback: int = 120) -> pd.DataFrame:
    recent = frame.tail(lookback).copy()
    recent["target_allocation_change"] = recent["target_allocation"].diff().fillna(0.0)
    recent["signal"] = np.where(
        recent["target_allocation"] <= recent["floor_allocation"] + 1e-9,
        "defensive",
        np.where(recent["target_allocation"] >= 0.95, "risk_on", "partial_risk"),
    )
    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "target_allocation",
        "target_allocation_change",
        "floor_allocation",
        "entry_signal",
        "exit_signal",
        "signal",
    ]
    return recent[[column for column in columns if column in recent.columns]].reset_index(drop=True)


def paper_track_crypto_strategy(
    *,
    model_run_id: str,
    start_date: str,
    config_path: str | Path = "config/crypto_btc.yaml",
    out_run_id: str | None = None,
    max_data_stale_days: int = 2,
) -> Path:
    config: CryptoConfig = load_crypto_config(config_path)
    data = load_crypto_feature_data(config)
    if data.empty:
        raise ValueError("No BTC feature data available. Run crypto-sync first.")

    candidate = _best_crypto_candidate(model_run_id)
    frame = build_crypto_strategy_frame(data, candidate["family"], candidate["params"])
    latest_date = pd.Timestamp(data["date"].iloc[-1])
    start_ts = pd.Timestamp(start_date)
    out_run_id = out_run_id or f"crypto-paper-{model_run_id}-{start_ts.date()}"
    run_dir = ensure_dir(Path("runs") / out_run_id)
    freshness = _data_freshness_status(
        latest_date=latest_date,
        as_of=pd.Timestamp(datetime.now(UTC).date()),
        max_stale_days=max_data_stale_days,
    )
    current_signal = build_current_crypto_signal(frame, candidate["family"], candidate["params"])
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_run_id": model_run_id,
        "family": candidate["family"],
        "params": candidate["params"],
        "requested_start_date": str(start_ts.date()),
        "latest_available_date": str(latest_date.date()),
        "features_path": config.data.features_path,
        "normalized_path": config.data.normalized_path,
        "data_freshness": freshness,
    }

    if start_ts > latest_date:
        summary = {
            "status": "waiting_for_future_data",
            "message": f"No data yet after {start_ts.date()}. Run crypto-sync and rerun crypto-paper-track.",
            "latest_signal": current_signal,
            "data_freshness": freshness,
        }
        write_json(run_dir / "manifest.json", {**manifest, "status": "waiting_for_future_data"})
        write_json(run_dir / "paper_track_summary.json", summary)
        write_json(run_dir / "current_signal.json", current_signal)
        recent = _recent_signal_frame(frame)
        recent.to_csv(run_dir / "current_signal_recent.csv", index=False)
        _write_current_signal_report(
            run_dir=run_dir,
            model_run_id=model_run_id,
            signal=current_signal,
            summary=summary,
            recent=recent,
        )
        return run_dir

    start_idx = int(data.index[data["date"] >= start_ts][0])
    strategy = run_strategy_backtest(frame, start_idx, len(data) - 1, config.backtest)
    monthly_dca = run_dca_benchmark(data, start_idx, len(data) - 1, config.backtest)
    weekly_dca = run_weekly_dca_benchmark(data, start_idx, len(data) - 1, config.backtest)
    buy_hold = run_buy_hold_benchmark(data, start_idx, len(data) - 1)
    buy_hold_equity = buy_hold * float(strategy.flows.iloc[0])
    recent = _recent_signal_frame(frame)

    daily = pd.DataFrame(
        {
            "date": data["date"].iloc[start_idx:].values,
            "close": data["close"].iloc[start_idx:].values,
            "target_allocation": frame["target_allocation"].iloc[start_idx:].values,
            "floor_allocation": frame["floor_allocation"].iloc[start_idx:].values,
            "signal": _recent_signal_frame(frame.iloc[start_idx:], lookback=len(frame) - start_idx)["signal"].values,
            "strategy_equity": strategy.equity.values,
            "monthly_dca_equity": monthly_dca.equity.values,
            "weekly_dca_equity": weekly_dca.equity.values,
            "buy_hold_equity": buy_hold_equity.values,
            "strategy_minus_monthly_dca": strategy.equity.values - monthly_dca.equity.values,
        }
    )
    daily.to_csv(run_dir / "paper_track_daily.csv", index=False)
    recent.to_csv(run_dir / "current_signal_recent.csv", index=False)
    strategy.trades.to_csv(run_dir / "paper_trades.csv", index=False)

    summary = {
        "status": "tracked",
        "model_run_id": model_run_id,
        "family": candidate["family"],
        "paper_accounting_start_date": str(pd.Timestamp(data["date"].iloc[start_idx]).date()),
        "latest_date": str(latest_date.date()),
        "strategy_twr_total_return": strategy.metrics["twr_total_return"],
        "monthly_dca_twr_total_return": monthly_dca.metrics["twr_total_return"],
        "weekly_dca_twr_total_return": weekly_dca.metrics["twr_total_return"],
        "buy_hold_total_return": float(buy_hold.iloc[-1] - 1.0),
        "excess_vs_monthly_dca": strategy.metrics["twr_total_return"] - monthly_dca.metrics["twr_total_return"],
        "excess_vs_weekly_dca": strategy.metrics["twr_total_return"] - weekly_dca.metrics["twr_total_return"],
        "strategy_final_equity": strategy.metrics["final_equity"],
        "monthly_dca_final_equity": monthly_dca.metrics["final_equity"],
        "weekly_dca_final_equity": weekly_dca.metrics["final_equity"],
        "closed_trades": strategy.metrics["closed_trades"],
        "trade_allowed": freshness["status"] == "fresh",
        "block_reasons": ["data_not_fresh"] if freshness["status"] != "fresh" else [],
        "data_freshness": freshness,
        "latest_signal": current_signal,
    }

    write_json(run_dir / "manifest.json", {**manifest, "status": "tracked"})
    write_json(run_dir / "paper_track_summary.json", summary)
    write_json(run_dir / "current_signal.json", current_signal)
    write_json(run_dir / "model_candidate.json", to_native(candidate))
    report_path = _write_current_signal_report(
        run_dir=run_dir,
        model_run_id=model_run_id,
        signal=current_signal,
        summary=summary,
        recent=recent,
    )
    write_json(
        run_dir / "current_signal_report_summary.json",
        {
            "run_id": out_run_id,
            "report_path": str(report_path),
            "signal": current_signal["signal"],
            "action": current_signal["action"],
            "target_allocation": current_signal["target_allocation"],
        },
    )
    return run_dir
