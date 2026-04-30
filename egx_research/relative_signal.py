from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egx_research.data import load_price_data
from egx_research.stock_rotation_config import (
    StockRotationConfig,
    load_stock_rotation_config,
)
from egx_research.utils import ensure_dir, write_json


DEFAULT_HORIZONS = (21, 63, 126)
SIGNAL_COLUMNS = [
    "rel_mom_21",
    "rel_mom_63",
    "rel_mom_126",
    "rel_mom_252",
    "ratio_vs_sma50",
    "ratio_vs_sma200",
    "sma50_vs_sma200",
    "ratio_breakout_20",
    "ratio_breakout_63",
    "ratio_breakout_126",
    "vol_adj_rel_mom_63",
    "volume_ratio_20_63",
    "mom63_volume_score",
    "down_day_excess_63",
    "down_day_excess_126",
    "beta_252",
    "corr_252",
    "residual_strength_21_126b",
    "residual_strength_63_126b",
    "residual_strength_126_126b",
]


@dataclass
class RelativeSignalRun:
    run_id: str
    run_dir: Path


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _pct(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


def _benchmark_path(config: StockRotationConfig, benchmark: str) -> tuple[Path, str]:
    normalized = benchmark.strip().lower()
    if normalized == "etf":
        return Path(config.benchmark.etf_symbol_path), "etf"
    if normalized == "index":
        return Path(config.benchmark.index_symbol_path), "index"
    return Path(benchmark), Path(benchmark).stem


def _load_stock_file(config: StockRotationConfig, symbol: str) -> pd.DataFrame:
    normalized_dir = Path(config.storage.root_dir) / config.storage.normalized_dir
    path = normalized_dir / f"{symbol.upper()}.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"])
    else:
        panel_path = Path(config.storage.root_dir) / config.storage.panel_filename
        if not panel_path.exists():
            raise FileNotFoundError(f"Stock file missing: {path}")
        panel = pd.read_csv(panel_path, parse_dates=["date"])
        frame = panel[panel["symbol"].astype(str).str.upper() == symbol.upper()].copy()
        if frame.empty:
            raise FileNotFoundError(f"Stock symbol missing: {symbol.upper()}")

    required = {"date", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{symbol.upper()} missing columns: {sorted(missing)}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["symbol"] = symbol.upper()
    for column in ("close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame[["date", "close", "volume", "symbol"]]
        .dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )


def _list_symbols(config: StockRotationConfig) -> list[str]:
    normalized_dir = Path(config.storage.root_dir) / config.storage.normalized_dir
    if normalized_dir.exists():
        symbols = sorted(path.stem.upper() for path in normalized_dir.glob("*.csv"))
        if symbols:
            return symbols

    panel_path = Path(config.storage.root_dir) / config.storage.panel_filename
    if panel_path.exists():
        panel = pd.read_csv(panel_path, usecols=["symbol"])
        return sorted(panel["symbol"].astype(str).str.upper().unique().tolist())

    raise FileNotFoundError(f"No stock data found under {normalized_dir}")


def _month_end_sample(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.sort_values("date")
        .groupby(frame["date"].dt.to_period("M"), group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )


def build_relative_signal_frame(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    stock_frame = stock.rename(
        columns={"close": "stock_close", "volume": "stock_volume"}
    )[["date", "stock_close", "stock_volume"]]
    benchmark_frame = benchmark.rename(columns={"close": "benchmark_close"})[
        ["date", "benchmark_close"]
    ]
    frame = (
        stock_frame.merge(benchmark_frame, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError(f"No benchmark overlap for {symbol.upper()}")

    frame["symbol"] = symbol.upper()
    frame["stock_ret"] = frame["stock_close"].pct_change()
    frame["benchmark_ret"] = frame["benchmark_close"].pct_change()
    frame["excess_ret"] = frame["stock_ret"] - frame["benchmark_ret"]
    frame["ratio"] = frame["stock_close"] / frame["benchmark_close"]

    for window in (21, 63, 126, 252):
        frame[f"rel_mom_{window}"] = frame["ratio"] / frame["ratio"].shift(window) - 1.0

    frame["ratio_sma50"] = frame["ratio"].rolling(50, min_periods=50).mean()
    frame["ratio_sma200"] = frame["ratio"].rolling(200, min_periods=200).mean()
    frame["ratio_vs_sma50"] = frame["ratio"] / frame["ratio_sma50"] - 1.0
    frame["ratio_vs_sma200"] = frame["ratio"] / frame["ratio_sma200"] - 1.0
    frame["sma50_vs_sma200"] = frame["ratio_sma50"] / frame["ratio_sma200"] - 1.0

    for window in (20, 63, 126):
        prior_high = frame["ratio"].rolling(window, min_periods=window).max().shift(1)
        frame[f"ratio_breakout_{window}"] = frame["ratio"] / prior_high - 1.0

    rel_vol_63 = frame["excess_ret"].rolling(63, min_periods=63).std(ddof=0)
    frame["vol_adj_rel_mom_63"] = frame["rel_mom_63"] / rel_vol_63.replace(0.0, np.nan)
    frame["volume_ratio_20_63"] = (
        frame["stock_volume"].rolling(20, min_periods=20).mean()
        / frame["stock_volume"].rolling(63, min_periods=63).mean()
    )
    frame["mom63_volume_score"] = frame["rel_mom_63"] * frame["volume_ratio_20_63"]

    down_excess = frame["excess_ret"].where(frame["benchmark_ret"] < 0.0)
    frame["down_day_excess_63"] = down_excess.rolling(63, min_periods=10).mean()
    frame["down_day_excess_126"] = down_excess.rolling(126, min_periods=20).mean()

    beta_126 = (
        frame["stock_ret"].rolling(126, min_periods=63).cov(frame["benchmark_ret"])
        / frame["benchmark_ret"].rolling(126, min_periods=63).var(ddof=0)
    )
    beta_252 = (
        frame["stock_ret"].rolling(252, min_periods=126).cov(frame["benchmark_ret"])
        / frame["benchmark_ret"].rolling(252, min_periods=126).var(ddof=0)
    )
    frame["beta_252"] = beta_252
    frame["corr_252"] = frame["stock_ret"].rolling(252, min_periods=126).corr(
        frame["benchmark_ret"]
    )
    frame["residual_ret_126b"] = (
        frame["stock_ret"] - beta_126.shift(1) * frame["benchmark_ret"]
    )
    for window in (21, 63, 126):
        frame[f"residual_strength_{window}_126b"] = (
            frame["residual_ret_126b"].rolling(window, min_periods=window).sum()
        )

    for horizon in DEFAULT_HORIZONS:
        frame[f"stock_fwd_{horizon}"] = (
            frame["stock_close"].shift(-horizon) / frame["stock_close"] - 1.0
        )
        frame[f"benchmark_fwd_{horizon}"] = (
            frame["benchmark_close"].shift(-horizon) / frame["benchmark_close"] - 1.0
        )
        frame[f"excess_fwd_{horizon}"] = (
            frame[f"stock_fwd_{horizon}"] - frame[f"benchmark_fwd_{horizon}"]
        )
        frame[f"outperform_fwd_{horizon}"] = frame[f"excess_fwd_{horizon}"] > 0.0

    return frame


def _rule_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "rel_mom_63_gt_0": frame["rel_mom_63"] > 0.0,
        "rel_mom_126_gt_0": frame["rel_mom_126"] > 0.0,
        "ratio_gt_sma50": frame["ratio_vs_sma50"] > 0.0,
        "ratio_gt_sma200": frame["ratio_vs_sma200"] > 0.0,
        "sma50_gt_sma200": frame["sma50_vs_sma200"] > 0.0,
        "rel_mom_63_volume_1_1": (frame["rel_mom_63"] > 0.0)
        & (frame["volume_ratio_20_63"] > 1.1),
        "residual_strength_63_gt_0": frame["residual_strength_63_126b"] > 0.0,
        "relative_breakout_63": frame["ratio_breakout_63"] > 0.0,
    }


def evaluate_signal_edges(
    frame: pd.DataFrame,
    *,
    symbol: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    sample = _month_end_sample(frame)
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target = f"excess_fwd_{horizon}"
        hit = f"outperform_fwd_{horizon}"
        valid_target = sample[target].notna()
        for rule_name, mask in _rule_masks(sample).items():
            valid = valid_target & mask.notna()
            on = valid & mask
            off = valid & ~mask
            avg_on = sample.loc[on, target].mean() if on.any() else np.nan
            avg_off = sample.loc[off, target].mean() if off.any() else np.nan
            hit_on = sample.loc[on, hit].mean() if on.any() else np.nan
            hit_off = sample.loc[off, hit].mean() if off.any() else np.nan
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "sample": "month_end",
                    "rule": rule_name,
                    "horizon_days": horizon,
                    "samples": int(valid.sum()),
                    "days_on": int(on.sum()),
                    "coverage": float(on.sum() / valid.sum()) if valid.any() else np.nan,
                    "avg_excess_on": avg_on,
                    "hit_rate_on": hit_on,
                    "avg_excess_off": avg_off,
                    "hit_rate_off": hit_off,
                    "edge_avg_excess": avg_on - avg_off,
                    "edge_hit_rate": hit_on - hit_off,
                }
            )
    return pd.DataFrame(rows)


def latest_signal_row(
    frame: pd.DataFrame,
    *,
    symbol: str,
    benchmark_label: str,
    edge_frame: pd.DataFrame,
) -> dict[str, Any]:
    latest = frame.iloc[-1]
    score = 0.0
    score += 25.0 * _pct(latest["rel_mom_63"] > 0.0)
    score += 20.0 * _pct(latest["ratio_vs_sma50"] > 0.0)
    score += 20.0 * _pct(latest["residual_strength_63_126b"] > 0.0)
    score += 15.0 * _pct(
        (latest["rel_mom_63"] > 0.0) and (latest["volume_ratio_20_63"] > 1.1)
    )
    score += 10.0 * _pct(latest["ratio_vs_sma200"] > 0.0)
    score += 10.0 * _pct(latest["rel_mom_126"] > 0.0)

    if score >= 70.0:
        state = "bullish"
    elif score >= 45.0:
        state = "watch"
    else:
        state = "weak"

    main_edge = edge_frame[
        (edge_frame["symbol"] == symbol.upper())
        & (edge_frame["rule"] == "rel_mom_63_volume_1_1")
        & (edge_frame["horizon_days"] == 63)
    ]
    edge = main_edge.iloc[0].to_dict() if not main_edge.empty else {}

    row: dict[str, Any] = {
        "symbol": symbol.upper(),
        "benchmark": benchmark_label,
        "as_of_date": latest["date"].date().isoformat(),
        "overlap_start": frame["date"].iloc[0].date().isoformat(),
        "overlap_end": frame["date"].iloc[-1].date().isoformat(),
        "overlap_rows": int(len(frame)),
        "stock_close": _safe_float(latest["stock_close"]),
        "benchmark_close": _safe_float(latest["benchmark_close"]),
        "signal_score": score,
        "signal_state": state,
        "main_rule_on": bool(
            (latest["rel_mom_63"] > 0.0) and (latest["volume_ratio_20_63"] > 1.1)
        ),
        "main_rule_edge_63d": _safe_float(edge.get("edge_avg_excess")),
        "main_rule_hit_rate_63d": _safe_float(edge.get("hit_rate_on")),
        "main_rule_samples_63d": int(edge.get("days_on", 0) or 0),
    }
    for column in SIGNAL_COLUMNS:
        row[column] = _safe_float(latest.get(column))
    return row


def _history_export(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    columns = [
        "date",
        "symbol",
        "stock_close",
        "benchmark_close",
        "ratio",
        *SIGNAL_COLUMNS,
        "excess_fwd_21",
        "excess_fwd_63",
        "excess_fwd_126",
    ]
    available = [column for column in columns if column in frame.columns]
    export = frame[available].copy()
    export["symbol"] = symbol.upper()
    return export


def run_relative_signal(
    *,
    config_path: str | Path = Path("config/stock_rotation.yaml"),
    symbol: str | None = None,
    all_symbols: bool = False,
    benchmark: str = "etf",
    run_id: str | None = None,
) -> RelativeSignalRun:
    if symbol and all_symbols:
        raise ValueError("Use either --symbol or --all, not both.")
    config = load_stock_rotation_config(config_path)
    benchmark_file, benchmark_label = _benchmark_path(config, benchmark)
    benchmark_frame = load_price_data(benchmark_file)

    symbols = _list_symbols(config) if all_symbols else [symbol.upper() if symbol else "AMOC"]
    date_tag = datetime.now(UTC).strftime("%Y%m%d")
    scope = "all" if all_symbols else symbols[0].lower()
    actual_run_id = run_id or f"relative-signal-{scope}-{benchmark_label}-{date_tag}"
    run_dir = ensure_dir(Path("runs") / actual_run_id)

    latest_rows: list[dict[str, Any]] = []
    edge_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for stock_symbol in symbols:
        try:
            stock_frame = _load_stock_file(config, stock_symbol)
            signal_frame = build_relative_signal_frame(
                stock_frame, benchmark_frame, symbol=stock_symbol
            )
            edges = evaluate_signal_edges(signal_frame, symbol=stock_symbol)
            latest_rows.append(
                latest_signal_row(
                    signal_frame,
                    symbol=stock_symbol,
                    benchmark_label=benchmark_label,
                    edge_frame=edges,
                )
            )
            edge_frames.append(edges)
            history_frames.append(_history_export(signal_frame, stock_symbol))
        except Exception as exc:
            errors.append({"symbol": stock_symbol, "error": str(exc)})

    if not latest_rows:
        raise ValueError(f"No relative signals generated. errors={errors}")

    latest = pd.DataFrame(latest_rows).sort_values(
        ["signal_score", "rel_mom_63", "residual_strength_63_126b"],
        ascending=[False, False, False],
    )
    latest["rank"] = range(1, len(latest) + 1)
    latest.to_csv(run_dir / "latest_signals.csv", index=False)

    edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame()
    edges.to_csv(run_dir / "signal_edges.csv", index=False)

    history = (
        pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    )
    history.to_csv(run_dir / "signal_history.csv", index=False)

    summary = {
        "run_id": actual_run_id,
        "benchmark": benchmark_label,
        "benchmark_path": str(benchmark_file),
        "symbols_requested": symbols,
        "symbols_completed": latest["symbol"].tolist(),
        "errors": errors,
        "latest_top": latest.head(10).to_dict(orient="records"),
        "artifact_paths": {
            "latest_signals": str(run_dir / "latest_signals.csv"),
            "signal_edges": str(run_dir / "signal_edges.csv"),
            "signal_history": str(run_dir / "signal_history.csv"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return RelativeSignalRun(run_id=actual_run_id, run_dir=run_dir)
