from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.relative_signal import build_relative_signal_frame


def _ohlcv(dates: pd.DatetimeIndex, close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": volume,
        }
    )


def _write_relative_signal_fixture(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation/normalized").mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2024-01-01", periods=360)
    idx = np.arange(len(dates), dtype=float)
    etf_close = 100.0 + idx * 0.04 + np.sin(idx / 11.0) * 0.8
    amoc_close = 20.0 + idx * 0.01 + np.sin(idx / 9.0) * 0.15
    amoc_close += np.maximum(idx - 250.0, 0.0) * 0.06
    weak_close = 30.0 + idx * 0.01 - np.maximum(idx - 230.0, 0.0) * 0.035

    etf_volume = 20_000.0 + idx
    amoc_volume = np.where(idx > 315.0, 3_000.0, 1_000.0 + idx)
    weak_volume = 1_500.0 + idx

    _ohlcv(dates, etf_close, etf_volume).to_csv(
        tmp_path / "data/normalized/EGX30_ETF.csv", index=False
    )
    _ohlcv(dates, etf_close, etf_volume).to_csv(
        tmp_path / "data/normalized/EGX30_INDEX.csv", index=False
    )
    _ohlcv(dates, amoc_close, amoc_volume).to_csv(
        tmp_path / "data/stock_rotation/normalized/AMOC.csv", index=False
    )
    _ohlcv(dates, weak_close, weak_volume).to_csv(
        tmp_path / "data/stock_rotation/normalized/WEAK.csv", index=False
    )

    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "  normalized_dir: normalized",
            ]
        ),
        encoding="utf-8",
    )


def test_build_relative_signal_frame_adds_predictive_features() -> None:
    dates = pd.bdate_range("2024-01-01", periods=300)
    idx = np.arange(len(dates), dtype=float)
    stock = _ohlcv(dates, 10.0 + idx * 0.04, 1_000.0 + idx)
    benchmark = _ohlcv(dates, 20.0 + idx * 0.02, 2_000.0 + idx)

    frame = build_relative_signal_frame(stock, benchmark, symbol="AMOC")

    assert frame.iloc[-1]["symbol"] == "AMOC"
    assert pd.notna(frame.iloc[-1]["rel_mom_63"])
    assert pd.notna(frame.iloc[-1]["ratio_vs_sma50"])
    assert "excess_fwd_63" in frame.columns


def test_relative_signal_cli_symbol_and_all(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_relative_signal_fixture(tmp_path)

    runner = CliRunner()
    single = runner.invoke(
        app,
        [
            "relative-signal",
            "--config",
            "config/stock_rotation.yaml",
            "--symbol",
            "AMOC",
            "--run-id",
            "relative-amoc-test",
        ],
    )
    assert single.exit_code == 0, single.stdout
    single_dir = Path("runs/relative-amoc-test")
    assert (single_dir / "latest_signals.csv").exists()
    assert (single_dir / "signal_edges.csv").exists()
    assert (single_dir / "signal_history.csv").exists()

    latest = pd.read_csv(single_dir / "latest_signals.csv")
    assert latest.iloc[0]["symbol"] == "AMOC"
    assert latest.iloc[0]["main_rule_on"]

    all_result = runner.invoke(
        app,
        [
            "relative-signal",
            "--config",
            "config/stock_rotation.yaml",
            "--all",
            "--run-id",
            "relative-all-test",
        ],
    )
    assert all_result.exit_code == 0, all_result.stdout
    all_dir = Path("runs/relative-all-test")
    all_latest = pd.read_csv(all_dir / "latest_signals.csv")
    assert set(all_latest["symbol"]) == {"AMOC", "WEAK"}
    assert all_latest.iloc[0]["symbol"] == "AMOC"

    with (all_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["benchmark"] == "etf"
    assert summary["symbols_completed"] == ["AMOC", "WEAK"]
