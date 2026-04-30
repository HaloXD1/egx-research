from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from egx_research.cli import app


def _ohlcv(dates: pd.DatetimeIndex, close: np.ndarray, volume: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        }
    )


def _write_fixture(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2020-01-01", periods=700)
    idx = np.arange(len(dates), dtype=float)
    etf_close = 100.0 * np.exp(0.0002 * idx)
    _ohlcv(dates, etf_close, 50_000.0).to_csv(
        tmp_path / "data/normalized/EGX30_ETF.csv", index=False
    )
    _ohlcv(dates, etf_close, 50_000.0).to_csv(
        tmp_path / "data/normalized/EGX30_INDEX.csv", index=False
    )

    profiles = {"AAA": 0.0010, "BBB": 0.0008, "CCC": 0.0001, "DDD": -0.0001}
    frames = []
    for symbol, drift in profiles.items():
        close = 20.0 * np.exp(drift * idx)
        frame = _ohlcv(dates, close, 30_000.0)
        frame["symbol"] = symbol
        frame["holding_name"] = f"{symbol} Holding"
        frame["weight"] = 0.25
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(
        tmp_path / "data/stock_rotation/panel.csv", index=False
    )
    pd.DataFrame(
        {
            "symbol": list(profiles),
            "is_member": [True] * len(profiles),
            "effective_date": [dates[0]] * len(profiles),
            "source": ["synthetic"] * len(profiles),
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_verified_partial.csv", index=False)
    pd.DataFrame(
        {
            "ticker": list(profiles),
            "holding_name": [f"{symbol} Holding" for symbol in profiles],
            "weight": [0.25] * len(profiles),
        }
    ).to_csv(tmp_path / "data/stock_rotation/universe.csv", index=False)

    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
                "  fee_bps: 0.0",
                "  slippage_bps: 0.0",
                "  share_precision: 3",
                "storage:",
                "  root_dir: data/stock_rotation",
                "  panel_filename: panel.csv",
                "portfolio:",
                "  top_n: 2",
                "  fixed_buy_fee_egp: 0.0",
                "selection:",
                "  liquidity_window_bars: 20",
                "  min_median_daily_value_egp: 1.0",
                "  min_median_daily_volume: 1.0",
                "validation:",
                "  min_history_bars: 20",
                "  coverage_lookback_bars: 20",
                "  min_coverage_ratio: 0.8",
            ]
        ),
        encoding="utf-8",
    )


def test_stock_momentum_pyramid_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_fixture(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "stock-momentum-pyramid-backtest",
            "--config",
            "config/stock_rotation.yaml",
            "--run-id",
            "pyramid-smoke",
            "--max-holdings",
            "2",
            "--focus-n",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout

    run_dir = Path("runs/pyramid-smoke")
    for filename in [
        "monthly_rankings.csv",
        "selected_holdings.csv",
        "equity_curve.csv",
        "trade_actions.csv",
        "turnover.csv",
        "summary.json",
    ]:
        assert (run_dir / filename).exists()

    rankings = pd.read_csv(run_dir / "monthly_rankings.csv")
    first = rankings[rankings["rebalance_date"] == rankings["rebalance_date"].min()]
    assert first.sort_values("rank").iloc[0]["symbol"] == "AAA"

    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["max_holdings"] == 2
    assert summary["momentum_pyramid_metrics"]["final_equity"] > 0
