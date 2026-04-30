from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from egx_research.annual_top10 import build_annual_top10_basket
from egx_research.cli import app


def _make_symbol_panel(
    symbol: str,
    holding_name: str,
    dates: pd.DatetimeIndex,
    drift: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 10.0 + seed
    trend = base + np.arange(len(dates)) * drift
    cycle = np.sin(np.linspace(0, 10 * np.pi, len(dates))) * 0.25
    close = trend + cycle
    open_ = close * (1 + rng.normal(0, 0.001, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, len(dates)))
    volume = 1000 + seed * 100 + np.arange(len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
            "holding_name": holding_name,
            "weight": 1.0 / 6.0,
        }
    )


def test_build_annual_top10_basket_selects_top_returners() -> None:
    panel_dates = pd.bdate_range("2019-01-01", periods=450)
    benchmark_dates = panel_dates[120:]
    membership = pd.DataFrame(
        {
            "effective_date": [panel_dates[0]] * 6,
            "symbol": list("ABCDEF"),
            "is_member": [True] * 6,
            "source": ["test"] * 6,
        }
    )
    panel = pd.concat(
        [
            _make_symbol_panel(symbol, f"Stock {symbol}", panel_dates, drift, seed)
            for seed, (symbol, drift) in enumerate(
                [("A", 0.01), ("B", 0.015), ("C", 0.02), ("D", 0.03), ("E", 0.04), ("F", 0.05)],
                start=1,
            )
        ],
        ignore_index=True,
    )

    basket, selections = build_annual_top10_basket(
        panel,
        membership,
        pd.Series(benchmark_dates),
        top_n=2,
        lookback_bars=63,
        hold_cash_when_few=True,
    )

    first_rebalance = selections["rebalance_date"].min()
    first_symbols = selections.loc[selections["rebalance_date"] == first_rebalance, "symbol"].tolist()
    assert first_symbols == ["F", "E"]
    assert not basket.empty
    assert list(basket.columns[:6]) == ["date", "open", "high", "low", "close", "volume"]


def test_annual_top10_cli_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs/pullback-seed").mkdir(parents=True, exist_ok=True)

    panel_dates = pd.bdate_range("2019-01-01", periods=520)
    benchmark_dates = panel_dates[140:]

    panel = pd.concat(
        [
            _make_symbol_panel(symbol, f"Stock {symbol}", panel_dates, drift, seed)
            for seed, (symbol, drift) in enumerate(
                [("A", 0.01), ("B", 0.015), ("C", 0.02), ("D", 0.03), ("E", 0.04), ("F", 0.05)],
                start=1,
            )
        ],
        ignore_index=True,
    )
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)

    membership = pd.DataFrame(
        {
            "symbol": list("ABCDEF"),
            "is_member": [True] * 6,
            "effective_date": [panel_dates[0]] * 6,
            "source": ["test"] * 6,
        }
    )
    membership.to_csv(tmp_path / "data/stock_rotation/membership_verified_partial.csv", index=False)

    universe = panel.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol"], keep="last")[
        ["symbol", "holding_name", "weight"]
    ].rename(columns={"symbol": "ticker"})
    universe["ric_code"] = universe["ticker"].str.lower() + ".ca"
    universe["stock_page_url"] = "https://example.com"
    universe["historical_csv_url"] = "https://example.com/history.csv"
    universe.to_csv(tmp_path / "data/stock_rotation/universe.csv", index=False)

    etf = pd.DataFrame(
        {
            "date": benchmark_dates,
            "open": np.linspace(100, 160, len(benchmark_dates)),
            "high": np.linspace(101, 161, len(benchmark_dates)),
            "low": np.linspace(99, 159, len(benchmark_dates)),
            "close": np.linspace(100, 160, len(benchmark_dates)),
            "volume": np.linspace(1000, 2000, len(benchmark_dates)),
        }
    )
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    etf.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)

    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "  universe_filename: universe.csv",
                "  panel_filename: panel.csv",
                "  membership_filename: membership_snapshots.csv",
                "  raw_dir: raw",
                "  normalized_dir: normalized",
                "portfolio:",
                "  top_n: 2",
                "  hold_cash_when_few: true",
                "backtest:",
                "  initial_cash: 100000.0",
                "  monthly_contribution: 10000.0",
                "  fee_bps: 0.0",
                "  slippage_bps: 0.0",
                "  share_precision: 0",
            ]
        ),
        encoding="utf-8",
    )

    with (tmp_path / "runs/pullback-seed/report_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "top_family": "dca_pullback_only",
                "top_params": {
                    "kama_len": 20,
                    "kama_fast": 2,
                    "kama_slow": 30,
                    "cci_len": 20,
                    "buy_threshold": -30.0,
                    "trend_buffer_atr": 0.5,
                    "atr_len": 14,
                },
            },
            handle,
        )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "annual-top10-backtest",
            "--stock-config",
            "config/stock_rotation.yaml",
            "--run-id",
            "annual-smoke",
            "--top-n",
            "2",
            "--lookback-bars",
            "63",
            "--pullback-run-id",
            "pullback-seed",
        ],
    )
    assert result.exit_code == 0, result.stdout

    summary_path = Path("runs/annual-smoke/summary.json")
    report_path = Path("runs/annual-smoke/annual_top10_report.html")
    selections_path = Path("runs/annual-smoke/annual_selections.csv")

    assert summary_path.exists()
    assert report_path.exists()
    assert selections_path.exists()

    selections = pd.read_csv(selections_path)
    first_rebalance = selections["rebalance_date"].min()
    first_symbols = selections.loc[selections["rebalance_date"] == first_rebalance, "symbol"].tolist()
    assert first_symbols == ["F", "E"]


def test_annual_top10_stock_cli_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs/pullback-seed").mkdir(parents=True, exist_ok=True)

    panel_dates = pd.bdate_range("2019-01-01", periods=520)
    benchmark_dates = panel_dates[140:]

    panel = pd.concat(
        [
            _make_symbol_panel(symbol, f"Stock {symbol}", panel_dates, drift, seed)
            for seed, (symbol, drift) in enumerate(
                [("A", 0.01), ("B", 0.015), ("C", 0.02), ("D", 0.03), ("E", 0.04), ("F", 0.05)],
                start=1,
            )
        ],
        ignore_index=True,
    )
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)

    membership = pd.DataFrame(
        {
            "symbol": list("ABCDEF"),
            "is_member": [True] * 6,
            "effective_date": [panel_dates[0]] * 6,
            "source": ["test"] * 6,
        }
    )
    membership.to_csv(tmp_path / "data/stock_rotation/membership_verified_partial.csv", index=False)

    universe = panel.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol"], keep="last")[
        ["symbol", "holding_name", "weight"]
    ].rename(columns={"symbol": "ticker"})
    universe["ric_code"] = universe["ticker"].str.lower() + ".ca"
    universe["stock_page_url"] = "https://example.com"
    universe["historical_csv_url"] = "https://example.com/history.csv"
    universe.to_csv(tmp_path / "data/stock_rotation/universe.csv", index=False)

    etf = pd.DataFrame(
        {
            "date": benchmark_dates,
            "open": np.linspace(100, 160, len(benchmark_dates)),
            "high": np.linspace(101, 161, len(benchmark_dates)),
            "low": np.linspace(99, 159, len(benchmark_dates)),
            "close": np.linspace(100, 160, len(benchmark_dates)),
            "volume": np.linspace(1000, 2000, len(benchmark_dates)),
        }
    )
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    etf.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)

    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "  universe_filename: universe.csv",
                "  panel_filename: panel.csv",
                "  membership_filename: membership_snapshots.csv",
                "  raw_dir: raw",
                "  normalized_dir: normalized",
                "portfolio:",
                "  top_n: 2",
                "  hold_cash_when_few: true",
                "  fixed_buy_fee_egp: 0.0",
                "backtest:",
                "  initial_cash: 100000.0",
                "  monthly_contribution: 10000.0",
                "  fee_bps: 0.0",
                "  slippage_bps: 0.0",
                "  share_precision: 0",
            ]
        ),
        encoding="utf-8",
    )

    with (tmp_path / "runs/pullback-seed/report_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "top_family": "dca_pullback_only",
                "top_params": {
                    "kama_len": 20,
                    "kama_fast": 2,
                    "kama_slow": 30,
                    "cci_len": 20,
                    "buy_threshold": -30.0,
                    "trend_buffer_atr": 0.5,
                    "atr_len": 14,
                },
            },
            handle,
        )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "annual-top10-stock-backtest",
            "--stock-config",
            "config/stock_rotation.yaml",
            "--run-id",
            "annual-stock-smoke",
            "--top-n",
            "2",
            "--lookback-bars",
            "63",
            "--pullback-run-id",
            "pullback-seed",
        ],
    )
    assert result.exit_code == 0, result.stdout

    assert Path("runs/annual-stock-smoke/summary_stock.json").exists()
    assert Path("runs/annual-stock-smoke/annual_top10_stock_report.html").exists()
    assert Path("runs/annual-stock-smoke/selection_coverage.csv").exists()
