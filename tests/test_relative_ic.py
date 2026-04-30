from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.relative_ic import learn_ic_weights


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


def test_learn_ic_weights_signs_and_neutral_features() -> None:
    rows = []
    for month in pd.date_range("2020-01-01", periods=3, freq="MS"):
        for value, symbol in enumerate(["A", "B", "C", "D"], start=1):
            rows.append(
                {
                    "rebalance_date": month,
                    "symbol": symbol,
                    "target_end_date_63": month + pd.Timedelta(days=20),
                    "future_63d_excess": float(value),
                    "positive_feature": float(value),
                    "inverse_feature": float(-value),
                    "constant_feature": 1.0,
                    "missing_feature": np.nan,
                }
            )
    panel = pd.DataFrame(rows)

    weights, diagnostics = learn_ic_weights(
        panel,
        "2020-06-01",
        feature_columns=[
            "positive_feature",
            "inverse_feature",
            "constant_feature",
            "missing_feature",
        ],
        warmup_months=1,
    )

    assert weights["positive_feature"] > 0.0
    assert weights["inverse_feature"] < 0.0
    assert weights["constant_feature"] == 0.0
    assert weights["missing_feature"] == 0.0
    assert abs(float(weights.abs().sum()) - 1.0) < 1e-12
    assert diagnostics["warmup_complete"].all()


def test_learn_ic_weights_excludes_targets_ending_after_rebalance() -> None:
    rows = []
    for value, symbol in enumerate(["A", "B", "C", "D"], start=1):
        rows.append(
            {
                "rebalance_date": "2020-01-01",
                "symbol": symbol,
                "target_end_date_63": "2020-03-01",
                "future_63d_excess": float(value),
                "signal_feature": float(value),
            }
        )
        rows.append(
            {
                "rebalance_date": "2020-02-01",
                "symbol": symbol,
                "target_end_date_63": "2020-05-01",
                "future_63d_excess": float(value),
                "signal_feature": float(-value),
            }
        )
    panel = pd.DataFrame(rows)

    weights, diagnostics = learn_ic_weights(
        panel,
        "2020-04-01",
        feature_columns=["signal_feature"],
        warmup_months=1,
    )

    assert weights["signal_feature"] == 1.0
    assert diagnostics.iloc[0]["training_months"] == 1
    assert diagnostics.iloc[0]["training_rows"] == 4
    assert diagnostics.iloc[0]["raw_ic_mean"] == 1.0


def _write_relative_ic_fixture(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2020-01-01", periods=1100)
    idx = np.arange(len(dates), dtype=float)
    etf_close = 100.0 * np.exp(0.0002 * idx)
    _ohlcv(dates, etf_close, 50_000.0).to_csv(
        tmp_path / "data/normalized/EGX30_ETF.csv", index=False
    )
    _ohlcv(dates, etf_close, 50_000.0).to_csv(
        tmp_path / "data/normalized/EGX30_INDEX.csv", index=False
    )

    profiles = {
        "STRONG": 0.0010,
        "GOOD": 0.00065,
        "OKAY": 0.00030,
        "WEAK": -0.00010,
    }
    panel_rows = []
    for symbol, drift in profiles.items():
        close = 20.0 * np.exp(drift * idx)
        stock = _ohlcv(dates, close, 25_000.0)
        stock["symbol"] = symbol
        stock["holding_name"] = f"{symbol} Holding"
        stock["weight"] = 0.25
        panel_rows.append(stock)
    pd.concat(panel_rows, ignore_index=True).to_csv(
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


def test_relative_ic_cli_smoke_writes_artifacts_and_selects_strongest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_relative_ic_fixture(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "relative-ic-backtest",
            "--config",
            "config/stock_rotation.yaml",
            "--run-id",
            "relative-ic-smoke",
        ],
    )
    assert result.exit_code == 0, result.stdout

    run_dir = Path("runs/relative-ic-smoke")
    for filename in [
        "ic_weights.csv",
        "monthly_rankings.csv",
        "selected_holdings.csv",
        "equity_curve.csv",
        "trade_actions.csv",
        "turnover.csv",
        "summary.json",
    ]:
        assert (run_dir / filename).exists()

    rankings = pd.read_csv(run_dir / "monthly_rankings.csv", parse_dates=["rebalance_date"])
    latest = rankings[rankings["rebalance_date"] == rankings["rebalance_date"].max()]
    top = latest.sort_values("rank").iloc[0]
    assert top["symbol"] == "STRONG"
    assert bool(top["selected"])

    selected = pd.read_csv(run_dir / "selected_holdings.csv")
    latest_selected = selected[selected["rebalance_date"] == selected["rebalance_date"].max()]
    assert "STRONG" in set(latest_selected["symbol"])

    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["number_of_rebalances"] > 0
    assert summary["latest_selected_stocks"][0] == "STRONG"
