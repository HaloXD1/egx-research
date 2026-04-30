from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.config import BacktestConfig
from egx_research.stock_rotation_config import StockRotationConfig
from egx_research.stock_strategy_research import (
    build_strategy_feature_panel,
    simulate_event_driven_strategy,
)
from tests.conftest import make_synthetic_ohlcv


def _panel(symbols: list[str], rows: int = 180) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols):
        frame = make_synthetic_ohlcv(rows=rows, seed=20 + idx)
        frame["symbol"] = symbol
        frame["holding_name"] = symbol.title()
        frame["weight"] = 1.0 / len(symbols)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_event_driven_exit_next_open_and_waits_for_next_setup() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "holding_name": ["AAA"] * len(dates) + ["BBB"] * len(dates),
            "open": [10, 10, 10, 10, 10, 11, 12, 12] + [20] * len(dates),
            "high": [10, 10, 10, 10, 10, 14, 12, 12] + [20] * len(dates),
            "low": [10, 10, 10, 10, 10, 10, 12, 12] + [20] * len(dates),
            "close": [10, 10, 10, 10, 10, 13, 12, 12] + [20] * len(dates),
            "volume": [1000] * len(dates) * 2,
            "weight": [0.5] * len(dates) * 2,
        }
    )
    features = panel.copy()
    features["atr"] = 1.0
    features["rank_score"] = 1.0
    features["entry_signal"] = False
    features["signal_fail"] = False
    features.loc[
        (features["symbol"] == "AAA") & (features["date"] == dates[4]),
        "entry_signal",
    ] = True
    membership = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "effective_date": [pd.Timestamp("2023-01-01")] * 2,
            "is_member": [True, True],
        }
    )
    config = StockRotationConfig()
    config.backtest = BacktestConfig(
        initial_cash=10_000.0,
        monthly_contribution=0.0,
        fee_bps=20.0,
        slippage_bps=0.0,
        share_precision=0,
    )
    config.portfolio.fixed_buy_fee_egp = 5.0

    sim = simulate_event_driven_strategy(
        panel=panel,
        features=features,
        calendar=pd.Series(dates),
        membership=membership,
        config=config,
        family="breakout",
        params={
            "scan_mode": "weekly",
            "top_n": 2,
            "max_positions": 1,
            "target_atr": 2.0,
            "stop_atr": 1.0,
            "trail_atr": 10.0,
            "max_hold_bars": 20,
        },
    )

    actions = sim.actions
    assert actions["action"].tolist() == ["BUY", "SELL"]
    assert actions.iloc[0]["date"] == str(dates[5].date())
    assert actions.iloc[1]["date"] == str(dates[6].date())
    assert actions.iloc[1]["reason"] == "target"
    assert actions["fee"].gt(0).all()


def test_event_driven_position_size_multiplier_scales_buy() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    panel = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * len(dates),
            "holding_name": ["AAA"] * len(dates),
            "open": [10.0] * len(dates),
            "high": [11.0] * len(dates),
            "low": [9.0] * len(dates),
            "close": [10.0] * len(dates),
            "volume": [1000] * len(dates),
            "weight": [1.0] * len(dates),
        }
    )
    features = panel.copy()
    features["atr"] = 1.0
    features["rank_score"] = 1.0
    features["entry_signal"] = False
    features["signal_fail"] = False
    features["position_size_mult"] = 0.5
    features.loc[features["date"] == dates[4], "entry_signal"] = True
    membership = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "effective_date": [pd.Timestamp("2023-01-01")],
            "is_member": [True],
        }
    )
    config = StockRotationConfig()
    config.backtest = BacktestConfig(
        initial_cash=10_000.0,
        monthly_contribution=0.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        share_precision=0,
    )
    config.portfolio.fixed_buy_fee_egp = 0.0

    sim = simulate_event_driven_strategy(
        panel=panel,
        features=features,
        calendar=pd.Series(dates),
        membership=membership,
        config=config,
        family="rebound",
        params={
            "scan_mode": "weekly",
            "top_n": 1,
            "max_positions": 1,
            "target_atr": 10.0,
            "stop_atr": 1.0,
            "trail_atr": 10.0,
            "max_hold_bars": 20,
            "max_position_weight": 1.0,
        },
    )

    buy = sim.actions[sim.actions["action"] == "BUY"].iloc[0]
    assert buy["value"] == 5000.0


def test_news_event_features_do_not_use_future_events() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    panel = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * len(dates),
            "holding_name": ["AAA"] * len(dates),
            "open": range(10, 18),
            "high": range(11, 19),
            "low": range(9, 17),
            "close": range(10, 18),
            "volume": [1000] * len(dates),
            "weight": [1.0] * len(dates),
        }
    )
    events = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "event_date": [dates[4]],
            "event_class": ["dividend"],
            "title": ["Dividend"],
            "summary": [""],
        }
    )
    features = build_strategy_feature_panel(
        panel,
        family="news_event_rules",
        params={
            "atr_len": 2,
            "trend_len": 2,
            "event_window_days": 30,
            "confirm_lookback": 2,
            "min_event_score": 1.0,
            "volume_short": 2,
            "volume_long": 3,
        },
        disclosure_events=events,
    )
    assert features.loc[features["date"] < dates[4], "event_score"].eq(0.0).all()
    assert features.loc[features["date"] >= dates[4], "event_score"].gt(0.0).any()


def test_stock_strategy_research_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True)

    etf = make_synthetic_ohlcv(rows=180, seed=3)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    panel = _panel(["AAA", "BBB", "CCC"], rows=180)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "effective_date": [pd.Timestamp("2019-01-01")] * 3,
            "is_member": [True, True, True],
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_snapshots.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "event_date": [pd.Timestamp("2020-04-01")],
            "event_class": ["dividend"],
            "title": ["Dividend"],
            "summary": [""],
            "source_url": ["https://example.com"],
        }
    ).to_csv(tmp_path / "data/stock_rotation/disclosure_events.csv", index=False)
    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
                "model_selection:",
                "  holdout_ratio: 0.2",
                "  max_drawdown: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "stock-strategy-research",
            "--config",
            "config/stock_rotation.yaml",
            "--trials",
            "1",
            "--run-id",
            "strategy-smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dir = Path("runs/strategy-smoke")
    assert (run_dir / "leaderboard.csv").exists()
    assert (run_dir / "stock_strategy_report.html").exists()
    assert (run_dir / "actions_breakout.csv").exists()
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["families"] == ["breakout", "rebound", "news_event_rules"]


def test_stock_strategy_validation_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True)

    etf = make_synthetic_ohlcv(rows=280, seed=13)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    panel = _panel(["AAA", "BBB", "CCC", "DDD"], rows=280)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "effective_date": [pd.Timestamp("2020-01-01")] * 4,
            "is_member": [True, True, True, True],
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_snapshots.csv", index=False)
    pd.DataFrame(
        columns=["symbol", "event_date", "event_class", "title", "summary", "source_url"]
    ).to_csv(tmp_path / "data/stock_rotation/disclosure_events.csv", index=False)
    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
                "model_selection:",
                "  holdout_ratio: 0.2",
                "  max_drawdown: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "stock-strategy-validate",
            "--config",
            "config/stock_rotation.yaml",
            "--trials",
            "1",
            "--max-positions",
            "3,4",
            "--train-end",
            "2020-06-30",
            "--run-id",
            "strategy-validate-smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dir = Path("runs/strategy-validate-smoke")
    assert (run_dir / "static_freeze_summary.csv").exists()
    assert (run_dir / "position_count_summary.csv").exists()
    assert (run_dir / "summary.json").exists()


def test_stock_strategy_v2_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True)

    etf = make_synthetic_ohlcv(rows=280, seed=23)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    panel = _panel(["AAA", "BBB", "CCC", "DDD", "EEE"], rows=280)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "effective_date": [pd.Timestamp("2020-01-01")] * 5,
            "is_member": [True, True, True, True, True],
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_snapshots.csv", index=False)
    pd.DataFrame(
        columns=["symbol", "event_date", "event_class", "title", "summary", "source_url"]
    ).to_csv(tmp_path / "data/stock_rotation/disclosure_events.csv", index=False)
    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "stock-strategy-v2",
            "--config",
            "config/stock_rotation.yaml",
            "--train-end",
            "2020-06-30",
            "--run-id",
            "strategy-v2-smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dir = Path("runs/strategy-v2-smoke")
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "actions.csv").exists()
    assert (run_dir / "yearly_performance.csv").exists()


def test_stock_strategy_v3_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True)

    etf = make_synthetic_ohlcv(rows=420, seed=33)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    index = make_synthetic_ohlcv(rows=420, seed=34)
    index.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)
    panel = _panel(["AAA", "BBB", "CCC", "DDD", "EEE"], rows=420)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "effective_date": [pd.Timestamp("2020-01-01")] * 5,
            "is_member": [True, True, True, True, True],
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_snapshots.csv", index=False)
    (tmp_path / "config/stock_rotation_multifactor.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
                "selection:",
                "  method: sector_multifactor",
                "  require_long_term_trend: true",
                "  max_drawdown_252: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "stock-strategy-v3",
            "--config",
            "config/stock_rotation_multifactor.yaml",
            "--train-end",
            "2020-12-31",
            "--run-id",
            "strategy-v3-smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dir = Path("runs/strategy-v3-smoke")
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "actions.csv").exists()
    assert (run_dir / "market_regime.csv").exists()
    assert (run_dir / "factor_scores.csv").exists()


def test_stock_strategy_robustness_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True)

    etf = make_synthetic_ohlcv(rows=420, seed=43)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    index = make_synthetic_ohlcv(rows=420, seed=44)
    index.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)
    panel = _panel(["AAA", "BBB", "CCC", "DDD", "EEE"], rows=420)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "effective_date": [pd.Timestamp("2020-01-01")] * 5,
            "is_member": [True, True, True, True, True],
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_snapshots.csv", index=False)
    (tmp_path / "config/stock_rotation_multifactor.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
                "selection:",
                "  method: sector_multifactor",
                "  require_long_term_trend: true",
                "  max_drawdown_252: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "stock-strategy-robustness",
            "--config",
            "config/stock_rotation_multifactor.yaml",
            "--train-end",
            "2020-12-31",
            "--start-samples",
            "2",
            "--run-id",
            "strategy-robustness-smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dir = Path("runs/strategy-robustness-smoke")
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "strategy_comparison.csv").exists()
    assert (run_dir / "cost_stress.csv").exists()
    assert (run_dir / "robustness_report.md").exists()


def test_stock_strategy_v4_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/normalized").mkdir(parents=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True)

    etf = make_synthetic_ohlcv(rows=420, seed=53)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    index = make_synthetic_ohlcv(rows=420, seed=54)
    index.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)
    panel = _panel(["AAA", "BBB", "CCC", "DDD", "EEE"], rows=420)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "effective_date": [pd.Timestamp("2020-01-01")] * 5,
            "is_member": [True, True, True, True, True],
        }
    ).to_csv(tmp_path / "data/stock_rotation/membership_snapshots.csv", index=False)
    (tmp_path / "config/stock_rotation_multifactor.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "backtest:",
                "  initial_cash: 10000.0",
                "  monthly_contribution: 1000.0",
                "selection:",
                "  method: sector_multifactor",
                "  require_long_term_trend: true",
                "  max_drawdown_252: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "stock-strategy-v4",
            "--config",
            "config/stock_rotation_multifactor.yaml",
            "--train-end",
            "2020-12-31",
            "--run-id",
            "strategy-v4-smoke",
        ],
    )

    assert result.exit_code == 0, result.stdout
    run_dir = Path("runs/strategy-v4-smoke")
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "actions.csv").exists()
    assert (run_dir / "v4_feature_audit.csv").exists()
