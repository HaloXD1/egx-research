from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.crypto_config import CryptoConfig, save_crypto_config
from egx_research.crypto_research import _optimize_family_prefix


def _features(rows: int = 240) -> pd.DataFrame:
    idx = np.arange(rows)
    close = 100 + idx * 0.15 + 5 * np.sin(idx / 16)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=rows, freq="D"),
            "open": close * 0.999,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000 + idx,
            "CapMVRVCur": 1.2 + 0.1 * np.sin(idx / 20),
            "AdrActCnt": 500000 + idx * 100,
            "TxCnt": 250000 + idx * 50,
            "HashRate": 1000000 + idx * 200,
            "FlowInExUSD": 1000000 + idx,
            "FlowOutExUSD": 1050000 + idx,
            "fear_greed_value": 45 + 20 * np.sin(idx / 18),
            "macro_nasdaq": 10000 + idx,
            "macro_us10y": 4.0,
            "macro_dollar": 100.0,
            "macro_fed_liquidity": 8000 + idx,
            "macro_vix": 20.0,
        }
    )


def test_crypto_research_and_report_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/crypto/features").mkdir(parents=True)

    config = CryptoConfig()
    config.data.raw_dir = "data/crypto/raw"
    config.data.normalized_dir = "data/crypto/normalized"
    config.data.features_dir = "data/crypto/features"
    config.search.families = ["crypto_dca_overlay"]
    config.search.trials_per_family = 2
    config.search.top_candidates_per_family = 1
    config.validation.holdout_ratio = 0.2
    config.validation.primary_train_bars = 100
    config.validation.primary_test_bars = 40
    config.validation.primary_step_bars = 40
    config.validation.fallback_train_bars = 80
    config.validation.fallback_test_bars = 40
    config.validation.fallback_step_bars = 20
    save_crypto_config("config/crypto_btc.yaml", config)
    _features().to_csv(Path(config.data.features_path), index=False)

    runner = CliRunner()
    research = runner.invoke(
        app,
        ["crypto-research", "--config", "config/crypto_btc.yaml", "--trials", "2", "--run-id", "crypto-smoke"],
    )
    assert research.exit_code == 0, research.stdout

    report = runner.invoke(app, ["crypto-report", "--run-id", "crypto-smoke"])
    assert report.exit_code == 0, report.stdout
    assert Path("runs/crypto-smoke/top10_overall.csv").exists()
    assert Path("runs/crypto-smoke/crypto_report.html").exists()

    paper = runner.invoke(
        app,
        [
            "crypto-paper-track",
            "--config",
            "config/crypto_btc.yaml",
            "--model-run-id",
            "crypto-smoke",
            "--start-date",
            "2021-05-01",
            "--run-id",
            "crypto-paper-smoke",
            "--max-data-stale-days",
            "9999",
        ],
    )
    assert paper.exit_code == 0, paper.stdout

    run_dir = Path("runs/crypto-paper-smoke")
    assert (run_dir / "paper_track_daily.csv").exists()
    assert (run_dir / "current_signal.json").exists()
    assert (run_dir / "current_signal_report.html").exists()
    summary = json.loads((run_dir / "paper_track_summary.json").read_text())
    assert summary["status"] == "tracked"
    assert 0.0 <= summary["latest_signal"]["target_allocation"] <= 1.0


def test_nested_inner_selection_is_unchanged_by_future_prices() -> None:
    config = CryptoConfig()
    config.validation.primary_train_bars = 80
    config.validation.primary_test_bars = 30
    config.validation.primary_step_bars = 30
    data = _features(rows=180)
    changed_future = data.copy()
    changed_future.loc[140:, ["open", "high", "low", "close"]] *= 10.0

    first, _, _ = _optimize_family_prefix(
        data,
        "crypto_dca_overlay",
        140,
        config,
        trials=3,
        seed=42,
    )
    second, _, _ = _optimize_family_prefix(
        changed_future,
        "crypto_dca_overlay",
        140,
        config,
        trials=3,
        seed=42,
    )

    assert first.best_trial.params == second.best_trial.params
    assert first.best_value == second.best_value


def test_crypto_bottom_score_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data/crypto/features").mkdir(parents=True)

    config = CryptoConfig()
    config.data.raw_dir = "data/crypto/raw"
    config.data.normalized_dir = "data/crypto/normalized"
    config.data.features_dir = "data/crypto/features"
    save_crypto_config("config/crypto_btc.yaml", config)
    _features(rows=420).to_csv(Path(config.data.features_path), index=False)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["crypto-bottom-score", "--config", "config/crypto_btc.yaml", "--run-id", "bottom-smoke"],
    )
    assert result.exit_code == 0, result.stdout

    run_dir = Path("runs/bottom-smoke")
    assert (run_dir / "bottom_score_summary.json").exists()
    assert (run_dir / "bottom_probability_grid.csv").exists()
    assert (run_dir / "bottom_feature_hitrates.csv").exists()
    assert (run_dir / "bottom_report.html").exists()
    summary = json.loads((run_dir / "bottom_score_summary.json").read_text())
    assert 0.0 <= summary["best_case"]["confidence"] <= 1.0
    assert summary["best_case"]["horizon_days"] in [30, 60, 90]
