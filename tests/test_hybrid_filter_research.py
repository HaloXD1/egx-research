from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.config import AppConfig, save_config
from tests.conftest import make_synthetic_ohlcv


def _stock_panel(rows: int = 420) -> pd.DataFrame:
    symbols = ["ALPHA", "BETA", "GAMMA"]
    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols):
        frame = make_synthetic_ohlcv(rows=rows, seed=40 + idx).copy()
        scale = 1.0 + idx * 0.08
        for column in ["open", "high", "low", "close"]:
            frame[column] = frame[column] * scale
        frame["symbol"] = symbol
        frame["holding_name"] = symbol.title()
        frame["weight"] = 1.0 / len(symbols)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_hybrid_filter_research_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True, exist_ok=True)

    etf = make_synthetic_ohlcv(rows=420, seed=13)
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)

    panel = _stock_panel()
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)

    config = AppConfig()
    config.validation.holdout_ratio = 0.2
    config.validation.primary_train_bars = 120
    config.validation.primary_test_bars = 60
    config.validation.primary_step_bars = 60
    config.validation.fallback_train_bars = 90
    config.validation.fallback_test_bars = 45
    config.validation.fallback_step_bars = 45
    save_config(tmp_path / "config/default.yaml", config)

    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "storage:",
                "  root_dir: data/stock_rotation",
                "validation:",
                "  min_history_bars: 180",
            ]
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hybrid-filter-research",
            "--config",
            "config/default.yaml",
            "--stock-config",
            "config/stock_rotation.yaml",
            "--run-id",
            "hybrid-smoke",
            "--min-stock-bars",
            "180",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert Path("runs/hybrid-smoke/leaderboard.csv").exists()
    assert Path("runs/hybrid-smoke/summary.json").exists()
    assert Path("runs/hybrid-smoke/stock_variant_summary.csv").exists()
