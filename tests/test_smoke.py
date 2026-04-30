from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.config import AppConfig, save_config
from tests.conftest import make_synthetic_ohlcv


def test_end_to_end_cli_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)

    data = make_synthetic_ohlcv(rows=1000)
    source = tmp_path / "sample.csv"
    data.to_csv(source, index=False)

    config = AppConfig()
    save_config(tmp_path / "config/default.yaml", config)

    runner = CliRunner()
    ingest = runner.invoke(app, ["ingest", "--input", str(source), "--config", "config/default.yaml"])
    assert ingest.exit_code == 0, ingest.stdout

    optimize = runner.invoke(
        app,
        ["optimize", "--config", "config/default.yaml", "--trials", "10", "--run-id", "smoke"],
    )
    assert optimize.exit_code == 0, optimize.stdout

    report = runner.invoke(app, ["report", "--run-id", "smoke"])
    assert report.exit_code == 0, report.stdout

    assert Path("runs/smoke/top10_overall.csv").exists()
    assert Path("runs/smoke/report.html").exists()
