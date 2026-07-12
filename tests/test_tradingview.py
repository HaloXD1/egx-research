from __future__ import annotations

import shutil
import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.advanced import compare_trades, export_parameters
from egx_research.tradingview.automation import cron_line
from egx_research.tradingview.browser import apply_account_request
from egx_research.tradingview.models import ExecutionModel, StrategyDefinition, SymbolMapping
from egx_research.tradingview.notifications import notify_run
from egx_research.tradingview.operations import audit_strategy, data_status, doctor, export_strategy_bundle
from egx_research.tradingview.parity import compare_events
from egx_research.tradingview.pine import render_template, validate_pine
from egx_research.tradingview.webhooks import accept_webhook, sign_payload
from tests.conftest import make_synthetic_ohlcv


def test_pine_validation_and_template_rendering(tmp_path: Path) -> None:
    pine = tmp_path / "sample.pine"
    pine.write_text("//@version=6\nindicator(\"Sample\", overlay=true)\n", encoding="utf-8")
    result = validate_pine(pine, expected_kind="indicator")
    assert result["valid"] is True
    template = tmp_path / "sample.tmpl"
    template.write_text("//@version={{ version }}\nindicator(\"{{ name }}\")\n", encoding="utf-8")
    assert render_template(template, {"version": 6, "name": "Generated"}) == "//@version=6\nindicator(\"Generated\")\n"


def test_parity_exact_and_one_bar_tolerance() -> None:
    dates = pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True)
    python_events = pd.DataFrame({"timestamp": dates, "action": ["buy", "sell"]})
    pine_events = pd.DataFrame({"timestamp": dates + pd.Timedelta(days=1), "action": ["buy", "sell"]})
    assert compare_events(python_events, python_events)["status"] == "pass"
    assert compare_events(python_events, pine_events, tolerance_bars=1)["status"] == "pass"
    assert compare_events(python_events, pine_events)["status"] == "fail"
    bars = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"], utc=True)
    weekend_shift = pd.DataFrame({"timestamp": [bars[2]], "action": ["buy"]})
    friday = pd.DataFrame({"timestamp": [bars[1]], "action": ["buy"]})
    result = compare_events(friday, weekend_shift, tolerance_bars=1, bar_dates=bars)
    assert result["status"] == "pass"
    assert result["tolerance_mode"] == "trading_bars"


def test_status_audit_and_notification(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    pd.DataFrame({"date": ["2026-07-11", "2026-07-12"], "open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [1, 2]}).to_csv(data_path, index=False)
    symbol = SymbolMapping(logical_symbol="TEST", local_path=str(data_path), timezone="UTC")
    status = data_status(symbol, max_stale_days=2)
    assert status["latest_date"] == "2026-07-12"
    pine = tmp_path / "strategy.pine"
    pine.write_text("//@version=6\nstrategy(\"Test\", process_orders_on_close = false)\n", encoding="utf-8")
    definition = StrategyDefinition(id="test", version="1.0.0", display_name="Test", pine_path=str(pine), script_kind="strategy", logical_symbol="TEST", execution=ExecutionModel(fill_model="next_open"))
    assert audit_strategy(definition)["valid"] is True
    run_dir = tmp_path / "runs" / "paper"
    run_dir.mkdir(parents=True)
    (run_dir / "current_signal.json").write_text('{"strategy_id":"test","logical_symbol":"TEST","target_exposure":0}', encoding="utf-8")
    path = notify_run("paper", str(tmp_path / "runs"))
    assert path.exists()


def test_tradingview_cli_local_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "tradingview/templates").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "tradingview/strategy.pine").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tradingview/strategy.pine").write_text(
        "//@version=6\nstrategy(\"Smoke\")\n", encoding="utf-8"
    )
    data = make_synthetic_ohlcv(rows=300)
    data.to_csv(tmp_path / "data/sample.csv", index=False)
    registry = {
        "strategies": [{
            "id": "trend_smoke",
            "version": "1.0.0",
            "display_name": "Trend Smoke",
            "pine_path": "tradingview/strategy.pine",
            "pine_version": 6,
            "script_kind": "strategy",
            "local_family": "trend",
            "logical_symbol": "SAMPLE",
            "params": {
                "fast_ma": 10,
                "slow_ma": 30,
                "ma_type": "SMA",
                "adx_len": 7,
                "adx_threshold": 15,
                "atr_stop": 2.0,
                "atr_trail": 1.0,
            },
        }]
    }
    (tmp_path / "tradingview/registry.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    symbols = {"symbols": [{"logical_symbol": "SAMPLE", "local_path": "data/sample.csv", "timezone": "UTC"}]}
    (tmp_path / "tradingview/symbols.yaml").write_text(yaml.safe_dump(symbols), encoding="utf-8")
    template = Path(__file__).parents[1] / "tradingview/templates/close_confirmed_strategy.pine.tmpl"
    shutil.copy2(template, tmp_path / "tradingview/templates/close_confirmed_strategy.pine.tmpl")
    config = {
        "registry_path": "tradingview/registry.yaml",
        "symbol_map_path": "tradingview/symbols.yaml",
        "template_dir": "tradingview/templates",
        "runs_dir": "runs",
        "backtest": {"initial_cash": 100000, "monthly_contribution": 1000, "fee_bps": 20, "slippage_bps": 5, "share_precision": 0},
    }
    (tmp_path / "config/tradingview.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    runner = CliRunner()
    scan = runner.invoke(app, ["tv", "scan", "--strategy", "trend_smoke", "--config", "config/tradingview.yaml", "--run-id", "scan-smoke", "--allow-stale"])
    assert scan.exit_code == 0, scan.stdout
    pine_validation = runner.invoke(app, ["tv", "strategy", "validate", "--strategy", "trend_smoke", "--config", "config/tradingview.yaml"])
    assert pine_validation.exit_code == 0, pine_validation.stdout
    backtest = runner.invoke(app, ["tv", "backtest", "--strategy", "trend_smoke", "--config", "config/tradingview.yaml", "--run-id", "backtest-smoke"])
    assert backtest.exit_code == 0, backtest.stdout
    report = runner.invoke(app, ["tv", "report", "--run-id", "backtest-smoke", "--config", "config/tradingview.yaml"])
    assert report.exit_code == 0, report.stdout
    diagnostic = runner.invoke(app, ["tv", "doctor", "--config", "config/tradingview.yaml", "--run-id", "doctor-smoke"])
    assert diagnostic.exit_code == 0, diagnostic.stdout
    exported = runner.invoke(app, ["tv", "strategy", "export", "--strategy", "trend_smoke", "--config", "config/tradingview.yaml", "--run-id", "export-smoke"])
    assert exported.exit_code == 0, exported.stdout
    assert (tmp_path / "runs/backtest-smoke/manifest.json").exists()
    assert (tmp_path / "runs/backtest-smoke/report.html").exists()
    assert (tmp_path / "runs/backtest-smoke/benchmarks.json").exists()
    assert (tmp_path / "runs/doctor-smoke/tradingview_doctor.json").exists()
    assert (tmp_path / "runs/export-smoke/tradingview_export/strategy.pine").exists()
    assert (tmp_path / "runs/export-smoke/tradingview_export/manifest.json").exists()


def test_doctor_strict_data_and_export_guard(tmp_path: Path) -> None:
    pine = tmp_path / "valid.pine"
    pine.write_text("//@version=6\nstrategy(\"Valid\", process_orders_on_close=false)\n", encoding="utf-8")
    definition = StrategyDefinition(
        id="valid",
        version="1.0.0",
        display_name="Valid",
        pine_path=str(pine),
        logical_symbol="MISSING",
        execution=ExecutionModel(fill_model="next_open"),
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"strategies": [{
        "id": definition.id,
        "version": definition.version,
        "display_name": definition.display_name,
        "pine_path": definition.pine_path,
        "logical_symbol": definition.logical_symbol,
    }]}), encoding="utf-8")
    symbols = tmp_path / "symbols.yaml"
    symbols.write_text(yaml.safe_dump({"symbols": [{
        "logical_symbol": "MISSING",
        "local_path": str(tmp_path / "missing.csv"),
    }]}), encoding="utf-8")
    config = TradingViewConfig(
        registry_path=str(registry),
        symbol_map_path=str(symbols),
        runs_dir=str(tmp_path / "runs"),
    )
    assert doctor(config)["valid"] is True
    strict = doctor(config, strict_data=True)
    assert strict["valid"] is False
    mapping = SymbolMapping(logical_symbol="MISSING", local_path=str(tmp_path / "missing.csv"))
    bundle = export_strategy_bundle(config, definition, mapping, "export")
    assert (bundle / "valid.pine").exists()


def test_trade_comparison_params_cron_and_webhook(tmp_path: Path) -> None:
    trades = pd.DataFrame({
        "entry_date": ["2026-01-01"], "exit_date": ["2026-01-03"],
        "entry_price": [100.0], "exit_price": [110.0], "shares": [2.0],
    })
    local_path = tmp_path / "local.csv"
    remote_path = tmp_path / "remote.csv"
    trades.to_csv(local_path, index=False)
    trades.rename(columns={"entry_date": "Entry Time", "exit_date": "Exit Time", "shares": "Qty"}).to_csv(remote_path, index=False)
    assert compare_trades(local_path, remote_path)["status"] == "pass"

    definition = StrategyDefinition(id="params", version="1.0.0", display_name="Params", pine_path="x.pine", params={"length": 20})
    output = export_parameters(definition, tmp_path / "params")
    assert json.loads((output / "pine_inputs.json").read_text())["length"] == 20
    assert "egx tv schedule run" in cron_line("config/tradingview.yaml", "params", "TEST", 18, 5)

    payload = b'{"id":"event-1","action":"buy"}'
    signature = sign_payload(payload, "secret")
    first = accept_webhook(payload, signature, "secret", tmp_path / "webhooks")
    second = accept_webhook(payload, signature, "secret", tmp_path / "webhooks")
    assert first["duplicate"] is False
    assert second["duplicate"] is True


def test_account_adapter_is_guarded(tmp_path: Path, monkeypatch) -> None:
    request = tmp_path / "request.json"
    request.write_text('{"live_order_execution":false,"ui_steps":[{"action":"wait","milliseconds":1}]}', encoding="utf-8")
    profile = tmp_path.parent / "outside-profile"
    monkeypatch.delenv("EGX_TV_ALLOW_ACCOUNT_MUTATIONS", raising=False)
    try:
        apply_account_request(request, profile, tmp_path / "artifacts", confirm=True)
    except ValueError as exc:
        assert "EGX_TV_ALLOW_ACCOUNT_MUTATIONS" in str(exc)
    else:
        raise AssertionError("account adapter should require the environment allow-switch")
