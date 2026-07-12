from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from egx_research.tradingview.advanced import archive_run, clean_runs, compare_inputs, compare_runs, compare_trades, export_parameters, list_runs, promote_strategy
from egx_research.tradingview.automation import cron_line, run_batch_backtests, run_daily_pipeline
from egx_research.tradingview.browser import apply_account_request, chart_snapshot
from egx_research.tradingview.config import load_tradingview_config
from egx_research.tradingview.notifications import notify_run
from egx_research.tradingview.operations import audit_strategy, data_status, doctor as run_doctor, export_strategy_bundle, refresh_symbol
from egx_research.tradingview.paper import run_paper_track
from egx_research.tradingview.pine import validate_pine
from egx_research.tradingview.registry import find_strategy, load_registry, load_symbols, strategy_dict
from egx_research.tradingview.research import run_backtest, run_parity, run_scan, run_scan_many
from egx_research.tradingview.reporting import generate_report
from egx_research.tradingview.templates import render_named_template
from egx_research.tradingview.validation_workflow import run_validation
from egx_research.tradingview.webhooks import accept_webhook, serve_webhooks
from egx_research.utils import write_json


tv_app = typer.Typer(help="Local-first TradingView and Pine research tools.")
strategy_app = typer.Typer(help="Manage registered Pine strategies.")
chart_app = typer.Typer(help="Optional Playwright chart operations.")
alert_app = typer.Typer(help="Safe alert preparation and audit artifacts.")
run_app = typer.Typer(help="Inspect, compare, archive, and clean TradingView runs.")
schedule_app = typer.Typer(help="Reproducible scheduled TradingView research workflows.")
webhook_app = typer.Typer(help="Verify and persist signed TradingView webhooks.")
account_app = typer.Typer(help="Guarded authenticated TradingView browser operations.")
tv_app.add_typer(strategy_app, name="strategy")
tv_app.add_typer(chart_app, name="chart")
tv_app.add_typer(alert_app, name="alert")
tv_app.add_typer(run_app, name="run")
tv_app.add_typer(schedule_app, name="schedule")
tv_app.add_typer(webhook_app, name="webhook")
tv_app.add_typer(account_app, name="account")


def _load(config_path: Path, strategy_id: str, symbol_name: str | None = None):
    config = load_tradingview_config(config_path)
    definition = find_strategy(load_registry(config.registry_path), strategy_id)
    symbols = load_symbols(config.symbol_map_path)
    logical_symbol = symbol_name or definition.logical_symbol
    if logical_symbol not in symbols:
        raise typer.BadParameter(f"Unknown logical symbol: {logical_symbol}")
    return config, definition, symbols[logical_symbol]


@strategy_app.command("list")
def strategy_list(config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config")) -> None:
    config = load_tradingview_config(config_path)
    for definition in load_registry(config.registry_path):
        typer.echo(f"{definition.id}@{definition.version} {definition.script_kind} {definition.logical_symbol} - {definition.display_name}")


@strategy_app.command("show")
def strategy_show(
    strategy_id: str = typer.Argument(...),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    version: str | None = typer.Option(None, "--version"),
) -> None:
    config = load_tradingview_config(config_path)
    definition = find_strategy(load_registry(config.registry_path), strategy_id, version)
    typer.echo(json.dumps(strategy_dict(definition), indent=2, default=str))


@strategy_app.command("new")
def strategy_new(
    template: str = typer.Option(..., "--template"),
    strategy_id: str = typer.Option(..., "--strategy-id"),
    version: str = typer.Option("0.1.0", "--version"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
    title: str | None = typer.Option(None, "--title"),
    fast_length: int = typer.Option(20, "--fast-length"),
    slow_length: int = typer.Option(50, "--slow-length"),
) -> None:
    config = load_tradingview_config(config_path)
    actual_run_id = run_id or f"tv-strategy-{strategy_id}-{version}"
    run_dir = Path(config.runs_dir) / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pine = render_named_template(
        config.template_dir,
        template,
        {
            "strategy_id": strategy_id,
            "strategy_title": title or strategy_id,
            "version": version,
            "fast_length": fast_length,
            "slow_length": slow_length,
        },
    )
    pine_path = run_dir / "pine" / f"{strategy_id}.pine"
    pine_path.parent.mkdir(parents=True, exist_ok=True)
    pine_path.write_text(pine, encoding="utf-8")
    metadata = {"id": strategy_id, "version": version, "pine_path": str(pine_path), "source_mode": "template", "template": template, "status": "draft"}
    write_json(run_dir / "strategy_snapshot.json", metadata)
    typer.echo(f"strategy_draft={pine_path}")


@strategy_app.command("validate")
def strategy_validate(
    path: Path | None = typer.Option(None, "--path"),
    strategy: str | None = typer.Option(None, "--strategy"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    expected_version: int = typer.Option(6, "--pine-version"),
    expected_kind: str | None = typer.Option(None, "--script-kind"),
) -> None:
    if (path is None) == (strategy is None):
        raise typer.BadParameter("Provide exactly one of --path or --strategy")
    if strategy:
        config = load_tradingview_config(config_path)
        definition = find_strategy(load_registry(config.registry_path), strategy)
        path = Path(definition.pine_path)
        expected_version = definition.pine_version
        expected_kind = definition.script_kind
    assert path is not None
    result = validate_pine(path, expected_version=expected_version, expected_kind=expected_kind)
    typer.echo(json.dumps(result, indent=2, default=str))
    if not result["valid"]:
        raise typer.Exit(code=1)


@strategy_app.command("templates")
def strategy_templates(
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    config = load_tradingview_config(config_path)
    template_dir = Path(config.template_dir)
    for path in sorted(template_dir.glob("*.pine.tmpl")):
        typer.echo(path.name.removesuffix(".pine.tmpl"))


@strategy_app.command("export")
def strategy_export(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str | None = typer.Option(None, "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    bundle = export_strategy_bundle(config, definition, mapping, run_id)
    typer.echo(f"tv_strategy_export={bundle}")


@strategy_app.command("params-export")
def strategy_params_export(
    strategy: str = typer.Option(..., "--strategy"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    params_json: str | None = typer.Option(None, "--params-json", help="Optimizer/candidate parameter overrides."),
    params_file: Path | None = typer.Option(None, "--params-file", help="JSON file containing optimizer/candidate parameters."),
) -> None:
    config = load_tradingview_config(config_path)
    definition = find_strategy(load_registry(config.registry_path), strategy)
    overrides = json.loads(params_file.read_text(encoding="utf-8")) if params_file else (json.loads(params_json) if params_json else {})
    if not isinstance(overrides, dict):
        raise typer.BadParameter("Parameter overrides must be a JSON object")
    typer.echo(f"tv_params_export={export_parameters(definition, output_dir, overrides)}")


@strategy_app.command("inputs-compare")
def strategy_inputs_compare(
    strategy: str = typer.Option(..., "--strategy"),
    tradingview_inputs: Path = typer.Option(..., "--tradingview-inputs"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    config = load_tradingview_config(config_path)
    result = compare_inputs(find_strategy(load_registry(config.registry_path), strategy), tradingview_inputs)
    typer.echo(json.dumps(result, indent=2, default=str))
    if result["status"] != "pass":
        raise typer.Exit(code=1)


@strategy_app.command("promote")
def strategy_promote(
    strategy: str = typer.Option(..., "--strategy"),
    validation_run_id: str = typer.Option(..., "--validation-run-id"),
    status: str = typer.Option("validated", "--status"),
    confirm: bool = typer.Option(False, "--confirm"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    if not confirm:
        raise typer.BadParameter("Registry promotion requires --confirm")
    config = load_tradingview_config(config_path)
    path = promote_strategy(config.registry_path, strategy, Path(config.runs_dir) / validation_run_id, status)
    typer.echo(f"tv_strategy_registry={path}")


@strategy_app.command("compile-record")
def strategy_compile_record(
    strategy: str = typer.Option(..., "--strategy"),
    status: str = typer.Option(..., "--status", help="TradingView compile result: pass or fail."),
    run_id: str = typer.Option(..., "--run-id"),
    evidence: str | None = typer.Option(None, "--evidence"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    if status not in {"pass", "fail"}:
        raise typer.BadParameter("--status must be pass or fail")
    config = load_tradingview_config(config_path)
    definition = find_strategy(load_registry(config.registry_path), strategy)
    target = Path(config.runs_dir) / run_id / "pine_compile.json"
    write_json(target, {"strategy_id": definition.id, "strategy_version": definition.version, "status": status, "evidence": evidence, "source": "tradingview_manual"})
    typer.echo(f"tv_compile_record={target}")


@tv_app.command("scan")
def scan(
    strategy: str = typer.Option(..., "--strategy"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    symbol: str | None = typer.Option(None, "--symbol"),
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated logical symbols."),
    all_symbols: bool = typer.Option(False, "--all-symbols"),
    refresh: bool = typer.Option(False, "--refresh", help="Refresh supported local data before scanning."),
    allow_stale: bool = typer.Option(False, "--allow-stale"),
    run_id: str | None = typer.Option(None, "--run-id"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config = load_tradingview_config(config_path)
    definition = find_strategy(load_registry(config.registry_path), strategy)
    mappings = load_symbols(config.symbol_map_path)
    selected_names = [part.strip() for part in symbols.split(",") if part.strip()] if symbols else []
    if all_symbols:
        selected_names = [name for name, mapping in mappings.items() if mapping.asset_class == definition.asset_class]
    if not selected_names:
        selected_names = [symbol or definition.logical_symbol]
    selected = []
    for name in selected_names:
        if name not in mappings:
            raise typer.BadParameter(f"Unknown logical symbol: {name}")
        selected.append(mappings[name])
    if refresh:
        refreshed = set()
        for mapping in selected:
            if mapping.logical_symbol not in refreshed:
                refresh_symbol(config, mapping)
                refreshed.add(mapping.logical_symbol)
    statuses = [data_status(mapping, config.default_max_stale_days) for mapping in selected]
    stale = [item for item in statuses if item.get("status") in {"stale", "missing", "empty"}]
    if stale and not allow_stale:
        raise typer.BadParameter("Data is stale or missing; use --refresh or --allow-stale")
    actual_run_id = run_id or f"tv-scan-{definition.id}"
    run_dir = run_scan(config, definition, selected[0], actual_run_id, params_json) if len(selected) == 1 else run_scan_many(config, definition, selected, actual_run_id, params_json)
    typer.echo(f"tv_scan={run_dir}")


@tv_app.command("sync")
def sync(
    symbol: str = typer.Option("BTCUSDT", "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    config = load_tradingview_config(config_path)
    mappings = load_symbols(config.symbol_map_path)
    if symbol not in mappings:
        raise typer.BadParameter(f"Unknown logical symbol: {symbol}")
    path = refresh_symbol(config, mappings[symbol])
    typer.echo(f"tv_sync={path}")


@tv_app.command("status")
def status(
    symbol: str | None = typer.Option(None, "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    max_stale_days: int | None = typer.Option(None, "--max-stale-days"),
) -> None:
    config = load_tradingview_config(config_path)
    mappings = load_symbols(config.symbol_map_path)
    names = [symbol] if symbol else list(mappings)
    for name in names:
        if name not in mappings:
            raise typer.BadParameter(f"Unknown logical symbol: {name}")
        typer.echo(json.dumps(data_status(mappings[name], max_stale_days or config.default_max_stale_days), sort_keys=True))


@tv_app.command("backtest")
def backtest(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str | None = typer.Option(None, "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    run_dir = run_backtest(config, definition, mapping, run_id, params_json)
    typer.echo(f"tv_backtest={run_dir}")


@tv_app.command("batch-backtest")
def batch_backtest(
    strategies: str = typer.Option(..., "--strategies", help="Comma-separated strategy ids."),
    symbols: str | None = typer.Option(None, "--symbols", help="Optional comma-separated symbol ids matched by asset class."),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config = load_tradingview_config(config_path)
    definitions = load_registry(config.registry_path)
    mappings = load_symbols(config.symbol_map_path)
    requested_symbols = [item.strip() for item in symbols.split(",") if item.strip()] if symbols else []
    jobs = []
    for strategy_id in [item.strip() for item in strategies.split(",") if item.strip()]:
        definition = find_strategy(definitions, strategy_id)
        names = requested_symbols or [definition.logical_symbol]
        for name in names:
            if name not in mappings or mappings[name].asset_class != definition.asset_class:
                continue
            jobs.append((definition, mappings[name]))
    if not jobs:
        raise typer.BadParameter("No compatible strategy/symbol jobs selected")
    typer.echo(f"tv_batch_backtest={run_batch_backtests(config, jobs, run_id, params_json)}")


@tv_app.command("batch-validate")
def batch_validate(
    strategies: str = typer.Option(..., "--strategies"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
    cost_stress: str = typer.Option("1,2,3", "--cost-stress"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config = load_tradingview_config(config_path)
    definitions = load_registry(config.registry_path)
    mappings = load_symbols(config.symbol_map_path)
    multipliers = [float(item.strip()) for item in cost_stress.split(",") if item.strip()]
    rows = []
    for strategy_id in [item.strip() for item in strategies.split(",") if item.strip()]:
        definition = find_strategy(definitions, strategy_id)
        child_id = f"{run_id}-{definition.id}".lower().replace("_", "-")
        child = run_validation(config, definition, mappings[definition.logical_symbol], child_id, params_json, multipliers)
        rows.append({"strategy_id": definition.id, "run_id": child_id, "path": str(child)})
    root = Path(config.runs_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "batch_validation.json", {"status": "complete", "jobs": rows})
    typer.echo(f"tv_batch_validation={root}")


@tv_app.command("parity")
def parity(
    strategy: str = typer.Option(..., "--strategy"),
    pine_events: Path = typer.Option(..., "--pine-events"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    symbol: str | None = typer.Option(None, "--symbol"),
    run_id: str = typer.Option(..., "--run-id"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    run_dir = run_parity(config, definition, mapping, pine_events, run_id, params_json)
    summary = json.loads((run_dir / "parity_summary.json").read_text(encoding="utf-8"))
    typer.echo(f"tv_parity={run_dir} status={summary['status']} matched={summary['matched_events']}")
    if summary["status"] != "pass":
        raise typer.Exit(code=1)


@tv_app.command("trade-parity")
def trade_parity(
    local_trades: Path = typer.Option(..., "--local-trades"),
    tradingview_trades: Path = typer.Option(..., "--tradingview-trades"),
    price_tolerance_bps: float = typer.Option(10.0, "--price-tolerance-bps"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    result = compare_trades(local_trades, tradingview_trades, price_tolerance_bps)
    if output:
        write_json(output, result)
    typer.echo(json.dumps(result, indent=2, default=str))
    if result["status"] != "pass":
        raise typer.Exit(code=1)


@tv_app.command("report")
def report(
    run_id: str = typer.Option(..., "--run-id"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    config = load_tradingview_config(config_path)
    report_path = generate_report(run_id, config.runs_dir)
    typer.echo(f"tv_report={report_path}")


@tv_app.command("validate")
def validate(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str | None = typer.Option(None, "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
    cost_stress: str = typer.Option("1,2,3", "--cost-stress", help="Comma-separated fee/slippage multipliers."),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    try:
        multipliers = [float(part.strip()) for part in cost_stress.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter("--cost-stress must be comma-separated numbers") from exc
    run_dir = run_validation(config, definition, mapping, run_id, params_json, multipliers)
    summary = json.loads((run_dir / "validation_summary.json").read_text(encoding="utf-8"))
    typer.echo(f"tv_validation={run_dir} status={summary['status']} holdout_excess={summary['holdout_excess_return_vs_dca']:.4f}")


@tv_app.command("paper-track")
def paper_track(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str | None = typer.Option(None, "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
    start_date: str | None = typer.Option(None, "--start-date"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    run_dir = run_paper_track(config, definition, mapping, run_id, start_date, params_json)
    typer.echo(f"tv_paper_track={run_dir}")


@tv_app.command("audit")
def audit(
    strategy: str = typer.Option(..., "--strategy"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    config = load_tradingview_config(config_path)
    definition = find_strategy(load_registry(config.registry_path), strategy)
    result = audit_strategy(definition)
    if run_id:
        run_dir = Path(config.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "execution_audit.json", result)
    typer.echo(json.dumps(result, indent=2, default=str))
    if not result["valid"]:
        raise typer.Exit(code=1)


@tv_app.command("doctor")
def doctor(
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    strict_data: bool = typer.Option(False, "--strict-data", help="Treat stale or missing local data as an error."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optionally save the diagnostic artifact under runs/."),
) -> None:
    config = load_tradingview_config(config_path)
    result = run_doctor(config, strict_data)
    if run_id:
        run_dir = Path(config.runs_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "tradingview_doctor.json", result)
    typer.echo(json.dumps(result, indent=2, default=str))
    if not result["valid"]:
        raise typer.Exit(code=1)


@tv_app.command("notify")
def notify(
    run_id: str = typer.Option(..., "--run-id"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    send: bool = typer.Option(False, "--send", help="Actually POST to the configured webhook."),
    webhook_url: str | None = typer.Option(None, "--webhook-url", help="Prefer EGX_TV_WEBHOOK_URL for secrets."),
    channel: str = typer.Option("webhook", "--channel", help="webhook, slack, discord, telegram, or email."),
) -> None:
    config = load_tradingview_config(config_path)
    path = notify_run(run_id, config.runs_dir, send, webhook_url, channel)
    typer.echo(f"tv_notification={path}")


@chart_app.command("open")
def chart_open(
    url: str = typer.Option(..., "--url"),
    screenshot: Path = typer.Option(..., "--screenshot"),
    profile_dir: Path | None = typer.Option(None, "--profile-dir", help="User-managed profile outside the repository."),
    headless: bool = typer.Option(False, "--headless"),
    wait_seconds: int = typer.Option(5, "--wait-seconds"),
) -> None:
    path = chart_snapshot(url, str(screenshot), str(profile_dir) if profile_dir else None, headless, wait_seconds)
    typer.echo(f"tv_chart_screenshot={path}")


@alert_app.command("prepare")
def alert_prepare(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str | None = typer.Option(None, "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    run_id: str = typer.Option(..., "--run-id"),
    condition: str = typer.Option("Any alert() function call", "--condition"),
    frequency: str = typer.Option("Once Per Bar Close", "--frequency"),
    operation: str = typer.Option("create", "--operation", help="list, create, update, pause, resume, or delete."),
    ui_steps_json: str | None = typer.Option(None, "--ui-steps-json", help="Explicit Playwright steps for authenticated apply."),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    run_dir = Path(config.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "strategy_id": definition.id,
        "strategy_version": definition.version,
        "logical_symbol": mapping.logical_symbol,
        "tradingview_symbol": mapping.tradingview_symbol,
        "condition": condition,
        "frequency": frequency,
        "operation": operation,
        "message": {"strategy": definition.id, "symbol": mapping.logical_symbol, "target_exposure": "{{plot_0}}"},
        "dry_run": True,
        "live_order_execution": False,
        "ui_steps": json.loads(ui_steps_json) if ui_steps_json else [],
    }
    write_json(run_dir / "alert_request.json", request)
    typer.echo(f"tv_alert_request={run_dir / 'alert_request.json'}")


@alert_app.command("create")
def alert_create(
    request_file: Path = typer.Option(..., "--request-file"),
    confirm: bool = typer.Option(False, "--confirm"),
    profile_dir: Path | None = typer.Option(None, "--profile-dir"),
    artifact_dir: Path = typer.Option(Path("runs/tradingview-account"), "--artifact-dir"),
    headless: bool = typer.Option(False, "--headless"),
) -> None:
    if not confirm:
        typer.echo(f"dry_run_alert_request={request_file}")
        return
    if profile_dir is None:
        raise typer.BadParameter("Authenticated apply requires --profile-dir")
    result = apply_account_request(request_file, profile_dir, artifact_dir, confirm=True, headless=headless)
    typer.echo(json.dumps(result, indent=2))


@account_app.command("apply")
def account_apply(
    request_file: Path = typer.Option(..., "--request-file"),
    profile_dir: Path = typer.Option(..., "--profile-dir"),
    artifact_dir: Path = typer.Option(..., "--artifact-dir"),
    confirm: bool = typer.Option(False, "--confirm"),
    headless: bool = typer.Option(False, "--headless"),
) -> None:
    typer.echo(json.dumps(apply_account_request(request_file, profile_dir, artifact_dir, confirm, headless), indent=2))


@account_app.command("prepare")
def account_prepare(
    operation: str = typer.Option(..., "--operation", help="list/create/update/pause/resume/delete/upload_script/update_script/export_strategy_tester"),
    ui_steps_json: str = typer.Option(..., "--ui-steps-json"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    allowed = {"list", "create", "update", "pause", "resume", "delete", "upload_script", "update_script", "export_strategy_tester"}
    if operation not in allowed:
        raise typer.BadParameter(f"Unsupported operation: {operation}")
    steps = json.loads(ui_steps_json)
    if not isinstance(steps, list):
        raise typer.BadParameter("--ui-steps-json must contain a JSON array")
    write_json(output, {"operation": operation, "live_order_execution": False, "ui_steps": steps})
    typer.echo(f"tv_account_request={output}")


@run_app.command("list")
def run_list(
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    typer.echo(json.dumps(list_runs(load_tradingview_config(config_path)), indent=2))


@run_app.command("show")
def run_show(
    run_id: str = typer.Argument(...),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    rows = compare_runs(load_tradingview_config(config_path), [run_id])
    typer.echo(json.dumps(rows[0], indent=2))


@run_app.command("compare")
def run_compare(
    run_ids: str = typer.Option(..., "--run-ids"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    typer.echo(json.dumps(compare_runs(load_tradingview_config(config_path), [item.strip() for item in run_ids.split(",") if item.strip()]), indent=2))


@run_app.command("archive")
def run_archive(
    run_id: str = typer.Argument(...),
    archive_dir: Path = typer.Option(Path("archives/tradingview"), "--archive-dir"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    typer.echo(f"tv_run_archive={archive_run(load_tradingview_config(config_path), run_id, archive_dir)}")


@run_app.command("clean")
def run_clean(
    older_than_days: int = typer.Option(30, "--older-than-days"),
    confirm: bool = typer.Option(False, "--confirm"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
) -> None:
    typer.echo(json.dumps({"deleted": confirm, "runs": clean_runs(load_tradingview_config(config_path), older_than_days, confirm)}, indent=2))


@schedule_app.command("run")
def schedule_run(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str | None = typer.Option(None, "--symbol"),
    run_id: str = typer.Option(..., "--run-id"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    refresh: bool = typer.Option(False, "--refresh"),
    send_notification: bool = typer.Option(False, "--notify"),
    channel: str = typer.Option("webhook", "--channel"),
    params_json: str | None = typer.Option(None, "--params-json"),
) -> None:
    config, definition, mapping = _load(config_path, strategy, symbol)
    typer.echo(f"tv_pipeline={run_daily_pipeline(config, definition, mapping, run_id, refresh, send_notification, channel, params_json)}")


@schedule_app.command("cron")
def schedule_cron(
    strategy: str = typer.Option(..., "--strategy"),
    symbol: str = typer.Option(..., "--symbol"),
    config_path: Path = typer.Option(Path("config/tradingview.yaml"), "--config"),
    hour: int = typer.Option(18, "--hour"),
    minute: int = typer.Option(0, "--minute"),
) -> None:
    typer.echo(cron_line(str(config_path), strategy, symbol, hour, minute))


@webhook_app.command("verify")
def webhook_verify(
    payload_file: Path = typer.Option(..., "--payload-file"),
    signature: str = typer.Option(..., "--signature"),
    store_dir: Path = typer.Option(Path("data/tradingview/webhooks"), "--store-dir"),
    secret_env: str = typer.Option("EGX_TV_WEBHOOK_SECRET", "--secret-env"),
) -> None:
    secret = os.getenv(secret_env)
    if not secret:
        raise typer.BadParameter(f"Missing environment variable: {secret_env}")
    typer.echo(json.dumps(accept_webhook(payload_file.read_bytes(), signature, secret, store_dir), indent=2))


@webhook_app.command("serve")
def webhook_serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    store_dir: Path = typer.Option(Path("data/tradingview/webhooks"), "--store-dir"),
    secret_env: str = typer.Option("EGX_TV_WEBHOOK_SECRET", "--secret-env"),
) -> None:
    secret = os.getenv(secret_env)
    if not secret:
        raise typer.BadParameter(f"Missing environment variable: {secret_env}")
    typer.echo(f"tv_webhook_listen=http://{host}:{port}")
    serve_webhooks(host, port, secret, store_dir)
