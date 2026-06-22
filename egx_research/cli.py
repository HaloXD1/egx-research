from __future__ import annotations

from pathlib import Path

import typer

from egx_research.annual_top10 import (
    generate_annual_top10_report,
    generate_annual_top10_stock_report,
    run_annual_top10_backtest,
    run_annual_top10_stock_backtest,
)
from egx_research.blackcat_research import run_blackcat_research
from egx_research.config import load_config
from egx_research.core_satellite import run_core_satellite_backtest
from egx_research.crypto_config import load_crypto_config
from egx_research.crypto_bottom import run_crypto_bottom_score
from egx_research.crypto_data import sync_crypto_data
from egx_research.crypto_paper_tracking import paper_track_crypto_strategy
from egx_research.crypto_reporting import generate_crypto_report
from egx_research.crypto_research import run_crypto_research
from egx_research.data import ingest_source
from egx_research.hybrid_filter_research import run_hybrid_filter_research
from egx_research.optimization import optimize_run
from egx_research.paper_tracking import paper_track_run, paper_track_stock_strategy
from egx_research.relative_ic import run_relative_ic_backtest
from egx_research.relative_signal import run_relative_signal
from egx_research.reporting import generate_report
from egx_research.soft_filter_research import run_soft_filter_research
from egx_research.stock_factor_research import run_stock_factor_research
from egx_research.stock_rotation import run_stock_rotation_backtest
from egx_research.stock_rotation import run_stock_selection_backtest
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.stock_rotation_data import sync_stock_rotation_data
from egx_research.stock_rotation_data import sync_stock_fundamentals
from egx_research.stock_rotation_data import sync_institutional_stage2_data
from egx_research.stock_rotation_reporting import (
    generate_stock_rotation_report,
    generate_stock_selection_report,
)
from egx_research.stock_strategy_research import run_stock_strategy_research
from egx_research.stock_strategy_robustness import run_stock_strategy_robustness
from egx_research.stock_strategy_validation import (
    run_rebound_max5_v2,
    run_rebound_max5_v3,
    run_rebound_max5_v4,
    run_rebound_max5_v5,
    run_stock_strategy_validation,
)
from egx_research.stock_momentum_pyramid import (
    run_stock_momentum_pyramid_backtest,
)


app = typer.Typer(help="EGX30 ETF local research toolkit.")


@app.command()
def ingest(
    input: str = typer.Option(
        ..., "--input", help="Local CSV path or HTTP(S) CSV URL."
    ),
    config_path: Path = typer.Option(
        Path("config/default.yaml"), "--config", help="Config path."
    ),
    symbol: str | None = typer.Option(None, "--symbol", help="Override output symbol."),
) -> None:
    config = load_config(config_path)
    result = ingest_source(input, config=config, symbol=symbol)
    typer.echo(
        f"ingested symbol={result['symbol']} rows={result['rows']} normalized={result['normalized_path']}"
    )


@app.command()
def optimize(
    config_path: Path = typer.Option(
        Path("config/default.yaml"), "--config", help="Config path."
    ),
    family: str | None = typer.Option(None, "--family", help="Single family override."),
    trials: int | None = typer.Option(None, "--trials", help="Trial override."),
    objective: str | None = typer.Option(
        None, "--objective", help="Objective override."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    config = load_config(config_path)
    actual_run_id = optimize_run(
        config=config,
        config_path=config_path,
        trials_override=trials,
        family_override=family,
        run_id=run_id,
        objective_mode_override=objective,
    )
    typer.echo(f"optimized run_id={actual_run_id}")


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", help="Run id under runs/."),
    config_path: Path | None = typer.Option(
        None, "--config", help="Optional config override."
    ),
) -> None:
    report_path = generate_report(run_id=run_id, config_path=config_path)
    typer.echo(f"report={report_path}")


@app.command("crypto-sync")
def crypto_sync(
    config_path: Path = typer.Option(
        Path("config/crypto_btc.yaml"), "--config", help="Crypto config path."
    ),
) -> None:
    config = load_crypto_config(config_path)
    features_path = sync_crypto_data(config)
    typer.echo(f"crypto_sync={features_path}")


@app.command("crypto-research")
def crypto_research(
    config_path: Path = typer.Option(
        Path("config/crypto_btc.yaml"), "--config", help="Crypto config path."
    ),
    family: str | None = typer.Option(None, "--family", help="Single family override."),
    trials: int | None = typer.Option(None, "--trials", help="Trial override."),
    objective: str | None = typer.Option(None, "--objective", help="Objective override."),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    config = load_crypto_config(config_path)
    actual_run_id = run_crypto_research(
        config=config,
        config_path=config_path,
        trials_override=trials,
        family_override=family,
        run_id=run_id,
        objective_mode_override=objective,
    )
    typer.echo(f"crypto_research_run={actual_run_id}")


@app.command("crypto-report")
def crypto_report(
    run_id: str = typer.Option(..., "--run-id", help="Run id under runs/."),
) -> None:
    report_path = generate_crypto_report(run_id)
    typer.echo(f"crypto_report={report_path}")


@app.command("crypto-bottom-score")
def crypto_bottom_score(
    config_path: Path = typer.Option(
        Path("config/crypto_btc.yaml"), "--config", help="Crypto config path."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Output run id override."),
    as_of_date: str | None = typer.Option(
        None, "--as-of-date", help="Score using data on or before YYYY-MM-DD."
    ),
) -> None:
    config = load_crypto_config(config_path)
    result = run_crypto_bottom_score(
        config=config,
        config_path=config_path,
        run_id=run_id,
        as_of_date=as_of_date,
    )
    best = result.summary["best_case"]
    typer.echo(
        "crypto_bottom_score="
        f"{result.run_dir} confidence={best['confidence_pct']:.1f}% "
        f"horizon={best['horizon_days']}d tolerance={best['tolerance_pct']:.0f}% "
        f"report={result.report_path}"
    )


@app.command("crypto-paper-track")
def crypto_paper_track(
    model_run_id: str = typer.Option(
        ..., "--model-run-id", help="Crypto research run id to source best model from."
    ),
    start_date: str = typer.Option(
        ..., "--start-date", help="Paper tracking start date YYYY-MM-DD."
    ),
    config_path: Path = typer.Option(
        Path("config/crypto_btc.yaml"), "--config", help="Crypto config path."
    ),
    out_run_id: str | None = typer.Option(
        None, "--run-id", help="Output run id override."
    ),
    max_data_stale_days: int = typer.Option(
        2,
        "--max-data-stale-days",
        help="Block action instructions if BTC data is older than this.",
    ),
) -> None:
    run_dir = paper_track_crypto_strategy(
        model_run_id=model_run_id,
        start_date=start_date,
        config_path=config_path,
        out_run_id=out_run_id,
        max_data_stale_days=max_data_stale_days,
    )
    typer.echo(f"crypto_paper_track={run_dir}")


@app.command("paper-track")
def paper_track(
    model_run_id: str | None = typer.Option(
        None, "--model-run-id", help="Run id to source best ETF model from."
    ),
    strategy: str | None = typer.Option(
        None, "--strategy", help="Stock strategy to paper-track, e.g. rebound_max5_v5."
    ),
    start_date: str = typer.Option(
        ..., "--start-date", help="Paper tracking start date YYYY-MM-DD."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help="Optional config override."
    ),
    out_run_id: str | None = typer.Option(
        None, "--run-id", help="Output run id override."
    ),
    stock_config_path: Path = typer.Option(
        Path("config/stock_rotation_multifactor.yaml"),
        "--stock-config",
        help="Stock strategy config path.",
    ),
    max_data_stale_days: int = typer.Option(
        7,
        "--max-data-stale-days",
        help="Block stock paper trade instructions if data is older than this.",
    ),
) -> None:
    if strategy:
        run_dir = paper_track_stock_strategy(
            strategy=strategy,
            start_date=start_date,
            config_path=stock_config_path,
            out_run_id=out_run_id,
            max_data_stale_days=max_data_stale_days,
        )
    else:
        if model_run_id is None:
            raise typer.BadParameter("--model-run-id is required unless --strategy is set.")
        run_dir = paper_track_run(
            model_run_id=model_run_id,
            start_date=start_date,
            config_path=config_path,
            out_run_id=out_run_id,
        )
    typer.echo(f"paper_track={run_dir}")


@app.command("stock-sync")
def stock_sync(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
) -> None:
    config = load_stock_rotation_config(config_path)
    root_dir = sync_stock_rotation_data(config)
    typer.echo(f"stock_sync={root_dir}")


@app.command("stock-sync-stage2")
def stock_sync_stage2(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    max_news_items: int = typer.Option(
        50, "--max-news-items", help="Max Mubasher news items per symbol."
    ),
) -> None:
    config = load_stock_rotation_config(config_path)
    root_dir = sync_institutional_stage2_data(
        config, max_news_items_per_symbol=max_news_items
    )
    typer.echo(f"stock_sync_stage2={root_dir}")


@app.command("stock-sync-fundamentals")
def stock_sync_fundamentals(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
) -> None:
    config = load_stock_rotation_config(config_path)
    root_dir = sync_stock_fundamentals(config)
    typer.echo(f"stock_sync_fundamentals={root_dir}")


@app.command("stock-rotate-backtest")
def stock_rotate_backtest(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_stock_rotation_backtest(config_path=config_path, run_id=run_id)
    typer.echo(f"stock_rotate_run={run.run_dir}")


@app.command("stock-rotate-report")
def stock_rotate_report(
    run_id: str = typer.Option(..., "--run-id", help="Stock rotation run id."),
) -> None:
    report_path = generate_stock_rotation_report(run_id)
    typer.echo(f"stock_rotate_report={report_path}")


@app.command("stock-select-backtest")
def stock_select_backtest(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    pullback_run_id: str | None = typer.Option(
        None, "--pullback-run-id", help="Use pullback params from an existing run."
    ),
    rebalance_mode: str = typer.Option(
        "monthly", "--rebalance-mode", help="Rebalance mode: monthly or annual."
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Optional start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Optional end date YYYY-MM-DD."
    ),
) -> None:
    run = run_stock_selection_backtest(
        config_path=config_path,
        run_id=run_id,
        pullback_run_id=pullback_run_id,
        rebalance_mode=rebalance_mode,
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo(f"stock_select_run={run.run_dir}")


@app.command("stock-select-report")
def stock_select_report(
    run_id: str = typer.Option(..., "--run-id", help="Stock selection run id."),
) -> None:
    report_path = generate_stock_selection_report(run_id)
    typer.echo(f"stock_select_report={report_path}")


@app.command("stock-factor-research")
def stock_factor_research(
    config_path: Path = typer.Option(
        Path("config/stock_rotation_multifactor.yaml"),
        "--config",
        help="Stock multifactor config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_stock_factor_research(config_path=config_path, run_id=run_id)
    typer.echo(f"stock_factor_research_run={run.run_dir}")


@app.command("relative-signal")
def relative_signal(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    symbol: str | None = typer.Option(
        None, "--symbol", help="Single stock symbol. Defaults to AMOC."
    ),
    all_symbols: bool = typer.Option(
        False, "--all", help="Run every normalized stock symbol."
    ),
    benchmark: str = typer.Option(
        "etf", "--benchmark", help="Benchmark: etf, index, or CSV path."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_relative_signal(
        config_path=config_path,
        symbol=symbol,
        all_symbols=all_symbols,
        benchmark=benchmark,
        run_id=run_id,
    )
    typer.echo(f"relative_signal_run={run.run_dir}")


@app.command("relative-ic-backtest")
def relative_ic_backtest(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_relative_ic_backtest(config_path=config_path, run_id=run_id)
    typer.echo(f"relative_ic_run={run.run_dir}")


@app.command("annual-top10-backtest")
def annual_top10_backtest(
    stock_config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--stock-config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    top_n: int | None = typer.Option(
        None, "--top-n", help="Override number of stocks."
    ),
    lookback_bars: int = typer.Option(
        252, "--lookback-bars", help="Ranking lookback bars."
    ),
    pullback_run_id: str | None = typer.Option(
        None, "--pullback-run-id", help="Use pullback params from an existing run."
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Optional start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Optional end date YYYY-MM-DD."
    ),
) -> None:
    run = run_annual_top10_backtest(
        stock_config_path=stock_config_path,
        run_id=run_id,
        top_n=top_n,
        lookback_bars=lookback_bars,
        pullback_run_id=pullback_run_id,
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo(f"annual_top10_run={run.run_dir}")


@app.command("annual-top10-report")
def annual_top10_report(
    run_id: str = typer.Option(..., "--run-id", help="Annual top10 run id."),
) -> None:
    report_path = generate_annual_top10_report(run_id)
    typer.echo(f"annual_top10_report={report_path}")


@app.command("stock-core-satellite-backtest")
def stock_core_satellite_backtest(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Optional start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Optional end date YYYY-MM-DD."
    ),
    top_n: int | None = typer.Option(
        None, "--top-n", help="Override number of stocks."
    ),
    core_weight: float = typer.Option(
        0.70, "--core-weight", help="ETF core weight."
    ),
    rebalance_mode: str = typer.Option(
        "annual", "--rebalance-mode", help="annual or semiannual."
    ),
    drawdown_guard: float = typer.Option(
        0.35, "--drawdown-guard", help="Stock sleeve drawdown guard."
    ),
) -> None:
    run = run_core_satellite_backtest(
        config_path=config_path,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        top_n=top_n,
        core_weight=core_weight,
        rebalance_mode=rebalance_mode,
        drawdown_guard=drawdown_guard,
    )
    typer.echo(f"stock_core_satellite_run={run.run_dir}")


@app.command("stock-momentum-pyramid-backtest")
def stock_momentum_pyramid_backtest(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Optional start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Optional end date YYYY-MM-DD."
    ),
    max_holdings: int = typer.Option(10, "--max-holdings", help="Max stocks held."),
    focus_n: int = typer.Option(5, "--focus-n", help="Primary buy focus count."),
    max_weight: float = typer.Option(0.25, "--max-weight", help="Max stock weight."),
    focus_fill_weight: float = typer.Option(
        0.20, "--focus-fill-weight", help="Weight where focus names are full."
    ),
    exit_rank: int = typer.Option(20, "--exit-rank", help="Slow-exit rank threshold."),
    exit_months: int = typer.Option(
        2, "--exit-months", help="Months outside exit rank before sell."
    ),
) -> None:
    run = run_stock_momentum_pyramid_backtest(
        config_path=config_path,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        max_holdings=max_holdings,
        focus_n=focus_n,
        max_weight=max_weight,
        focus_fill_weight=focus_fill_weight,
        exit_rank=exit_rank,
        exit_months=exit_months,
    )
    typer.echo(f"stock_momentum_pyramid_run={run.run_dir}")


@app.command("annual-top10-stock-backtest")
def annual_top10_stock_backtest(
    stock_config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--stock-config",
        help="Stock rotation config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    top_n: int | None = typer.Option(
        None, "--top-n", help="Override number of stocks."
    ),
    lookback_bars: int = typer.Option(
        252, "--lookback-bars", help="Ranking lookback bars."
    ),
    pullback_run_id: str | None = typer.Option(
        None, "--pullback-run-id", help="Use pullback params from an existing run."
    ),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Optional start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Optional end date YYYY-MM-DD."
    ),
) -> None:
    run = run_annual_top10_stock_backtest(
        stock_config_path=stock_config_path,
        run_id=run_id,
        top_n=top_n,
        lookback_bars=lookback_bars,
        pullback_run_id=pullback_run_id,
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo(f"annual_top10_stock_run={run.run_dir}")


@app.command("annual-top10-stock-report")
def annual_top10_stock_report(
    run_id: str = typer.Option(..., "--run-id", help="Annual top10 stock run id."),
) -> None:
    report_path = generate_annual_top10_stock_report(run_id)
    typer.echo(f"annual_top10_stock_report={report_path}")


@app.command("blackcat-research")
def blackcat_research(
    config_path: Path = typer.Option(
        Path("config/default.yaml"), "--config", help="ETF research config path."
    ),
    stock_config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--stock-config",
        help="Stock panel config path.",
    ),
    trials: int | None = typer.Option(None, "--trials", help="Trial override per family."),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    min_stock_bars: int | None = typer.Option(
        None,
        "--min-stock-bars",
        help="Override minimum stock history bars.",
    ),
) -> None:
    run = run_blackcat_research(
        config_path=config_path,
        stock_config_path=stock_config_path,
        trials_override=trials,
        run_id=run_id,
        min_stock_bars=min_stock_bars,
    )
    typer.echo(f"blackcat_research_run={run.run_dir}")


@app.command("hybrid-filter-research")
def hybrid_filter_research(
    config_path: Path = typer.Option(
        Path("config/default.yaml"), "--config", help="ETF research config path."
    ),
    stock_config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--stock-config",
        help="Stock panel config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    pullback_run_id: str | None = typer.Option(
        None, "--pullback-run-id", help="Optional pullback source run id."
    ),
    overlay_run_id: str | None = typer.Option(
        None, "--overlay-run-id", help="Optional overlay source run id."
    ),
    blackcat_run_id: str | None = typer.Option(
        None, "--blackcat-run-id", help="Optional Blackcat source run id."
    ),
    min_stock_bars: int | None = typer.Option(
        None,
        "--min-stock-bars",
        help="Override minimum stock history bars.",
    ),
) -> None:
    run = run_hybrid_filter_research(
        config_path=config_path,
        stock_config_path=stock_config_path,
        run_id=run_id,
        pullback_run_id=pullback_run_id,
        overlay_run_id=overlay_run_id,
        blackcat_run_id=blackcat_run_id,
        min_stock_bars=min_stock_bars,
    )
    typer.echo(f"hybrid_filter_research_run={run.run_dir}")


@app.command("soft-filter-research")
def soft_filter_research(
    config_path: Path = typer.Option(
        Path("config/default.yaml"), "--config", help="ETF research config path."
    ),
    stock_config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--stock-config",
        help="Stock panel config path.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    pullback_run_id: str | None = typer.Option(
        None, "--pullback-run-id", help="Optional pullback source run id."
    ),
    overlay_run_id: str | None = typer.Option(
        None, "--overlay-run-id", help="Optional overlay source run id."
    ),
    blackcat_run_id: str | None = typer.Option(
        None, "--blackcat-run-id", help="Optional Blackcat source run id."
    ),
    min_stock_bars: int | None = typer.Option(
        None,
        "--min-stock-bars",
        help="Override minimum stock history bars.",
    ),
) -> None:
    run = run_soft_filter_research(
        config_path=config_path,
        stock_config_path=stock_config_path,
        run_id=run_id,
        pullback_run_id=pullback_run_id,
        overlay_run_id=overlay_run_id,
        blackcat_run_id=blackcat_run_id,
        min_stock_bars=min_stock_bars,
    )
    typer.echo(f"soft_filter_research_run={run.run_dir}")


@app.command("stock-strategy-research")
def stock_strategy_research(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    families: str | None = typer.Option(
        None,
        "--families",
        help="Comma-separated families: breakout,rebound,news_event_rules.",
    ),
    trials: int | None = typer.Option(None, "--trials", help="Trial override per family."),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
    start_date: str | None = typer.Option(
        None, "--start-date", help="Optional start date YYYY-MM-DD."
    ),
    end_date: str | None = typer.Option(
        None, "--end-date", help="Optional end date YYYY-MM-DD."
    ),
    fee_bps: float | None = typer.Option(
        None, "--fee-bps", help="Override percent fee (bps) per trade."
    ),
    slippage_bps: float | None = typer.Option(
        None, "--slippage-bps", help="Override slippage (bps) per fill."
    ),
    fixed_fee_egp: float | None = typer.Option(
        None, "--fixed-fee-egp", help="Override fixed fee (EGP) per BUY."
    ),
    fixed_fee_on_sell: bool = typer.Option(
        False, "--fixed-fee-on-sell", help="Also charge fixed fee on SELL."
    ),
) -> None:
    selected_families = (
        None
        if families is None
        else [part.strip() for part in families.split(",") if part.strip()]
    )
    run = run_stock_strategy_research(
        config_path=config_path,
        families=selected_families,
        trials_override=trials,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        fee_bps_override=fee_bps,
        slippage_bps_override=slippage_bps,
        fixed_fee_egp_override=fixed_fee_egp,
        fixed_fee_on_sell=fixed_fee_on_sell,
    )
    typer.echo(f"stock_strategy_research_run={run.run_dir}")


@app.command("stock-strategy-validate")
def stock_strategy_validate(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    trials: int = typer.Option(
        80,
        "--trials",
        help="Trials per static position-count search.",
    ),
    max_positions: str = typer.Option(
        "3,4,5,6,8",
        "--max-positions",
        help="Comma-separated max-position counts to compare.",
    ),
    train_end: str = typer.Option(
        "2023-12-31",
        "--train-end",
        help="Static validation train end date YYYY-MM-DD.",
    ),
    fixed_fee_on_sell: bool = typer.Option(
        True,
        "--fixed-fee-on-sell/--no-fixed-fee-on-sell",
        help="Charge fixed EGP fee on sells too.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    counts = [int(part.strip()) for part in max_positions.split(",") if part.strip()]
    run = run_stock_strategy_validation(
        config_path=config_path,
        trials=trials,
        max_positions=counts,
        train_end=train_end,
        fixed_fee_on_sell=fixed_fee_on_sell,
        run_id=run_id,
    )
    typer.echo(f"stock_strategy_validation_run={run.run_dir}")


@app.command("stock-strategy-v2")
def stock_strategy_v2(
    config_path: Path = typer.Option(
        Path("config/stock_rotation.yaml"),
        "--config",
        help="Stock rotation config path.",
    ),
    train_end: str = typer.Option(
        "2023-12-31",
        "--train-end",
        help="Train/test split marker YYYY-MM-DD.",
    ),
    market_filter: bool = typer.Option(
        True,
        "--market-filter/--no-market-filter",
        help="Use ETF regime filter for new buys.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_rebound_max5_v2(
        config_path=config_path,
        train_end=train_end,
        use_market_filter=market_filter,
        run_id=run_id,
    )
    typer.echo(f"stock_strategy_v2_run={run.run_dir}")


@app.command("stock-strategy-v3")
def stock_strategy_v3(
    config_path: Path = typer.Option(
        Path("config/stock_rotation_multifactor.yaml"),
        "--config",
        help="Stock rotation multifactor config path.",
    ),
    train_end: str = typer.Option(
        "2023-12-31",
        "--train-end",
        help="Train/test split marker YYYY-MM-DD.",
    ),
    market_filter: bool = typer.Option(
        True,
        "--market-filter/--no-market-filter",
        help="Use ETF/index/breadth regime filter for new buys.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_rebound_max5_v3(
        config_path=config_path,
        train_end=train_end,
        use_market_filter=market_filter,
        run_id=run_id,
    )
    typer.echo(f"stock_strategy_v3_run={run.run_dir}")


@app.command("stock-strategy-robustness")
def stock_strategy_robustness(
    config_path: Path = typer.Option(
        Path("config/stock_rotation_multifactor.yaml"),
        "--config",
        help="Stock rotation multifactor config path.",
    ),
    train_end: str = typer.Option(
        "2023-12-31",
        "--train-end",
        help="Train/test split marker YYYY-MM-DD.",
    ),
    start_samples: int = typer.Option(
        12,
        "--start-samples",
        help="Deterministic random start-date samples.",
    ),
    target_strategy: str = typer.Option(
        "rebound_max5_v4",
        "--target-strategy",
        help="Strategy to stress in cost/start/contribution tests.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_stock_strategy_robustness(
        config_path=config_path,
        train_end=train_end,
        start_samples=start_samples,
        target_strategy=target_strategy,
        run_id=run_id,
    )
    typer.echo(f"stock_strategy_robustness_run={run.run_dir}")


@app.command("stock-strategy-v4")
def stock_strategy_v4(
    config_path: Path = typer.Option(
        Path("config/stock_rotation_multifactor.yaml"),
        "--config",
        help="Stock rotation multifactor config path.",
    ),
    train_end: str = typer.Option(
        "2023-12-31",
        "--train-end",
        help="Train/test split marker YYYY-MM-DD.",
    ),
    market_filter: bool = typer.Option(
        True,
        "--market-filter/--no-market-filter",
        help="Use ETF/index/breadth regime filter for new buys.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_rebound_max5_v4(
        config_path=config_path,
        train_end=train_end,
        use_market_filter=market_filter,
        run_id=run_id,
    )
    typer.echo(f"stock_strategy_v4_run={run.run_dir}")


@app.command("stock-strategy-v5")
def stock_strategy_v5(
    config_path: Path = typer.Option(
        Path("config/stock_rotation_multifactor.yaml"),
        "--config",
        help="Stock rotation multifactor config path.",
    ),
    train_end: str = typer.Option(
        "2023-12-31",
        "--train-end",
        help="Train/test split marker YYYY-MM-DD.",
    ),
    max_candidates: int = typer.Option(
        16,
        "--max-candidates",
        help="Max deterministic v5 robustness candidates.",
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    run = run_rebound_max5_v5(
        config_path=config_path,
        train_end=train_end,
        max_candidates=max_candidates,
        run_id=run_id,
    )
    typer.echo(f"stock_strategy_v5_run={run.run_dir}")


if __name__ == "__main__":
    app()
