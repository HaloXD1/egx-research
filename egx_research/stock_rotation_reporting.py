from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
import plotly.graph_objects as go


def generate_stock_rotation_report(run_id: str) -> Path:
    run_dir = Path("runs") / run_id
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    holdings = pd.read_csv(run_dir / "selected_holdings_monthly.csv")
    turnover = pd.read_csv(run_dir / "turnover_monthly.csv")
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity["date"], y=equity["strategy_equity"], name="Stock Rotation")
    )
    if "equal_weight_equity" in equity.columns:
        fig.add_trace(
            go.Scatter(
                x=equity["date"],
                y=equity["equal_weight_equity"],
                name="Equal-Weight Stocks",
            )
        )
    fig.add_trace(
        go.Scatter(x=equity["date"], y=equity["etf_dca_equity"], name="ETF DCA")
    )
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["etf_buy_hold_norm"],
            name="ETF Buy & Hold (norm)",
        )
    )
    fig.update_layout(
        title="Stock Rotation vs ETF Benchmarks",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    summary_rows = [
        {
            "strategy": "Stock Rotation",
            **summary["strategy_metrics"],
            "excess_vs_etf_dca": summary["excess_vs_etf_dca"],
            "etf_buy_hold_return": summary["etf_buy_hold_return"],
        }
    ]
    if "equal_weight_metrics" in summary:
        summary_rows.append(
            {
                "strategy": "Equal-Weight Stocks",
                **summary["equal_weight_metrics"],
                "excess_vs_etf_dca": summary.get("equal_weight_excess_vs_etf_dca", 0.0),
                "etf_buy_hold_return": summary["etf_buy_hold_return"],
            }
        )
    summary_rows.append(
        {
            "strategy": "ETF DCA",
            **summary["etf_dca_metrics"],
            "excess_vs_etf_dca": 0.0,
            "etf_buy_hold_return": summary["etf_buy_hold_return"],
        }
    )

    html = [
        "<html><head><title>EGX30 Stock Rotation Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>Run {run_id}</h1>",
        f"<p>Excess vs ETF DCA: {summary['excess_vs_etf_dca']:.4f}</p>",
        (
            f"<p>Equal-weight stocks excess vs ETF DCA: {summary['equal_weight_excess_vs_etf_dca']:.4f} "
            f"| Method: {summary['equal_weight_method']} | Buy count: {summary['equal_weight_buy_count']}</p>"
            if "equal_weight_metrics" in summary
            else ""
        ),
        "<h2>Summary</h2>",
        pd.DataFrame(summary_rows).to_html(index=False),
        "<h2>Equity Curve</h2>",
        fig.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Selected Holdings by Month</h2>",
        holdings.to_html(index=False),
        "<h2>Turnover</h2>",
        turnover.to_html(index=False),
        "</body></html>",
    ]

    path = run_dir / "stock_rotation_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def generate_stock_selection_report(run_id: str) -> Path:
    run_dir = Path("runs") / run_id
    with (run_dir / "summary_selection.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    diagnostics = pd.read_csv(run_dir / "selection_diagnostics.csv")
    selections = pd.read_csv(run_dir / "selected_holdings_rebalance.csv")
    equity = pd.read_csv(run_dir / "equity_curve_selection.csv", parse_dates=["date"])
    factor_coverage_path = run_dir / "factor_coverage_rebalance.csv"
    warnings_path = run_dir / "missing_fundamental_warnings.csv"
    factor_coverage = (
        pd.read_csv(factor_coverage_path)
        if factor_coverage_path.exists()
        else pd.DataFrame()
    )
    warnings = pd.read_csv(warnings_path) if warnings_path.exists() else pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=equity["date"], y=equity["immediate_equity"], name="Immediate")
    )
    fig.add_trace(
        go.Scatter(x=equity["date"], y=equity["pullback_equity"], name="Pullback")
    )
    fig.add_trace(
        go.Scatter(x=equity["date"], y=equity["etf_dca_equity"], name="ETF DCA")
    )
    fig.update_layout(
        title="Enhanced Stock Selection vs ETF DCA",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    candidates = pd.DataFrame(summary["candidate_evaluation"])
    if not candidates.empty:
        candidates["holdout_cagr"] = candidates["holdout_metrics"].apply(
            lambda x: x.get("cagr", 0.0)
        )
        candidates["holdout_max_drawdown"] = candidates["holdout_metrics"].apply(
            lambda x: x.get("max_drawdown", 0.0)
        )
        candidates["holdout_twr_total_return"] = candidates["holdout_metrics"].apply(
            lambda x: x.get("twr_total_return", 0.0)
        )
        candidate_table = candidates[
            [
                "style",
                "passes_constraints",
                "holdout_excess_vs_etf_dca",
                "holdout_twr_total_return",
                "holdout_cagr",
                "holdout_max_drawdown",
                "neighbor_pass_rate",
                "mean_rebalance_turnover_pct",
                "fee_to_contributions_ratio",
            ]
        ].to_html(index=False)
    else:
        candidate_table = "<p>No candidate evaluation rows.</p>"

    html = [
        "<html><head><title>Enhanced Stock Selection Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>Run {run_id}</h1>",
        f"<p>Selection method: {summary.get('selection_method')}</p>",
        f"<p>Selected style: {summary.get('selected_style')} | Best holdout style: {summary.get('best_holdout_style')}</p>",
        f"<p>Reason: {summary.get('selection_reason')}</p>",
        "<h2>Candidate Evaluation</h2>",
        candidate_table,
        "<h2>Equity Curve</h2>",
        fig.to_html(full_html=False, include_plotlyjs=False),
        "<h2>Selection Diagnostics</h2>",
        diagnostics.to_html(index=False),
        "<h2>Factor Coverage by Rebalance</h2>",
        (
            factor_coverage.to_html(index=False)
            if not factor_coverage.empty
            else "<p>No factor coverage rows.</p>"
        ),
        "<h2>Missing Fundamental Warnings</h2>",
        (
            warnings.to_html(index=False)
            if not warnings.empty
            else "<p>No missing fundamental warnings.</p>"
        ),
        "<h2>Selected Holdings Per Rebalance</h2>",
        selections.to_html(index=False),
        "</body></html>",
    ]

    path = run_dir / "stock_selection_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def generate_stock_strategy_report(run_id: str) -> Path:
    run_dir = Path("runs") / run_id
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    leaderboard = pd.read_csv(run_dir / "leaderboard.csv")
    fig = go.Figure()
    for family in summary.get("families", []):
        equity_path = run_dir / f"equity_curve_{family}.csv"
        if not equity_path.exists():
            continue
        equity = pd.read_csv(equity_path, parse_dates=["date"])
        fig.add_trace(
            go.Scatter(
                x=equity["date"],
                y=equity["strategy_equity"],
                name=family,
            )
        )
        if "ETF DCA" not in [trace.name for trace in fig.data]:
            fig.add_trace(
                go.Scatter(
                    x=equity["date"],
                    y=equity["etf_dca_equity"],
                    name="ETF DCA",
                )
            )
    fig.update_layout(
        title="Event-Driven Stock Strategy Research",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    setup_sections: list[str] = []
    for family in summary.get("families", []):
        setup_path = run_dir / f"latest_setups_{family}.csv"
        actions_path = run_dir / f"actions_{family}.csv"
        try:
            setups = pd.read_csv(setup_path) if setup_path.exists() else pd.DataFrame()
        except EmptyDataError:
            setups = pd.DataFrame()
        try:
            actions = pd.read_csv(actions_path) if actions_path.exists() else pd.DataFrame()
        except EmptyDataError:
            actions = pd.DataFrame()
        setup_sections.extend(
            [
                f"<h2>{family} Latest Setups</h2>",
                setups.head(20).to_html(index=False)
                if not setups.empty
                else "<p>No latest setups.</p>",
                f"<h2>{family} Actions</h2>",
                actions.tail(50).to_html(index=False)
                if not actions.empty
                else "<p>No actions.</p>",
            ]
        )

    html = [
        "<html><head><title>EGX Stock Strategy Research</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>Run {run_id}</h1>",
        f"<p>Best family: {summary.get('best_family')}</p>",
        f"<p>Holdout start: {summary.get('holdout_start_date')}</p>",
        "<h2>Leaderboard</h2>",
        leaderboard.to_html(index=False),
        "<h2>Equity Curves</h2>",
        fig.to_html(full_html=False, include_plotlyjs=False),
        *setup_sections,
        "</body></html>",
    ]
    path = run_dir / "stock_strategy_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path
