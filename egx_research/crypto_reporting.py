from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from egx_research.utils import write_json


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _table(path: Path) -> str:
    if not path.exists():
        return "<p>Missing.</p>"
    frame = pd.read_csv(path)
    if frame.empty:
        return "<p>Empty.</p>"
    return frame.to_html(index=False)


def _equity_chart(path: Path) -> str:
    if not path.exists():
        return "<p>Missing equity curve.</p>"
    equity = pd.read_csv(path, parse_dates=["date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity["date"], y=equity["strategy_equity"], name="Strategy"))
    fig.add_trace(go.Scatter(x=equity["date"], y=equity["monthly_dca_equity"], name="Monthly DCA"))
    fig.add_trace(go.Scatter(x=equity["date"], y=equity["weekly_dca_equity"], name="Weekly DCA"))
    fig.add_trace(go.Scatter(x=equity["date"], y=equity["buy_hold_equity"], name="Buy & Hold"))
    if "target_allocation" in equity.columns:
        fig.add_trace(
            go.Scatter(
                x=equity["date"],
                y=equity["target_allocation"],
                name="Target allocation",
                yaxis="y2",
                opacity=0.35,
            )
        )
        fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "range": [0, 1]})
    fig.update_layout(title="BTC holdout equity", xaxis_title="Date", yaxis_title="Value")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def generate_crypto_report(run_id: str) -> Path:
    run_dir = Path("runs") / run_id
    manifest = _read_json(run_dir / "manifest.json")
    summary = _read_json(run_dir / "crypto_research_summary.json")

    body = [
        "<html><head><title>BTC Crypto Research Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>BTC Research Run {run_id}</h1>",
        f"<p>Top family: {summary['top_family']} | holdout excess vs monthly DCA: {summary['holdout_excess_return_vs_dca']:.4f}</p>",
        "<h2>Research Setup</h2>",
        pd.DataFrame(
            [
                {
                    "features_path": manifest.get("features_path"),
                    "window_scheme": manifest.get("window_scheme"),
                    "objective_mode": manifest.get("objective_mode"),
                    "benchmark_primary": manifest.get("benchmark_primary"),
                }
            ]
        ).to_html(index=False),
        "<h2>Top 10 Overall</h2>",
        _table(run_dir / "top10_overall.csv"),
        "<h2>Top 3 Per Family</h2>",
        _table(run_dir / "top3_per_family.csv"),
        "<h2>Holdout Equity</h2>",
        _equity_chart(run_dir / "crypto_equity_curve_holdout.csv"),
        "<h2>Ablation Summary</h2>",
        _table(run_dir / "crypto_ablation_summary.csv"),
        "<h2>Cost Stress</h2>",
        _table(run_dir / "crypto_cost_stress.csv"),
        "<h2>Crash/Regime Review</h2>",
        _table(run_dir / "crypto_regime_summary.csv"),
        "<h2>Feature Coverage</h2>",
        _table(run_dir / "crypto_feature_coverage.csv"),
        *(
            [
                "<h2>Institutional Sleeve Ablation</h2>",
                _table(run_dir / "institutional_sleeve_ablation.csv"),
                "<h2>Institutional Latest State</h2>",
                _table(run_dir / "institutional_current_state.csv"),
            ]
            if (run_dir / "institutional_current_state.csv").exists()
            else []
        ),
        "<h2>Parameter Importance</h2>",
        _table(run_dir / "parameter_importance.csv"),
        "</body></html>",
    ]
    report_path = run_dir / "crypto_report.html"
    report_path.write_text("\n".join(body), encoding="utf-8")
    write_json(
        run_dir / "crypto_report_summary.json",
        {
            "run_id": run_id,
            "report_path": str(report_path),
            "top_family": summary["top_family"],
            "holdout_excess_return_vs_dca": summary["holdout_excess_return_vs_dca"],
        },
    )
    return report_path
