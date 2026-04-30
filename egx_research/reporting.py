from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from egx_research.backtest import run_buy_hold_benchmark, run_dca_benchmark, run_strategy_backtest
from egx_research.config import AppConfig, load_config
from egx_research.data import load_price_data
from egx_research.optimization import evaluate_windows
from egx_research.strategies import PARAMETER_SPACES, build_strategy_frame, normalize_params
from egx_research.utils import write_json
from egx_research.validation import Window, build_walk_forward_windows, split_holdout


def _load_candidates(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "candidates.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["candidates"]


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _windows_from_frame(frame: pd.DataFrame) -> list[Window]:
    return [
        Window(
            train_start=int(row.train_start),
            train_end=int(row.train_end),
            test_start=int(row.test_start),
            test_end=int(row.test_end),
        )
        for row in frame.itertuples(index=False)
    ]


def _build_stability_heatmap(
    data: pd.DataFrame,
    family: str,
    params: dict[str, Any],
    windows: list[Window],
    config: AppConfig,
) -> pd.DataFrame:
    numeric_params = [
        name
        for name, spec in PARAMETER_SPACES[family].items()
        if spec["type"] in {"int", "float"}
    ]
    if not numeric_params:
        return pd.DataFrame()

    primary = numeric_params[0]
    secondary = numeric_params[1] if len(numeric_params) > 1 else numeric_params[0]
    primary_spec = PARAMETER_SPACES[family][primary]
    secondary_spec = PARAMETER_SPACES[family][secondary]

    primary_values = sorted(
        {
            max(primary_spec["low"], min(primary_spec["high"], params[primary] - primary_spec["step"])),
            params[primary],
            max(primary_spec["low"], min(primary_spec["high"], params[primary] + primary_spec["step"])),
        }
    )
    secondary_values = sorted(
        {
            max(secondary_spec["low"], min(secondary_spec["high"], params[secondary] - secondary_spec["step"])),
            params[secondary],
            max(secondary_spec["low"], min(secondary_spec["high"], params[secondary] + secondary_spec["step"])),
        }
    )

    benchmarks = {
        index: {"dca": run_dca_benchmark(data, window.test_start, window.test_end, config.backtest)}
        for index, window in enumerate(windows)
    }
    rows = []
    for first in primary_values:
        for second in secondary_values:
            candidate = dict(params)
            candidate[primary] = first
            candidate[secondary] = second
            metrics, _ = evaluate_windows(data, family, candidate, windows, config, benchmarks)
            rows.append(
                {
                    primary: first,
                    secondary: second,
                    "score": metrics["score"],
                }
            )
    return pd.DataFrame(rows)


def generate_report(run_id: str, config_path: str | Path | None = None) -> Path:
    run_dir = Path("runs") / run_id
    manifest = _load_manifest(run_dir)
    config = load_config(config_path or run_dir / "config_snapshot.yaml")
    data = load_price_data(manifest["normalized_path"])
    candidates = _load_candidates(run_dir)
    candidates.sort(
        key=lambda item: (item["passed_filters"], item["rank_score"], item["holdout_metrics"]["excess_return_vs_dca"]),
        reverse=True,
    )

    windows_frame = pd.read_csv(run_dir / "walk_forward_windows.csv")
    windows = _windows_from_frame(windows_frame)
    research_end, _ = split_holdout(len(data), config.validation.holdout_ratio)
    holdout_start = research_end

    top_candidate = candidates[0]
    strategy_frame = build_strategy_frame(data, top_candidate["family"], normalize_params(top_candidate["family"], top_candidate["params"]))
    strategy_holdout = run_strategy_backtest(strategy_frame, holdout_start, len(data) - 1, config.backtest)
    dca_holdout = run_dca_benchmark(data, holdout_start, len(data) - 1, config.backtest)
    buy_hold = run_buy_hold_benchmark(data, holdout_start, len(data) - 1)
    buy_hold_equity = buy_hold * float(strategy_holdout.flows.iloc[0])

    equity_curves = pd.DataFrame(
        {
            "date": data["date"].iloc[holdout_start:].values,
            "strategy_equity": strategy_holdout.equity.values,
            "dca_equity": dca_holdout.equity.values,
            "buy_hold_equity": buy_hold_equity.values,
        }
    )
    equity_curves.to_csv(run_dir / "equity_curve_holdout.csv", index=False)

    holdout_summary = pd.DataFrame(
        [
            {
                "family": candidate["family"],
                "rank_score": candidate["rank_score"],
                "passed_filters": candidate["passed_filters"],
                "holdout_cagr": candidate["holdout_metrics"]["cagr"],
                "holdout_profit_factor": candidate["holdout_metrics"]["profit_factor"],
                "holdout_max_drawdown": candidate["holdout_metrics"]["max_drawdown"],
                "holdout_excess_return_vs_dca": candidate["holdout_metrics"]["excess_return_vs_dca"],
                "neighbor_pass_rate": candidate["neighbor_pass_rate"],
                "params": str(candidate["params"]),
            }
            for candidate in candidates[:10]
        ]
    )
    holdout_summary.to_csv(run_dir / "holdout_summary.csv", index=False)

    heatmap_sections = []
    for candidate in candidates:
        family = candidate["family"]
        if any(section["family"] == family for section in heatmap_sections):
            continue
        heatmap_frame = _build_stability_heatmap(data, family, normalize_params(family, candidate["params"]), windows, config)
        if heatmap_frame.empty:
            continue
        heatmap_frame.to_csv(run_dir / f"stability_{family}.csv", index=False)
        numeric_columns = [column for column in heatmap_frame.columns if column != "score"]
        pivot = heatmap_frame.pivot(index=numeric_columns[0], columns=numeric_columns[1], values="score")
        fig = px.imshow(pivot, text_auto=".2f", aspect="auto", title=f"{family} stability heatmap")
        heatmap_sections.append({"family": family, "html": fig.to_html(full_html=False, include_plotlyjs=False)})

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=equity_curves["date"], y=equity_curves["strategy_equity"], name="Strategy"))
    line_fig.add_trace(go.Scatter(x=equity_curves["date"], y=equity_curves["dca_equity"], name="Monthly DCA"))
    line_fig.add_trace(go.Scatter(x=equity_curves["date"], y=equity_curves["buy_hold_equity"], name="ETF Buy & Hold"))
    line_fig.update_layout(title="Holdout equity curve", xaxis_title="Date", yaxis_title="Value")

    top10_html = pd.read_csv(run_dir / "top10_overall.csv").to_html(index=False)
    top3_html = pd.read_csv(run_dir / "top3_per_family.csv").to_html(index=False)
    windows_html = windows_frame.to_html(index=False)
    holdout_html = holdout_summary.to_html(index=False)

    importance_path = run_dir / "parameter_importance.csv"
    importance_html = ""
    if importance_path.exists():
        importance_html = pd.read_csv(importance_path).to_html(index=False)

    body = [
        "<html><head><title>EGX Research Report</title>",
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>',
        "</head><body>",
        f"<h1>Run {run_id}</h1>",
        f"<p>Top candidate: {top_candidate['family']} | passed={top_candidate['passed_filters']}</p>",
        "<h2>Top 10 overall</h2>",
        top10_html,
        "<h2>Top 3 per family</h2>",
        top3_html,
        "<h2>Walk-forward windows</h2>",
        windows_html,
        "<h2>Holdout summary</h2>",
        holdout_html,
        "<h2>Holdout equity curve</h2>",
        line_fig.to_html(full_html=False, include_plotlyjs=False),
    ]
    if importance_html:
        body.extend(["<h2>Parameter importance</h2>", importance_html])
    for section in heatmap_sections:
        body.extend([f"<h2>{section['family']} stability heatmap</h2>", section["html"]])
    body.append("</body></html>")

    report_path = run_dir / "report.html"
    report_path.write_text("\n".join(body), encoding="utf-8")
    write_json(
        run_dir / "report_summary.json",
        {
            "run_id": run_id,
            "top_family": top_candidate["family"],
            "top_params": top_candidate["params"],
            "top_rank_score": top_candidate["rank_score"],
            "holdout_excess_return_vs_dca": top_candidate["holdout_metrics"]["excess_return_vs_dca"],
        },
    )
    return report_path
