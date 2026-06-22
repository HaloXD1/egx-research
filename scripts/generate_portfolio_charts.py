from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
OUTPUT_DIR = ROOT / "docs" / "portfolio_charts"

INK = "#18212f"
MUTED = "#667085"
GRID = "#e6e8ee"
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
SLATE = "#64748b"
LIGHT = "#cbd5e1"


def pct(value: float) -> float:
    return value * 100.0


def parse_pct(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict, keys: list[str]) -> dict:
    item = data
    for key in keys:
        item = item[key]
    return item


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(float) / float(series.iloc[0]) * 100.0


def make_title(title: str, subtitle: str) -> str:
    return f"{title}<br><span style='font-size:15px;color:{MUTED}'>{subtitle}</span>"


def base_layout(fig: go.Figure, title: str, height: int = 720, width: int = 1280) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=32, color=INK)),
        template="plotly_white",
        width=width,
        height=height,
        font=dict(family="Inter, IBM Plex Sans, Arial, sans-serif", size=16, color=INK),
        margin=dict(l=86, r=54, t=108, b=86),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.18, x=0),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, showline=True, linecolor=GRID, ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=True, linecolor=GRID, ticks="outside")
    return fig


def write_chart(fig: go.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    fig.write_image(output_path.with_suffix(".png"), width=fig.layout.width, height=fig.layout.height, scale=2)


def load_backtest_inventory(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_idx = next(idx for idx, line in enumerate(lines) if line.strip().startswith("| Run |"))
    headers = [part.strip() for part in lines[header_idx].strip().strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[header_idx + 2 :]:
        if not line.strip().startswith("|"):
            break
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        rows.append((parts + [""] * len(headers))[: len(headers)])
    return pd.DataFrame(rows, columns=headers)


def chart_membership_impact(output_path: Path) -> None:
    rows = []
    configs = [
        ("Live membership", "stock-rotation-live-20260411", "strategy_metrics", TEAL),
        ("Historical fee", "stock-rotation-historical-fee-20260411", "strategy_metrics", BLUE),
        ("Partial membership", "stock-rotation-partial-membership-20260413", "strategy_metrics", RED),
        ("Equal weight", "stock-rotation-equal-weight-20260414", "strategy_metrics", AMBER),
    ]
    dca_return = None
    for label, run_id, metrics_key, color in configs:
        data = read_json(RUNS_DIR / run_id / "summary.json")
        metric = data[metrics_key]
        dca = data["etf_dca_metrics"]
        dca_return = pct(dca["twr_total_return"])
        total_return = pct(metric["twr_total_return"])
        rows.append(
            {
                "label": label,
                "total_return": total_return,
                "excess_pp": total_return - dca_return,
                "color": color,
            }
        )

    df = pd.DataFrame(rows)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("Total return by membership assumption", "Excess vs ETF DCA baseline"),
    )
    fig.add_trace(
        go.Bar(
            x=df["label"],
            y=df["total_return"],
            text=df["total_return"].map(lambda v: f"{v:.0f}%"),
            textposition="outside",
            marker_color=df["color"],
            hovertemplate="%{x}<br>Total return: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["label"],
            y=df["excess_pp"],
            text=df["excess_pp"].map(lambda v: f"{v:+.1f} pp"),
            textposition="outside",
            marker_color=[TEAL if v >= 0 else RED for v in df["excess_pp"]],
            hovertemplate="%{x}<br>Excess vs DCA: %{y:+.1f} percentage points<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=dca_return, line_dash="dot", line_color=SLATE, row=1, col=1)
    fig.add_hline(y=0, line_color=SLATE, line_width=1, row=1, col=2)
    fig.update_yaxes(title="Total return (%)", row=1, col=1)
    fig.update_yaxes(title="Percentage points vs DCA", row=1, col=2)
    fig.update_xaxes(title=None, tickangle=-15)
    fig = base_layout(
        fig,
        make_title(
            "Data Provenance Can Flip the Result",
            f"Same stock-rotation idea; different membership assumptions produce different conclusions. ETF DCA baseline = {dca_return:.0f}%.",
        ),
        height=760,
    )
    fig.update_layout(showlegend=False, margin=dict(l=86, r=54, t=122, b=120))
    write_chart(fig, output_path)


def chart_holdout_validation(output_path: Path) -> None:
    df = pd.read_csv(RUNS_DIR / "dca-pullback-only-20260411/equity_curve_holdout.csv", parse_dates=["date"])
    last_date = df["date"].iloc[-1].to_pydatetime()
    curves = {
        "Pullback rule": ("strategy_equity", BLUE),
        "Monthly DCA": ("dca_equity", SLATE),
        "Buy & hold": ("buy_hold_equity", AMBER),
    }
    fig = go.Figure()
    for label, (column, color) in curves.items():
        indexed = normalize(df[column])
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=indexed,
                mode="lines",
                name=label,
                line=dict(color=color, width=3, dash="dot" if label == "Monthly DCA" else "solid"),
                hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>Indexed value: %{{y:.1f}}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=last_date,
            y=indexed.iloc[-1],
            text=f"{indexed.iloc[-1]:.0f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=13, color=color),
        )

    summary = read_json(RUNS_DIR / "dca-pullback-only-20260411/report_summary.json")
    excess_pp = pct(summary["holdout_excess_return_vs_dca"])
    fig.add_annotation(
        x=0.02,
        y=0.92,
        xref="paper",
        yref="paper",
        text=f"Holdout excess vs DCA: {excess_pp:+.2f} pp",
        showarrow=False,
        align="left",
        bgcolor="#f8fafc",
        bordercolor=GRID,
        borderwidth=1,
        font=dict(size=15, color=INK),
    )
    fig.update_xaxes(title="Holdout period", tickformat="%b %Y")
    fig.update_yaxes(title="Indexed value (start = 100)")
    fig = base_layout(
        fig,
        make_title(
            "Holdout Validation, Not a Victory Lap",
            "The selected ETF pullback rule only slightly beats monthly DCA over the held-out window.",
        ),
        height=760,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    write_chart(fig, output_path)


def chart_factor_ic_heatmap(output_path: Path) -> None:
    df = pd.read_csv(RUNS_DIR / "stock-factor-research-20260429/factor_ic_summary.csv")
    factor_labels = {
        "score_growth": "Growth",
        "score_momentum": "Momentum",
        "score_low_risk": "Low risk",
        "score_quality": "Quality",
        "score_value": "Value",
        "score_liquidity": "Liquidity",
    }
    order = ["score_growth", "score_momentum", "score_low_risk", "score_quality", "score_value", "score_liquidity"]
    horizons = ["1m", "3m", "6m"]
    df = df[df["factor"].isin(order) & df["horizon"].isin(horizons)].copy()
    df["factor_label"] = pd.Categorical(df["factor"].map(factor_labels), [factor_labels[f] for f in order], ordered=True)
    df["horizon"] = pd.Categorical(df["horizon"], horizons, ordered=True)
    pivot = df.pivot(index="factor_label", columns="horizon", values="mean_ic")
    max_abs = max(abs(float(pivot.min().min())), abs(float(pivot.max().max())))
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            colorscale="RdBu",
            text=pivot.values,
            texttemplate="%{text:.2f}",
            colorbar=dict(title="Mean IC"),
            hovertemplate="Factor: %{y}<br>Horizon: %{x}<br>Mean IC: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Forward return horizon")
    fig.update_yaxes(title=None)
    fig = base_layout(
        fig,
        make_title("Which Factors Had Signal?", "Mean IC measures rank correlation between factor score and future return."),
        height=700,
    )
    write_chart(fig, output_path)


def chart_data_quality_flags(output_path: Path) -> None:
    df = pd.read_csv(RUNS_DIR / "stock-factor-research-20260429/data_quality_flags.csv")
    counts = (
        df.groupby("flag")
        .agg(affected_symbols=("symbol", "nunique"), rows=("symbol", "size"))
        .reset_index()
        .sort_values("affected_symbols", ascending=False)
    )
    fig = go.Figure(
        go.Bar(
            x=counts["flag"],
            y=counts["affected_symbols"],
            text=counts["affected_symbols"],
            textposition="outside",
            marker_color=[AMBER, BLUE, RED, TEAL][: len(counts)],
            hovertemplate="%{x}<br>Affected symbols: %{y}<extra></extra>",
        )
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Affected symbols")
    fig = base_layout(
        fig,
        make_title("Data Quality Flags", "Research output records stale fundamentals and outliers instead of hiding them."),
        height=650,
    )
    fig.update_layout(showlegend=False)
    write_chart(fig, output_path)


def chart_topn_sensitivity(inventory_df: pd.DataFrame, output_path: Path) -> None:
    df = inventory_df[inventory_df["Run"].str.contains("probe-stock-only-top", na=False)].copy()
    df["top_n"] = df["Run"].str.extract(r"top(\d+)").astype(float)
    df["mode"] = df["Run"].apply(lambda x: "semiannual" if "semiannual" in x else "annual")
    df["CAGR_pct"] = df["CAGR"].map(parse_pct)
    df["Max_DD_pct"] = df["Max DD"].map(parse_pct)
    df = df.dropna(subset=["top_n", "CAGR_pct", "Max_DD_pct"])
    df = df.drop_duplicates(["top_n", "mode"]).sort_values(["mode", "top_n"])
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CAGR by selection size", "Max drawdown by selection size"))
    for mode, color in [("annual", BLUE), ("semiannual", AMBER)]:
        subset = df[df["mode"] == mode]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(x=subset["top_n"], y=subset["CAGR_pct"], mode="lines+markers", name=f"{mode} CAGR", line=dict(color=color, width=3)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=subset["top_n"], y=subset["Max_DD_pct"], mode="lines+markers", name=f"{mode} Max DD", line=dict(color=color, width=3, dash="dot")),
            row=1,
            col=2,
        )
    fig.update_xaxes(title="Top N")
    fig.update_yaxes(title="CAGR (%)", row=1, col=1)
    fig.update_yaxes(title="Max drawdown (%)", row=1, col=2)
    fig = base_layout(
        fig,
        make_title("Portfolio Rules Change Outcomes", "Changing selection size and rebalance frequency shifts both return and risk."),
        height=700,
    )
    write_chart(fig, output_path)


def chart_signal_consistency(output_path: Path) -> None:
    df_2020 = pd.read_csv(RUNS_DIR / "per-stock-returns-2020-to-date-latest.csv")
    df_2025 = pd.read_csv(RUNS_DIR / "per-stock-returns-2025-to-date-latest.csv")
    rows = []
    for label, df in [("2020 to 2026", df_2020), ("2025 to 2026", df_2025)]:
        positive = int((df["pullback_minus_dca_pct"] > 0).sum())
        total = len(df)
        rows.append({"period": label, "share": positive / total * 100.0, "positive": positive, "total": total})
    summary = pd.DataFrame(rows)
    fig = go.Figure(
        go.Bar(
            x=summary["period"],
            y=summary["share"],
            text=summary.apply(lambda r: f"{r['share']:.0f}%<br>{int(r['positive'])}/{int(r['total'])} stocks", axis=1),
            textposition="outside",
            marker_color=[TEAL, AMBER],
            hovertemplate="%{x}<br>Positive stock share: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Stocks where pullback > DCA (%)", range=[0, 100])
    fig = base_layout(
        fig,
        make_title("Signal Consistency Changed by Period", "The pullback rule helped fewer stocks in the shorter recent window."),
        height=650,
    )
    fig.update_layout(showlegend=False)
    write_chart(fig, output_path)


def chart_holdout_excess(inventory_df: pd.DataFrame, output_path: Path) -> None:
    selected = {
        "dca-pullback-only-20260411": ("Pullback only", "ETF timing"),
        "dca-overlay-20260411": ("Tactical overlay", "ETF timing"),
        "dca-pullback-topup-20260411": ("Pullback top-up", "ETF timing"),
        "dca-zone-overlay-20260411": ("Zone overlay", "ETF timing"),
        "stock-rotation-live-20260411": ("Live membership", "Stock rotation"),
        "stock-rotation-historical-fee-20260411": ("Historical fee", "Stock rotation"),
        "stock-rotation-partial-membership-20260413": ("Partial history", "Stock rotation"),
        "stock-rotation-equal-weight-20260414": ("Equal weight", "Stock rotation"),
    }
    df = inventory_df[inventory_df["Run"].isin(selected)].copy()
    df["label"] = df["Run"].map(lambda run: selected[run][0])
    df["group"] = df["Run"].map(lambda run: selected[run][1])
    df["holdout_excess_pp"] = df["Holdout excess"].map(parse_pct)
    df = df.dropna(subset=["holdout_excess_pp"])

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.28,
        subplot_titles=("ETF timing tests", "Stock rotation assumptions"),
    )
    for col, group in [(1, "ETF timing"), (2, "Stock rotation")]:
        subset = df[df["group"] == group].sort_values("holdout_excess_pp")
        positive = subset[subset["holdout_excess_pp"] >= 0]
        negative = subset[subset["holdout_excess_pp"] < 0]
        if not positive.empty:
            fig.add_trace(
                go.Bar(
                    x=positive["holdout_excess_pp"],
                    y=positive["label"],
                    orientation="h",
                    cliponaxis=False,
                    marker_color=TEAL,
                    hovertemplate="%{y}<br>Holdout excess: %{x:+.1f} percentage points<extra></extra>",
                ),
                row=1,
                col=col,
            )
        if not negative.empty:
            fig.add_trace(
                go.Bar(
                    x=negative["holdout_excess_pp"],
                    y=negative["label"],
                    orientation="h",
                    marker_color=RED,
                    hovertemplate="%{y}<br>Holdout excess: %{x:+.1f} percentage points<extra></extra>",
                ),
                row=1,
                col=col,
            )
        offset = 0.28 if group == "ETF timing" else 2.8
        for _, row in subset.iterrows():
            value = float(row["holdout_excess_pp"])
            if value < 0 and group == "Stock rotation":
                x = 4
                xanchor = "left"
                font_color = INK
            elif value < 0 and abs(value) >= 1 and group == "ETF timing":
                x = value / 2
                xanchor = "center"
                font_color = "white"
            elif value < 0:
                x = value - offset
                xanchor = "right"
                font_color = INK
            else:
                x = value + offset
                xanchor = "left"
                font_color = INK
            fig.add_annotation(
                x=x,
                y=row["label"],
                text=f"{value:+.1f} pp",
                showarrow=False,
                xanchor=xanchor,
                font=dict(size=13, color=font_color),
                row=1,
                col=col,
            )
        fig.add_vline(x=0, line_color=SLATE, line_width=1, row=1, col=col)
        fig.update_yaxes(categoryorder="array", categoryarray=subset["label"].tolist(), row=1, col=col)
    fig.update_xaxes(title="Percentage points vs DCA")
    fig.update_xaxes(range=[-9.5, 1.8], row=1, col=1)
    fig.update_xaxes(range=[-22, 225], row=1, col=2)
    fig.update_yaxes(title=None)
    fig = base_layout(
        fig,
        make_title(
            "Holdout Excess Uses Percentage Points",
            "This fixes the old chart's ratio math and keeps small ETF tests separate from larger stock-rotation effects.",
        ),
        height=760,
    )
    fig.update_layout(showlegend=False, margin=dict(l=190, r=84, t=122, b=90))
    write_chart(fig, output_path)


def chart_risk_map(output_path: Path) -> None:
    configs = [
        ("ETF DCA baseline", "annual-top10-20260414/summary.json", ["etf_dca_metrics"], "Baseline", SLATE),
        ("Annual top 10", "annual-top10-20260414/summary.json", ["basket_pullback_metrics"], "High return", BLUE),
        ("Core/Satellite 50-50", "core-satellite-fixed-50-50-20260424/summary.json", ["core_satellite_metrics"], "Balanced", TEAL),
        ("Fundamental MF", "fundamental-multifactor-20260429/summary_selection.json", ["immediate_metrics"], "Factor", AMBER),
        ("Stock factor research", "stock-factor-research-20260429/summary_factor_research.json", ["best_candidate", "metrics"], "Factor", "#b45309"),
        ("Rotation partial history", "stock-rotation-partial-membership-20260413/summary.json", ["strategy_metrics"], "Data caveat", RED),
        ("Rebound max5 v4", "rebound-max5-v4-candidate-20260430/summary.json", ["metrics"], "Event strategy", PURPLE),
    ]
    rows = []
    for label, rel_path, keys, group, color in configs:
        metrics = nested(read_json(RUNS_DIR / rel_path), keys)
        rows.append(
            {
                "label": label,
                "group": group,
                "color": color,
                "CAGR_pct": pct(metrics["cagr"]),
                "Max_DD_pct": pct(metrics["max_drawdown"]),
                "Sharpe": metrics["sharpe"],
                "TWR_pct": pct(metrics["twr_total_return"]),
            }
        )
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Max_DD_pct"]],
                y=[row["CAGR_pct"]],
                mode="markers+text",
                name=row["label"],
                text=[row["label"]],
                textposition="top center",
                marker=dict(
                    size=18 + row["Sharpe"] * 10,
                    color=row["color"],
                    opacity=0.88,
                    line=dict(color="white", width=1.2),
                ),
                hovertemplate=(
                    f"{row['label']}<br>Max DD: %{{x:.1f}}%<br>CAGR: %{{y:.1f}}%"
                    f"<br>Sharpe: {row['Sharpe']:.2f}<br>TWR: {row['TWR_pct']:.1f}%<extra></extra>"
                ),
            )
        )
    fig.add_annotation(
        x=0.02,
        y=0.96,
        xref="paper",
        yref="paper",
        text="Better direction: up and left",
        showarrow=False,
        bgcolor="#f8fafc",
        bordercolor=GRID,
        borderwidth=1,
        font=dict(size=14, color=INK),
    )
    fig.update_xaxes(title="Max drawdown (%)", range=[20, 56])
    fig.update_yaxes(title="CAGR (%)", range=[18, 52])
    fig = base_layout(
        fig,
        make_title(
            "Risk Map: Return Is Not Enough",
            "Full-period summaries; point size reflects Sharpe ratio.",
        ),
        height=780,
    )
    fig.update_layout(showlegend=False)
    write_chart(fig, output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_df = load_backtest_inventory(RUNS_DIR / "backtest_inventory_20260429.md")

    chart_membership_impact(OUTPUT_DIR / "01_membership_equity_curve.html")
    chart_holdout_validation(OUTPUT_DIR / "02_holdout_validation.html")
    chart_factor_ic_heatmap(OUTPUT_DIR / "03_factor_ic_heatmap.html")
    chart_data_quality_flags(OUTPUT_DIR / "04_data_quality_flags.html")
    chart_topn_sensitivity(inventory_df, OUTPUT_DIR / "05_topn_sensitivity.html")
    chart_signal_consistency(OUTPUT_DIR / "06_signal_consistency.html")
    chart_holdout_excess(inventory_df, OUTPUT_DIR / "07_holdout_excess_ladder.html")
    chart_risk_map(OUTPUT_DIR / "08_risk_adjusted_map.html")

    print(f"Charts generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
