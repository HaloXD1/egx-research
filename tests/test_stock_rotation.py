from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from typer.testing import CliRunner

from egx_research.cli import app
from egx_research.stock_rotation_config import StockRotationConfig
from egx_research.stock_rotation import (
    active_members_on_date,
    build_etf_features,
    build_stock_features,
    find_overlap_start_date,
    run_equal_weight_benchmark,
    score_snapshot,
    select_rebalance_portfolio,
)
from egx_research.stock_rotation_config import load_stock_rotation_config
from egx_research.stock_rotation_data import (
    build_stage2_datasets_from_disclosures,
    classify_disclosure_event,
    fetch_current_constituents,
    parse_mubasher_financial_statements,
    parse_mubasher_history_csv,
    parse_mubasher_news_page,
    parse_mubasher_stock_overview,
    resolve_historical_csv_url,
    sync_stock_fundamentals,
)


def _make_workbook_bytes() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Basket Constituents"
    ws.append(["XT-MISR Basket Constituents"])
    ws.append(["Last Update:", "2026-02-01"])
    ws.append([])
    ws.append(
        [
            "RIC Code",
            "Holding Name",
            "Number of Shares per Basket",
            "Last Price",
            "FX",
            "Value (EGP)",
            "Weight",
        ]
    )
    for idx in range(12):
        ws.append(
            [f"stk{idx}.ca", f"Stock {idx}", 1, 10 + idx, 1, 10 + idx, 0.01 * (idx + 1)]
        )
    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _make_csv_text(seed: int = 0, rows: int = 900) -> str:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    base = 10 + seed
    values = []
    for i, date in enumerate(dates):
        close = base + i * (0.01 + seed * 0.0005)
        open_ = close - 0.05
        high = close + 0.1
        low = close - 0.1
        volume = 1000 + seed * 10 + i
        values.append(
            f"{date.strftime('%Y-%m-%d')}/00:00:00,{open_:.2f},{high:.2f},{low:.2f},{close:.2f},{volume}"
        )
    return "\n".join(values)


class _DummyResponse:
    def __init__(
        self, text: str = "", content: bytes | None = None, status_code: int = 200
    ) -> None:
        self.text = text
        self.content = content if content is not None else text.encode()
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        import json as _json

        return _json.loads(self.text)


def test_fetch_current_constituents_parses_workbook(monkeypatch, tmp_path) -> None:
    workbook_bytes = _make_workbook_bytes()
    monkeypatch.setattr(
        "egx_research.stock_rotation_data._get",
        lambda url: _DummyResponse(content=workbook_bytes),
    )
    config_path = tmp_path / "stock.yaml"
    config_path.write_text(
        "sources:\n  etf_rebalancing_url: https://example.com/workbook.xlsx\n  mubasher_stock_url_template: https://example.com/{ticker}\n",
        encoding="utf-8",
    )
    config = load_stock_rotation_config(config_path)
    frame, effective_date = fetch_current_constituents(config)
    assert list(frame.columns) == [
        "ticker",
        "ric_code",
        "holding_name",
        "weight",
        "stock_page_url",
        "historical_csv_url",
    ]
    assert len(frame) == 12
    assert frame.iloc[0]["ticker"] == "STK0"
    assert str(pd.Timestamp(effective_date).date()) == "2026-02-01"


def test_resolve_historical_csv_url_extracts_attribute(monkeypatch) -> None:
    html = '<div historical-data-url="https://static.mubasher.info/file.csv"></div>'
    monkeypatch.setattr(
        "egx_research.stock_rotation_data._get", lambda url: _DummyResponse(text=html)
    )
    config = StockRotationConfig()
    resolved_page, resolved_csv = resolve_historical_csv_url(
        config, "STK0", "Stock 0", "https://example.com/STK0"
    )
    assert resolved_page == "https://example.com/STK0"
    assert resolved_csv == "https://static.mubasher.info/file.csv"


def test_parse_mubasher_history_csv_normalizes_rows() -> None:
    csv_text = "2020-01-01/00:00:00,10,11,9,10.5,1000\n2020-01-02/00:00:00,10.5,11.5,10,11,2000"
    frame = parse_mubasher_history_csv(csv_text)
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(frame) == 2
    assert frame.iloc[0]["date"].strftime("%Y-%m-%d") == "2020-01-01"
    assert frame.iloc[1]["close"] == 11.0


def test_parse_mubasher_stock_overview_extracts_fundamental_stats() -> None:
    html = """
    <span class="stock-overview__text">Market Cap</span>
    <span class="stock-overview__value"><span class="number number--aligned">10,000.00</span></span>
    <span class="stock-overview__text">Book Value (BVPS)</span>
    <span class="stock-overview__value">
      <span class="number number--aligned">2.00</span>
      <span class="market-summary__date">Based on: First Quarter 2024</span>
    </span>
    <span class="stock-overview__text">EPS</span>
    <span class="stock-overview__value">
      <span class="number number--aligned">1.00</span>
      <span class="market-summary__date">Based on: First Quarter 2024</span>
    </span>
    <span class="stock-overview__text">Current Total Shares</span>
    <span class="stock-overview__value number number--aligned">1,000</span>
    """

    stats = parse_mubasher_stock_overview(html)

    assert stats["market_cap"] == 10000.0
    assert stats["book_value_per_share"] == 2.0
    assert stats["book_value_period"] == ("First Quarter", 2024)
    assert stats["eps"] == 1.0
    assert stats["eps_period"] == ("First Quarter", 2024)
    assert stats["shares_outstanding"] == 1000.0


def test_parse_mubasher_financial_statements_builds_quarterly_rows() -> None:
    html = """
    <script>
    midata.financialStatement = {'currency':'Egyptian Pound(EGP)','periods':[{
      'attachments':{'2024':'https://example.com/q1.pdf'},
      'label':'First Quarter',
      'sections':[
        {'label':'Balance Sheet','records':[
          {'label':'Total Owners\\' Equity & Minority Interest Equity','values':{'2024':2000.0}},
          {'label':'Total Assets','values':{'2024':5000.0}},
          {'label':'Total Liabilities','values':{'2024':3000.0}}
        ]},
        {'label':'Income Statement','records':[
          {'label':'Net Income or Loss','values':{'2024':250.0}},
          {'label':'Total Revenues','values':{'2024':1000.0}}
        ]},
        {'label':'Cash Flow','records':[
          {'label':'Net Cash Flow from (Used In) Operating Activities','values':{'2024':300.0}}
        ]}
      ],
      'year':'2024',
      'years':['2024']
    }],'yearStart':'January'};
    </script>
    """
    stats = {
        "shares_outstanding": 1000.0,
        "eps": 1.0,
        "eps_period": ("First Quarter", 2024),
        "book_value_per_share": 2.0,
        "book_value_period": ("First Quarter", 2024),
    }

    frame = parse_mubasher_financial_statements(
        html,
        symbol="abc",
        source_url="https://example.com/ABC/financial-statements",
        stock_overview=stats,
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["symbol"] == "ABC"
    assert str(pd.Timestamp(row["period_end"]).date()) == "2024-03-31"
    assert str(pd.Timestamp(row["filing_date"]).date()) == "2024-05-30"
    assert row["fiscal_period"] == "Q1 2024"
    assert row["currency"] == "EGP"
    assert row["source_url"] == "https://example.com/q1.pdf"
    assert row["net_income"] == 250.0
    assert row["shares_outstanding"] == 1000.0


def test_parse_mubasher_financial_statements_smooths_unit_outliers() -> None:
    html = """
    <script>
    midata.financialStatement = {'currency':'Egyptian Pound(EGP)','periods':[{
      'attachments':{},'label':'First Quarter',
      'sections':[{'label':'Balance Sheet','records':[
        {'label':'Total Owners\\' Equity & Minority Interest Equity','values':{'2024':200000.0}}
      ]}],
      'year':'2024','years':['2024']
    },{
      'attachments':{},'label':'Second Quarter',
      'sections':[{'label':'Balance Sheet','records':[
        {'label':'Total Owners\\' Equity & Minority Interest Equity','values':{'2024':210.0}}
      ]}],
      'year':'2024','years':['2024']
    }],'yearStart':'January'};
    </script>
    """

    frame = parse_mubasher_financial_statements(
        html,
        symbol="abc",
        source_url="https://example.com/ABC/financial-statements",
        stock_overview={"shares_outstanding": 1000.0},
    )

    q2 = frame[frame["fiscal_period"] == "Q2 2024"].iloc[0]
    assert q2["equity"] == 210000.0


def test_sync_stock_fundamentals_writes_internet_dataset(monkeypatch, tmp_path) -> None:
    root = tmp_path / "data/stock_rotation"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "ABC",
                "holding_name": "ABC Co",
                "stock_page_url": "https://example.com/ABC",
            }
        ]
    ).to_csv(root / "universe.csv", index=False)
    config = StockRotationConfig()
    config.storage.root_dir = str(root)

    stock_html = """
    <span class="stock-overview__text">EPS</span>
    <span class="stock-overview__value">
      <span class="number number--aligned">1.00</span>
      <span class="market-summary__date">Based on: First Quarter 2024</span>
    </span>
    <span class="stock-overview__text">Current Total Shares</span>
    <span class="stock-overview__value number number--aligned">1,000</span>
    """
    statement_html = """
    <script>
    midata.financialStatement = {'currency':'Egyptian Pound(EGP)','periods':[{
      'attachments':{},'label':'First Quarter',
      'sections':[{'label':'Income Statement','records':[
        {'label':'Net Income or Loss','values':{'2024':250.0}}
      ]}],
      'year':'2024','years':['2024']
    }],'yearStart':'January'};
    </script>
    """

    def fake_get(url):
        if url.endswith("/financial-statements"):
            return _DummyResponse(text=statement_html)
        return _DummyResponse(text=stock_html)

    monkeypatch.setattr("egx_research.stock_rotation_data._get", fake_get)

    sync_stock_fundamentals(config)

    fundamentals = pd.read_csv(root / "fundamentals.csv")
    assert len(fundamentals) == 1
    assert fundamentals.iloc[0]["symbol"] == "ABC"
    assert Path(root / "fundamental_sync_summary.json").exists()


def test_parse_mubasher_news_page_extracts_events() -> None:
    html = """
    <div class="mi-article-media-block__content">
      <span class="mi-article-media-block__date">17 March 2026 02:01 PM</span>
      <a class="mi-article-media-block__title" href="/news/4579290/CIB-unveils-dividends-for-2025-capital-hike-key-appointments/">CIB unveils dividends for 2025, capital hike, key appointments</a>
      <div class="mi-hide-for-small mi-article-media-block__text">Cairo - Mubasher: The general assembly approved cash dividends worth EGP 6 per share for 2025.</div>
    </div>
    <div class="mi-article-media-block__content">
      <span class="mi-article-media-block__date">10 February 2026 11:00 AM</span>
      <a class="mi-article-media-block__title" href="/news/4500000/Company-announces-capital-increase/">Company announces capital increase</a>
      <div class="mi-hide-for-small mi-article-media-block__text">The board approved a capital increase through a rights issue.</div>
    </div>
    """
    frame = parse_mubasher_news_page(
        html,
        symbol="COMI",
        source_url="https://english.mubasher.info/markets/EGX/stocks/COMI/news",
    )
    assert len(frame) == 2
    assert list(frame["event_class"]) == ["capital_action", "dividend"]
    assert frame.iloc[1]["cash_amount"] == 6.0


def test_build_stage2_datasets_from_disclosures_derives_templates() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "event_date": pd.to_datetime(["2026-03-17", "2026-02-10", "2026-01-01"]),
            "event_class": ["dividend", "capital_action", "governance"],
            "title": ["Dividend", "Capital increase", "AGM resolutions"],
            "summary": ["Cash dividends worth EGP 2 per share", "Rights issue approved", "Minutes approved"],
            "source_url": ["u1", "u2", "u3"],
            "source": ["mubasher_news"] * 3,
            "source_page_url": ["p"] * 3,
            "cash_amount": [2.0, None, None],
        }
    )
    dividends, corporate = build_stage2_datasets_from_disclosures(events)
    assert len(dividends) == 1
    assert dividends.iloc[0]["cash_amount"] == 2.0
    assert len(corporate) == 2
    assert corporate.iloc[0]["event_type"] in {"capital_action", "governance"}


def test_score_snapshot_and_selection_are_deterministic(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    config.portfolio.top_n = 3
    snapshot = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "holding_name": ["A", "B", "C", "D"],
            "close": [10, 10, 10, 10],
            "kama": [9, 9, 9, 9],
            "kama_rising": [True, True, True, True],
            "ret_3m": [0.3, 0.2, 0.1, 0.05],
            "ret_6m": [0.4, 0.25, 0.2, 0.1],
        }
    )
    scored = score_snapshot(snapshot, etf_ret_3m=0.01, config=config)
    assert scored.iloc[0]["symbol"] == "A"
    selected = select_rebalance_portfolio(
        snapshot, etf_ret_3m=0.01, config=config, active_members={"A", "B", "C", "D"}
    )
    assert list(selected["symbol"]) == ["A", "B", "C"]
    assert all(selected["target_weight"] == (1 / 3))


def test_score_snapshot_multifactor_core_uses_v0_factor_mix(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    config.selection.method = "multi_factor_core"
    config.portfolio.top_n = 2

    snapshot = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "holding_name": ["A", "B", "C"],
            "close": [10.0, 12.0, 8.0],
            "median_daily_value": [1_500_000.0, 900_000.0, 700_000.0],
            "median_daily_volume": [10_000.0, 9_000.0, 8_000.0],
            "coverage_ratio": [0.99, 0.99, 0.99],
            "ret_12_1": [0.40, 0.25, 0.10],
            "ret_6_1": [0.20, 0.10, 0.02],
            "ret_1m": [-0.03, 0.01, 0.05],
            "ret_3m": [0.30, 0.20, 0.15],
            "ret_6m": [0.35, 0.25, 0.20],
            "vol_63": [0.18, 0.24, 0.30],
            "beta_252": [0.70, 0.95, 1.20],
            "dividend_yield_1y": [0.08, 0.04, 0.00],
            "dividend_event_count_3y": [5.0, 3.0, 0.0],
            "dilutive_event_count_3y": [0.0, 1.0, 3.0],
        }
    )

    scored = score_snapshot(snapshot, etf_ret_3m=0.01, config=config)
    assert list(scored["symbol"][:2]) == ["A", "B"]
    assert "score_momentum" in scored.columns
    assert "score_quality" in scored.columns
    assert scored.iloc[0]["score"] > scored.iloc[1]["score"] > scored.iloc[2]["score"]


def test_find_overlap_start_date_requires_minimum_history(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    config.portfolio.top_n = 2
    config.validation.min_history_bars = 20
    dates = pd.bdate_range("2022-01-01", periods=60)
    panel_rows = []
    for symbol in ["A", "B"]:
        for i, date in enumerate(dates):
            panel_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": 10 + i,
                    "kama": 9 + i,
                    "kama_rising": True,
                    "ret_3m": 0.1,
                    "ret_6m": 0.2,
                }
            )
    features = pd.DataFrame(panel_rows)
    etf = pd.DataFrame({"date": dates, "close": range(len(dates))})
    etf_features = build_etf_features(etf, config)
    membership = pd.DataFrame(
        {
            "effective_date": [dates[0], dates[0]],
            "symbol": ["A", "B"],
            "is_member": [True, True],
            "source": ["test", "test"],
        }
    )
    start = find_overlap_start_date(features, etf_features, membership, config)
    assert pd.Timestamp(start) >= dates[0]


def test_active_members_on_date_uses_latest_snapshot() -> None:
    membership = pd.DataFrame(
        {
            "effective_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-06-01"]
            ),
            "symbol": ["A", "B", "B"],
            "is_member": [True, True, False],
            "source": ["s", "s", "s"],
        }
    )
    assert active_members_on_date(membership, pd.Timestamp("2020-03-01")) == {"A", "B"}
    assert active_members_on_date(membership, pd.Timestamp("2020-07-01")) == {"A"}


def test_equal_weight_benchmark_applies_fixed_fee_per_stock_buy(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    config.backtest.initial_cash = 0.0
    config.backtest.monthly_contribution = 100.0
    config.backtest.fee_bps = 0.0
    config.backtest.slippage_bps = 0.0
    config.backtest.share_precision = 0
    config.portfolio.fixed_buy_fee_egp = 5.0

    dates = pd.to_datetime(["2020-01-01", "2020-02-03"])
    panel = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "holding_name": ["Alpha", "Beta", "Alpha", "Beta"],
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
            "volume": [1000, 1000, 1000, 1000],
        }
    )
    membership = pd.DataFrame(
        {
            "effective_date": [dates[0], dates[0]],
            "symbol": ["A", "B"],
            "is_member": [True, True],
            "source": ["test", "test"],
        }
    )

    result = run_equal_weight_benchmark(panel, membership, pd.Series(dates), config)

    assert result.metrics["final_equity"] == 180.0
    assert len(result.actions) == 4
    assert result.monthly_rows[0]["buy_fees"] == 10.0
    assert result.monthly_rows[0]["cash_balance"] == 10.0
    assert result.monthly_rows[1]["cash_balance"] == 0.0


def test_stock_cli_flow_with_mocked_scraper(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)

    etf_dates = pd.bdate_range("2020-01-01", periods=900)
    etf_df = pd.DataFrame(
        {
            "date": etf_dates,
            "open": [10 + i * 0.01 for i in range(len(etf_dates))],
            "high": [10.1 + i * 0.01 for i in range(len(etf_dates))],
            "low": [9.9 + i * 0.01 for i in range(len(etf_dates))],
            "close": [10 + i * 0.01 for i in range(len(etf_dates))],
            "volume": [1000 + i for i in range(len(etf_dates))],
        }
    )
    etf_df.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    etf_df.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)

    config_path = tmp_path / "config/stock_rotation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "sources:",
                "  etf_rebalancing_url: https://example.com/workbook.xlsx",
                "  mubasher_stock_url_template: https://example.com/{ticker}",
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "backtest:",
                "  initial_cash: 100000.0",
                "  monthly_contribution: 10000.0",
                "  fee_bps: 0.0",
                "  slippage_bps: 0.0",
                "  share_precision: 0",
                "storage:",
                "  root_dir: data/stock_rotation",
                "  universe_filename: universe.csv",
                "  panel_filename: panel.csv",
                "  membership_filename: membership_snapshots.csv",
                "  raw_dir: raw",
                "  normalized_dir: normalized",
                "portfolio:",
                "  top_n: 10",
                "  fixed_buy_fee_egp: 5.0",
                "selection:",
                "  rel_strength_window_3m: 63",
                "  rel_strength_window_6m: 126",
                "validation:",
                "  min_history_bars: 200",
            ]
        ),
        encoding="utf-8",
    )

    workbook_bytes = _make_workbook_bytes()
    csv_map = {f"STK{i}": _make_csv_text(i, rows=900) for i in range(12)}
    news_html = """
    <div class="mi-article-media-block__content">
      <span class="mi-article-media-block__date">17 March 2026 02:01 PM</span>
      <a class="mi-article-media-block__title" href="/news/4579290/CIB-unveils-dividends-for-2025-capital-hike-key-appointments/">CIB unveils dividends for 2025, capital hike, key appointments</a>
      <div class="mi-hide-for-small mi-article-media-block__text">Cairo - Mubasher: The general assembly approved cash dividends worth EGP 6 per share for 2025 and a capital hike.</div>
    </div>
    """

    def fake_get(url, headers=None, verify=None, timeout=None, params=None):
        if "workbook.xlsx" in url:
            return _DummyResponse(content=workbook_bytes)
        if "companySearch" in url:
            return _DummyResponse(text="[]")
        if "example.com/STK" in url and url.endswith("/news"):
            return _DummyResponse(text=news_html)
        if "example.com/STK" in url:
            ticker = url.rsplit("/", 1)[-1]
            return _DummyResponse(
                text=f'<div historical-data-url="https://csv.local/{ticker}.csv"></div>'
            )
        if "csv.local" in url:
            ticker = Path(url).stem
            return _DummyResponse(text=csv_map[ticker])
        raise AssertionError(url)

    monkeypatch.setattr("egx_research.stock_rotation_data.requests.get", fake_get)

    runner = CliRunner()
    sync_result = runner.invoke(
        app, ["stock-sync", "--config", "config/stock_rotation.yaml"]
    )
    assert sync_result.exit_code == 0, sync_result.stdout
    backtest_result = runner.invoke(
        app,
        [
            "stock-rotate-backtest",
            "--config",
            "config/stock_rotation.yaml",
            "--run-id",
            "stock-rotation-smoke",
        ],
    )
    assert backtest_result.exit_code == 0, backtest_result.stdout
    report_result = runner.invoke(
        app, ["stock-rotate-report", "--run-id", "stock-rotation-smoke"]
    )
    assert report_result.exit_code == 0, report_result.stdout
    assert Path("runs/stock-rotation-smoke/summary.json").exists()
    assert Path("runs/stock-rotation-smoke/stock_rotation_report.html").exists()
    with Path("runs/stock-rotation-smoke/summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    equity = pd.read_csv(Path("runs/stock-rotation-smoke/equity_curve.csv"))
    assert "equal_weight_metrics" in summary
    assert "equal_weight_excess_vs_etf_dca" in summary
    assert "equal_weight_equity" in equity.columns
    assert Path("runs/stock-rotation-smoke/equal_weight_trade_actions.csv").exists()
    assert Path("data/stock_rotation/membership_snapshots.csv").exists()

    stage2_result = runner.invoke(
        app,
        [
            "stock-sync-stage2",
            "--config",
            "config/stock_rotation.yaml",
            "--max-news-items",
            "5",
        ],
    )
    assert stage2_result.exit_code == 0, stage2_result.stdout
    assert Path("data/stock_rotation/disclosure_events.csv").exists()
    assert Path("data/stock_rotation/dividend_actions.csv").exists()
    assert Path("data/stock_rotation/corporate_actions.csv").exists()
    assert Path("data/stock_rotation/institutional_stage2_summary.json").exists()


def test_stock_selection_config_defaults_and_loader(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)

    assert config.selection.liquidity_window_bars == 63
    assert config.selection.min_median_daily_value_egp == 100_000.0
    assert config.selection.min_median_daily_volume == 1_000.0
    assert config.selection.method == "relative_strength"
    assert config.storage.fundamentals_filename == "fundamentals.csv"
    assert config.storage.dividend_actions_filename == "dividend_actions.csv"
    assert config.storage.corporate_actions_filename == "corporate_actions.csv"
    assert config.validation.coverage_lookback_bars == 126
    assert config.validation.min_coverage_ratio == 0.92
    assert config.portfolio.turnover_buffer_score == 0.03
    assert config.model_selection.holdout_ratio == 0.2
    assert config.model_selection.min_neighbor_pass_rate == 0.6


def test_stock_select_cli_flow_on_local_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/normalized").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data/stock_rotation").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs/pullback-seed").mkdir(parents=True, exist_ok=True)

    dates = pd.bdate_range("2019-01-01", periods=540)
    symbols = ["A", "B", "C", "D", "E", "F"]

    panel_rows = []
    for s_idx, symbol in enumerate(symbols):
        base = 20 + s_idx * 2
        for i, date in enumerate(dates):
            close = base + i * (0.03 + s_idx * 0.002)
            panel_rows.append(
                {
                    "symbol": symbol,
                    "holding_name": f"Stock {symbol}",
                    "date": date,
                    "open": close * 0.998,
                    "high": close * 1.004,
                    "low": close * 0.996,
                    "close": close,
                    "volume": 5000 + i + s_idx * 100,
                    "weight": 1.0 / len(symbols),
                }
            )
    panel = pd.DataFrame(panel_rows)
    panel.to_csv(tmp_path / "data/stock_rotation/panel.csv", index=False)

    membership = pd.DataFrame(
        {
            "effective_date": [dates[0]] * len(symbols),
            "symbol": symbols,
            "is_member": [True] * len(symbols),
            "source": ["test"] * len(symbols),
        }
    )
    membership.to_csv(
        tmp_path / "data/stock_rotation/membership_verified_partial.csv", index=False
    )
    (tmp_path / "data/stock_rotation/dividend_actions.csv").write_text(
        "symbol,event_date,cash_amount,currency,source\n",
        encoding="utf-8",
    )
    (tmp_path / "data/stock_rotation/corporate_actions.csv").write_text(
        "symbol,event_date,event_type,is_dilutive,source\n",
        encoding="utf-8",
    )

    universe = pd.DataFrame(
        {
            "ticker": symbols,
            "ric_code": [s.lower() + ".ca" for s in symbols],
            "holding_name": [f"Stock {s}" for s in symbols],
            "weight": [1.0 / len(symbols)] * len(symbols),
            "stock_page_url": ["https://example.com"] * len(symbols),
            "historical_csv_url": ["https://example.com/history.csv"] * len(symbols),
        }
    )
    universe.to_csv(tmp_path / "data/stock_rotation/universe.csv", index=False)

    etf = pd.DataFrame(
        {
            "date": dates,
            "open": [100 + i * 0.02 for i in range(len(dates))],
            "high": [100.5 + i * 0.02 for i in range(len(dates))],
            "low": [99.5 + i * 0.02 for i in range(len(dates))],
            "close": [100 + i * 0.02 for i in range(len(dates))],
            "volume": [20_000 + i for i in range(len(dates))],
        }
    )
    etf.to_csv(tmp_path / "data/normalized/EGX30_ETF.csv", index=False)
    etf.to_csv(tmp_path / "data/normalized/EGX30_INDEX.csv", index=False)

    (tmp_path / "config/stock_rotation.yaml").write_text(
        "\n".join(
            [
                "benchmark:",
                "  etf_symbol_path: data/normalized/EGX30_ETF.csv",
                "  index_symbol_path: data/normalized/EGX30_INDEX.csv",
                "storage:",
                "  root_dir: data/stock_rotation",
                "  universe_filename: universe.csv",
                "  panel_filename: panel.csv",
                "  membership_filename: membership_snapshots.csv",
                "  dividend_actions_filename: dividend_actions.csv",
                "  corporate_actions_filename: corporate_actions.csv",
                "portfolio:",
                "  top_n: 3",
                "  turnover_buffer_score: 0.02",
                "  fixed_buy_fee_egp: 0.0",
                "selection:",
                "  method: relative_strength",
                "  rel_strength_window_3m: 63",
                "  rel_strength_window_6m: 126",
                "  liquidity_window_bars: 40",
                "  min_median_daily_value_egp: 1000.0",
                "  min_median_daily_volume: 1000.0",
                "validation:",
                "  min_history_bars: 200",
                "  coverage_lookback_bars: 60",
                "  min_coverage_ratio: 0.9",
                "model_selection:",
                "  holdout_ratio: 0.25",
                "  min_holdout_excess_return: -0.5",
                "  max_drawdown: 1.0",
                "  min_neighbor_pass_rate: 0.0",
                "  max_mean_rebalance_turnover_pct: 2.0",
                "  max_fee_to_contributions_ratio: 1.0",
                "backtest:",
                "  initial_cash: 100000.0",
                "  monthly_contribution: 10000.0",
                "  fee_bps: 0.0",
                "  slippage_bps: 0.0",
                "  share_precision: 0",
            ]
        ),
        encoding="utf-8",
    )

    with (tmp_path / "runs/pullback-seed/report_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "top_family": "dca_pullback_only",
                "top_params": {
                    "kama_len": 20,
                    "kama_fast": 2,
                    "kama_slow": 40,
                    "cci_len": 20,
                    "buy_threshold": -30.0,
                    "trend_buffer_atr": 0.5,
                    "atr_len": 14,
                },
            },
            handle,
        )

    runner = CliRunner()
    backtest_result = runner.invoke(
        app,
        [
            "stock-select-backtest",
            "--config",
            "config/stock_rotation.yaml",
            "--run-id",
            "stock-select-smoke",
            "--pullback-run-id",
            "pullback-seed",
            "--rebalance-mode",
            "monthly",
        ],
    )
    assert backtest_result.exit_code == 0, backtest_result.stdout

    report_result = runner.invoke(
        app, ["stock-select-report", "--run-id", "stock-select-smoke"]
    )
    assert report_result.exit_code == 0, report_result.stdout

    summary_path = Path("runs/stock-select-smoke/summary_selection.json")
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["rebalance_mode"] == "monthly"
    assert "candidate_evaluation" in summary
    assert Path("runs/stock-select-smoke/stock_selection_report.html").exists()
    assert Path("runs/stock-select-smoke/selection_diagnostics.csv").exists()
    assert Path("runs/stock-select-smoke/factor_coverage_rebalance.csv").exists()
    assert Path("runs/stock-select-smoke/missing_fundamental_warnings.csv").exists()
    assert Path("runs/stock-select-smoke/equity_curve_selection.csv").exists()


def test_fundamentals_merge_uses_filing_date_not_period_end(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    dates = pd.bdate_range("2024-01-01", periods=8)
    panel = pd.DataFrame(
        {
            "symbol": ["A"] * len(dates),
            "holding_name": ["Alpha"] * len(dates),
            "date": dates,
            "open": [10.0 + i for i in range(len(dates))],
            "high": [10.5 + i for i in range(len(dates))],
            "low": [9.5 + i for i in range(len(dates))],
            "close": [10.0 + i for i in range(len(dates))],
            "volume": [1000] * len(dates),
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["A"],
            "period_end": pd.to_datetime(["2023-12-31"]),
            "filing_date": pd.to_datetime(["2024-01-05"]),
            "fiscal_period": ["FY2023"],
            "currency": ["EGP"],
            "source_url": ["https://example.com/a"],
            "revenue": [1000.0],
            "net_income": [100.0],
            "equity": [500.0],
            "assets": [1000.0],
            "operating_cf": [120.0],
            "capex": [-30.0],
            "debt": [100.0],
            "cash": [50.0],
            "shares_outstanding": [10.0],
        }
    )

    features = build_stock_features(panel, config, fundamentals=fundamentals)
    before = features[features["date"] < pd.Timestamp("2024-01-05")]
    after = features[features["date"] >= pd.Timestamp("2024-01-05")].iloc[0]

    assert not before["has_fundamentals"].any()
    assert pd.isna(before["earnings_yield"]).all()
    assert bool(after["has_fundamentals"])
    assert str(pd.Timestamp(after["fundamental_period_end"]).date()) == "2023-12-31"
    assert after["fundamental_filing_date"] == pd.Timestamp("2024-01-05")


def test_multifactor_missing_fundamentals_stay_neutral(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    config.selection.method = "multi_factor_core"

    snapshot = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "holding_name": ["A", "B"],
            "close": [10.0, 10.0],
            "median_daily_value": [1_000_000.0, 1_000_000.0],
            "median_daily_volume": [10_000.0, 10_000.0],
            "coverage_ratio": [0.99, 0.99],
            "ret_12_1": [0.20, 0.20],
            "ret_6_1": [0.10, 0.10],
            "ret_1m": [0.01, 0.01],
            "ret_3m": [0.15, 0.15],
            "ret_6m": [0.20, 0.20],
            "vol_63": [0.20, 0.20],
            "beta_252": [1.0, 1.0],
        }
    )

    scored = score_snapshot(snapshot, etf_ret_3m=0.01, config=config)
    assert scored["score_value"].tolist() == [0.5, 0.5]
    assert scored["score_quality"].tolist() == [0.5, 0.5]


def test_multifactor_ranking_changes_with_real_fundamentals(tmp_path) -> None:
    config_path = tmp_path / "stock.yaml"
    config_path.write_text("", encoding="utf-8")
    config = load_stock_rotation_config(config_path)
    config.selection.method = "multi_factor_core"

    base = {
        "holding_name": ["Alpha", "Beta"],
        "close": [10.0, 10.0],
        "median_daily_value": [1_000_000.0, 1_000_000.0],
        "median_daily_volume": [10_000.0, 10_000.0],
        "coverage_ratio": [0.99, 0.99],
        "ret_12_1": [0.20, 0.20],
        "ret_6_1": [0.10, 0.10],
        "ret_1m": [0.01, 0.01],
        "ret_3m": [0.15, 0.15],
        "ret_6m": [0.20, 0.20],
        "vol_63": [0.20, 0.20],
        "beta_252": [1.0, 1.0],
        "dividend_yield_1y": [0.05, 0.05],
        "dividend_event_count_3y": [3.0, 3.0],
        "dilutive_event_count_3y": [0.0, 0.0],
    }
    neutral = pd.DataFrame({"symbol": ["A", "B"], **base})
    with_fundamentals = neutral.assign(
        earnings_yield=[0.20, 0.02],
        book_to_price=[1.2, 0.3],
        roe=[0.30, 0.03],
        roa=[0.15, 0.01],
        cash_conversion=[1.3, 0.2],
        leverage_inverse=[0.85, 0.40],
        positive_earnings_count_4p=[4.0, 1.0],
        margin_stability_4p=[0.95, 0.50],
    )

    neutral_scored = score_snapshot(neutral, etf_ret_3m=0.01, config=config)
    fundamental_scored = score_snapshot(
        with_fundamentals, etf_ret_3m=0.01, config=config
    )

    assert neutral_scored["score"].nunique() == 1
    assert fundamental_scored.iloc[0]["symbol"] == "A"
    assert fundamental_scored.iloc[0]["score"] > fundamental_scored.iloc[1]["score"]
