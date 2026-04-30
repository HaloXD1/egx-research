from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "symbol",
    "period_end",
    "filing_date",
    "fiscal_period",
    "currency",
    "source_url",
    "revenue",
    "net_income",
    "equity",
    "assets",
    "operating_cf",
    "capex",
    "debt",
    "cash",
    "shares_outstanding",
]

NUMERIC_COLUMNS = [
    "revenue",
    "net_income",
    "equity",
    "assets",
    "operating_cf",
    "capex",
    "debt",
    "cash",
    "shares_outstanding",
]

RAW_MERGE_COLUMNS = [
    "fundamental_period_end",
    "fundamental_filing_date",
    "fiscal_period",
    "currency",
    "source_url",
    *NUMERIC_COLUMNS,
]

DERIVED_FACTOR_COLUMNS = [
    "revenue_ttm",
    "net_income_ttm",
    "operating_cf_ttm",
    "earnings_yield",
    "pe_ratio",
    "book_to_price",
    "pb_ratio",
    "roe",
    "roa",
    "cash_conversion",
    "leverage_inverse",
    "revenue_growth",
    "net_income_growth",
    "positive_earnings_count_4p",
    "margin_stability_4p",
    "fundamental_row_age_days",
]

MERGED_FUNDAMENTAL_COLUMNS = [
    *RAW_MERGE_COLUMNS,
    *DERIVED_FACTOR_COLUMNS,
    "has_fundamentals",
]


def empty_fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def load_fundamentals(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return empty_fundamentals_frame()

    frame = pd.read_csv(path)
    if frame.empty:
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"Fundamentals missing columns: {sorted(missing)}")
        return empty_fundamentals_frame()

    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Fundamentals missing columns: {sorted(missing)}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce")
    frame["fiscal_period"] = frame["fiscal_period"].astype(str).str.strip()
    frame["currency"] = frame["currency"].astype(str).str.strip()
    frame["source_url"] = frame["source_url"].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bad_required = frame[
        frame["symbol"].eq("")
        | frame["period_end"].isna()
        | frame["filing_date"].isna()
    ]
    if not bad_required.empty:
        raise ValueError("Fundamentals rows need symbol, period_end, filing_date.")

    lookahead = frame[frame["filing_date"] < frame["period_end"]]
    if not lookahead.empty:
        symbols = sorted(lookahead["symbol"].dropna().unique().tolist())
        raise ValueError(
            "Fundamentals filing_date before period_end; would leak data: "
            + ", ".join(symbols)
        )

    return (
        frame.sort_values(["symbol", "filing_date", "period_end", "fiscal_period"])
        .drop_duplicates(
            subset=["symbol", "period_end", "filing_date", "fiscal_period"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0.0, np.nan)
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan)


def _ttm_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if column not in frame.columns:
        return result

    fiscal = frame["fiscal_period"].astype(str).str.upper()
    quarterly = fiscal.str.match(r"Q[1-4]\b", na=False)
    annual = fiscal.str.match(r"FY\b", na=False) | fiscal.str.contains(
        "ANNUAL", na=False
    )

    for _, group in frame[quarterly].groupby("symbol", sort=False):
        result.loc[group.index] = group[column].rolling(4, min_periods=1).sum()

    result.loc[annual] = frame.loc[annual, column]
    missing = result.isna()
    result.loc[missing] = frame.loc[missing, column]
    return result


def add_derived_fundamental_fields(fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        frame = fundamentals.copy()
        for column in DERIVED_FACTOR_COLUMNS:
            if column != "fundamental_row_age_days":
                frame[column] = pd.Series(dtype=float)
        return frame

    frame = fundamentals.sort_values(["symbol", "period_end", "filing_date"]).copy()
    frame["revenue_ttm"] = _ttm_metric(frame, "revenue")
    frame["net_income_ttm"] = _ttm_metric(frame, "net_income")
    frame["operating_cf_ttm"] = _ttm_metric(frame, "operating_cf")
    frame["earnings_yield"] = np.nan
    frame["pe_ratio"] = np.nan
    frame["book_to_price"] = np.nan
    frame["pb_ratio"] = np.nan
    frame["roe"] = _safe_div(frame["net_income_ttm"], frame["equity"])
    frame["roa"] = _safe_div(frame["net_income_ttm"], frame["assets"])
    frame["cash_conversion"] = _safe_div(
        frame["operating_cf_ttm"],
        frame["net_income_ttm"].where(frame["net_income_ttm"] > 0.0),
    )
    frame["leverage_inverse"] = 1.0 - _safe_div(frame["debt"], frame["assets"])
    frame["revenue_growth"] = frame.groupby("symbol")["revenue_ttm"].pct_change()
    frame["net_income_growth"] = frame.groupby("symbol")[
        "net_income_ttm"
    ].pct_change()
    frame[["revenue_growth", "net_income_growth"]] = frame[
        ["revenue_growth", "net_income_growth"]
    ].replace([np.inf, -np.inf], np.nan)

    frame["net_margin"] = _safe_div(frame["net_income"], frame["revenue"])
    frame["positive_earnings_count_4p"] = (
        frame.groupby("symbol")["net_income"]
        .transform(
            lambda values: values.gt(0)
            .astype(float)
            .rolling(4, min_periods=1)
            .sum()
        )
        .astype(float)
    )
    margin_std = frame.groupby("symbol")["net_margin"].transform(
        lambda values: values.rolling(4, min_periods=2).std(ddof=0)
    )
    frame["margin_stability_4p"] = 1.0 / (1.0 + margin_std)
    frame = frame.drop(columns=["net_margin"])
    return frame


def merge_fundamentals_asof(
    prices: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()

    base = prices.copy()
    if fundamentals.empty:
        for column in MERGED_FUNDAMENTAL_COLUMNS:
            base[column] = False if column == "has_fundamentals" else np.nan
        return base

    enriched = add_derived_fundamental_fields(fundamentals)
    merged_frames: list[pd.DataFrame] = []

    for symbol, group in base.groupby("symbol", sort=False):
        left = group.sort_values("date").copy()
        right = enriched[enriched["symbol"] == str(symbol)].copy()
        if right.empty:
            for column in MERGED_FUNDAMENTAL_COLUMNS:
                left[column] = False if column == "has_fundamentals" else np.nan
            merged_frames.append(left)
            continue

        right = (
            right.rename(
                columns={
                    "period_end": "fundamental_period_end",
                    "filing_date": "fundamental_filing_date",
                }
            )
            .sort_values("fundamental_filing_date")
            .reset_index(drop=True)
        )
        right["has_fundamentals"] = True
        keep = [column for column in MERGED_FUNDAMENTAL_COLUMNS if column in right]

        merged = pd.merge_asof(
            left,
            right[keep],
            left_on="date",
            right_on="fundamental_filing_date",
            direction="backward",
        )
        merged["has_fundamentals"] = (
            merged["has_fundamentals"].fillna(False).astype(bool)
        )
        merged["fundamental_row_age_days"] = (
            merged["date"] - merged["fundamental_filing_date"]
        ).dt.days
        merged_frames.append(merged)

    result = pd.concat(merged_frames, ignore_index=True)
    market_cap = result["shares_outstanding"] * result["close"]
    net_income_base = (
        result["net_income_ttm"]
        if "net_income_ttm" in result.columns
        else result["net_income"]
    )
    result["earnings_yield"] = _safe_div(net_income_base, market_cap)
    result["pe_ratio"] = _safe_div(market_cap, net_income_base.where(net_income_base > 0.0))
    result["book_to_price"] = _safe_div(result["equity"], market_cap)
    result["pb_ratio"] = _safe_div(market_cap, result["equity"].where(result["equity"] > 0.0))
    return result
