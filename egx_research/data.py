from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import re

from egx_research.config import AppConfig


COLUMN_ALIASES = {
    "date": "date",
    "time": "date",
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "price": "close",
    "last": "close",
    "volume": "volume",
    "vol.": "volume",
    "vol": "volume",
}


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _is_stockanalysis_history_url(value: str) -> bool:
    return value.startswith("https://stockanalysis.com/quote/egx/") and value.rstrip("/").endswith("/history")


def _parse_numeric(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "null", "none"}:
        return None
    if text.endswith("%"):
        text = text[:-1]

    multiplier = 1.0
    if text[-1:].upper() in {"K", "M", "B"}:
        suffix = text[-1:].upper()
        text = text[:-1]
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[suffix]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    slash_mask = text.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    if slash_mask.any():
        parts = text[slash_mask].str.split("/", expand=True).astype(int)
        first_gt_12 = bool((parts[0] > 12).any())
        second_gt_12 = bool((parts[1] > 12).any())
        if second_gt_12 and not first_gt_12:
            return pd.to_datetime(series, errors="coerce", dayfirst=False)
        if first_gt_12 and not second_gt_12:
            return pd.to_datetime(series, errors="coerce", dayfirst=True)

    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().mean() > 0.25:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        key = column.strip().lower()
        if key in COLUMN_ALIASES:
            renamed[column] = COLUMN_ALIASES[key]
    df = df.rename(columns=renamed).copy()
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _canonicalize_columns(df)
    if "date" not in df.columns:
        raise ValueError("Missing date column.")

    df["date"] = _parse_dates(df["date"])
    for column in ("open", "high", "low", "close", "volume"):
        if column in df.columns:
            df[column] = df[column].map(_parse_numeric)

    if "close" not in df.columns:
        raise ValueError("Missing close/price column.")

    for column in ("open", "high", "low"):
        if column not in df.columns:
            df[column] = df["close"]
        else:
            df[column] = df[column].fillna(df["close"])

    if "volume" not in df.columns:
        df["volume"] = 0.0

    normalized = (
        df[["date", "open", "high", "low", "close", "volume"]]
        .dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    normalized = normalized[normalized["close"] > 0].copy()
    normalized["high"] = normalized[["high", "open", "close"]].max(axis=1)
    normalized["low"] = normalized[["low", "open", "close"]].min(axis=1)
    normalized["volume"] = normalized["volume"].fillna(0.0)
    return normalized


def load_csv(source: str) -> pd.DataFrame:
    return pd.read_csv(source)


def load_stockanalysis_history(source: str) -> pd.DataFrame:
    response = requests.get(source, timeout=30)
    response.raise_for_status()
    html = response.text
    pattern = re.compile(
        r'\{a:(?P<adj>-?\d+(?:\.\d+)?),c:(?P<close>-?\d+(?:\.\d+)?),h:(?P<high>-?\d+(?:\.\d+)?),'
        r'l:(?P<low>-?\d+(?:\.\d+)?),o:(?P<open>-?\d+(?:\.\d+)?),t:"(?P<date>\d{4}-\d{2}-\d{2})",'
        r'v:(?P<volume>-?\d+(?:\.\d+)?)'
    )
    rows = [match.groupdict() for match in pattern.finditer(html)]
    if not rows:
        raise ValueError("Could not parse StockAnalysis history payload.")
    df = pd.DataFrame(rows)
    return df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})


def ingest_source(source: str, config: AppConfig, symbol: str | None = None) -> dict[str, str]:
    symbol = symbol or config.data.symbol
    raw_dir = Path(config.data.raw_dir)
    normalized_dir = Path(config.data.normalized_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    raw_target = raw_dir / f"{symbol}.csv"
    normalized_target = normalized_dir / f"{symbol}.csv"

    if _is_stockanalysis_history_url(source):
        raw_df = load_stockanalysis_history(source)
        raw_df.to_csv(raw_target, index=False)
    elif _is_url(source):
        raw_df = load_csv(source)
        raw_df.to_csv(raw_target, index=False)
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source_path, raw_target)
        raw_df = load_csv(str(source_path))

    normalized = normalize_ohlcv(raw_df)
    normalized.to_csv(normalized_target, index=False)
    return {
        "symbol": symbol,
        "raw_path": str(raw_target),
        "normalized_path": str(normalized_target),
        "rows": str(len(normalized)),
    }


def load_price_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df
