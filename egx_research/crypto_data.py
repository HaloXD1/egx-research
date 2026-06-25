from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from egx_research.crypto_config import CryptoConfig
from egx_research.utils import ensure_dir, write_json
from egx_research.crypto_sources import (
    SOURCE_REGISTRY,
    get_source_api_key,
    get_source_env_var,
    is_source_enabled,
    is_source_required,
    source_requires_credentials,
)



BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OPTIONAL_FEATURE_PREFIXES = {
    "open_interest.csv": "derivatives",
    "futures_positioning.csv": "derivatives",
    "coinbase_premium.csv": "spot",
    "stablecoin_supply.csv": "liquidity",
    "exchange_stablecoin_reserves.csv": "liquidity",
    "options_skew.csv": "options",
    "btc_etf_flows.csv": "etf",
    "liquidations.csv": "derivatives",
}

OPTIONAL_CANONICAL_COLUMNS = {
    "open_interest.csv": {"derivatives_open_interest", "derivatives_open_interest_value"},
    "futures_positioning.csv": {
        "derivatives_open_interest",
        "derivatives_basis",
        "derivatives_taker_buy_sell_ratio",
        "derivatives_long_short_ratio",
        "derivatives_leverage_ratio",
    },
    "coinbase_premium.csv": {"spot_coinbase_close", "spot_coinbase_premium"},
    "stablecoin_supply.csv": {"liquidity_stablecoin_supply"},
    "exchange_stablecoin_reserves.csv": {"liquidity_exchange_stablecoin_reserves"},
    "options_skew.csv": {
        "options_options_skew",
        "options_put_call_ratio",
        "options_dvol",
        "options_25d_skew",
        "options_put_call_oi",
        "options_put_call_volume",
        "options_iv_30d",
        "options_term_structure",
    },
    "btc_etf_flows.csv": {
        "etf_flow_ibit",
        "etf_flow_fbtc",
        "etf_flow_arkb",
        "etf_flow_bitb",
        "etf_flow_gbtc",
        "etf_net_flow_usd",
    },
    "liquidations.csv": {
        "derivatives_long_liq_usd",
        "derivatives_short_liq_usd",
        "derivatives_total_liq_usd",
        "derivatives_liq_imbalance",
        "derivatives_heatmap_nearest_down_liq",
        "derivatives_heatmap_nearest_up_liq",
    },
}

CURRENT_SNAPSHOT_COLUMNS = {"options_options_skew", "options_put_call_ratio"}


def _date_to_ms(value: str | pd.Timestamp | datetime) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _utc_day(value: Any, unit: str | None = None) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, unit=unit, utc=True)
    return pd.Timestamp(timestamp).tz_convert(None).normalize()


def parse_binance_klines(payload: list[list[Any]]) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    rows = []
    for item in payload:
        if len(item) < 6:
            raise ValueError("Binance kline row has fewer than 6 fields.")
        rows.append(
            {
                "date": _utc_day(item[0], unit="ms"),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]) if len(item) > 7 else np.nan,
                "trade_count": int(item[8]) if len(item) > 8 else 0,
            }
        )

    frame = pd.DataFrame(rows)
    frame = (
        frame.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    frame["high"] = frame[["high", "open", "close"]].max(axis=1)
    frame["low"] = frame[["low", "open", "close"]].min(axis=1)
    return frame


def parse_coinmetrics_payload(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("data", []):
        row = {"date": _utc_day(item["time"])}
        for key, value in item.items():
            if key in {"asset", "time"} or key.endswith("-status") or key.endswith("-status-time"):
                continue
            row[key] = pd.to_numeric(value, errors="coerce")
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["date"])
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def parse_fear_greed_payload(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in payload.get("data", []):
        rows.append(
            {
                "date": _utc_day(int(item["timestamp"]), unit="s"),
                "fear_greed_value": float(item["value"]),
                "fear_greed_classification": str(item.get("value_classification", "")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "fear_greed_value", "fear_greed_classification"])
    return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")


def parse_fred_csv(text: str, series_id: str, column_name: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(text))
    if "observation_date" not in frame.columns or series_id not in frame.columns:
        raise ValueError(f"Unexpected FRED CSV for {series_id}.")
    frame = frame.rename(columns={"observation_date": "date", series_id: column_name})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[column_name] = pd.to_numeric(frame[column_name].replace(".", np.nan), errors="coerce")
    return (
        frame[["date", column_name]]
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def parse_funding_payload(payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in payload:
        rows.append(
            {
                "date": _utc_day(item["fundingTime"], unit="ms"),
                "funding_rate": float(item["fundingRate"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["date", "funding_rate_mean", "funding_rate_sum", "funding_rate_count"])
    raw = pd.DataFrame(rows)
    daily = raw.groupby("date", as_index=False).agg(
        funding_rate_mean=("funding_rate", "mean"),
        funding_rate_sum=("funding_rate", "sum"),
        funding_rate_count=("funding_rate", "count"),
    )
    return daily.sort_values("date").reset_index(drop=True)


def _get_json(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_text(url: str, params: dict[str, Any] | None = None) -> str:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_binance_klines(config: CryptoConfig) -> pd.DataFrame:
    url = f"{config.sources.binance_spot_base_url.rstrip('/')}/api/v3/klines"
    start_ms = _date_to_ms(config.sources.price_start)
    end_ms = _date_to_ms(pd.Timestamp.utcnow().normalize())
    all_rows: list[list[Any]] = []

    while start_ms <= end_ms:
        payload = _get_json(
            url,
            {
                "symbol": config.data.symbol,
                "interval": config.data.interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not payload:
            break
        all_rows.extend(payload)
        next_start = int(payload[-1][0]) + 24 * 60 * 60 * 1000
        if next_start <= start_ms:
            break
        start_ms = next_start

    return parse_binance_klines(all_rows)


def fetch_coinmetrics(config: CryptoConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    skipped = []
    url = f"{config.sources.coinmetrics_base_url.rstrip('/')}/v4/timeseries/asset-metrics"

    for metric in config.sources.coinmetrics_metrics:
        params = {
            "assets": config.data.asset,
            "metrics": metric,
            "frequency": "1d",
            "start_time": config.sources.onchain_start,
            "page_size": 10000,
        }
        metric_rows = []
        next_url: str | None = url
        next_params: dict[str, Any] | None = params
        try:
            while next_url:
                payload = _get_json(next_url, next_params)
                metric_rows.extend(payload.get("data", []))
                next_url = payload.get("next_page_url")
                next_params = None
        except requests.HTTPError as exc:
            skipped.append({"metric": metric, "error": str(exc)})
            continue

        frame = parse_coinmetrics_payload({"data": metric_rows})
        if metric in frame.columns:
            frames.append(frame[["date", metric]])
        else:
            skipped.append({"metric": metric, "error": "empty"})

    if not frames:
        return pd.DataFrame(columns=["date"]), {"skipped_metrics": skipped}

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged, {"skipped_metrics": skipped, "metrics_loaded": [c for c in merged.columns if c != "date"]}


def fetch_fear_greed(config: CryptoConfig) -> pd.DataFrame:
    payload = _get_json(config.sources.fear_greed_url, {"limit": 0, "format": "json"})
    return parse_fear_greed_payload(payload)


def _parse_flow_value(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    parsed = pd.to_numeric(text, errors="coerce")
    if pd.isna(parsed):
        return 0.0
    amount = float(parsed)
    return -amount if negative else amount


def _validate_etf_data(df: pd.DataFrame) -> list[str]:
    warnings = []
    required_cols = [
        "date", "etf_flow_ibit", "etf_flow_fbtc", "etf_flow_arkb",
        "etf_flow_bitb", "etf_flow_gbtc", "etf_net_flow_usd"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required ETF column '{col}' is missing.")

    # Check if all numeric columns are actually numeric
    for col in required_cols:
        if col != "date":
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"ETF column '{col}' is not numeric.")
            if df[col].isna().all():
                raise ValueError(f"ETF column '{col}' contains only NaN values.")

    # Check dates are valid, sorted, and not in the future
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("ETF 'date' column is not a datetime type.")

    today_utc = pd.Timestamp.now(UTC).tz_localize(None).normalize()
    future_dates = df[df["date"] > today_utc]
    if not future_dates.empty:
        raise ValueError(f"ETF data contains future dates: {future_dates['date'].tolist()}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("ETF data dates are not sorted in ascending order.")

    if df["date"].duplicated().any():
        raise ValueError("ETF data contains duplicate dates.")

    # Unit check (warning only)
    non_zeros = df.loc[df["etf_net_flow_usd"] != 0, "etf_net_flow_usd"]
    if not non_zeros.empty:
        avg_abs_flow = non_zeros.abs().mean()
        if avg_abs_flow < 100_000.0:
            warnings.append(
                f"ETF net flow values seem unscaled or incorrect (avg absolute non-zero flow is ${avg_abs_flow:.2f})."
            )

    return warnings


def fetch_btc_etf_flows(config: CryptoConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = config.sources.bitcoin_etf_flows_url
    raw_dir = Path(config.data.raw_dir)
    fallback_path = raw_dir / "btc_etf_flows.csv"

    # Try fetching public URL
    try:
        text = _get_text(url)
        tables = pd.read_html(StringIO(text))

        parsed_df = None
        warnings = []
        for table in tables:
            # Match columns case-insensitively and with substring matches
            col_mapping = {}
            for original_col in table.columns:
                cleaned = str(original_col).strip().replace("US$m", "").replace(" ", "").lower()
                if cleaned == "date":
                    col_mapping["date"] = original_col
                elif cleaned == "total" or cleaned == "netflow":
                    col_mapping["total"] = original_col
                elif "ibit" in cleaned:
                    col_mapping["ibit"] = original_col
                elif "fbtc" in cleaned:
                    col_mapping["fbtc"] = original_col
                elif "arkb" in cleaned:
                    col_mapping["arkb"] = original_col
                elif "bitb" in cleaned:
                    col_mapping["bitb"] = original_col
                elif "gbtc" in cleaned:
                    col_mapping["gbtc"] = original_col

            # Check if all required columns were found in this table
            required_keys = ["date", "total", "ibit", "fbtc", "arkb", "bitb", "gbtc"]
            if all(k in col_mapping for k in required_keys):
                frame = pd.DataFrame(index=table.index)
                frame["date"] = pd.to_datetime(table[col_mapping["date"]], errors="coerce")
                frame["etf_flow_ibit"] = table[col_mapping["ibit"]].map(_parse_flow_value) * 1_000_000.0
                frame["etf_flow_fbtc"] = table[col_mapping["fbtc"]].map(_parse_flow_value) * 1_000_000.0
                frame["etf_flow_arkb"] = table[col_mapping["arkb"]].map(_parse_flow_value) * 1_000_000.0
                frame["etf_flow_bitb"] = table[col_mapping["bitb"]].map(_parse_flow_value) * 1_000_000.0
                frame["etf_flow_gbtc"] = table[col_mapping["gbtc"]].map(_parse_flow_value) * 1_000_000.0
                frame["etf_net_flow_usd"] = table[col_mapping["total"]].map(_parse_flow_value) * 1_000_000.0

                # Tolerates missing days (weekends, holidays) - drop NaT dates
                frame = frame.dropna(subset=["date"])
                if frame.empty:
                    continue

                frame = (
                    frame.sort_values("date")
                    .drop_duplicates("date", keep="last")
                    .reset_index(drop=True)
                )

                # Perform validation
                warnings = _validate_etf_data(frame)
                parsed_df = frame
                break

        if parsed_df is not None:
            return parsed_df, {
                "source_url": url,
                "status": "success",
                "rows": len(parsed_df),
                "columns": list(parsed_df.columns),
                "warnings": warnings,
            }
        else:
            raise ValueError("No table in public ETF URL matched the required schema.")

    except Exception as exc:
        # Fallback to local CSV (explicit status: fallback_local)
        if fallback_path.exists():
            try:
                fallback_df = pd.read_csv(fallback_path, parse_dates=["date"])
                warnings = _validate_etf_data(fallback_df)
                return fallback_df, {
                    "source_url": url,
                    "status": "fallback_local",
                    "rows": len(fallback_df),
                    "columns": list(fallback_df.columns),
                    "warnings": warnings,
                    "error": str(exc),
                }
            except Exception as fallback_exc:
                raise ValueError(
                    f"Public ETF fetch failed: {exc}. Local CSV fallback read failed: {fallback_exc}"
                ) from exc
        else:
            raise ValueError(
                f"Public ETF fetch failed: {exc}. No local CSV fallback found at {fallback_path}"
            ) from exc


def fetch_macro(config: CryptoConfig) -> pd.DataFrame:
    frames = []
    for series_id, label in config.sources.macro_series.items():
        text = _get_text(config.sources.fred_base_url, {"id": series_id})
        frame = parse_fred_csv(text, series_id=series_id, column_name=f"macro_{label}")
        frame = frame[frame["date"] >= pd.Timestamp(config.sources.macro_start)]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def fetch_funding_rates(config: CryptoConfig) -> pd.DataFrame:
    url = f"{config.sources.binance_futures_base_url.rstrip('/')}/fapi/v1/fundingRate"
    start_ms = _date_to_ms(config.sources.funding_start)
    end_ms = _date_to_ms(pd.Timestamp.utcnow().normalize())
    rows: list[dict[str, Any]] = []

    while start_ms <= end_ms:
        payload = _get_json(
            url,
            {
                "symbol": config.data.symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not payload:
            break
        rows.extend(payload)
        next_start = int(payload[-1]["fundingTime"]) + 1
        if next_start <= start_ms:
            break
        start_ms = next_start

    return parse_funding_payload(rows)


def fetch_open_interest(config: CryptoConfig) -> pd.DataFrame:
    """Fetch daily BTC open interest from Binance Futures."""
    url = f"{config.sources.binance_futures_base_url.rstrip('/')}/futures/data/openInterestHist"
    end_ms = _date_to_ms(pd.Timestamp.utcnow().normalize())
    start_ms = _date_to_ms("2020-01-01")
    all_rows: list[dict[str, Any]] = []

    cursor_ms = end_ms
    while cursor_ms >= start_ms:
        payload = _get_json(
            url,
            {
                "symbol": config.data.symbol,
                "period": "1d",
                "limit": 500,
                "endTime": cursor_ms,
            },
        )
        if not payload:
            break
        all_rows.extend(payload)
        earliest = min(int(item["timestamp"]) for item in payload)
        next_cursor = earliest - 1
        if next_cursor >= cursor_ms:
            break
        cursor_ms = next_cursor

    if not all_rows:
        return pd.DataFrame(columns=["date", "open_interest", "open_interest_value"])

    rows = []
    for item in all_rows:
        rows.append(
            {
                "date": _utc_day(int(item["timestamp"]), unit="ms"),
                "open_interest": float(item["sumOpenInterest"]),
                "open_interest_value": float(item["sumOpenInterestValue"]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _load_futures_positioning_csv(path: Path) -> pd.DataFrame:
    columns = [
        "date",
        "derivatives_open_interest",
        "derivatives_basis",
        "derivatives_taker_buy_sell_ratio",
        "derivatives_long_short_ratio",
        "derivatives_leverage_ratio",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, parse_dates=["date"])
    aliases = {
        "open_interest": "derivatives_open_interest",
        "basis": "derivatives_basis",
        "taker_buy_sell_ratio": "derivatives_taker_buy_sell_ratio",
        "long_short_ratio": "derivatives_long_short_ratio",
        "leverage_ratio": "derivatives_leverage_ratio",
    }
    frame = frame.rename(columns=aliases)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _fetch_binance_futures_data(
    config: CryptoConfig,
    endpoint: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    url = f"{config.sources.binance_futures_base_url.rstrip('/')}{endpoint}"
    start_ms = _date_to_ms(config.sources.funding_start)
    cursor_ms = _date_to_ms(pd.Timestamp.utcnow().normalize())
    rows: list[dict[str, Any]] = []

    while cursor_ms >= start_ms:
        payload = _get_json(
            url,
            {
                **params,
                "period": "1d",
                "limit": 500,
                "endTime": cursor_ms,
            },
        )
        if not payload:
            break
        rows.extend(payload)
        earliest = min(int(item["timestamp"]) for item in payload if "timestamp" in item)
        next_cursor = earliest - 1
        if next_cursor >= cursor_ms:
            break
        cursor_ms = next_cursor

    return rows


def fetch_futures_positioning(config: CryptoConfig) -> pd.DataFrame:
    """Fetch daily futures positioning with local CSV fallback."""
    fallback_path = Path(config.data.raw_dir) / "futures_positioning.csv"
    symbol = config.data.symbol
    frames: list[pd.DataFrame] = []

    try:
        oi_rows = _fetch_binance_futures_data(config, "/futures/data/openInterestHist", {"symbol": symbol})
        if oi_rows:
            frames.append(
                pd.DataFrame(
                    {
                        "date": [_utc_day(int(item["timestamp"]), unit="ms") for item in oi_rows],
                        "derivatives_open_interest": [float(item["sumOpenInterest"]) for item in oi_rows],
                    }
                )
            )

        basis_rows = _fetch_binance_futures_data(
            config,
            "/futures/data/basis",
            {"pair": symbol, "contractType": "PERPETUAL"},
        )
        if basis_rows:
            frames.append(
                pd.DataFrame(
                    {
                        "date": [_utc_day(int(item["timestamp"]), unit="ms") for item in basis_rows],
                        "derivatives_basis": [
                            float(item.get("basisRate", item.get("basis", np.nan)))
                            for item in basis_rows
                        ],
                    }
                )
            )

        taker_rows = _fetch_binance_futures_data(
            config,
            "/futures/data/takerlongshortRatio",
            {"symbol": symbol},
        )
        if taker_rows:
            frames.append(
                pd.DataFrame(
                    {
                        "date": [_utc_day(int(item["timestamp"]), unit="ms") for item in taker_rows],
                        "derivatives_taker_buy_sell_ratio": [
                            float(item["buySellRatio"]) for item in taker_rows
                        ],
                    }
                )
            )

        long_short_rows = _fetch_binance_futures_data(
            config,
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol},
        )
        if long_short_rows:
            frames.append(
                pd.DataFrame(
                    {
                        "date": [_utc_day(int(item["timestamp"]), unit="ms") for item in long_short_rows],
                        "derivatives_long_short_ratio": [
                            float(item["longShortRatio"]) for item in long_short_rows
                        ],
                    }
                )
            )
    except Exception:
        if is_source_required(config, "futures_positioning"):
            raise
        return _load_futures_positioning_csv(fallback_path)

    if not frames:
        return _load_futures_positioning_csv(fallback_path)

    merged = frames[0].sort_values("date").drop_duplicates("date", keep="last")
    for frame in frames[1:]:
        merged = merged.merge(
            frame.sort_values("date").drop_duplicates("date", keep="last"),
            on="date",
            how="outer",
        )
    for column in OPTIONAL_CANONICAL_COLUMNS["futures_positioning.csv"]:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[["date", *sorted(OPTIONAL_CANONICAL_COLUMNS["futures_positioning.csv"])]].sort_values("date").reset_index(drop=True)


def fetch_coinbase_premium(config: CryptoConfig) -> pd.DataFrame:
    """Compute Coinbase premium from Coinbase daily close vs Binance daily close."""
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    cb_start = pd.Timestamp("2019-01-01", tz="UTC")
    cb_end = pd.Timestamp.utcnow().normalize()
    all_rows: list[dict[str, Any]] = []

    cursor = cb_start
    while cursor < cb_end:
        window_end = min(cursor + pd.Timedelta(days=300), cb_end)
        payload = _get_json(
            url,
            {
                "granularity": 86400,
                "start": cursor.isoformat(),
                "end": window_end.isoformat(),
            },
        )
        if not payload:
            cursor = window_end
            continue
        for candle in payload:
            # Coinbase candle: [time, low, high, open, close, volume]
            all_rows.append(
                {
                    "date": _utc_day(int(candle[0]), unit="s"),
                    "coinbase_close": float(candle[4]),
                }
            )
        cursor = window_end

    if not all_rows:
        return pd.DataFrame(columns=["date", "coinbase_close", "coinbase_premium"])

    cb_frame = (
        pd.DataFrame(all_rows)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    # Load Binance normalized price for the close
    binance = pd.read_csv(config.data.normalized_path, parse_dates=["date"])
    binance = binance[["date", "close"]].rename(columns={"close": "binance_close"})

    merged = cb_frame.merge(binance, on="date", how="inner")
    merged["coinbase_premium"] = (
        (merged["coinbase_close"] - merged["binance_close"]) / merged["binance_close"]
    )
    return (
        merged[["date", "coinbase_close", "coinbase_premium"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_stablecoin_supply(config: CryptoConfig) -> pd.DataFrame:
    """Fetch total stablecoin supply (USDT + USDC) from DefiLlama."""
    fallback_path = Path(config.data.raw_dir) / "stablecoin_supply.csv"
    try:
        frames: list[pd.DataFrame] = []
        for stablecoin_id in (1, 2):  # 1=USDT, 2=USDC
            payload = _get_json(
                f"https://stablecoins.llama.fi/stablecoincharts/all?stablecoin={stablecoin_id}"
            )
            if not payload:
                continue
            rows = []
            for item in payload:
                pegged = item.get("totalCirculatingUSD", {}).get("peggedUSD")
                if pegged is None:
                    continue
                rows.append(
                    {
                        "date": _utc_day(int(item["date"]), unit="s"),
                        "supply": float(pegged),
                    }
                )
            if rows:
                frame = (
                    pd.DataFrame(rows)
                    .sort_values("date")
                    .drop_duplicates("date", keep="last")
                    .reset_index(drop=True)
                )
                frames.append(frame)

        if not frames:
            raise ValueError("No stablecoin supply frames retrieved from DefiLlama")

        if len(frames) == 1:
            result = frames[0].rename(columns={"supply": "stablecoin_supply"})
        else:
            merged = frames[0].merge(frames[1], on="date", how="outer", suffixes=("_usdt", "_usdc"))
            merged["stablecoin_supply"] = merged["supply_usdt"].fillna(0) + merged["supply_usdc"].fillna(0)
            result = merged[["date", "stablecoin_supply"]]

        return result.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        if fallback_path.exists():
            print(f"Warning: DefiLlama stablecoin fetch failed: {exc}. Using cached CSV fallback.")
            try:
                df = pd.read_csv(fallback_path)
                if "date" in df.columns and not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    return df
            except Exception:
                pass
        print(f"Warning: DefiLlama stablecoin fetch failed and no cache exists: {exc}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["date", "stablecoin_supply"])


def fetch_exchange_stablecoin_reserves(config: CryptoConfig) -> pd.DataFrame:
    """Fetch exchange stablecoin reserves (CryptoQuant or local CSV fallback)."""
    import os
    fallback_path = Path(config.data.raw_dir) / "exchange_stablecoin_reserves.csv"

    api_key = get_source_api_key("exchange_stablecoin_reserves", get_source_env_var(config, "exchange_stablecoin_reserves"))
    if not api_key:
        api_key = os.getenv("CRYPTOQUANT_API_KEY")

    if api_key:
        url = "https://api.cryptoquant.com/v1/btc/exchange-flows/stablecoin-reserves"
        try:
            payload = _get_json(url, {"limit": 1000, "token": api_key})
            rows = []
            data_list = []
            if isinstance(payload, dict) and "result" in payload:
                data_list = payload["result"].get("data", [])
            elif isinstance(payload, list):
                data_list = payload
            for item in data_list:
                date_val = item.get("date") or item.get("time") or item.get("timestamp")
                reserves = item.get("exchange_stablecoin_reserves") or item.get("reserves") or item.get("value")
                if date_val is not None and reserves is not None:
                    rows.append({
                        "date": _utc_day(date_val),
                        "exchange_stablecoin_reserves": float(reserves)
                    })
            if rows:
                return pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        except Exception as exc:
            print(f"Warning: Exchange stablecoin reserves API fetch failed: {exc}")

    if fallback_path.exists():
        try:
            df = pd.read_csv(fallback_path)
            for col in ["liquidity_exchange_stablecoin_reserves", "exchange_stablecoin_reserves"]:
                if col in df.columns:
                    df = df.rename(columns={col: "exchange_stablecoin_reserves"})
                    df["date"] = pd.to_datetime(df["date"])
                    return df[["date", "exchange_stablecoin_reserves"]].sort_values("date").reset_index(drop=True)
        except Exception as exc:
            print(f"Warning: Failed to read local exchange stablecoin reserves CSV: {exc}")

    print("Warning: Exchange stablecoin reserves fetch failed and no cache exists. Returning empty DataFrame.")
    return pd.DataFrame(columns=["date", "exchange_stablecoin_reserves"])


def fetch_deribit_options(config: CryptoConfig) -> pd.DataFrame:
    """Fetch daily options data. Falls back to limited DVOL/snapshot if no credentials present."""
    from egx_research.crypto_sources import get_source_env_var, get_source_api_key
    env_var = get_source_env_var(config, "options_skew")
    api_key = get_source_api_key("options_skew", env_var)

    if api_key:
        url = getattr(config.sources, "historical_options_url", "https://www.deribit.com/api/v2/private/get_historical_options")
        try:
            payload = _get_json(url, {"currency": "BTC"}, headers={"Authorization": f"Bearer {api_key}"})
            if payload and "result" in payload:
                rows = []
                for entry in payload["result"]:
                    rows.append({
                        "date": _utc_day(entry["timestamp"], unit="ms" if isinstance(entry["timestamp"], int) else None),
                        "options_skew": float(entry.get("skew", np.nan)),
                        "put_call_ratio": float(entry.get("put_call_ratio", np.nan)),
                        "dvol": float(entry.get("dvol", np.nan)),
                        "options_25d_skew": float(entry.get("options_25d_skew", np.nan)),
                        "options_put_call_oi": float(entry.get("options_put_call_oi", np.nan)),
                        "options_put_call_volume": float(entry.get("options_put_call_volume", np.nan)),
                        "options_iv_30d": float(entry.get("options_iv_30d", np.nan)),
                        "options_term_structure": float(entry.get("options_term_structure", np.nan)),
                        "options_dvol": float(entry.get("options_dvol", np.nan)),
                    })
                return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        except Exception as exc:
            if is_source_required(config, "options_skew"):
                raise
            pass

    # --- Limited DVOL/snapshot fallback only ---
    # --- DVOL history (last 365 days) ---
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_ms = now_ms - 365 * 24 * 60 * 60 * 1000
    dvol_payload = _get_json(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        {
            "currency": "BTC",
            "resolution": 86400,
            "start_timestamp": start_ms,
            "end_timestamp": now_ms,
        },
    )
    dvol_rows = []
    for entry in dvol_payload.get("result", {}).get("data", []):
        # entry: [timestamp, open, high, low, close]
        dvol_rows.append(
            {
                "date": _utc_day(int(entry[0]), unit="ms"),
                "dvol": float(entry[4]),
            }
        )
    dvol_frame = (
        pd.DataFrame(dvol_rows, columns=["date", "dvol"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    # --- Put/Call snapshot (current state only) ---
    book_payload = _get_json(
        "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
        {"currency": "BTC", "kind": "option"},
    )
    instruments = book_payload.get("result", [])
    put_oi = 0.0
    call_oi = 0.0
    put_volume = 0.0
    call_volume = 0.0
    for inst in instruments:
        name = str(inst.get("instrument_name", ""))
        oi = float(inst.get("open_interest", 0) or 0)
        vol = float(inst.get("volume", 0) or 0)
        if name.endswith("-P"):
            put_oi += oi
            put_volume += vol
        elif name.endswith("-C"):
            call_oi += oi
            call_volume += vol

    put_call_ratio = put_oi / call_oi if call_oi > 0 else np.nan
    options_skew = (put_call_ratio - 1.0) if not np.isnan(put_call_ratio) else np.nan
    put_call_volume = put_volume / call_volume if call_volume > 0 else np.nan
    today = pd.Timestamp.utcnow().normalize().tz_convert(None)

    snapshot = pd.DataFrame(
        [{
            "date": today,
            "options_skew": options_skew,
            "put_call_ratio": put_call_ratio,
            "options_25d_skew": options_skew,
            "options_put_call_oi": put_call_ratio,
            "options_put_call_volume": put_call_volume,
            "options_iv_30d": np.nan,
            "options_term_structure": np.nan,
        }]
    )

    # Merge DVOL history with today's snapshot
    if dvol_frame.empty:
        result = snapshot.copy()
        result["dvol"] = np.nan
        result["options_dvol"] = np.nan
    else:
        dvol_frame["options_dvol"] = dvol_frame["dvol"]
        result = dvol_frame.merge(snapshot, on="date", how="outer")
        result = result.sort_values("date").reset_index(drop=True)
        if "options_dvol_x" in result.columns or "options_dvol_y" in result.columns:
            result["options_dvol"] = result.get("options_dvol_x", pd.Series(np.nan, index=result.index)).combine_first(
                result.get("options_dvol_y", pd.Series(np.nan, index=result.index))
            )
            result = result.drop(columns=[c for c in ("options_dvol_x", "options_dvol_y") if c in result.columns])

    for col in (
        "options_skew", "put_call_ratio", "dvol",
        "options_25d_skew", "options_put_call_oi", "options_put_call_volume",
        "options_iv_30d", "options_term_structure", "options_dvol"
    ):
        if col not in result.columns:
            result[col] = np.nan

    return result[[
        "date", "options_skew", "put_call_ratio", "dvol",
        "options_25d_skew", "options_put_call_oi", "options_put_call_volume",
        "options_iv_30d", "options_term_structure", "options_dvol"
    ]].reset_index(drop=True)


def parse_liquidations_payload(payload: Any) -> pd.DataFrame:
    data_list = payload
    if isinstance(payload, dict):
        data_list = payload.get("data", payload)

    if not isinstance(data_list, list):
        raise ValueError("Liquidations payload data is not a list.")

    rows = []
    for item in data_list:
        if not isinstance(item, dict):
            continue
        date_raw = item.get("date") or item.get("time") or item.get("timestamp")
        if date_raw is None:
            continue
        if isinstance(date_raw, (int, float)):
            if date_raw > 1e11:
                date_val = _utc_day(date_raw, unit="ms")
            else:
                date_val = _utc_day(date_raw, unit="s")
        else:
            date_val = _utc_day(date_raw)

        long_liq = float(item.get("long_liq_usd") or item.get("buyVolUsd") or item.get("long_liquidations") or 0.0)
        short_liq = float(item.get("short_liq_usd") or item.get("sellVolUsd") or item.get("short_liquidations") or 0.0)
        total_liq = float(item.get("total_liq_usd") or item.get("totalVolUsd") or (long_liq + short_liq))

        if "liq_imbalance" in item:
            liq_imbalance = float(item["liq_imbalance"])
        else:
            denom = long_liq + short_liq
            liq_imbalance = (long_liq - short_liq) / denom if denom > 0 else 0.0

        down_liq = item.get("heatmap_nearest_down_liq") or item.get("nearest_down_liq") or item.get("heatmap_down")
        up_liq = item.get("heatmap_nearest_up_liq") or item.get("nearest_up_liq") or item.get("heatmap_up")

        row = {
            "date": date_val,
            "long_liq_usd": long_liq,
            "short_liq_usd": short_liq,
            "total_liq_usd": total_liq,
            "liq_imbalance": liq_imbalance,
            "heatmap_nearest_down_liq": float(down_liq) if down_liq is not None else np.nan,
            "heatmap_nearest_up_liq": float(up_liq) if up_liq is not None else np.nan,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "date", "long_liq_usd", "short_liq_usd", "total_liq_usd",
            "liq_imbalance", "heatmap_nearest_down_liq", "heatmap_nearest_up_liq"
        ])

    df = pd.DataFrame(rows)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_liquidations(config: CryptoConfig) -> pd.DataFrame:
    """Fetch liquidations from Coinglass or configured provider URL."""
    columns = [
        "date",
        "long_liq_usd",
        "short_liq_usd",
        "total_liq_usd",
        "liq_imbalance",
        "heatmap_nearest_down_liq",
        "heatmap_nearest_up_liq",
    ]
    fallback_path = Path(config.data.raw_dir) / "liquidations.csv"

    def load_local_fallback() -> pd.DataFrame:
        if not fallback_path.exists():
            return pd.DataFrame(columns=columns)
        frame = pd.read_csv(fallback_path, parse_dates=["date"])
        rename_map = {
            "derivatives_long_liq_usd": "long_liq_usd",
            "derivatives_short_liq_usd": "short_liq_usd",
            "derivatives_total_liq_usd": "total_liq_usd",
            "derivatives_liq_imbalance": "liq_imbalance",
            "derivatives_heatmap_nearest_down_liq": "heatmap_nearest_down_liq",
            "derivatives_heatmap_nearest_up_liq": "heatmap_nearest_up_liq",
        }
        frame = frame.rename(columns=rename_map)
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[columns].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    url = getattr(config.sources, "liquidations_url", None)
    if not url:
        return load_local_fallback()

    api_key = get_source_api_key("liquidations", get_source_env_var(config, "liquidations"))
    if not api_key:
        return load_local_fallback()

    headers = {}
    headers["coinglassSecret"] = api_key
    headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(url, params={"symbol": "BTC"}, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return parse_liquidations_payload(payload)
    except Exception:
        if is_source_required(config, "liquidations"):
            raise
        return load_local_fallback()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _canonical_optional_columns(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    prefix = OPTIONAL_FEATURE_PREFIXES.get(path.name)
    if prefix is None:
        return frame
    canonical_columns = OPTIONAL_CANONICAL_COLUMNS.get(path.name, set())
    return frame.rename(
        columns={
            column: f"{prefix}_{column}"
            for column in frame.columns
            if column != "date" and column not in canonical_columns
        }
    )


def _should_keep_current_snapshot(panel: pd.DataFrame, column: str) -> bool:
    if column not in CURRENT_SNAPSHOT_COLUMNS:
        return False
    non_null = panel[column].notna()
    if int(non_null.sum()) != 1:
        return False
    latest_date = pd.Timestamp(panel["date"].max())
    latest_value = panel.loc[pd.to_datetime(panel["date"]) == latest_date, column]
    return bool(latest_value.notna().any())


def load_crypto_price_data(config: CryptoConfig) -> pd.DataFrame:
    frame = pd.read_csv(config.data.normalized_path, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def build_crypto_feature_panel(config: CryptoConfig) -> pd.DataFrame:
    price = load_crypto_price_data(config)
    price = price.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    panel = price.copy()
    raw_dir = Path(config.data.raw_dir)

    optional_files = [
        raw_dir / "coinmetrics_btc.csv",
        raw_dir / "fear_greed.csv",
        raw_dir / "macro_fred.csv",
        raw_dir / "funding_rates.csv",
        raw_dir / "btc_etf_flows.csv",
        raw_dir / "open_interest.csv",
        raw_dir / "futures_positioning.csv",
        raw_dir / "coinbase_premium.csv",
        raw_dir / "stablecoin_supply.csv",
        raw_dir / "exchange_stablecoin_reserves.csv",
        raw_dir / "options_skew.csv",
        raw_dir / "liquidations.csv",
    ]
    for path in optional_files:
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = _canonical_optional_columns(path, frame)
        panel = panel.merge(frame, on="date", how="left")

    macro_columns = [column for column in panel.columns if column.startswith("macro_")]
    if macro_columns:
        panel[macro_columns] = panel[macro_columns].ffill()

    base_columns = {"date", "open", "high", "low", "close", "volume", "quote_volume", "trade_count"}
    external_columns = [column for column in panel.columns if column not in base_columns]
    for column in external_columns:
        if _should_keep_current_snapshot(panel, column):
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
        elif column == "fear_greed_classification":
            panel[column] = panel[column].shift(1)
        else:
            panel[column] = pd.to_numeric(panel[column], errors="coerce").shift(1)

    # Compute lag-safe dry powder ratio after shifting loop
    if "liquidity_stablecoin_supply" in panel.columns:
        sply = panel["SplyCur"] if "SplyCur" in panel.columns else pd.Series(19500000.0, index=panel.index)
        sply = sply.fillna(19500000.0).replace(0, 19500000.0)
        prev_close = panel["close"].shift(1)
        btc_mcap_lagged = prev_close * sply
        panel["liquidity_dry_powder_ratio"] = panel["liquidity_stablecoin_supply"] / btc_mcap_lagged.replace(0, np.nan)
    else:
        panel["liquidity_dry_powder_ratio"] = np.nan

    if "liquidity_dry_powder_ratio" not in external_columns:
        external_columns.append("liquidity_dry_powder_ratio")

    panel["return_1d"] = panel["close"].pct_change()
    panel["return_7d"] = panel["close"].pct_change(7)
    panel["realized_vol_30d"] = panel["return_1d"].rolling(30, min_periods=15).std(ddof=0) * np.sqrt(365)
    panel["btc_feature_complete"] = panel[external_columns].notna().mean(axis=1) if external_columns else 0.0

    features_path = Path(config.data.features_path)
    _write_csv(features_path, panel)
    _write_feature_quality(config, panel, optional_files)
    return panel


def _write_feature_quality(config: CryptoConfig, panel: pd.DataFrame, optional_files: list[Path]) -> None:
    rows = []
    for column in panel.columns:
        if column == "date":
            continue
        non_null = panel[column].notna()
        rows.append(
            {
                "column": column,
                "non_null_rows": int(non_null.sum()),
                "coverage": float(non_null.mean()) if len(panel) else 0.0,
                "first_date": str(panel.loc[non_null, "date"].min().date()) if non_null.any() else "",
                "last_date": str(panel.loc[non_null, "date"].max().date()) if non_null.any() else "",
            }
        )
    quality = pd.DataFrame(rows)
    _write_csv(Path(config.data.features_dir) / "BTCUSDT_feature_coverage.csv", quality)
    from egx_research.crypto_sources import get_source_env_var, get_source_api_key
    env_var = get_source_env_var(config, "options_skew")
    options_source_type = "historical_api" if get_source_api_key("options_skew", env_var) else "limited/proxy"

    write_json(
        Path(config.data.features_dir) / "BTCUSDT_feature_quality.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "rows": int(len(panel)),
            "start_date": str(panel["date"].min().date()) if not panel.empty else "",
            "end_date": str(panel["date"].max().date()) if not panel.empty else "",
            "optional_files_present": {str(path): path.exists() for path in optional_files},
            "options_source_type": options_source_type,
        },
    )


def _sync_optional_source(
    name: str,
    fetch_fn: Any,
    config: CryptoConfig,
    raw_dir: Path,
    filename: str,
    summary: dict[str, Any],
) -> None:
    source_statuses = summary.setdefault("source_statuses", {})
    if not is_source_enabled(config, name):
        source_statuses[name] = "disabled"
        return

    env_var = get_source_env_var(config, name)
    if env_var and source_requires_credentials(config, name) and not get_source_api_key(name, env_var):
        if is_source_required(config, name):
            raise ValueError(f"Missing credentials for required source '{name}'")
        source_statuses[name] = "missing_credentials"
        return

    try:
        if name == "coinmetrics":
            data, cm_summary = fetch_fn(config)
            summary["coinmetrics"] = cm_summary
            _write_csv(raw_dir / filename, data)
            status = "success"
        elif name == "btc_etf_flows":
            data, etf_summary = fetch_fn(config)
            summary["btc_etf_flows"] = etf_summary
            status = etf_summary.get("status", "success")
            if status != "fallback_local" and len(data) > 0:
                _write_csv(raw_dir / filename, data)
        else:
            data = fetch_fn(config)
            if len(data) == 0 and not SOURCE_REGISTRY.get(name, {}).get("critical", False):
                status = "missing_optional"
            else:
                _write_csv(raw_dir / filename, data)
                status = "success"
        if name == "funding_rates":
            rows_key = "funding_rows"
        elif name == "btc_etf_flows":
            rows_key = "btc_etf_flow_rows"
        else:
            rows_key = f"{name}_rows"
        summary[rows_key] = int(len(data))
        source_statuses[name] = status
    except Exception as exc:
        if is_source_required(config, name):
            raise
        source_statuses[name] = "failed"
        summary["errors"].append({"source": name, "error": str(exc)})


def sync_crypto_data(config: CryptoConfig) -> Path:
    raw_dir = ensure_dir(config.data.raw_dir)
    normalized_dir = ensure_dir(config.data.normalized_dir)
    ensure_dir(config.data.features_dir)
    summary: dict[str, Any] = {"created_at": datetime.now(UTC).isoformat(), "errors": []}

    price = fetch_binance_klines(config)
    _write_csv(raw_dir / f"binance_{config.data.symbol}_{config.data.interval}.csv", price)
    normalized = price[["date", "open", "high", "low", "close", "volume"]].copy()
    _write_csv(normalized_dir / config.data.normalized_filename, normalized)
    summary["price_rows"] = int(len(normalized))
    summary.setdefault("source_statuses", {})["binance"] = "success"

    _sync_optional_source("coinmetrics", fetch_coinmetrics, config, raw_dir, "coinmetrics_btc.csv", summary)
    _sync_optional_source("fear_greed", fetch_fear_greed, config, raw_dir, "fear_greed.csv", summary)
    _sync_optional_source("macro", fetch_macro, config, raw_dir, "macro_fred.csv", summary)
    _sync_optional_source("funding_rates", fetch_funding_rates, config, raw_dir, "funding_rates.csv", summary)
    _sync_optional_source("btc_etf_flows", fetch_btc_etf_flows, config, raw_dir, "btc_etf_flows.csv", summary)
    _sync_optional_source("open_interest", fetch_open_interest, config, raw_dir, "open_interest.csv", summary)
    _sync_optional_source("futures_positioning", fetch_futures_positioning, config, raw_dir, "futures_positioning.csv", summary)
    _sync_optional_source("coinbase_premium", fetch_coinbase_premium, config, raw_dir, "coinbase_premium.csv", summary)
    _sync_optional_source("stablecoin_supply", fetch_stablecoin_supply, config, raw_dir, "stablecoin_supply.csv", summary)
    _sync_optional_source("exchange_stablecoin_reserves", fetch_exchange_stablecoin_reserves, config, raw_dir, "exchange_stablecoin_reserves.csv", summary)
    _sync_optional_source("options_skew", fetch_deribit_options, config, raw_dir, "options_skew.csv", summary)
    _sync_optional_source("liquidations", fetch_liquidations, config, raw_dir, "liquidations.csv", summary)

    panel = build_crypto_feature_panel(config)
    summary["feature_rows"] = int(len(panel))

    env_var = get_source_env_var(config, "options_skew")
    options_source_type = "historical_api" if get_source_api_key("options_skew", env_var) else "limited/proxy"
    summary["options_source_type"] = options_source_type

    write_json(Path(config.data.features_dir) / "sync_summary.json", summary)
    return Path(config.data.features_path)
