from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class UniversePolicy:
    venues: tuple[str, ...] = ("binance_spot",)
    allowed_quotes: tuple[str, ...] = ("USDT", "USDC")
    minimum_history_days: int = 90
    liquidity_lookback_days: int = 30
    minimum_median_quote_volume: float = 1_000_000.0
    stablecoin_assets: tuple[str, ...] = ("USDT", "USDC", "DAI")
    delisting_policy: str = "liquidate_on_last_eligible_bar"


MEMBERSHIP_COLUMNS = {
    "asset_id",
    "venue",
    "symbol",
    "base_asset",
    "quote_asset",
    "valid_from",
    "valid_to",
}


def normalize_membership_history(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(MEMBERSHIP_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"membership history missing columns: {', '.join(missing)}")
    result = frame.copy()
    result["valid_from"] = pd.to_datetime(result["valid_from"], errors="raise").dt.normalize()
    result["valid_to"] = pd.to_datetime(result["valid_to"], errors="coerce").dt.normalize()
    invalid = result["valid_to"].notna() & (result["valid_to"] < result["valid_from"])
    if invalid.any():
        raise ValueError("membership valid_to cannot precede valid_from")
    result = result.sort_values(["venue", "symbol", "valid_from"]).reset_index(drop=True)
    for (_, _), group in result.groupby(["venue", "symbol"], sort=False):
        previous_end: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            if previous_end is None:
                previous_end = row.valid_to
                continue
            if pd.isna(previous_end) or row.valid_from <= previous_end:
                raise ValueError("membership intervals overlap for a venue symbol")
            previous_end = row.valid_to
    return result


def membership_as_of(history: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    normalized = normalize_membership_history(history)
    day = pd.Timestamp(as_of).normalize()
    active = (normalized["valid_from"] <= day) & (
        normalized["valid_to"].isna() | (normalized["valid_to"] >= day)
    )
    return normalized.loc[active].reset_index(drop=True)


def build_point_in_time_panel(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    policy: UniversePolicy,
) -> pd.DataFrame:
    required_prices = {"date", "venue", "symbol", "open", "high", "low", "close", "volume"}
    missing = sorted(required_prices - set(prices.columns))
    if missing:
        raise ValueError(f"price panel missing columns: {', '.join(missing)}")
    history = normalize_membership_history(membership)
    raw = prices.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="raise").dt.normalize()
    if raw.duplicated(["date", "venue", "symbol"]).any():
        raise ValueError("price panel contains duplicate venue-symbol dates")
    numeric = ["open", "high", "low", "close", "volume"]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="raise")
    if "quote_volume" not in raw:
        raw["quote_volume"] = raw["close"] * raw["volume"]

    segments: list[pd.DataFrame] = []
    for row in history.itertuples(index=False):
        mask = (
            (raw["venue"] == row.venue)
            & (raw["symbol"] == row.symbol)
            & (raw["date"] >= row.valid_from)
            & (pd.isna(row.valid_to) | (raw["date"] <= row.valid_to))
        )
        segment = raw.loc[mask].copy()
        if segment.empty:
            continue
        segment["asset_id"] = row.asset_id
        segment["base_asset"] = row.base_asset
        segment["quote_asset"] = row.quote_asset
        segment["membership_from"] = row.valid_from
        segment["membership_to"] = row.valid_to
        segment["forced_exit"] = False
        if pd.notna(row.valid_to) and policy.delisting_policy == "liquidate_on_last_eligible_bar":
            segment.loc[segment["date"].idxmax(), "forced_exit"] = True
        segments.append(segment)
    if not segments:
        return pd.DataFrame()
    panel = pd.concat(segments, ignore_index=True).sort_values(
        ["asset_id", "venue", "date"]
    )
    panel["history_days"] = (
        panel.groupby(["asset_id", "venue"]).cumcount() + 1
    )
    panel["median_quote_volume"] = panel.groupby(["asset_id", "venue"])[
        "quote_volume"
    ].transform(
        lambda values: values.rolling(
            policy.liquidity_lookback_days,
            min_periods=policy.liquidity_lookback_days,
        ).median()
    )
    panel["is_stablecoin"] = panel["base_asset"].isin(policy.stablecoin_assets)
    panel["eligible"] = (
        panel["venue"].isin(policy.venues)
        & panel["quote_asset"].isin(policy.allowed_quotes)
        & (panel["history_days"] >= policy.minimum_history_days)
        & (panel["median_quote_volume"] >= policy.minimum_median_quote_volume)
        & ~panel["is_stablecoin"]
    )
    panel.loc[panel["forced_exit"], "eligible"] = False
    return panel.reset_index(drop=True)


def eligible_universe_as_of(
    panel: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    day = pd.Timestamp(as_of).normalize()
    available = panel.loc[panel["date"] <= day]
    if available.empty:
        return available
    latest = (
        available.sort_values("date")
        .groupby(["asset_id", "venue"], as_index=False)
        .tail(1)
    )
    return latest.loc[latest["eligible"]].reset_index(drop=True)
