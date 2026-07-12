from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ALIASES = {
    "date": "timestamp", "time": "timestamp", "datetime": "timestamp", "timestamp": "timestamp",
    "action": "action", "signal": "action", "event": "action", "target": "target_exposure",
    "target_exposure": "target_exposure", "exposure": "target_exposure", "close": "close", "price": "close",
    "symbol": "logical_symbol", "logical_symbol": "logical_symbol",
}


def _normalize_action(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "buy", "buying", "entry", "add", "long", "up"}:
        return "buy"
    if text in {"-1", "sell", "selling", "exit", "trim", "stop", "down"}:
        return "sell"
    if text in {"0", "none", ""}:
        return "none"
    return text


def normalize_events(path: str | Path, source: str = "pine", timezone: str = "UTC") -> pd.DataFrame:
    frame = pd.read_csv(path)
    renamed: dict[str, str] = {}
    for column in frame.columns:
        key = column.strip().lower().replace(" ", "_")
        if key in ALIASES:
            renamed[column] = ALIASES[key]
    frame = frame.rename(columns=renamed)
    if "timestamp" not in frame.columns or "action" not in frame.columns:
        raise ValueError("Parity CSV requires timestamp/date and action/signal columns")
    normalized_timestamps = []
    for value in frame["timestamp"]:
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError):
            raise ValueError("Parity CSV contains invalid timestamps") from None
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(timezone)
        normalized_timestamps.append(parsed.tz_convert("UTC").normalize())
    frame["timestamp"] = normalized_timestamps
    frame["action"] = frame["action"].map(_normalize_action)
    frame = frame[frame["action"] != "none"].copy()
    if "target_exposure" not in frame.columns:
        frame["target_exposure"] = pd.NA
    frame["target_exposure"] = pd.to_numeric(frame["target_exposure"], errors="coerce")
    if "logical_symbol" not in frame.columns:
        frame["logical_symbol"] = ""
    if "close" not in frame.columns:
        frame["close"] = pd.NA
    frame["source"] = source
    return frame[["timestamp", "logical_symbol", "action", "target_exposure", "close", "source"]].sort_values("timestamp").reset_index(drop=True)


def compare_events(
    python_events: pd.DataFrame,
    pine_events: pd.DataFrame,
    tolerance_bars: int = 0,
    bar_dates: pd.Series | list[Any] | None = None,
) -> dict[str, Any]:
    left = python_events.copy()
    right = pine_events.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True).dt.normalize()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True).dt.normalize()
    left_keys = {(row.timestamp, row.action) for row in left.itertuples()}
    right_keys = {(row.timestamp, row.action) for row in right.itertuples()}
    unmatched_right = set(right_keys)
    matched: set[tuple[pd.Timestamp, str]] = set()
    missing: list[tuple[pd.Timestamp, str]] = []
    bar_positions: dict[pd.Timestamp, int] = {}
    if bar_dates is not None:
        normalized_bars = pd.to_datetime(pd.Series(bar_dates), utc=True).dt.normalize()
        bar_positions = {timestamp: index for index, timestamp in enumerate(normalized_bars)}

    def within_tolerance(left_timestamp: pd.Timestamp, right_timestamp: pd.Timestamp) -> bool:
        if bar_positions:
            if left_timestamp not in bar_positions or right_timestamp not in bar_positions:
                return left_timestamp == right_timestamp
            return abs(bar_positions[left_timestamp] - bar_positions[right_timestamp]) <= max(0, tolerance_bars)
        return abs(right_timestamp - left_timestamp) <= pd.Timedelta(days=max(0, tolerance_bars))
    for timestamp, action in sorted(left_keys):
        candidates = [
            item for item in unmatched_right
            if item[1] == action and within_tolerance(timestamp, item[0])
        ]
        if candidates:
            selected = min(candidates, key=lambda item: abs(item[0] - timestamp))
            unmatched_right.remove(selected)
            matched.add((timestamp, action))
        else:
            missing.append((timestamp, action))
    extra = sorted(unmatched_right)
    return {
        "exact_match": not missing and not extra and tolerance_bars == 0,
        "matched_events": len(matched),
        "python_events": len(left_keys),
        "pine_events": len(right_keys),
        "missing_python_events_in_pine": [{"timestamp": str(ts), "action": action} for ts, action in missing],
        "extra_pine_events": [{"timestamp": str(ts), "action": action} for ts, action in extra],
        "date_tolerance_bars": tolerance_bars,
        "tolerance_mode": "trading_bars" if bar_positions else "calendar_days",
        "status": "pass" if not missing and not extra else "fail",
    }
