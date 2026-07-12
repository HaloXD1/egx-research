from __future__ import annotations

from typing import Any

import pandas as pd

from egx_research.crypto_strategies import build_crypto_strategy_frame
from egx_research.strategies import build_strategy_frame
from egx_research.tradingview.models import StrategyDefinition


CRYPTO_FAMILIES = {
    "crypto_price_trend", "crypto_trend_adx", "crypto_donchian_breakout", "crypto_supertrend_combo",
    "crypto_pullback_combo", "crypto_dca_overlay", "crypto_onchain_overlay", "crypto_sentiment_overlay",
    "crypto_macro_overlay", "crypto_ensemble_overlay", "crypto_hierarchy_combo", "crypto_multisignal_score",
}


def build_local_frame(data: pd.DataFrame, definition: StrategyDefinition, params: dict[str, Any] | None = None) -> pd.DataFrame:
    if not definition.local_family:
        raise ValueError(f"Strategy has no local_family adapter: {definition.id}")
    merged = dict(definition.params)
    merged.update(params or {})
    if definition.local_family in CRYPTO_FAMILIES:
        return build_crypto_strategy_frame(data, definition.local_family, merged)
    return build_strategy_frame(data, definition.local_family, merged)


def canonical_events(frame: pd.DataFrame, logical_symbol: str, source: str = "python", timezone: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    previous_target = 0.0
    for index, row in frame.iterrows():
        target = float(row.get("target_allocation", previous_target))
        entry = bool(row.get("entry_signal", False))
        exit_ = bool(row.get("exit_signal", False))
        if target > previous_target + 1e-9:
            action = "buy"
        elif target < previous_target - 1e-9:
            action = "sell"
        elif entry and not exit_:
            action, target = "buy", max(target, 1.0)
        elif exit_ and not entry:
            action, target = "sell", 0.0
        else:
            previous_target = target
            continue
        timestamp = pd.Timestamp(row["date"])
        if timezone and timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone)
        rows.append({
            "timestamp": timestamp,
            "logical_symbol": logical_symbol,
            "action": action,
            "target_exposure": target,
            "close": float(row["close"]),
            "source": source,
            "bar_index": int(index),
        })
        previous_target = target
    return pd.DataFrame(rows, columns=["timestamp", "logical_symbol", "action", "target_exposure", "close", "source", "bar_index"])
