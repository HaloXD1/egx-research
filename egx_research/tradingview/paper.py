from __future__ import annotations

from pathlib import Path

import pandas as pd

from egx_research.tradingview.config import TradingViewConfig
from egx_research.tradingview.models import StrategyDefinition, SymbolMapping
from egx_research.tradingview.research import _load_quality, _manifest, _params
from egx_research.tradingview.signals import build_local_frame, canonical_events
from egx_research.utils import ensure_dir, write_json


def run_paper_track(config: TradingViewConfig, definition: StrategyDefinition, symbol: SymbolMapping, run_id: str, start_date: str | None = None, params_json: str | None = None) -> Path:
    data, quality = _load_quality(symbol.local_path)
    frame = build_local_frame(data, definition, _params(params_json))
    events = canonical_events(frame, symbol.logical_symbol, timezone=symbol.timezone)
    daily = frame[["date", "close"]].copy()
    if "target_allocation" in frame.columns:
        daily["target_exposure"] = pd.to_numeric(frame["target_allocation"], errors="coerce").ffill().fillna(0.0)
    else:
        daily["target_exposure"] = 0.0
        for event in events.itertuples():
            mask = daily["date"] >= pd.Timestamp(event.timestamp).tz_localize(None)
            daily.loc[mask, "target_exposure"] = float(event.target_exposure)
    action_by_date = {pd.Timestamp(row.timestamp).tz_localize(None).normalize(): row.action for row in events.itertuples()}
    daily["action"] = daily["date"].map(action_by_date).fillna("")
    daily["signal_changed"] = daily["action"] != ""
    if start_date:
        daily = daily[daily["date"] >= pd.Timestamp(start_date)].copy()
    run_dir = ensure_dir(Path(config.runs_dir) / run_id)
    daily.to_csv(run_dir / "paper_log.csv", index=False)
    latest = daily.iloc[-1]
    current = {"strategy_id": definition.id, "logical_symbol": symbol.logical_symbol, "as_of": str(latest["date"]), "close": float(latest["close"]), "target_exposure": float(latest["target_exposure"]), "action": str(latest["action"]), "signal_changed": bool(latest["signal_changed"]), "status": "paper_only"}
    write_json(run_dir / "current_signal.json", current)
    write_json(run_dir / "data_quality.json", quality)
    write_json(run_dir / "manifest.json", _manifest(definition, symbol, run_id, symbol.local_path, quality, "paper-track"))
    return run_dir
