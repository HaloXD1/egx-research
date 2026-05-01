from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_native(item) for item in value]
    if isinstance(value, tuple):
        return [to_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_native(payload), handle, indent=2, sort_keys=False)
