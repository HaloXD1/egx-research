from __future__ import annotations

from pathlib import Path
from typing import Any

from egx_research.tradingview.pine import render_template


def render_named_template(template_dir: str | Path, template_id: str, context: dict[str, Any]) -> str:
    path = Path(template_dir) / f"{template_id}.pine.tmpl"
    if not path.exists():
        raise FileNotFoundError(f"TradingView template missing: {path}")
    return render_template(path, context)
