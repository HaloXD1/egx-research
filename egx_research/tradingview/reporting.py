from __future__ import annotations

import html
import json
from pathlib import Path


def generate_report(run_id: str, runs_dir: str = "runs") -> Path:
    run_dir = Path(runs_dir) / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"TradingView run missing: {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sections = [f"<h1>TradingView research run {html.escape(run_id)}</h1>", f"<pre>{html.escape(json.dumps(manifest, indent=2))}</pre>"]
    for name in ("scan.json", "metrics.json", "parity_summary.json", "validation_summary.json", "current_signal.json", "notification.json", "execution_audit.json", "data_quality.json"):
        path = run_dir / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            sections.extend([f"<h2>{html.escape(name)}</h2>", f"<pre>{html.escape(json.dumps(payload, indent=2))}</pre>"])
    report = run_dir / "report.html"
    report.write_text("<html><body>" + "\n".join(sections) + "</body></html>", encoding="utf-8")
    return report
