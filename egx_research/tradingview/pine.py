from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"^\s*//@version=(\d+)\s*$", re.MULTILINE)
ALERT_RE = re.compile(r"\balertcondition\s*\(", re.IGNORECASE)
ALERT_CALL_RE = re.compile(r"\balert\s*\(", re.IGNORECASE)
INPUT_RE = re.compile(r"\binput\.(?:int|float|bool|string)\s*\(", re.IGNORECASE)


def source_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_pine(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    text = target.read_text(encoding="utf-8")
    version_match = VERSION_RE.search(text)
    version = int(version_match.group(1)) if version_match else None
    if re.search(r"\bstrategy\s*\(", text):
        script_kind = "strategy"
    elif re.search(r"\bindicator\s*\(", text):
        script_kind = "indicator"
    else:
        script_kind = "unknown"
    return {
        "path": str(target),
        "pine_version": version,
        "script_kind": script_kind,
        "alertcondition_count": len(ALERT_RE.findall(text)),
        "alert_call_count": len(ALERT_CALL_RE.findall(text)),
        "input_count": len(INPUT_RE.findall(text)),
        "source_hash": source_hash(target),
        "uses_request_security": "request.security" in text,
        "uses_barstate_confirmed": "barstate.isconfirmed" in text,
        "calc_on_every_tick": bool(re.search(r"calc_on_every_tick\s*=\s*true", text)),
        "warnings": _warnings(text, script_kind),
    }


def _warnings(text: str, script_kind: str) -> list[str]:
    warnings: list[str] = []
    if "request.security" in text and "lookahead_on" in text:
        warnings.append("request.security uses lookahead_on; review for lookahead bias")
    if re.search(r"calc_on_every_tick\s*=\s*true", text):
        warnings.append("calc_on_every_tick is enabled; intrabar results may differ from close-only research")
    if re.search(r"\boffset\s*=\s*-\d+", text):
        warnings.append("A negative plot offset may display values before they are known")
    if script_kind == "strategy" and ALERT_RE.search(text) and not ALERT_CALL_RE.search(text):
        warnings.append("alertcondition has no effect in strategy scripts; add alert() or use order-fill alerts")
    return warnings


def validate_pine(path: str | Path, expected_version: int | None = 6, expected_kind: str | None = None) -> dict[str, Any]:
    details = inspect_pine(path)
    errors: list[str] = []
    if details["pine_version"] is None:
        errors.append("Missing //@version declaration")
    elif expected_version is not None and details["pine_version"] != expected_version:
        errors.append(f"Expected Pine v{expected_version}, found v{details['pine_version']}")
    if details["script_kind"] == "unknown":
        errors.append("Missing indicator() or strategy() declaration")
    if expected_kind and details["script_kind"] != expected_kind:
        errors.append(f"Expected script kind {expected_kind}, found {details['script_kind']}")
    return {**details, "errors": errors, "valid": not errors}


def render_template(template_path: str | Path, context: dict[str, Any]) -> str:
    text = Path(template_path).read_text(encoding="utf-8")
    for key, value in sorted(context.items()):
        text = text.replace("{{ " + key + " }}", str(value))
        text = text.replace("{{" + key + "}}", str(value))
    unresolved = re.findall(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", text)
    if unresolved:
        raise ValueError(f"Unresolved Pine template variables: {', '.join(sorted(set(unresolved)))}")
    return text
