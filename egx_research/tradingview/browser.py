from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def require_playwright() -> object:
    """Load the optional browser dependency without affecting local commands."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Browser automation requires the optional tradingview-browser extra") from exc
    return sync_playwright


def chart_snapshot(url: str, screenshot_path: str, profile_dir: str | None = None, headless: bool = False, wait_seconds: int = 5) -> str:
    """Open a chart and capture visible state; profile_dir must be user-managed."""
    from pathlib import Path

    target = Path(screenshot_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if profile_dir and Path(profile_dir).resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Browser profile must be outside the repository")
    playwright_factory = require_playwright()
    with playwright_factory() as playwright:
        if profile_dir:
            context = playwright.chromium.launch_persistent_context(profile_dir, headless=headless)
            page = context.new_page()
            close = context.close
        else:
            browser = playwright.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            close = browser.close
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(max(0, wait_seconds) * 1000)
            page.screenshot(path=str(target), full_page=True)
        finally:
            close()
    return str(target)


def apply_account_request(
    request_file: str | Path,
    profile_dir: str | Path,
    artifact_dir: str | Path,
    confirm: bool = False,
    headless: bool = False,
) -> dict[str, Any]:
    """Execute an explicit, auditable browser request against a user-owned profile."""
    if not confirm or os.getenv("EGX_TV_ALLOW_ACCOUNT_MUTATIONS") != "1":
        raise ValueError("Account mutations require --confirm and EGX_TV_ALLOW_ACCOUNT_MUTATIONS=1")
    profile = Path(profile_dir).resolve()
    if profile.is_relative_to(Path.cwd().resolve()):
        raise ValueError("Browser profile must be outside the repository")
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    if request.get("live_order_execution"):
        raise ValueError("Live order execution is not supported")
    allowed = {"goto", "click", "fill", "select", "upload", "wait"}
    steps = request.get("ui_steps", [])
    if not steps or any(step.get("action") not in allowed for step in steps):
        raise ValueError("Request requires non-empty ui_steps using allowed actions")
    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    playwright_factory = require_playwright()
    completed: list[dict[str, Any]] = []
    with playwright_factory() as playwright:
        context = playwright.chromium.launch_persistent_context(str(profile), headless=headless)
        context.tracing.start(screenshots=True, snapshots=True)
        page = context.pages[0] if context.pages else context.new_page()
        try:
            for index, step in enumerate(steps):
                action = step["action"]
                if action == "goto":
                    page.goto(step["url"], wait_until="domcontentloaded", timeout=60_000)
                elif action == "click":
                    page.locator(step["selector"]).click()
                elif action == "fill":
                    page.locator(step["selector"]).fill(str(step.get("value", "")))
                elif action == "select":
                    page.locator(step["selector"]).select_option(str(step.get("value", "")))
                elif action == "upload":
                    page.locator(step["selector"]).set_input_files(str(step["path"]))
                elif action == "wait":
                    page.wait_for_timeout(max(0, int(step.get("milliseconds", 1000))))
                completed.append({"index": index, "action": action})
            page.screenshot(path=str(target / "final.png"), full_page=True)
        except Exception:
            page.screenshot(path=str(target / "failure.png"), full_page=True)
            raise
        finally:
            context.tracing.stop(path=str(target / "trace.zip"))
            context.close()
    result = {"status": "complete", "request_file": str(request_file), "completed_steps": completed, "artifact_dir": str(target)}
    (target / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
