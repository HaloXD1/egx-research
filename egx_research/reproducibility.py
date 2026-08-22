from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def repository_state() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    status = _git_output("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(status) if status is not None else None,
    }


def build_run_provenance(paths: Iterable[str | Path]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for value in paths:
        path = Path(value)
        artifacts[str(path)] = (
            {
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            if path.is_file()
            else {"exists": False}
        )
    return {"repository": repository_state(), "artifacts": artifacts}
