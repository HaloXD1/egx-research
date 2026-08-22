from pathlib import Path

from egx_research.reproducibility import build_run_provenance, sha256_file


def test_run_provenance_hashes_existing_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "input.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    missing = tmp_path / "missing.csv"

    provenance = build_run_provenance([artifact, missing])

    assert provenance["artifacts"][str(artifact)]["sha256"] == sha256_file(artifact)
    assert provenance["artifacts"][str(missing)] == {"exists": False}
    assert "git_commit" in provenance["repository"]
