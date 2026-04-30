from __future__ import annotations

from egx_research.config import ValidationConfig
from egx_research.validation import build_walk_forward_windows, split_holdout


def test_split_holdout_and_walk_forward_boundaries() -> None:
    research_end, holdout_bars = split_holdout(1000, 0.2)
    assert research_end == 800
    assert holdout_bars == 200

    windows, label = build_walk_forward_windows(800, ValidationConfig())
    assert label == "fallback"
    assert windows[0].train_start == 0
    assert windows[0].train_end == 503
    assert windows[0].test_start == 504
    assert windows[0].test_end == 629
    assert windows[-1].test_end <= 799
