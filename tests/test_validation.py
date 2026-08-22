from __future__ import annotations

from egx_research.config import ValidationConfig
from egx_research.nested_validation import (
    SealedHoldout,
    build_nested_expanding_windows,
    multiple_testing_adjusted_score,
)
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


def test_nested_windows_are_expanding_purged_and_non_overlapping() -> None:
    config = ValidationConfig(
        primary_train_bars=100,
        primary_test_bars=40,
        outer_test_bars=30,
        outer_step_bars=30,
        purge_bars=5,
        embargo_bars=3,
    )
    windows = build_nested_expanding_windows(260, config)
    assert windows[0].train_start == 0
    assert windows[0].train_end == 139
    assert windows[0].test_start == 145
    assert windows[0].test_end == 174
    assert windows[1].train_end > windows[0].train_end
    assert windows[1].test_start > windows[0].test_end


def test_sealed_holdout_can_only_be_evaluated_once() -> None:
    holdout: SealedHoldout[tuple[int, int]] = SealedHoldout(80, 99)
    assert holdout.evaluate_once(lambda start, end: (start, end)) == (80, 99)
    try:
        holdout.evaluate_once(lambda start, end: (start, end))
    except RuntimeError as exc:
        assert "already been evaluated" in str(exc)
    else:
        raise AssertionError("second sealed-holdout access should fail")


def test_multiple_testing_penalty_increases_with_search_size() -> None:
    small = multiple_testing_adjusted_score(1.0, trials=10, independent_observations=5)
    large = multiple_testing_adjusted_score(1.0, trials=1000, independent_observations=5)
    assert large < small < 1.0
