from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Callable, Generic, TypeVar

from egx_research.config import ValidationConfig
from egx_research.validation import Window


T = TypeVar("T")


@dataclass
class SealedHoldout(Generic[T]):
    start: int
    end: int
    _used: bool = False

    def evaluate_once(self, evaluator: Callable[[int, int], T]) -> T:
        if self._used:
            raise RuntimeError("sealed holdout has already been evaluated")
        self._used = True
        return evaluator(self.start, self.end)

    @property
    def used(self) -> bool:
        return self._used


def build_nested_expanding_windows(
    length: int,
    config: ValidationConfig,
) -> list[Window]:
    test_bars = config.outer_test_bars or config.primary_test_bars
    step_bars = config.outer_step_bars or test_bars
    if min(test_bars, step_bars) <= 0:
        raise ValueError("outer test and step sizes must be positive")
    if min(config.purge_bars, config.embargo_bars) < 0:
        raise ValueError("purge and embargo bars cannot be negative")

    minimum_inner_history = config.primary_train_bars + config.primary_test_bars
    if length < minimum_inner_history + config.purge_bars + test_bars:
        fallback_history = config.fallback_train_bars + config.fallback_test_bars
        minimum_inner_history = fallback_history
    first_test = minimum_inner_history + config.purge_bars
    windows: list[Window] = []
    test_start = first_test
    while test_start + test_bars <= length:
        train_end = test_start - config.purge_bars - 1
        windows.append(
            Window(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_start + test_bars - 1,
            )
        )
        test_start += max(step_bars, test_bars) + config.embargo_bars
    if not windows:
        raise ValueError("not enough history for nested expanding validation")
    return windows


def multiple_testing_adjusted_score(
    score: float,
    *,
    trials: int,
    independent_observations: int,
    penalty_scale: float = 0.05,
) -> float:
    if trials <= 0 or independent_observations <= 0:
        raise ValueError("trials and independent observations must be positive")
    penalty = penalty_scale * sqrt(2.0 * log(max(2, trials)) / independent_observations)
    return float(score - penalty)
