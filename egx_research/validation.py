from __future__ import annotations

from dataclasses import asdict, dataclass

from egx_research.config import ValidationConfig


@dataclass
class Window:
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def split_holdout(length: int, holdout_ratio: float) -> tuple[int, int]:
    holdout_bars = max(1, int(length * holdout_ratio))
    research_end = max(0, length - holdout_bars)
    return research_end, holdout_bars


def build_walk_forward_windows(length: int, config: ValidationConfig) -> tuple[list[Window], str]:
    scheme = (
        config.primary_train_bars,
        config.primary_test_bars,
        config.primary_step_bars,
        "primary",
    )
    if length < config.primary_train_bars + config.primary_test_bars:
        scheme = (
            config.fallback_train_bars,
            config.fallback_test_bars,
            config.fallback_step_bars,
            "fallback",
        )

    train_bars, test_bars, step_bars, label = scheme
    if length < train_bars + test_bars:
        raise ValueError("Not enough history for walk-forward windows.")

    windows: list[Window] = []
    start = 0
    while start + train_bars + test_bars <= length:
        train_start = start
        train_end = start + train_bars - 1
        test_start = train_end + 1
        test_end = test_start + test_bars - 1
        windows.append(Window(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
        start += step_bars

    return windows, label
