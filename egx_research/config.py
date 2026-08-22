from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    symbol: str = "EGX30_ETF"
    raw_dir: str = "data/raw"
    normalized_dir: str = "data/normalized"
    normalized_filename: str = "EGX30_ETF.csv"

    @property
    def normalized_path(self) -> str:
        return str(Path(self.normalized_dir) / self.normalized_filename)


@dataclass
class BacktestConfig:
    initial_cash: float = 100000.0
    monthly_contribution: float = 10000.0
    monthly_contribution_day_offset: int = 0
    fee_bps: float = 20.0
    slippage_bps: float = 5.0
    share_precision: int = 0
    annualization_periods: float = 252.0


@dataclass
class SearchConfig:
    families: list[str] = field(default_factory=lambda: ["trend", "mean_reversion", "breakout"])
    trials_per_family: int = 500
    top_candidates_per_family: int = 10
    random_seed: int = 42
    robustness_neighbor_steps: int = 1
    objective_mode: str = "walk_forward_score"


@dataclass
class ValidationConfig:
    holdout_ratio: float = 0.2
    primary_train_bars: int = 756
    primary_test_bars: int = 252
    primary_step_bars: int = 126
    fallback_train_bars: int = 504
    fallback_test_bars: int = 126
    fallback_step_bars: int = 63
    outer_test_bars: int = 0
    outer_step_bars: int = 0
    purge_bars: int = 0
    embargo_bars: int = 0


@dataclass
class RankingWeights:
    return_dd: float = 0.4
    cagr: float = 0.2
    profit_factor: float = 0.15
    sharpe: float = 0.15
    excess_return_vs_dca: float = 0.1


@dataclass
class RankingFilters:
    min_closed_trades: int = 25
    max_drawdown: float = 0.35
    min_profit_factor: float = 1.1
    min_holdout_excess_return: float = 0.0
    min_neighbor_pass_rate: float = 0.75


@dataclass
class RankingConfig:
    weights: RankingWeights = field(default_factory=RankingWeights)
    filters: RankingFilters = field(default_factory=RankingFilters)


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_dict(data: dict[str, Any]) -> AppConfig:
    ranking = data.get("ranking", {})
    return AppConfig(
        data=DataConfig(**data.get("data", {})),
        backtest=BacktestConfig(**data.get("backtest", {})),
        search=SearchConfig(**data.get("search", {})),
        validation=ValidationConfig(**data.get("validation", {})),
        ranking=RankingConfig(
            weights=RankingWeights(**ranking.get("weights", {})),
            filters=RankingFilters(**ranking.get("filters", {})),
        ),
    )


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    base = AppConfig().to_dict()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _from_dict(_merge_dict(base, raw))


def save_config(path: str | Path, config: AppConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
