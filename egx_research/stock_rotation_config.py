from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from egx_research.config import BacktestConfig


@dataclass
class StockSourceConfig:
    etf_rebalancing_url: str = (
        "https://www.egx30etf.com/Content/Excel%20Files/ETF_Rebalancing.xlsx"
    )
    mubasher_stock_url_template: str = (
        "https://english.mubasher.info/markets/EGX/stocks/{ticker}"
    )
    mubasher_financial_statements_path: str = "financial-statements"
    mubasher_company_search_url: str = (
        "https://english.mubasher.info/api/1/companySearch"
    )
    fundamental_quarterly_lag_days: int = 60
    fundamental_annual_lag_days: int = 90


@dataclass
class StockBenchmarkConfig:
    etf_symbol_path: str = "data/normalized/EGX30_ETF.csv"
    index_symbol_path: str = "data/normalized/EGX30_INDEX.csv"


@dataclass
class StockStorageConfig:
    root_dir: str = "data/stock_rotation"
    universe_filename: str = "universe.csv"
    panel_filename: str = "panel.csv"
    membership_filename: str = "membership_snapshots.csv"
    fundamentals_filename: str = "fundamentals.csv"
    dividend_actions_filename: str = "dividend_actions.csv"
    corporate_actions_filename: str = "corporate_actions.csv"
    disclosure_events_filename: str = "disclosure_events.csv"
    macro_regime_filename: str = "macro_regime_monthly.csv"
    raw_dir: str = "raw"
    normalized_dir: str = "normalized"


@dataclass
class StockPortfolioConfig:
    top_n: int = 10
    weighting: str = "equal"
    rebalance: str = "monthly"
    hold_cash_when_few: bool = True
    fixed_buy_fee_egp: float = 5.0
    turnover_buffer_score: float = 0.03


@dataclass
class StockSelectionWeights:
    return_3m_rank: float = 0.5
    return_6m_rank: float = 0.3
    rel_strength_spread_3m: float = 0.2


@dataclass
class StockMultiFactorWeights:
    momentum: float = 0.35
    value: float = 0.20
    quality: float = 0.20
    growth: float = 0.0
    low_risk: float = 0.15
    liquidity: float = 0.10


@dataclass
class StockSelectionConfig:
    method: str = "relative_strength"
    trend_indicator: str = "KAMA"
    rel_strength_window_3m: int = 63
    rel_strength_window_6m: int = 126
    score_weights: StockSelectionWeights = field(default_factory=StockSelectionWeights)
    factor_weights: StockMultiFactorWeights = field(default_factory=StockMultiFactorWeights)
    liquidity_window_bars: int = 63
    min_median_daily_value_egp: float = 100_000.0
    min_median_daily_volume: float = 1_000.0
    use_total_return_features: bool = True
    require_long_term_trend: bool = False
    max_drawdown_252: float = 1.0
    max_sector_weight: float = 1.0
    max_position_weight: float = 1.0


@dataclass
class StockValidationConfig:
    min_history_bars: int = 756
    coverage_lookback_bars: int = 126
    min_coverage_ratio: float = 0.92


@dataclass
class StockModelSelectionConfig:
    holdout_ratio: float = 0.2
    min_holdout_excess_return: float = 0.0
    max_drawdown: float = 0.8
    min_neighbor_pass_rate: float = 0.6
    max_mean_rebalance_turnover_pct: float = 0.95
    max_fee_to_contributions_ratio: float = 0.2


@dataclass
class StockRotationConfig:
    sources: StockSourceConfig = field(default_factory=StockSourceConfig)
    benchmark: StockBenchmarkConfig = field(default_factory=StockBenchmarkConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    storage: StockStorageConfig = field(default_factory=StockStorageConfig)
    portfolio: StockPortfolioConfig = field(default_factory=StockPortfolioConfig)
    selection: StockSelectionConfig = field(default_factory=StockSelectionConfig)
    validation: StockValidationConfig = field(default_factory=StockValidationConfig)
    model_selection: StockModelSelectionConfig = field(
        default_factory=StockModelSelectionConfig
    )


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_dict(config: StockRotationConfig) -> dict[str, Any]:
    return {
        "sources": config.sources.__dict__,
        "benchmark": config.benchmark.__dict__,
        "backtest": config.backtest.__dict__,
        "storage": config.storage.__dict__,
        "portfolio": config.portfolio.__dict__,
        "selection": {
            "method": config.selection.method,
            "trend_indicator": config.selection.trend_indicator,
            "rel_strength_window_3m": config.selection.rel_strength_window_3m,
            "rel_strength_window_6m": config.selection.rel_strength_window_6m,
            "liquidity_window_bars": config.selection.liquidity_window_bars,
            "min_median_daily_value_egp": config.selection.min_median_daily_value_egp,
            "min_median_daily_volume": config.selection.min_median_daily_volume,
            "use_total_return_features": config.selection.use_total_return_features,
            "require_long_term_trend": config.selection.require_long_term_trend,
            "max_drawdown_252": config.selection.max_drawdown_252,
            "max_sector_weight": config.selection.max_sector_weight,
            "max_position_weight": config.selection.max_position_weight,
            "score_weights": config.selection.score_weights.__dict__,
            "factor_weights": config.selection.factor_weights.__dict__,
        },
        "validation": config.validation.__dict__,
        "model_selection": config.model_selection.__dict__,
    }


def load_stock_rotation_config(path: str | Path) -> StockRotationConfig:
    path = Path(path)
    base = _to_dict(StockRotationConfig())
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    merged = _merge_dict(base, raw)
    return StockRotationConfig(
        sources=StockSourceConfig(**merged.get("sources", {})),
        benchmark=StockBenchmarkConfig(**merged.get("benchmark", {})),
        backtest=BacktestConfig(**merged.get("backtest", {})),
        storage=StockStorageConfig(**merged.get("storage", {})),
        portfolio=StockPortfolioConfig(**merged.get("portfolio", {})),
        selection=StockSelectionConfig(
            method=merged.get("selection", {}).get("method", "relative_strength"),
            trend_indicator=merged.get("selection", {}).get("trend_indicator", "KAMA"),
            rel_strength_window_3m=merged.get("selection", {}).get(
                "rel_strength_window_3m", 63
            ),
            rel_strength_window_6m=merged.get("selection", {}).get(
                "rel_strength_window_6m", 126
            ),
            liquidity_window_bars=merged.get("selection", {}).get(
                "liquidity_window_bars", 63
            ),
            min_median_daily_value_egp=merged.get("selection", {}).get(
                "min_median_daily_value_egp", 100_000.0
            ),
            min_median_daily_volume=merged.get("selection", {}).get(
                "min_median_daily_volume", 1_000.0
            ),
            use_total_return_features=merged.get("selection", {}).get(
                "use_total_return_features", True
            ),
            require_long_term_trend=merged.get("selection", {}).get(
                "require_long_term_trend", False
            ),
            max_drawdown_252=merged.get("selection", {}).get(
                "max_drawdown_252", 1.0
            ),
            max_sector_weight=merged.get("selection", {}).get(
                "max_sector_weight", 1.0
            ),
            max_position_weight=merged.get("selection", {}).get(
                "max_position_weight", 1.0
            ),
            score_weights=StockSelectionWeights(
                **merged.get("selection", {}).get("score_weights", {})
            ),
            factor_weights=StockMultiFactorWeights(
                **merged.get("selection", {}).get("factor_weights", {})
            ),
        ),
        validation=StockValidationConfig(**merged.get("validation", {})),
        model_selection=StockModelSelectionConfig(**merged.get("model_selection", {})),
    )
