from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from egx_research.config import BacktestConfig, RankingConfig, ValidationConfig


CRYPTO_FAMILIES = [
    "crypto_price_trend",
    "crypto_trend_adx",
    "crypto_donchian_breakout",
    "crypto_supertrend_combo",
    "crypto_pullback_combo",
    "crypto_dca_overlay",
    "crypto_onchain_overlay",
    "crypto_sentiment_overlay",
    "crypto_macro_overlay",
    "crypto_ensemble_overlay",
    "crypto_hierarchy_combo",
    "crypto_multisignal_score",
]


@dataclass
class CryptoDataConfig:
    symbol: str = "BTCUSDT"
    asset: str = "btc"
    quote: str = "USDT"
    interval: str = "1d"
    raw_dir: str = "data/crypto/raw"
    normalized_dir: str = "data/crypto/normalized"
    features_dir: str = "data/crypto/features"
    normalized_filename: str = "BTCUSDT_1d.csv"
    features_filename: str = "BTCUSDT_daily_features.csv"

    @property
    def normalized_path(self) -> str:
        return str(Path(self.normalized_dir) / self.normalized_filename)

    @property
    def features_path(self) -> str:
        return str(Path(self.features_dir) / self.features_filename)


@dataclass
class CryptoSourceConfig:
    price_start: str = "2017-01-01"
    onchain_start: str = "2017-01-01"
    macro_start: str = "2017-01-01"
    funding_start: str = "2019-09-10"
    binance_spot_base_url: str = "https://api.binance.com"
    binance_futures_base_url: str = "https://fapi.binance.com"
    coinmetrics_base_url: str = "https://community-api.coinmetrics.io"
    fear_greed_url: str = "https://api.alternative.me/fng/"
    fred_base_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    bitcoin_etf_flows_url: str = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
    coinmetrics_metrics: list[str] = field(
        default_factory=lambda: [
            "PriceUSD",
            "CapMVRVCur",
            "AdrActCnt",
            "TxCnt",
            "HashRate",
            "SplyCur",
            "FlowInExUSD",
            "FlowOutExUSD",
        ]
    )
    macro_series: dict[str, str] = field(
        default_factory=lambda: {
            "NASDAQCOM": "nasdaq",
            "DGS10": "us10y",
            "DTWEXBGS": "dollar",
            "WALCL": "fed_liquidity",
            "VIXCLS": "vix",
        }
    )
    optional_sources: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "coinmetrics": {"enabled": True, "env_var": "COINMETRICS_API_KEY", "required": False},
            "fear_greed": {"enabled": True, "env_var": "", "required": False},
            "macro": {"enabled": True, "env_var": "", "required": False},
            "funding_rates": {"enabled": True, "env_var": "", "required": False},
            "btc_etf_flows": {"enabled": True, "env_var": "", "required": False},
            "open_interest": {"enabled": True, "env_var": "BINANCE_API_KEY", "required": False},
            "futures_positioning": {"enabled": True, "env_var": "", "required": False},
            "coinbase_premium": {"enabled": True, "env_var": "COINBASE_API_KEY", "required": False},
            "stablecoin_supply": {"enabled": True, "env_var": "DEFILLAMA_API_KEY", "required": False},
            "exchange_stablecoin_reserves": {"enabled": True, "env_var": "EXCHANGE_STABLECOIN_RESERVES_API_KEY", "required": False},
            "options_skew": {"enabled": True, "env_var": "DERIBIT_API_KEY", "required": False},
        }
    )



@dataclass
class CryptoSearchConfig:
    families: list[str] = field(default_factory=lambda: list(CRYPTO_FAMILIES))
    trials_per_family: int = 150
    top_candidates_per_family: int = 8
    random_seed: int = 42
    robustness_neighbor_steps: int = 1
    objective_mode: str = "walk_forward_score"


@dataclass
class CryptoConfig:
    data: CryptoDataConfig = field(default_factory=CryptoDataConfig)
    sources: CryptoSourceConfig = field(default_factory=CryptoSourceConfig)
    backtest: BacktestConfig = field(
        default_factory=lambda: BacktestConfig(
            initial_cash=100000.0,
            monthly_contribution=10000.0,
            fee_bps=10.0,
            slippage_bps=5.0,
            share_precision=8,
        )
    )
    search: CryptoSearchConfig = field(default_factory=CryptoSearchConfig)
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(
            holdout_ratio=0.2,
            primary_train_bars=1095,
            primary_test_bars=365,
            primary_step_bars=180,
            fallback_train_bars=365,
            fallback_test_bars=180,
            fallback_step_bars=90,
        )
    )
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


def _from_dict(data: dict[str, Any]) -> CryptoConfig:
    ranking = data.get("ranking", {})
    return CryptoConfig(
        data=CryptoDataConfig(**data.get("data", {})),
        sources=CryptoSourceConfig(**data.get("sources", {})),
        backtest=BacktestConfig(**data.get("backtest", {})),
        search=CryptoSearchConfig(**data.get("search", {})),
        validation=ValidationConfig(**data.get("validation", {})),
        ranking=RankingConfig(
            weights=RankingConfig().weights.__class__(**ranking.get("weights", {})),
            filters=RankingConfig().filters.__class__(**ranking.get("filters", {})),
        ),
    )


def load_crypto_config(path: str | Path) -> CryptoConfig:
    path = Path(path)
    base = CryptoConfig().to_dict()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _from_dict(_merge_dict(base, raw))


def save_crypto_config(path: str | Path, config: CryptoConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
