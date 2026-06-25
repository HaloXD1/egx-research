import os
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

from egx_research.crypto_config import CryptoConfig

SOURCE_REGISTRY = {
    "binance": {
        "provider": "Binance",
        "category": "price",
        "paid_free": "free",
        "freshness_sla_hours": 24,
        "required_columns": ["open", "high", "low", "close", "volume"],
        "filename": "binance_BTCUSDT_1d.csv",
        "env_var": "",
        "critical": True,
    },
    "coinmetrics": {
        "provider": "CoinMetrics",
        "category": "onchain",
        "paid_free": "free",
        "freshness_sla_hours": 48,
        "required_columns": [
            "CapMVRVCur",
            "FlowInExUSD",
            "FlowOutExUSD",
            "AdrActCnt",
            "TxCnt",
            "HashRate",
            "SplyCur",
        ],
        "filename": "coinmetrics_btc.csv",
        "env_var": "COINMETRICS_API_KEY",
        "critical": True,
    },
    "fear_greed": {
        "provider": "Alternative.me",
        "category": "sentiment",
        "paid_free": "free",
        "freshness_sla_hours": 36,
        "required_columns": ["fear_greed_value"],
        "filename": "fear_greed.csv",
        "env_var": "",
        "critical": True,
    },
    "macro": {
        "provider": "FRED",
        "category": "macro",
        "paid_free": "free",
        "freshness_sla_hours": 240,
        "required_columns": [
            "macro_nasdaq",
            "macro_us10y",
            "macro_dollar",
            "macro_fed_liquidity",
            "macro_vix",
        ],
        "filename": "macro_fred.csv",
        "env_var": "",
        "critical": False,
    },
    "funding_rates": {
        "provider": "Binance Futures",
        "category": "funding",
        "paid_free": "free",
        "freshness_sla_hours": 36,
        "required_columns": ["funding_rate_mean"],
        "filename": "funding_rates.csv",
        "env_var": "",
        "critical": True,
    },
    "btc_etf_flows": {
        "provider": "Farside Investors",
        "category": "etf",
        "paid_free": "free",
        "freshness_sla_hours": 48,
        "required_columns": ["etf_net_flow_usd"],
        "filename": "btc_etf_flows.csv",
        "env_var": "",
        "critical": False,
    },
    "open_interest": {
        "provider": "Binance Futures",
        "category": "derivatives",
        "paid_free": "free",
        "freshness_sla_hours": 36,
        "required_columns": ["derivatives_open_interest"],
        "filename": "open_interest.csv",
        "env_var": "BINANCE_API_KEY",
        "critical": True,
    },
    "coinbase_premium": {
        "provider": "Coinbase",
        "category": "spot",
        "paid_free": "free",
        "freshness_sla_hours": 36,
        "required_columns": ["spot_coinbase_premium"],
        "filename": "coinbase_premium.csv",
        "env_var": "COINBASE_API_KEY",
        "critical": False,
    },
    "stablecoin_supply": {
        "provider": "DefiLlama",
        "category": "liquidity",
        "paid_free": "free",
        "freshness_sla_hours": 48,
        "required_columns": ["liquidity_stablecoin_supply"],
        "filename": "stablecoin_supply.csv",
        "env_var": "DEFILLAMA_API_KEY",
        "critical": False,
    },
    "options_skew": {
        "provider": "Deribit",
        "category": "options",
        "paid_free": "free",
        "freshness_sla_hours": 36,
        "required_columns": ["options_options_skew", "options_put_call_ratio"],
        "filename": "options_skew.csv",
        "env_var": "DERIBIT_API_KEY",
        "critical": False,
    },
    "exchange_flows": {
        "provider": "CryptoQuant",
        "category": "onchain",
        "paid_free": "paid",
        "freshness_sla_hours": 48,
        "required_columns": [
            "onchain_exchange_reserve_btc",
            "onchain_exchange_netflow_btc",
            "onchain_exchange_netflow_usd",
            "onchain_whale_inflow_usd",
            "onchain_realized_profit_loss_exchange",
        ],
        "filename": "exchange_flows.csv",
        "env_var": "CRYPTOQUANT_API_KEY",
        "critical": False,
    },
}


def get_source_api_key(source_name: str, env_var: str) -> str | None:
    """Read API key from environment variable without logging it."""
    if not env_var:
        return None
    return os.getenv(env_var)


def get_source_config(config: CryptoConfig, source_name: str) -> dict[str, Any]:
    """Return config for a source, with registry metadata as defaults."""
    meta = SOURCE_REGISTRY.get(source_name, {})
    optional_srcs = getattr(config.sources, "optional_sources", {})
    src_config = optional_srcs.get(source_name, {})
    return {
        "enabled": src_config.get("enabled", True),
        "required": src_config.get("required", False),
        "env_var": src_config.get("env_var", meta.get("env_var", "")),
        "requires_credentials": src_config.get("requires_credentials", meta.get("requires_credentials", False)),
    }


def is_source_enabled(config: CryptoConfig, source_name: str) -> bool:
    """Check if a source is enabled in config."""
    return get_source_config(config, source_name)["enabled"]


def is_source_required(config: CryptoConfig, source_name: str) -> bool:
    """Check if a source is explicitly marked as required in config."""
    return get_source_config(config, source_name)["required"]


def get_source_env_var(config: CryptoConfig, source_name: str) -> str:
    """Return configured env var for a source, allowing config overrides."""
    return get_source_config(config, source_name)["env_var"]


def source_requires_credentials(config: CryptoConfig, source_name: str) -> bool:
    """Return whether missing credentials should prevent a fetch attempt."""
    source_config = get_source_config(config, source_name)
    return bool(source_config["required"] or source_config["requires_credentials"])


def source_missing_required_credentials(config: CryptoConfig, source_name: str) -> bool:
    """Check whether a source needs credentials and they are unavailable."""
    env_var = get_source_env_var(config, source_name)
    return bool(env_var and source_requires_credentials(config, source_name) and not get_source_api_key(source_name, env_var))



def run_data_quality_checks(
    config: CryptoConfig, panel: pd.DataFrame, as_of: pd.Timestamp
) -> dict[str, Any]:
    """Calculate freshness, coverage, and data quality/reliability metrics."""
    panel_as_of = panel[panel["date"] <= as_of]
    total_len = len(panel_as_of)

    source_statuses = {}
    column_coverage = {}
    warnings = []

    reliability_factor = 1.0

    for name, meta in SOURCE_REGISTRY.items():
        enabled = is_source_enabled(config, name)
        critical = meta["critical"]
        sla = meta["freshness_sla_hours"]
        required_cols = meta["required_columns"]

        # Check if file exists on disk
        if name == "binance":
            file_exists = Path(config.data.normalized_path).exists()
        else:
            file_exists = (Path(config.data.raw_dir) / meta["filename"]).exists()

        # Check status
        if not enabled:
            status = "disabled"
            lag_hours = None
            latest_date = None
            coverage_pct = 0.0
            if critical:
                warnings.append(f"Critical source '{name}' is disabled by config.")
        elif not file_exists:
            status = "missing"
            lag_hours = None
            latest_date = None
            coverage_pct = 0.0
            if critical:
                reliability_factor -= 0.15
                warnings.append(f"Critical source '{name}' is missing (raw file not found).")
        else:
            # Check columns in panel
            present_cols = [c for c in required_cols if c in panel_as_of.columns]
            if not present_cols:
                status = "missing"
                lag_hours = None
                latest_date = None
                coverage_pct = 0.0
                if critical:
                    reliability_factor -= 0.15
                    warnings.append(f"Critical source '{name}' is missing columns in features panel.")
            else:
                # Find latest date for this source
                # We check non-nulls among present columns
                source_non_nulls = panel_as_of[present_cols].notna()
                any_non_null = source_non_nulls.any(axis=1)

                if not any_non_null.any():
                    status = "missing"
                    lag_hours = None
                    latest_date = None
                    coverage_pct = 0.0
                    if critical:
                        reliability_factor -= 0.15
                        warnings.append(f"Critical source '{name}' is missing (no non-null data).")
                else:
                    latest_date = panel_as_of.loc[any_non_null, "date"].max()
                    # Calculate lag
                    lag_hours = float((as_of - latest_date).total_seconds() / 3600)
                    # Coverage
                    coverages = []
                    for c in required_cols:
                        if c in panel_as_of.columns:
                            pct = float(panel_as_of[c].notna().mean())
                            column_coverage[c] = pct
                            coverages.append(pct)
                        else:
                            column_coverage[c] = 0.0
                            coverages.append(0.0)

                    coverage_pct = float(np.mean(coverages)) if coverages else 0.0
                    
                    # Check recent coverage in last 30 rows
                    recent_panel = panel_as_of.tail(30)
                    recent_coverages = []
                    for c in required_cols:
                        if c in recent_panel.columns:
                            recent_coverages.append(float(recent_panel[c].notna().mean()))
                        else:
                            recent_coverages.append(0.0)
                    recent_coverage_pct = float(np.mean(recent_coverages)) if recent_coverages else 0.0

                    if lag_hours > sla:
                        status = "stale"
                        if critical:
                            reliability_factor -= 0.10
                            warnings.append(
                                f"Critical source '{name}' is stale (lag of {lag_hours:.1f}h > SLA of {sla}h)."
                            )
                        else:
                            warnings.append(
                                f"Optional source '{name}' is stale (lag of {lag_hours:.1f}h > SLA of {sla}h)."
                            )
                    elif len(present_cols) < len(required_cols) or recent_coverage_pct < 0.90 or coverage_pct < 0.50:
                        status = "partial"
                        if critical:
                            reliability_factor -= 0.05
                            warnings.append(
                                f"Critical source '{name}' has partial coverage (overall: {coverage_pct * 100:.1f}%, recent: {recent_coverage_pct * 100:.1f}%)."
                            )
                    else:
                        status = "success"

        source_statuses[name] = {
            "provider": meta["provider"],
            "category": meta["category"],
            "status": status,
            "lag_hours": lag_hours,
            "latest_date": str(latest_date.date()) if latest_date is not None else None,
            "coverage": coverage_pct,
            "critical": critical,
        }

    # Ensure reliability factor is bounded [0.5, 1.0]
    reliability_factor = max(0.5, min(1.0, reliability_factor))

    if reliability_factor >= 0.95:
        reliability_rating = "High"
    elif reliability_factor >= 0.80:
        reliability_rating = "Warning"
    else:
        reliability_rating = "Degraded"

    if reliability_rating == "High":
        reliability_note = "High reliability: all critical sources are fresh and complete."
    else:
        reliability_note = f"{reliability_rating} reliability: stale or missing sources detected."

    return {
        "reliability_score": float(reliability_factor),
        "reliability_rating": reliability_rating,
        "reliability_note": reliability_note,
        "warnings": warnings,
        "source_statuses": source_statuses,
        "column_coverage": column_coverage,
    }
