# BTC Bottom Score Guide

## Commands

```bash
egx crypto-sync --config config/crypto_btc.yaml
egx crypto-bottom-score --config config/crypto_btc.yaml
egx crypto-bottom-score --config config/crypto_btc.yaml --as-of-date 2025-01-01
egx crypto-bottom-score --config config/crypto_btc.yaml --refresh-first
```

Useful guards:

```bash
egx crypto-bottom-score --config config/crypto_btc.yaml --require-source coinmetrics
egx crypto-bottom-score --config config/crypto_btc.yaml --no-html
```

## Local CSV Fallbacks

Optional paid/API sources are never required for sync. If credentials are absent or a fetch fails, the workflow uses local CSV files under `data/crypto/raw/` when present, otherwise marks the source as missing/optional.

Common optional files:

- `btc_etf_flows.csv`
- `futures_positioning.csv`
- `liquidations.csv`
- `options_skew.csv`
- `exchange_flows.csv`
- `glassnode_sth_sopr.csv`
- `stablecoin_supply.csv`
- `exchange_stablecoin_reserves.csv`

All optional rows must have a `date` column. Feature columns are lagged by one day when merged into the scoring panel.

## Env Vars

- `DERIBIT_API_KEY`
- `COINGLASS_API_KEY`
- `CRYPTOQUANT_API_KEY`
- `GLASSNODE_API_KEY`
- `COINMETRICS_API_KEY`
- `COINBASE_API_KEY`
- `DEFILLAMA_API_KEY`
- `EXCHANGE_STABLECOIN_RESERVES_API_KEY`

## Outputs

`egx crypto-bottom-score` writes a run under `runs/<run_id>/` with:

- `bottom_score_summary.json`
- `bottom_probability_grid.csv`
- `bottom_component_scores.csv`
- `bottom_feature_hitrates.csv`
- `bottom_derivatives_audit.csv`
- `bottom_walkforward_validation.csv`
- `bottom_confidence_buckets.csv`
- `bottom_driver_attribution.json`
- `bottom_validation_summary.json`
- `bottom_type_audit.csv`
- `bottom_report.html`

## Interpretation

Confidence estimates whether the recent low holds for the selected horizon/tolerance. It is not a guarantee and can fail on exogenous shocks.

Confidence bands:

- `low`: weak evidence
- `medium`: probe only
- `high`: constructive
- `very_high`: strong, still tranche
- `extreme`: rare, still use invalidation

Bottom type augments confidence:

- `local bounce`
- `tradable swing bottom`
- `cycle bottom`
- `dead-cat bounce risk`

Recommendations are conservative mappings from confidence, regime, and bottom type. They are not financial advice.

## Acceptance Checklist

- Run `egx crypto-sync --config config/crypto_btc.yaml`.
- Run current `egx crypto-bottom-score --config config/crypto_btc.yaml`.
- Run at least one historical `--as-of-date`.
- Confirm missing paid sources are explicit in summaries.
- Run `pytest -q`.
