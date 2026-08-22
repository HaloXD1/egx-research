# EGX Research Toolkit

Local research toolkit for ingesting market data, normalizing OHLCV datasets, testing EGX/BTC strategies, and generating reproducible research reports.

## Why It Exists

Public financial data is often messy, inconsistent, and hard to compare across sources. This project creates a repeatable local workflow for cleaning EGX and BTC data, running strategy experiments, validating assumptions, and saving reports for later review.

## Highlights

- CSV and URL-based market data ingestion
- OHLCV normalization for different data export formats
- Config-driven strategy research with YAML
- Backtesting and optimization workflows
- BTC daily crypto research with price, on-chain, sentiment, macro, and funding features
- BTC paper-track/current-signal reports
- Research reports saved under `runs/`
- Validation utilities and pytest coverage
- CLI-first workflow for reproducible experiments

## Tech Stack

- Python
- pandas / NumPy
- Typer CLI
- Optuna
- Plotly
- YAML configuration
- pytest

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

## Example Commands

Ingest a local CSV:

```bash
egx ingest --input /path/to/egx30_etf.csv
```

Ingest a URL:

```bash
egx ingest --input https://stockanalysis.com/quote/egx/EGX30ETF/history/
```

Run optimization:

```bash
egx optimize --config config/default.yaml --trials 10
```

Generate a report:

```bash
egx report --run-id <run_id> --config config/default.yaml
```

Sync BTC data:

```bash
egx crypto-sync --config config/crypto_btc.yaml
```

Run BTC strategy research:

```bash
egx crypto-research --config config/crypto_btc.yaml --run-id crypto-btc-combo-lab-20260601
egx crypto-report --run-id crypto-btc-combo-lab-20260601
```

Track the latest BTC paper signal:

```bash
egx crypto-paper-track --config config/crypto_btc.yaml --model-run-id crypto-btc-combo-lab-20260601 --start-date 2026-06-01 --run-id crypto-paper-btc-combo-lab
```

## Data Format

The ingestion layer supports common OHLCV exports, including:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`

Normalized data is stored under:

```text
data/normalized/
```

BTC data is stored under:

```text
data/crypto/raw/
data/crypto/normalized/
data/crypto/features/
```

## Project Structure

```text
egx_research/     # Core package and CLI commands
config/           # YAML strategy and research configurations
data/             # Raw and normalized local datasets
runs/             # Generated experiment outputs and reports
tests/            # pytest coverage
docs/             # Research notes and methodology references
```

## What The Research Shows

- **Data quality can flip conclusions.** The stock-rotation strategy looks strong under a live membership snapshot and weak when partial historical membership is applied, so provenance and coverage are part of the result, not just metadata.
- **This is about sensitivity, not a single winner.** The toolkit is built to test how outcomes move when assumptions change, not to claim one strategy "wins."
- **Rules drive outcomes.** Changing selection size, rebalance frequency, or core/satellite weights shifts both return and drawdown; runs document the tradeoffs rather than hiding them.
- **Signals are uneven at the stock level.** Per-stock pullback vs DCA results show a mix of positive and negative deltas across names (for example, some names improve while others lag), which argues for portfolio-level evaluation.
- **Short windows can tell a different story.** 2025-to-date deltas are often weaker than 2020-to-date, reminding us that recent regimes can diverge from long samples.
- **The audit trail is the product.** Each run captures assumptions, metrics, holdout behavior, and data-quality notes so conclusions can be reviewed instead of treated as a black box.
- **BTC trend systems are currently strongest.** In the BTC combo lab, price-based trend/breakout families beat pure on-chain, sentiment, and macro overlays. The selected live candidate is `crypto_donchian_breakout`; `crypto_supertrend_combo` led 2022-to-now ROI, and `crypto_trend_adx` had the cleanest risk profile.

Example outputs: `runs/backtest_inventory_20260429.md`, `runs/per-stock-returns-2020-to-date-latest.csv`, `runs/per-stock-returns-2025-to-date-latest.csv`.

BTC docs: `docs/btc_first_research_run_summary.md`, `docs/btc_strategy_kpi_guide.md`, `docs/btc_mechanics_memo.md`, `docs/btc_bottom_score_guide.md`, `docs/btc_institutional_ensemble.md`.
