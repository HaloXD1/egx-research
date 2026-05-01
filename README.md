# EGX Research Toolkit

Local research toolkit for ingesting Egyptian Exchange market data, normalizing OHLCV datasets, testing investment strategies, and generating reproducible research reports.

## Why It Exists

Public financial data is often messy, inconsistent, and hard to compare across sources. This project creates a repeatable local workflow for cleaning EGX data, running strategy experiments, validating assumptions, and saving reports for later review.

## Highlights

- CSV and URL-based market data ingestion
- OHLCV normalization for different data export formats
- Config-driven strategy research with YAML
- Backtesting and optimization workflows
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

## Data Format

The ingestion layer supports common OHLCV exports, including:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`

Normalized data is stored under:

```text
data/normalized/
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

## Portfolio Note

This project demonstrates data cleaning, time-series analysis, reproducible research, CLI design, and test-backed analytical workflows.
