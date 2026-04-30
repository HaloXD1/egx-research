# egx-research

Daily local EGX30 ETF research toolkit.

## Commands

```bash
python3 -m pip install -e .
egx ingest --input /path/to/egx30_etf.csv
egx ingest --input https://stockanalysis.com/quote/egx/EGX30ETF/history/
egx optimize --config config/default.yaml --trials 10
egx report --run-id <run_id> --config config/default.yaml
```

## CSV expectations

Works with common OHLCV exports, including columns like:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- `Date`, `Price`, `Open`, `High`, `Low`, `Vol.`, `Change %`

The normalized file is stored at `data/normalized/EGX30_ETF.csv`.
