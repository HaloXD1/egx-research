# Stock Strategy Robustness Pack

## Headline

- v3 OOS TWR: `124.47%` vs v2 `110.50%`.
- v3 OOS CAGR: `45.34%`.
- v3 OOS max DD: `23.75%`.
- v3 OOS Sharpe: `2.31`.

## Stress Notes

- Worst v3 yearly excess TWR vs ETF DCA: `-50.49%`.
- x3 cost OOS CAGR: `31.49%`.
- Start-date sensitivity median CAGR: `28.88%`.
- Worst contribution-day OOS CAGR: `45.28%`.

## Files

- `strategy_comparison.csv`
- `yearly_walk_forward.csv`
- `cost_stress.csv`
- `start_date_sensitivity.csv`
- `contribution_day_sensitivity.csv`
- `data_freshness.csv`
- `summary.json`

## Data Freshness

| dataset | rows | start_date | end_date | days_stale | status | latest_filing_date |
| --- | --- | --- | --- | --- | --- | --- |
| etf | 1766 | 2015-01-15 | 2026-04-05 | 25 | stale |  |
| egx_index | 6901 | 1998-01-04 | 2026-04-30 | 0 | fresh |  |
| stock_panel | 77069 | 2015-01-15 | 2026-04-09 | 21 | stale |  |
| fundamentals | 733 | 2020-12-31 | 2026-12-31 | -305 | future_filing_date | 2027-03-01 |
