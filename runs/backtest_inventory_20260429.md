# Backtest Inventory (2026-04-29)

Metrics are from each run's summary file. Blank cells mean the run is a scan/artifact folder without comparable portfolio metrics in JSON.

| Run | Strategy | Selected | TWR | CAGR | Max DD | Sharpe | Holdout excess | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| annual-top10-20260414 | annual_top_n_by_prior_return | basket_pullback | 1380.97% | 46.94% | 46.98% | 0.7645 | 970.10% | top_n=10; years=12 |
| annual-top10-stock-20260414 | annual-top10-stock | stock_immediate | 566.64% | 31.11% | 76.76% | 0.6127 |  |  |
| annual-top10-stock-from-2025 | annual-top10-stock | stock_immediate | 69.49% | 55.08% | 9.79% | 2.2529 |  |  |
| annual-top10-stock-pullback-from-2019-20260424 | annual-top10-stock | stock_immediate | 413.22% | 37.14% | 76.03% | 0.6641 |  |  |
| annual-top10-stock-to-2025-01 | annual-top10-stock | stock_immediate | 296.50% | 26.41% | 76.76% | 0.5654 |  |  |
| best-strategy-research-20260414 | best-strategy-research | annual_top10_stock_pullback | 684.12% | 334.99% | 23.02% | 2.9042 | 628.01% | holdout-only comparison |
| blackcat-egx-20260416 | blackcat_ichimoku | blackcat_ichimoku |  |  |  |  |  | research scan; see leaderboard.csv |
| core-satellite-70-30-annual-guard-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 292.15% | 30.20% | 46.43% | 1.1970 | 52.78% | core=0.7; top_n=10; mode=annual |
| core-satellite-adaptive-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 292.15% | 30.20% | 46.43% | 1.1970 | 52.78% | core=0.7; top_n=10; mode=annual |
| core-satellite-fixed-30-70-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 362.42% | 34.41% | 51.01% | 1.2678 | 123.05% | core=0.3; top_n=10; mode=annual |
| core-satellite-fixed-50-50-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 327.53% | 32.39% | 48.74% | 1.2711 | 88.16% | core=0.5; top_n=10; mode=annual |
| core-satellite-fixed-60-40-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 310.15% | 31.33% | 47.59% | 1.2444 | 70.77% | core=0.6; top_n=10; mode=annual |
| dca-overlay-20260411 | dca_tactical_overlay | dca_tactical_overlay |  |  |  |  | -2.70% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| dca-pullback-only-20260411 | dca_pullback_only | dca_pullback_only |  |  |  |  | 0.86% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| dca-pullback-topup-20260411 | dca_pullback_topup | dca_pullback_topup |  |  |  |  | -5.94% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| dca-zone-overlay-20260411 | dca_zone_overlay | dca_zone_overlay |  |  |  |  | -8.62% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| etf-full-20260405 | breakout | breakout |  |  |  |  | -28.94% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| fdi-supertrend-20260405 | fdi_supertrend | fdi_supertrend |  |  |  |  | -16.76% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| fundamental-multifactor-20260426 | institutional_multi_factor_core_v0 | immediate | 414.41% | 26.51% | 43.90% | 1.2063 | 43.26% | top_n=8; holdout_start=2024-10-28 |
| fundamental-multifactor-20260429 | institutional_multi_factor_core_v0 | immediate | 414.41% | 26.51% | 43.90% | 1.2063 | 43.26% | top_n=8; holdout_start=2024-10-28 |
| fundamentals-v1 | institutional_multi_factor_core_v0 | immediate | 323.00% | 23.01% | 42.21% | 1.0790 | 11.95% | top_n=8; holdout_start=2024-10-28 |
| hierarchy-combo-20260405 | no-summary |  |  |  |  |  |  | no summary json |
| hierarchy-combo-20260405b | no-summary |  |  |  |  |  |  | no summary json |
| hierarchy-combo-20260405c | no-summary |  |  |  |  |  |  | no summary json |
| hierarchy-smoke-20260405 | no-summary |  |  |  |  |  |  | no summary json |
| hybrid-filter-egx-20260416 | pullback_base | pullback_base | 0.86% | 38.15% | 13.44% | 1.9159 | 0.86% | research scan; metrics are ETF holdout row |
| institutional-mf-20260416 | institutional_multi_factor_core_v0 | pullback | 295.48% | 21.83% | 40.37% | 0.8697 | 75.82% | top_n=8; holdout_start=2024-10-28 |
| institutional-mf-stage2-20260416 | institutional_multi_factor_core_v0 | pullback | 326.01% | 23.13% | 37.40% | 0.9512 | 28.70% | top_n=8; holdout_start=2024-10-28 |
| kama-cci-atr-alloc-20260411 | kama_cci_atr_allocation | kama_cci_atr_allocation |  |  |  |  | -6.55% | ETF optimizer; summary stores holdout excess/rank, not full metrics |
| paper-track-dca-pullback-only-20260412 | paper-track |  |  |  |  |  |  | paper tracking, not backtest |
| per-stock-pullback-vs-dca-20260414T133630Z | no-summary |  |  |  |  |  |  | no summary json |
| per-stock-pullback-vs-dca-cleaned-20260414T133934Z | no-summary |  |  |  |  |  |  | no summary json |
| probe-stock-only-top10-semiannual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 203.88% | 23.94% | 67.83% | 0.6403 | -35.50% | core=0.5; top_n=10; mode=semiannual |
| probe-stock-only-top10-semiannual-30guard-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 203.88% | 23.94% | 67.83% | 0.6403 | -35.50% | core=0.5; top_n=10; mode=semiannual |
| probe-stock-only-top12-annual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 204.07% | 23.96% | 64.01% | 0.6872 | -35.30% | core=0.5; top_n=12; mode=annual |
| probe-stock-only-top15-annual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 212.05% | 24.58% | 62.43% | 0.7152 | -27.32% | core=0.5; top_n=15; mode=annual |
| probe-stock-only-top3-annual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 264.16% | 28.35% | 47.99% | 1.1107 | 24.78% | core=0.5; top_n=3; mode=annual |
| probe-stock-only-top5-annual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 244.18% | 26.96% | 49.08% | 1.0997 | 4.81% | core=0.5; top_n=5; mode=annual |
| probe-stock-only-top5-semiannual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 245.34% | 27.04% | 48.18% | 1.0975 | 5.97% | core=0.5; top_n=5; mode=semiannual |
| probe-stock-only-top7-annual-20260424 | etf_core_stock_satellite_drawdown_guard | core_satellite | 278.80% | 29.33% | 47.61% | 1.1865 | 39.42% | core=0.5; top_n=7; mode=annual |
| relative-ic-etf-20260424 | ic_weighted_walk_forward_relative_strength | strategy | 176.65% | 21.71% | 55.81% | 0.7884 | -62.72% |  |
| relative-signal-all-etf-20260424 | relative-signal-all-etf-20260424 |  |  |  |  |  |  |  |
| relative-signal-amoc-etf-20260424 | relative-signal-amoc-etf-20260424 |  |  |  |  |  |  |  |
| relative-strength-stockselect-20260429 | kama_plus_3m6mrs_with_liquidity_coverage_gates | pullback | 1155.95% | 43.81% | 50.58% | 1.0016 | 100.27% | top_n=10; holdout_start=2024-10-28 |
| sector-multifactor-20260429 | sector_aware_multi_factor_core_v1 | pullback | 448.29% | 27.68% | 44.49% | 0.9775 | 54.28% | top_n=8; holdout_start=2024-10-28 |
| soft-filter-egx-20260416 | pullback_base | pullback_base | 0.86% | 38.15% | 13.44% | 1.9159 | 0.86% | research scan; metrics are ETF holdout row |
| stock-factor-research-20260429 | stock-factor-research | walk_forward_ic_immediate | 452.00% | 27.62% | 31.37% | 1.2611 | 49.59% | qa_flags=12; best by balanced CAGR/DD |
| stock-momentum-pyramid-20260424 | stock-momentum-pyramid | momentum_pyramid | 84.95% | 12.61% | 66.15% | 0.5267 | -154.42% |  |
| stock-rotation-equal-weight-20260414 | stock-rotation | strategy | 407.85% | 26.28% | 39.96% | 1.0868 | -0.06% |  |
| stock-rotation-historical-fee-20260411 | stock-rotation | strategy | 607.74% | 32.44% | 40.12% | 1.2190 | 199.83% |  |
| stock-rotation-live-20260411 | stock-rotation | strategy | 615.80% | 32.66% | 40.02% | 1.2254 | 207.90% |  |
| stock-rotation-partial-membership-20260413 | stock-rotation | strategy | 397.76% | 25.92% | 40.10% | 1.0765 | -10.14% |  |
| stock-scorecard-20260414T145050Z | no-summary |  |  |  |  |  |  | no summary json |
| stock-select-annual-20260414 | kama_plus_3m6mrs_with_liquidity_coverage_gates | immediate | 654.92% | 33.68% | 81.14% | 0.6303 | 16.28% | top_n=10; holdout_start=2024-10-28 |
| stock-select-monthly-20260414 | kama_plus_3m6mrs_with_liquidity_coverage_gates | pullback | 1155.95% | 43.81% | 50.58% | 1.0016 | 100.27% | top_n=10; holdout_start=2024-10-28 |
| stock-strategy-event-driven-20260429 | no-summary |  |  |  |  |  |  | no summary json |
| stock-strategy-event-driven-20260429-v2 | no-summary |  |  |  |  |  |  | no summary json |

## Output Files
### annual-top10-20260414
- Summary: `runs/annual-top10-20260414/summary.json`
- `runs/annual-top10-20260414/annual_top10_report.html`
- `runs/annual-top10-20260414/equity_curve.csv`
- `runs/annual-top10-20260414/summary.json`
- `runs/annual-top10-20260414/annual_selections.csv`

### annual-top10-stock-20260414
- Summary: `runs/annual-top10-stock-20260414/summary_stock.json`
- `runs/annual-top10-stock-20260414/annual_top10_stock_report.html`
- `runs/annual-top10-stock-20260414/equity_curve_stock.csv`
- `runs/annual-top10-stock-20260414/summary_stock.json`
- `runs/annual-top10-stock-20260414/annual_selections.csv`

### annual-top10-stock-from-2025
- Summary: `runs/annual-top10-stock-from-2025/summary_stock.json`
- `runs/annual-top10-stock-from-2025/annual_top10_stock_report.html`
- `runs/annual-top10-stock-from-2025/equity_curve_stock.csv`
- `runs/annual-top10-stock-from-2025/summary_stock.json`
- `runs/annual-top10-stock-from-2025/annual_selections.csv`

### annual-top10-stock-pullback-from-2019-20260424
- Summary: `runs/annual-top10-stock-pullback-from-2019-20260424/summary_stock.json`
- `runs/annual-top10-stock-pullback-from-2019-20260424/annual_top10_stock_report.html`
- `runs/annual-top10-stock-pullback-from-2019-20260424/equity_curve_stock.csv`
- `runs/annual-top10-stock-pullback-from-2019-20260424/summary_stock.json`
- `runs/annual-top10-stock-pullback-from-2019-20260424/annual_selections.csv`

### annual-top10-stock-to-2025-01
- Summary: `runs/annual-top10-stock-to-2025-01/summary_stock.json`
- `runs/annual-top10-stock-to-2025-01/annual_top10_stock_report.html`
- `runs/annual-top10-stock-to-2025-01/equity_curve_stock.csv`
- `runs/annual-top10-stock-to-2025-01/summary_stock.json`
- `runs/annual-top10-stock-to-2025-01/annual_selections.csv`

### best-strategy-research-20260414
- Summary: `runs/best-strategy-research-20260414/summary.json`
- `runs/best-strategy-research-20260414/summary.json`
- `runs/best-strategy-research-20260414/leaderboard.csv`

### blackcat-egx-20260416
- Summary: `runs/blackcat-egx-20260416/summary.json`
- `runs/blackcat-egx-20260416/summary.json`
- `runs/blackcat-egx-20260416/leaderboard.csv`

### core-satellite-70-30-annual-guard-20260424
- Summary: `runs/core-satellite-70-30-annual-guard-20260424/summary.json`
- `runs/core-satellite-70-30-annual-guard-20260424/equity_curve.csv`
- `runs/core-satellite-70-30-annual-guard-20260424/summary.json`
- `runs/core-satellite-70-30-annual-guard-20260424/selected_holdings.csv`

### core-satellite-adaptive-20260424
- Summary: `runs/core-satellite-adaptive-20260424/summary.json`
- `runs/core-satellite-adaptive-20260424/equity_curve.csv`
- `runs/core-satellite-adaptive-20260424/summary.json`
- `runs/core-satellite-adaptive-20260424/selected_holdings.csv`

### core-satellite-fixed-30-70-20260424
- Summary: `runs/core-satellite-fixed-30-70-20260424/summary.json`
- `runs/core-satellite-fixed-30-70-20260424/equity_curve.csv`
- `runs/core-satellite-fixed-30-70-20260424/summary.json`
- `runs/core-satellite-fixed-30-70-20260424/selected_holdings.csv`

### core-satellite-fixed-50-50-20260424
- Summary: `runs/core-satellite-fixed-50-50-20260424/summary.json`
- `runs/core-satellite-fixed-50-50-20260424/equity_curve.csv`
- `runs/core-satellite-fixed-50-50-20260424/summary.json`
- `runs/core-satellite-fixed-50-50-20260424/selected_holdings.csv`

### core-satellite-fixed-60-40-20260424
- Summary: `runs/core-satellite-fixed-60-40-20260424/summary.json`
- `runs/core-satellite-fixed-60-40-20260424/equity_curve.csv`
- `runs/core-satellite-fixed-60-40-20260424/summary.json`
- `runs/core-satellite-fixed-60-40-20260424/selected_holdings.csv`

### dca-overlay-20260411
- Summary: `runs/dca-overlay-20260411/report_summary.json`
- `runs/dca-overlay-20260411/report.html`
- `runs/dca-overlay-20260411/equity_curve_holdout.csv`
- `runs/dca-overlay-20260411/report_summary.json`

### dca-pullback-only-20260411
- Summary: `runs/dca-pullback-only-20260411/report_summary.json`
- `runs/dca-pullback-only-20260411/report.html`
- `runs/dca-pullback-only-20260411/equity_curve_holdout.csv`
- `runs/dca-pullback-only-20260411/report_summary.json`

### dca-pullback-topup-20260411
- Summary: `runs/dca-pullback-topup-20260411/report_summary.json`
- `runs/dca-pullback-topup-20260411/report.html`
- `runs/dca-pullback-topup-20260411/equity_curve_holdout.csv`
- `runs/dca-pullback-topup-20260411/report_summary.json`

### dca-zone-overlay-20260411
- Summary: `runs/dca-zone-overlay-20260411/report_summary.json`
- `runs/dca-zone-overlay-20260411/report.html`
- `runs/dca-zone-overlay-20260411/equity_curve_holdout.csv`
- `runs/dca-zone-overlay-20260411/report_summary.json`

### etf-full-20260405
- Summary: `runs/etf-full-20260405/report_summary.json`
- `runs/etf-full-20260405/report.html`
- `runs/etf-full-20260405/equity_curve_holdout.csv`
- `runs/etf-full-20260405/report_summary.json`

### fdi-supertrend-20260405
- Summary: `runs/fdi-supertrend-20260405/report_summary.json`
- `runs/fdi-supertrend-20260405/report.html`
- `runs/fdi-supertrend-20260405/equity_curve_holdout.csv`
- `runs/fdi-supertrend-20260405/report_summary.json`

### fundamental-multifactor-20260426
- Summary: `runs/fundamental-multifactor-20260426/summary_selection.json`
- `runs/fundamental-multifactor-20260426/equity_curve_selection.csv`
- `runs/fundamental-multifactor-20260426/summary_selection.json`
- `runs/fundamental-multifactor-20260426/selected_holdings_rebalance.csv`

### fundamental-multifactor-20260429
- Summary: `runs/fundamental-multifactor-20260429/summary_selection.json`
- `runs/fundamental-multifactor-20260429/stock_selection_report.html`
- `runs/fundamental-multifactor-20260429/equity_curve_selection.csv`
- `runs/fundamental-multifactor-20260429/summary_selection.json`
- `runs/fundamental-multifactor-20260429/selected_holdings_rebalance.csv`

### fundamentals-v1
- Summary: `runs/fundamentals-v1/summary_selection.json`
- `runs/fundamentals-v1/stock_selection_report.html`
- `runs/fundamentals-v1/equity_curve_selection.csv`
- `runs/fundamentals-v1/summary_selection.json`
- `runs/fundamentals-v1/selected_holdings_rebalance.csv`

### hierarchy-combo-20260405
- No standard output files matched inventory patterns.

### hierarchy-combo-20260405b
- No standard output files matched inventory patterns.

### hierarchy-combo-20260405c
- No standard output files matched inventory patterns.

### hierarchy-smoke-20260405
- No standard output files matched inventory patterns.

### hybrid-filter-egx-20260416
- Summary: `runs/hybrid-filter-egx-20260416/summary.json`
- `runs/hybrid-filter-egx-20260416/summary.json`
- `runs/hybrid-filter-egx-20260416/leaderboard.csv`

### institutional-mf-20260416
- Summary: `runs/institutional-mf-20260416/summary_selection.json`
- `runs/institutional-mf-20260416/stock_selection_report.html`
- `runs/institutional-mf-20260416/equity_curve_selection.csv`
- `runs/institutional-mf-20260416/summary_selection.json`
- `runs/institutional-mf-20260416/selected_holdings_rebalance.csv`

### institutional-mf-stage2-20260416
- Summary: `runs/institutional-mf-stage2-20260416/summary_selection.json`
- `runs/institutional-mf-stage2-20260416/stock_selection_report.html`
- `runs/institutional-mf-stage2-20260416/equity_curve_selection.csv`
- `runs/institutional-mf-stage2-20260416/summary_selection.json`
- `runs/institutional-mf-stage2-20260416/selected_holdings_rebalance.csv`

### kama-cci-atr-alloc-20260411
- Summary: `runs/kama-cci-atr-alloc-20260411/report_summary.json`
- `runs/kama-cci-atr-alloc-20260411/report.html`
- `runs/kama-cci-atr-alloc-20260411/equity_curve_holdout.csv`
- `runs/kama-cci-atr-alloc-20260411/report_summary.json`

### paper-track-dca-pullback-only-20260412
- Summary: `runs/paper-track-dca-pullback-only-20260412/paper_track_summary.json`
- No standard output files matched inventory patterns.

### per-stock-pullback-vs-dca-20260414T133630Z
- No standard output files matched inventory patterns.

### per-stock-pullback-vs-dca-cleaned-20260414T133934Z
- No standard output files matched inventory patterns.

### probe-stock-only-top10-semiannual-20260424
- Summary: `runs/probe-stock-only-top10-semiannual-20260424/summary.json`
- `runs/probe-stock-only-top10-semiannual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top10-semiannual-20260424/summary.json`
- `runs/probe-stock-only-top10-semiannual-20260424/selected_holdings.csv`

### probe-stock-only-top10-semiannual-30guard-20260424
- Summary: `runs/probe-stock-only-top10-semiannual-30guard-20260424/summary.json`
- `runs/probe-stock-only-top10-semiannual-30guard-20260424/equity_curve.csv`
- `runs/probe-stock-only-top10-semiannual-30guard-20260424/summary.json`
- `runs/probe-stock-only-top10-semiannual-30guard-20260424/selected_holdings.csv`

### probe-stock-only-top12-annual-20260424
- Summary: `runs/probe-stock-only-top12-annual-20260424/summary.json`
- `runs/probe-stock-only-top12-annual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top12-annual-20260424/summary.json`
- `runs/probe-stock-only-top12-annual-20260424/selected_holdings.csv`

### probe-stock-only-top15-annual-20260424
- Summary: `runs/probe-stock-only-top15-annual-20260424/summary.json`
- `runs/probe-stock-only-top15-annual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top15-annual-20260424/summary.json`
- `runs/probe-stock-only-top15-annual-20260424/selected_holdings.csv`

### probe-stock-only-top3-annual-20260424
- Summary: `runs/probe-stock-only-top3-annual-20260424/summary.json`
- `runs/probe-stock-only-top3-annual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top3-annual-20260424/summary.json`
- `runs/probe-stock-only-top3-annual-20260424/selected_holdings.csv`

### probe-stock-only-top5-annual-20260424
- Summary: `runs/probe-stock-only-top5-annual-20260424/summary.json`
- `runs/probe-stock-only-top5-annual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top5-annual-20260424/summary.json`
- `runs/probe-stock-only-top5-annual-20260424/selected_holdings.csv`

### probe-stock-only-top5-semiannual-20260424
- Summary: `runs/probe-stock-only-top5-semiannual-20260424/summary.json`
- `runs/probe-stock-only-top5-semiannual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top5-semiannual-20260424/summary.json`
- `runs/probe-stock-only-top5-semiannual-20260424/selected_holdings.csv`

### probe-stock-only-top7-annual-20260424
- Summary: `runs/probe-stock-only-top7-annual-20260424/summary.json`
- `runs/probe-stock-only-top7-annual-20260424/equity_curve.csv`
- `runs/probe-stock-only-top7-annual-20260424/summary.json`
- `runs/probe-stock-only-top7-annual-20260424/selected_holdings.csv`

### relative-ic-etf-20260424
- Summary: `runs/relative-ic-etf-20260424/summary.json`
- `runs/relative-ic-etf-20260424/equity_curve.csv`
- `runs/relative-ic-etf-20260424/summary.json`
- `runs/relative-ic-etf-20260424/selected_holdings.csv`
- `runs/relative-ic-etf-20260424/monthly_rankings.csv`

### relative-signal-all-etf-20260424
- Summary: `runs/relative-signal-all-etf-20260424/summary.json`
- `runs/relative-signal-all-etf-20260424/summary.json`

### relative-signal-amoc-etf-20260424
- Summary: `runs/relative-signal-amoc-etf-20260424/summary.json`
- `runs/relative-signal-amoc-etf-20260424/summary.json`

### relative-strength-stockselect-20260429
- Summary: `runs/relative-strength-stockselect-20260429/summary_selection.json`
- `runs/relative-strength-stockselect-20260429/equity_curve_selection.csv`
- `runs/relative-strength-stockselect-20260429/summary_selection.json`
- `runs/relative-strength-stockselect-20260429/selected_holdings_rebalance.csv`

### sector-multifactor-20260429
- Summary: `runs/sector-multifactor-20260429/summary_selection.json`
- `runs/sector-multifactor-20260429/stock_selection_report.html`
- `runs/sector-multifactor-20260429/equity_curve_selection.csv`
- `runs/sector-multifactor-20260429/summary_selection.json`
- `runs/sector-multifactor-20260429/selected_holdings_rebalance.csv`

### soft-filter-egx-20260416
- Summary: `runs/soft-filter-egx-20260416/summary.json`
- `runs/soft-filter-egx-20260416/summary.json`
- `runs/soft-filter-egx-20260416/leaderboard.csv`

### stock-factor-research-20260429
- Summary: `runs/stock-factor-research-20260429/summary_factor_research.json`
- `runs/stock-factor-research-20260429/equity_curve_factor_research.csv`
- `runs/stock-factor-research-20260429/summary_factor_research.json`
- `runs/stock-factor-research-20260429/ensemble_selected_holdings.csv`
- `runs/stock-factor-research-20260429/walk_forward_ic_selected_holdings.csv`
- `runs/stock-factor-research-20260429/factor_ic_summary.csv`
- `runs/stock-factor-research-20260429/data_quality_flags.csv`

### stock-momentum-pyramid-20260424
- Summary: `runs/stock-momentum-pyramid-20260424/summary.json`
- `runs/stock-momentum-pyramid-20260424/equity_curve.csv`
- `runs/stock-momentum-pyramid-20260424/summary.json`
- `runs/stock-momentum-pyramid-20260424/selected_holdings.csv`
- `runs/stock-momentum-pyramid-20260424/annual_top10_selected_holdings.csv`
- `runs/stock-momentum-pyramid-20260424/monthly_rankings.csv`

### stock-rotation-equal-weight-20260414
- Summary: `runs/stock-rotation-equal-weight-20260414/summary.json`
- `runs/stock-rotation-equal-weight-20260414/stock_rotation_report.html`
- `runs/stock-rotation-equal-weight-20260414/equity_curve.csv`
- `runs/stock-rotation-equal-weight-20260414/summary.json`
- `runs/stock-rotation-equal-weight-20260414/selected_holdings_monthly.csv`

### stock-rotation-historical-fee-20260411
- Summary: `runs/stock-rotation-historical-fee-20260411/summary.json`
- `runs/stock-rotation-historical-fee-20260411/stock_rotation_report.html`
- `runs/stock-rotation-historical-fee-20260411/equity_curve.csv`
- `runs/stock-rotation-historical-fee-20260411/summary.json`
- `runs/stock-rotation-historical-fee-20260411/selected_holdings_monthly.csv`

### stock-rotation-live-20260411
- Summary: `runs/stock-rotation-live-20260411/summary.json`
- `runs/stock-rotation-live-20260411/stock_rotation_report.html`
- `runs/stock-rotation-live-20260411/equity_curve.csv`
- `runs/stock-rotation-live-20260411/summary.json`
- `runs/stock-rotation-live-20260411/selected_holdings_monthly.csv`

### stock-rotation-partial-membership-20260413
- Summary: `runs/stock-rotation-partial-membership-20260413/summary.json`
- `runs/stock-rotation-partial-membership-20260413/stock_rotation_report.html`
- `runs/stock-rotation-partial-membership-20260413/equity_curve.csv`
- `runs/stock-rotation-partial-membership-20260413/summary.json`
- `runs/stock-rotation-partial-membership-20260413/selected_holdings_monthly.csv`

### stock-scorecard-20260414T145050Z
- No standard output files matched inventory patterns.

### stock-select-annual-20260414
- Summary: `runs/stock-select-annual-20260414/summary_selection.json`
- `runs/stock-select-annual-20260414/stock_selection_report.html`
- `runs/stock-select-annual-20260414/equity_curve_selection.csv`
- `runs/stock-select-annual-20260414/summary_selection.json`
- `runs/stock-select-annual-20260414/selected_holdings_rebalance.csv`

### stock-select-monthly-20260414
- Summary: `runs/stock-select-monthly-20260414/summary_selection.json`
- `runs/stock-select-monthly-20260414/stock_selection_report.html`
- `runs/stock-select-monthly-20260414/equity_curve_selection.csv`
- `runs/stock-select-monthly-20260414/summary_selection.json`
- `runs/stock-select-monthly-20260414/selected_holdings_rebalance.csv`

### stock-strategy-event-driven-20260429
- `runs/stock-strategy-event-driven-20260429/equity_curve_breakout.csv`
- `runs/stock-strategy-event-driven-20260429/equity_curve_rebound.csv`

### stock-strategy-event-driven-20260429-v2
- `runs/stock-strategy-event-driven-20260429-v2/equity_curve_breakout.csv`
