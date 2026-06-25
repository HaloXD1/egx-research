# BTC Research Run Summary

Status: expanded combo lab completed on 2026-06-01.

## Commands

```bash
egx crypto-sync --config config/crypto_btc.yaml
egx crypto-research --config config/crypto_btc.yaml --run-id crypto-btc-first
egx crypto-report --run-id crypto-btc-first
egx crypto-paper-track --config config/crypto_btc.yaml --model-run-id crypto-btc-first --start-date 2026-06-01 --run-id crypto-paper-btc-live
egx crypto-research --config config/crypto_btc.yaml --run-id crypto-btc-combo-lab-20260601
egx crypto-report --run-id crypto-btc-combo-lab-20260601
egx crypto-paper-track --config config/crypto_btc.yaml --model-run-id crypto-btc-combo-lab-20260601 --start-date 2026-06-01 --run-id crypto-paper-btc-combo-lab
```

## First Run Record

- Data coverage: 2017-08-17 to 2026-06-01, 3211 BTCUSDT daily rows.
- Top family: `crypto_price_trend`.
- Top params: EMA 28 / EMA 96 / trend MA 110 / ATR 14 / stop 3.2 / trail 4.0.
- Holdout excess vs monthly DCA: +7.13 pp.
- Holdout excess vs weekly DCA: +6.09 pp.
- Worst regime: late 2017 and 2024 bull phases, where DCA captured upside better.
- Cost stress result: edge survives 1x/2x, fails 3x.
- Ablation result: price-only won; sentiment/on-chain did not add enough.
- Caveats: daily spot-only research; no live execution; exchange/USDT/tax risk not modeled.

## Combo Lab Record

- Run id: `crypto-btc-combo-lab-20260601`.
- Report: `runs/crypto-btc-combo-lab-20260601/crypto_report.html`.
- Current signal report: `runs/crypto-paper-btc-combo-lab/current_signal_report.html`.
- Data coverage: 2017-08-17 to 2026-06-01.
- Families tested: price trend, ADX trend, Donchian breakout, Supertrend combo, Bollinger/RSI/CCI pullback, DCA overlay, on-chain overlay, sentiment overlay, macro overlay, ensemble overlay, hierarchy combo, multisignal score.
- Selected candidate: `crypto_donchian_breakout`.
- Selected params: entry lookback 109 / exit lookback 88 / regime MA 110 / ATR 10 / stop 3.9 / trail 4.6.
- Holdout CAGR: 11.47%.
- Holdout Sharpe: 0.652.
- Holdout max drawdown: 16.79%.
- Holdout excess vs monthly DCA: +12.13 pp.
- Holdout excess vs weekly DCA: +11.10 pp.
- Neighbor robustness: 100%.
- 3x cost stress excess vs monthly DCA: +10.74 pp.
- Current signal on 2026-06-01: defensive, target 0% BTC, hold cash/floor.

## Top 5 Since 2022

Period: 2022-01-01 to 2026-06-01. ROI is flow-adjusted TWR.

| Strategy | ROI | CAGR | Max DD | Sharpe | Sortino | Omega | Calmar | Vol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `crypto_supertrend_combo` | 247.93% | 21.52% | 15.94% | 0.982 | 0.682 | 1.373 | 1.350 | 22.38% |
| `crypto_donchian_breakout` | 198.03% | 18.62% | 20.20% | 0.833 | 0.577 | 1.280 | 0.922 | 23.87% |
| `crypto_price_trend` | 190.57% | 18.15% | 25.56% | 0.757 | 0.609 | 1.218 | 0.710 | 26.67% |
| `crypto_trend_adx` | 190.19% | 18.12% | 15.24% | 1.001 | 0.554 | 1.513 | 1.189 | 18.28% |
| `crypto_multisignal_score` | 161.44% | 16.21% | 21.27% | 0.834 | 0.697 | 1.235 | 0.762 | 20.54% |

## Tail And Trade KPIs

| Strategy | Ulcer | Longest DD | 5% VaR | 5% CVaR | Exposure | Trades | Win Rate | Profit Factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `crypto_supertrend_combo` | 9.47 | 531d | 1.56% | 3.19% | 31.56% | 26 | 38.46% | 5.11 |
| `crypto_donchian_breakout` | 8.85 | 292d | 2.01% | 3.54% | 36.21% | 4 | 100.00% | inf |
| `crypto_price_trend` | 12.77 | 299d | 2.36% | 3.94% | 48.61% | 22 | 27.27% | 2.52 |
| `crypto_trend_adx` | 5.44 | 375d | 0.90% | 2.41% | 19.28% | 12 | 66.67% | 5.89 |
| `crypto_multisignal_score` | 10.15 | 289d | 1.83% | 2.99% | 39.05% | 31 | 25.81% | 2.68 |

## Yearly Behavior

| Strategy | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---|---:|---:|---:|---:|---:|
| `crypto_supertrend_combo` | 0.00% | 77.03% | 107.43% | -5.25% | 0.00% |
| `crypto_donchian_breakout` | 0.00% | 44.64% | 100.42% | 2.81% | 0.00% |
| `crypto_price_trend` | -8.05% | 73.94% | 91.31% | 0.35% | -5.38% |
| `crypto_trend_adx` | 0.00% | 70.00% | 70.30% | 7.57% | -6.82% |
| `crypto_multisignal_score` | -3.87% | 67.92% | 63.30% | 0.23% | -1.04% |

## Interpretation

- `crypto_supertrend_combo` had the best 2022-to-now ROI.
- `crypto_trend_adx` had the cleanest risk profile: best Sharpe/Omega, lowest max drawdown, lowest tail loss.
- `crypto_donchian_breakout` remains the main candidate because it had the best holdout/robustness balance in the full lab.
- Pure on-chain, sentiment, and macro overlays lagged price-based trend systems.
- The pullback combo looked strong in walk-forward scoring but failed holdout, so it is treated as overfit/inactive.

## Next Actions

- Paper-track `crypto_donchian_breakout`, `crypto_supertrend_combo`, and `crypto_trend_adx` side by side for 30-60 days.
- Add a report section that ranks by composite score: excess vs DCA, Calmar, Sharpe, CVaR, Ulcer, yearly consistency, and trade count.
- Run a stricter validation split: train through 2021, test 2022-now, then rolling yearly OOS.
- Add BTC position-sizing examples for actual capital, monthly contribution, and max acceptable drawdown.
- Build TradingView/Pine alerts only after paper-track behavior matches local signals.

## Generated KPI Files

- `runs/crypto-btc-combo-lab-20260601/btc_top5_2022_now_kpis.csv`
- `runs/crypto-btc-combo-lab-20260601/btc_top5_2022_now_yearly_returns.csv`
- `runs/crypto-btc-combo-lab-20260601/btc_top5_2022_now_trade_tail_kpis.csv`
- `runs/crypto-btc-combo-lab-20260601/btc_2022_now_benchmark_kpis.csv`

## Unresolved Questions

- Actual capital size?
- Max tolerable drawdown?
- Core DCA plus tactical sleeve, or tactical only?
