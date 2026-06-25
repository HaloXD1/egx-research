# BTC Strategy KPI Guide

Use ROI as only the first screen. BTC strategies can look excellent because one bull year carried them, while still being hard to hold or fragile out of sample.

## Core Return KPIs

- ROI/TWR: flow-adjusted total return. Good for comparing strategy returns when monthly contributions exist.
- CAGR: annualized growth rate. Better than ROI for comparing different time spans.
- Excess vs DCA: strategy TWR minus DCA TWR. This is the main hurdle because DCA is the BTC baseline.

## Risk-Adjusted KPIs

- Sharpe: return per unit of total volatility. Higher is better, but upside volatility is penalized too.
- Sortino: return per unit of downside volatility. Better for BTC because upside spikes are not treated as bad.
- Omega: gain/loss ratio across daily returns. Above 1 means gains outweighed losses.
- Calmar: CAGR divided by max drawdown. High Calmar means returns were earned without too much worst-case pain.

## Drawdown And Tail KPIs

- Max DD: worst peak-to-trough loss.
- Ulcer: depth plus duration of drawdowns. Lower means easier to hold emotionally.
- Longest DD: longest days below the prior equity high.
- 5% VaR: bad-day threshold. If VaR is 2%, the worst 5% of days lost at least about 2%.
- 5% CVaR: average loss inside the worst 5% of days. More important than VaR for crash risk.

## Behavior KPIs

- Exposure: percent of days or average allocation in BTC. Lower exposure with strong returns is efficient.
- Trades: completed trades. Too few can be fragile; too many can be cost-sensitive.
- Win rate: percent of profitable trades. Useful, but not enough alone.
- Profit factor: gross profit divided by gross loss. Above 1 is profitable; above 2 is strong.
- Yearly returns: shows whether a strategy works across regimes or only one year.

## Latest BTC Top 5 Snapshot

Period: 2022-01-01 to 2026-06-01.

| Strategy | ROI | CAGR | Max DD | Sharpe | Sortino | Omega | Calmar | Vol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `crypto_supertrend_combo` | 247.93% | 21.52% | 15.94% | 0.982 | 0.682 | 1.373 | 1.350 | 22.38% |
| `crypto_donchian_breakout` | 198.03% | 18.62% | 20.20% | 0.833 | 0.577 | 1.280 | 0.922 | 23.87% |
| `crypto_price_trend` | 190.57% | 18.15% | 25.56% | 0.757 | 0.609 | 1.218 | 0.710 | 26.67% |
| `crypto_trend_adx` | 190.19% | 18.12% | 15.24% | 1.001 | 0.554 | 1.513 | 1.189 | 18.28% |
| `crypto_multisignal_score` | 161.44% | 16.21% | 21.27% | 0.834 | 0.697 | 1.235 | 0.762 | 20.54% |

## Decision Rule

Prefer strategies that pass all of these:

- Positive excess vs monthly and weekly DCA.
- Holdout survives, not just walk-forward.
- Max DD and CVaR are acceptable for actual capital.
- Calmar and Omega beat DCA.
- Yearly results do not rely on one single regime.
- Neighbor robustness stays high.
- Trade count is enough to trust but not so high that costs dominate.

Current interpretation: `crypto_donchian_breakout` is the main candidate, `crypto_supertrend_combo` is the aggressive candidate, and `crypto_trend_adx` is the defensive/risk-quality candidate.

## Unresolved Questions

- Actual account size?
- Max tolerable drawdown?
- Core DCA sleeve size?
