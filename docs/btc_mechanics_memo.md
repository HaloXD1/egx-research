# BTC Mechanics Memo

## Purpose

This memo records the BTC assumptions behind `egx crypto-*` research. The first BTC track is spot-only, daily, long-only, free-first data, and benchmarked against DCA.

## How BTC Operates

- Supply: Bitcoin has a fixed terminal supply schedule of 21 million BTC. New supply enters through block rewards, and the subsidy halves roughly every 210,000 blocks.
- Mining: Miners secure blocks with proof of work. Hash rate and miner economics can affect sell pressure, but they are slow-moving regime inputs, not trade triggers alone.
- Settlement: Bitcoin uses a UTXO ledger. On-chain transaction count, active addresses, fees, and exchange flows are useful activity/liquidity proxies, but they have reporting delays and entity-clustering noise.
- Market structure: BTC trades 24/7 globally. Daily candles use UTC dates, so signals fire on UTC daily close and fill at the next daily open.
- Quote risk: The default pair is `BTCUSDT`. USDT is liquid, but it is a stablecoin quote, not bank USD.
- Liquidity: Binance spot is the default source because it has deep BTCUSDT liquidity and public historical klines.
- Derivatives: Funding rates help identify crowded long/short regimes. Open interest is not a full-history v1 feature because free coverage is inconsistent.
- Macro: BTC often behaves like a high-beta liquidity asset. Nasdaq trend, dollar strength, rates, Fed balance sheet, and VIX are risk-regime inputs.

## Indicator Hypotheses

- Price trend should be the base signal: EMA/KAMA trend, Donchian breakouts, ATR risk, and pullback entries have the strongest prior.
- DCA overlay should be the default live-style structure: keep a core BTC accumulation sleeve, vary only the tactical sleeve.
- On-chain should filter sizing: MVRV zones, active-address/tx/hash-rate trends, and exchange net flows can improve regime awareness.
- Sentiment should be contrarian only when price regime is not broken: extreme fear can allow top-ups; extreme greed can trim tactical exposure.
- Macro should reduce risk, not fully block core DCA: weak Nasdaq, strong dollar, rising yields, weak liquidity, or high VIX reduce tactical exposure.

## Latest Research Conclusion

- The expanded combo lab completed on 2026-06-01 favored price-based trend/breakout systems.
- Main selected candidate: `crypto_donchian_breakout` with entry lookback 109, exit lookback 88, regime MA 110, ATR 10, stop 3.9, trail 4.6.
- 2022-to-now raw ROI leader: `crypto_supertrend_combo`.
- 2022-to-now cleanest risk profile: `crypto_trend_adx`.
- Pure on-chain, sentiment, and macro overlays did not beat the stronger price-only families. They remain useful as diagnostics/filters, not standalone trade engines.
- Pullback-only logic can avoid drawdown, but the latest pullback combo failed holdout excess vs DCA and should not be the main BTC engine.

## Validation Rules

- No same-day external feature use. On-chain, sentiment, funding, and macro are shifted one daily bar.
- Compare against monthly DCA, weekly DCA, and buy-and-hold.
- Rank by risk-adjusted excess vs DCA, not raw CAGR alone.
- Require holdout, walk-forward, parameter-neighbor checks, feature ablations, cost stress, and regime tables.
- For BTC, also review Calmar, Sortino, Omega, Ulcer, VaR/CVaR, yearly returns, exposure, trade count, win rate, and profit factor.

## Caveats

- Free data can revise or lag, especially on-chain and macro.
- Binance history is venue-specific, not a consolidated global BTC index.
- MVRV and exchange-flow definitions vary by provider.
- A strategy that avoids drawdown can lag badly in BTC bull regimes.
- This is research output, not live execution.
