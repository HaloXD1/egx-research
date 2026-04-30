# Institutional EGX Blueprint Pack

This pack turns institutional-style ideas into 5 concrete EGX blueprints that are public-feasible, long-only, daily-compatible, and decision-complete for later implementation.

## Fixed Card Schema

Every blueprint in this pack follows the same schema:

1. Thesis
2. Why institutions use it
3. Why it may fit EGX specifically
4. Universe and exclusions
5. Data inputs and exact public source candidates
6. Lag assumptions and survivorship controls
7. Signal definitions and rank formula
8. Allocation and risk-budget rules
9. Rebalance frequency and turnover controls
10. Benchmark set
11. Acceptance criteria
12. Kill criteria
13. Public-feasible v0
14. Phase-2 upgrades
15. References

## 1. EGX Multi-Factor Stock Selection Core

### Thesis

Build a long-only EGX stock-selection engine that ranks names on a diversified mix of value, momentum, quality, low-risk, and liquidity instead of relying on one timing signal. The objective is to outperform ETF DCA and the repo's current stock-selection outputs through cross-sectional selection and disciplined turnover control.

### Why institutions use it

Large systematic equity managers typically stack multiple weak but robust premia instead of searching for one dominant indicator. The institutional edge comes from combining value, momentum, quality, low-risk, and liquidity into a repeatable ranking process.

### Why it may fit EGX specifically

EGX is concentrated, capacity-constrained, and less efficiently arbitraged than large developed markets. That should make cross-sectional ranking more promising than pure ETF timing, especially when liquidity and implementation frictions are handled explicitly.

### Universe and exclusions

- Eligible universe on each rebalance date:
  - All symbols in `data/stock_rotation/membership_verified_partial.csv` that are active on the rebalance date.
  - Current names in `data/stock_rotation/universe.csv` may be included only when membership history is missing.
- Minimum history:
  - 252 trading days for entry.
  - 756 trading days preferred for stable beta and quality proxies.
- Liquidity gate:
  - Median daily value traded over the last 63 bars must exceed `min_median_daily_value_egp`.
  - Median daily volume over the last 63 bars must exceed `min_median_daily_volume`.
- Exclude:
  - Names with missing OHLCV on the rebalance date.
  - Names with less than 92% trading-day coverage over the last 126 bars.
  - Duplicate economic exposures when both foreign-currency and EGP share lines exist; keep the more liquid line unless a later version proves both should coexist.

### Data inputs and exact public source candidates

| Dataset | Public source candidate |
| --- | --- |
| Daily OHLCV | Mubasher historical CSV URLs already captured in `data/stock_rotation/universe.csv` |
| Current constituents | EGX30 ETF rebalancing workbook via `config/stock_rotation.yaml` |
| Historical membership | Mubasher and ArabFinance review articles already logged in `data/stock_rotation/historical_review_notes.md` |
| Dividend actions | Mubasher stock pages, company IR pages, EGX disclosures |
| Capital increase / rights issue history | EGX disclosures, Mubasher news, company IR pages |
| Balance-sheet quality fields for v1 | Company annual reports, EGX filing archive, Mubasher financial statement pages when available |

### Lag assumptions and survivorship controls

- Membership:
  - Use only constituents known as of the rebalance date.
  - Verified additions/removals become active from their published effective date.
- Price signals:
  - Computed from close data available through rebalance close; executed next open.
- Dividends and capital actions:
  - Apply a minimum 3 trading-day lag after public posting in v0.
- Annual-report fundamentals:
  - Apply a conservative 45 calendar-day lag after filing date in v1.
- Missing fields:
  - Missing sub-signals score to cross-sectional median, not zero.
- Survivorship:
  - Never use the current universe alone; membership history must drive inclusion.

### Signal definitions and rank formula

Use percentile ranks each rebalance date. Missing signals receive the median rank of `0.50`.

- Momentum sleeve: 35%
  - `mom_12_1`: 12-month return excluding the most recent 21 bars, weight 20%.
  - `mom_6_1`: 6-month return excluding the most recent 21 bars, weight 10%.
  - `rev_1m_penalty`: 21-bar return, ranked inverse, weight 5%.
- Value sleeve: 20%
  - `dividend_yield_rank`: trailing 12-month cash dividend yield, weight 10%.
  - `valuation_proxy_rank`: inverse of trailing P/E or price-to-book when backfilled, weight 10%.
  - Until valuation backfill exists, reassign the 10% valuation weight to `dividend_yield_rank`.
- Quality sleeve: 20%
  - `payout_stability_rank`: fewer dividend omissions over trailing 3 years, weight 10%.
  - `capital_discipline_rank`: penalize frequent dilutive rights issues / capital raises, weight 10%.
  - Until full filing history exists, `capital_discipline_rank` uses event history only.
- Low-risk sleeve: 15%
  - `beta_rank`: inverse 252-day beta vs EGX30 ETF or EGX30 index, weight 10%.
  - `vol_rank`: inverse 63-day realized volatility, weight 5%.
- Liquidity sleeve: 10%
  - `adv_rank`: 63-day median daily traded value rank, weight 10%.

Final rank:

`total_score = 0.35 * momentum + 0.20 * value + 0.20 * quality + 0.15 * low_risk + 0.10 * liquidity`

### Allocation and risk-budget rules

- Rebalance monthly by default; quarterly is the backup variant.
- Hold top 8 names equal-weight.
- Maximum name weight: 15%.
- Sector cap:
  - 35% by economic sector when sector tags are available.
  - Until then, apply issuer-group concentration cap of 20% where duplicate lines exist.
- Cash handling:
  - Unfilled names due to liquidity or missing data leave residual cash uninvested in v0.
- Turnover buffer:
  - Existing holdings stay in portfolio if they remain in top 12.
  - New names enter only if ranked in top 8.

### Rebalance frequency and turnover controls

- Primary cadence: first trading day of each month.
- Backup cadence: first trading day of each quarter.
- Max one-way turnover target: 35% per rebalance.
- If forecast turnover exceeds 35%, defer the lowest-conviction swaps first until turnover falls below target.

### Benchmark set

- ETF monthly DCA.
- ETF buy-and-hold.
- Current `dca_pullback_only`.
- Current `stock-select-monthly`.
- Current `annual-top10-stock`.

### Acceptance criteria

- ETF holdout excess vs DCA is positive.
- Median stock out-of-sample excess vs DCA is positive or materially less negative than current stock-selection baselines.
- Pass rate across symbols exceeds the current `pullback_base` stock pass rate or the strategy beats current stock-selection outputs on median return and drawdown.
- Max drawdown and turnover remain implementable under EGX liquidity constraints.

### Kill criteria

- ETF holdout excess vs DCA stays negative after fee/slippage stress.
- Median stock holdout excess vs DCA is negative by more than 50 bps per month-equivalent or worse than the current stock-selection baseline.
- Turnover consistently breaches the 35% target without improving returns.
- Performance depends on fewer than 3 names or a single rebalance episode.

### Public-feasible v0

- Use OHLCV, membership history, dividend continuity, capital-event penalties, beta, volatility, and liquidity only.
- Skip accounting-heavy quality and valuation fields unless publicly backfilled with verified filing lags.
- Implement monthly and quarterly variants; keep the better of the two only if both survive holdout and stock cross-section tests.

### Phase-2 upgrades

- Add book-to-price, EBIT/EV, ROE, gross profitability, accruals, net leverage, and interest coverage.
- Add sector-neutral ranking.
- Add macro-regime risk budgeting from Blueprint 3 as an outer overlay.

### References

- [Value and Momentum Everywhere](https://www.aqr.com/insights/research/journal-article/value-and-momentum-everywhere)
- [Quality Minus Junk](https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk)
- [Betting Against Beta](https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta)

## 2. EGX Event and Disclosure Reaction

### Thesis

Build a rules-based long-only event engine that captures positive post-disclosure drift after earnings, dividend, contract, index, and corporate-action events, while explicitly avoiding discretionary narrative trading.

### Why institutions use it

Institutional event-driven desks exploit slow information diffusion, mechanical flows, and behavioral underreaction around corporate events. Text, disclosures, and event metadata are often as important as price patterns.

### Why it may fit EGX specifically

EGX has meaningful retail participation, uneven analyst coverage, and slower disclosure digestion than larger markets. Corporate events and index-review changes can create durable follow-through if filters separate real information from low-quality noise.

### Universe and exclusions

- Eligible universe:
  - Current and historical EGX30 names first.
  - Expand to liquid non-index names only after v1.
- Event-eligible only if:
  - 63-day median traded value exceeds the repo liquidity minimum.
  - At least 252 bars of history exist.
- Exclude:
  - Rights issues or capital increases flagged as distress financing unless the event is categorized positive by rule.
  - Names with unresolved duplicate listing-line ambiguity.

### Data inputs and exact public source candidates

| Dataset | Public source candidate |
| --- | --- |
| Company disclosures | EGX disclosure archive |
| Company-specific news | Mubasher stock news pages |
| Financial headlines and summaries | ArabFinance company and market news |
| Dividend, rights issue, capital increase events | Mubasher, EGX disclosures, company IR pages |
| Index review adds/removes | Mubasher and ArabFinance index-review articles |
| Price/volume confirmation | Mubasher OHLCV |

### Lag assumptions and survivorship controls

- Every event enters the signal inventory only after public timestamp plus 1 full trading day.
- Index-review events use the article publication date unless the effective date is explicitly provided; implementation uses the later of the two.
- Text-derived labels in v0 must be dictionary-based and deterministic.
- No retrospective event relabeling after observing price reaction.

### Signal definitions and rank formula

Event score is additive. Start at `0`.

Positive event points:

- Earnings / profit beat or clear profit acceleration vs prior comparable period: `+3`
- Dividend initiation or dividend increase: `+2`
- Major contract, export award, expansion announcement, or capacity increase: `+2`
- Confirmed EGX30 add or major benchmark inclusion: `+2`
- Share buyback or management insider buy disclosure: `+2`

Negative event filters:

- Distress-like rights issue, emergency capital raise, qualified auditor note, or missed guidance: `-3`
- EGX30 deletion or forced index exit: `-2`
- Dividend cancellation after prior regular payouts: `-2`

Price/volume confirmation:

- Require 3-day cumulative return between `+2%` and `+12%` after the event.
- Require 3-day average volume at least `1.5x` the trailing 20-day median.
- Reject signals with >`+12%` 3-day move as likely already over-discounted in v0.

Entry score:

- `event_entry_score = event_points + confirmation_points`
- Confirmation adds `+1` if both price and volume conditions pass.
- Enter only if `event_entry_score >= 3`.

### Allocation and risk-budget rules

- Hold at most 5 concurrent event positions.
- Equal weight across active positions.
- Maximum per-name weight: 12%.
- If more than 5 names qualify, rank by:
  1. event score
  2. 20-day ADV rank
  3. 63-day volatility rank inverse

### Rebalance frequency and turnover controls

- Review signals daily after close.
- Enter next open.
- Default holding period: 20 trading days.
- Early exit triggers:
  - close below 20-day EMA
  - trailing stop of 10%
  - negative follow-up disclosure
- Cooldown: no re-entry for 10 trading days after exit unless a new event class appears.

### Benchmark set

- ETF monthly DCA.
- ETF buy-and-hold.
- `dca_pullback_only`.
- Multi-Factor Stock Selection Core once built.

### Acceptance criteria

- Positive ETF holdout excess vs DCA when aggregated into an equal-weight event basket.
- Positive median event-trade expectancy net of fees/slippage.
- Positive pass rate across symbols that is not driven solely by index-review events.

### Kill criteria

- Net expectancy turns negative after realistic fees/slippage.
- More than 50% of alpha comes from one event class or one calendar year.
- Signal count is too sparse to support diversified deployment.

### Public-feasible v0

- Use manually classified event categories from Mubasher, EGX, ArabFinance, and company IR.
- Use deterministic keyword dictionaries for positive/negative text classification.
- Limit to EGX30 names and high-liquidity names only.

### Phase-2 upgrades

- Arabic/English disclosure NLP.
- Named-entity and topic extraction.
- Broker-revision and management-language sentiment.
- Event surprise normalization against company history.

### References

- [Reimagining alpha with data and AI](https://www.blackrock.com/us/financial-professionals/insights/data-driven-investing)
- [Alternative data for systematic investing](https://www.blackrock.com/gls-download/literature/whitepaper/alt-data-for-systematic-investing.pdf)
- [The Rise of Machine Learning at Man AHL](https://www.man.com/insights/the-rise-of-machine-learning)

## 3. EGX Macro-Regime Risk Budgeter

### Thesis

Use slow-moving macro and market-internals data to set the risk budget of the stock sleeves, rather than trying to flip all-in/all-out directional timing signals.

### Why institutions use it

Institutional allocators commonly let macro and cross-asset information control position size, leverage, and diversification instead of replacing underlying alpha sleeves. The objective is exposure management, not heroic market calls.

### Why it may fit EGX specifically

EGX is highly sensitive to inflation shocks, FX moves, rates, imported input costs, and liquidity conditions. A slow risk-budget overlay can reduce exposure during locally adverse regimes without destroying the stock-selection process.

### Universe and exclusions

- This blueprint does not pick names directly.
- It scales exposure to Blueprints 1 and 2.
- ETF sleeve and stock sleeve can use separate risk budgets in v1.

### Data inputs and exact public source candidates

| Dataset | Public source candidate |
| --- | --- |
| Core and headline inflation | CBE CPI notes and CAPMAS CPI releases |
| Official exchange rates | CBE exchange-rate historical data |
| Policy rates | CBE MPC press releases |
| Local breadth | EGX30 membership panel and stock OHLCV |
| ETF/index trend | `data/normalized/EGX30_ETF.csv`, `data/normalized/EGX30_INDEX.csv` |
| Commodity stress proxies | Brent and fertilizer/industrial commodity public series from public market data portals |

### Lag assumptions and survivorship controls

- Inflation and policy rates use official publication dates plus a 1-trading-day lag.
- FX uses end-of-day official close.
- Breadth uses only stocks alive and known in the universe at that date.
- Commodity proxies enter only after local market close when sourced from end-of-day public feeds.

### Signal definitions and rank formula

Create 5 sub-scores, each scaled `0.0` to `1.0`.

- Breadth score: 30%
  - Fraction of active universe above 100-day KAMA.
- Inflation score: 25%
  - Higher when 3-month annualized headline CPI and core CPI are falling.
- FX stability score: 20%
  - Higher when 21-day EGP depreciation is within a stable band.
- Rate score: 15%
  - Higher when the last 2 MPC actions are flat or easing and inflation momentum is not re-accelerating.
- Input-cost score: 10%
  - Higher when Brent / relevant commodity proxies are stable to falling.

Composite regime score:

`regime_score = 0.30 * breadth + 0.25 * inflation + 0.20 * fx + 0.15 * rates + 0.10 * input_costs`

Risk budget:

`stock_sleeve_target = 0.40 + 0.60 * regime_score`

This means the stock sleeves run between 40% and 100% of target exposure. No hard all-out regime is allowed in core implementation.

### Allocation and risk-budget rules

- Apply the risk budget multiplicatively to Blueprint 1 and Blueprint 2 target weights.
- Cap monthly change in risk budget at 20 percentage points.
- If breadth and FX stability are both in their lowest bucket, cap exposure at 60% regardless of other scores.

### Rebalance frequency and turnover controls

- Update monthly after the latest macro releases are available.
- Emergency intra-month change allowed only after an MPC decision or FX regime break greater than a preset threshold.

### Benchmark set

- Underlying sleeve with no macro overlay.
- ETF monthly DCA.
- ETF buy-and-hold.
- `dca_pullback_only`.

### Acceptance criteria

- Improves drawdown-adjusted returns for Blueprint 1 and/or 2.
- Reduces worst-case drawdown without fully erasing underlying alpha.
- Remains stable under 1-month data-lag stress.

### Kill criteria

- Overlay improves drawdown only by permanently shrinking exposure and destroying return.
- Results depend on one macro input alone.
- Exposure changes are too frequent to be credible under publication lags.

### Public-feasible v0

- Use breadth, ETF/index trend, CBE rates, CBE/CAPMAS inflation, and official FX data.
- Commodity proxies are optional if public series are inconsistent.

### Phase-2 upgrades

- Better commodity-input maps by issuer.
- Sovereign-risk and CDS proxies.
- Local fund-flow or auction-depth proxies.
- Separate bank / exporter / importer sector budgets.

### References

- [A Century of Evidence on Trend-Following Investing](https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing)
- [How Do Factor Premia Vary Over Time? A Century of Evidence](https://www.aqr.com/Insights/Research/Journal-Article/How-Do-Factor-Premia-Vary-Over-Time-A-Century-of-Evidence)

## 4. EGX Defensive Quality Compounder

### Thesis

Build a long-only “boring winners” portfolio that systematically overweights profitable, stable, lower-beta, less-dilutive EGX names and avoids lottery-like risk.

### Why institutions use it

Institutional investors often pursue quality and low-risk premia because they compound more smoothly and can be implemented in long-only form. The edge comes from avoiding fragile names as much as from owning strong ones.

### Why it may fit EGX specifically

EGX frequently swings between high-beta speculation and periods where stable cash-generative names hold up better. A defensive-quality sleeve may outperform broad market exposure on a drawdown-adjusted basis even if raw upside is lower.

### Universe and exclusions

- Universe:
  - EGX30 and liquid near-index names once v1 arrives.
- Minimum history:
  - 756 bars preferred.
- Exclude:
  - Illiquid names below ADV threshold.
  - Frequent capital-raising names without clear growth funding rationale.
  - Names with missing price history or repeated trading suspensions.

### Data inputs and exact public source candidates

| Dataset | Public source candidate |
| --- | --- |
| Daily OHLCV | Mubasher historical CSV |
| Dividends and payout continuity | Mubasher stock pages, company IR pages |
| Capital-event history | EGX disclosures, Mubasher, company IR |
| Public filings for profitability and leverage | Annual reports, EGX filing archive |
| Beta and drawdown history | Local OHLCV panel |

### Lag assumptions and survivorship controls

- Price-based signals use next-open execution.
- Dividend continuity uses only actions known by the rebalance date.
- Filing-based profitability and leverage metrics use 45-calendar-day reporting lag in v1.
- Missing accounting metrics score to neutral, not bad.

### Signal definitions and rank formula

Use quarterly rebalance and percentile ranks.

- Defensive sleeve: 45%
  - inverse 252-day beta, 20%
  - inverse 126-day downside volatility, 15%
  - inverse 252-day max drawdown, 10%
- Quality sleeve: 35%
  - dividend continuity, 15%
  - dilution penalty inverse, 10%
  - payout consistency, 10%
- Stability/liquidity sleeve: 20%
  - inverse gap risk / daily jumpiness, 10%
  - ADV rank, 10%

Phase-2 quality formula adds:

- ROE / ROA
- gross profitability
- net leverage inverse
- interest coverage

### Allocation and risk-budget rules

- Hold top 10 names equal weight.
- Maximum weight: 12%.
- Cash allowed when fewer than 10 names pass minimum gates.
- Optional overlay from Blueprint 3 may scale exposure later.

### Rebalance frequency and turnover controls

- Quarterly rebalance.
- Hold existing names if they remain in top 15.
- Max one-way turnover target: 25%.

### Benchmark set

- ETF monthly DCA.
- ETF buy-and-hold.
- `dca_pullback_only`.
- Multi-Factor Stock Selection Core.

### Acceptance criteria

- Lower max drawdown than ETF buy-and-hold and Multi-Factor Stock Selection Core.
- Comparable or better return/drawdown ratio than the repo’s best current strategy family.
- Median stock out-of-sample excess vs DCA not materially worse than the Multi-Factor core.

### Kill criteria

- Strategy becomes a hidden sector bet.
- Return comes entirely from one bull window and disappears after costs.
- Quarterly turnover still exceeds 25% without benefit.

### Public-feasible v0

- Use price stability, beta, drawdown, liquidity, dividend continuity, and capital-event penalties only.
- Mark accounting-heavy quality fields as phase-2 unless public filing backfill is actually done.

### Phase-2 upgrades

- Add filing-derived profitability and leverage.
- Add cash-flow quality and accrual measures.
- Add board/governance and ownership-stability proxies where public and legal.

### References

- [Quality Minus Junk](https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk)
- [Betting Against Beta](https://www.aqr.com/Insights/Research/Journal-Article/Betting-Against-Beta)

## 5. EGX Alternative-Data / NLP Lite

### Thesis

Create a public-feasible alternative-data and NLP sleeve that extracts incremental alpha from text, attention, hiring, and thematic-input proxies, without assuming access to expensive institutional vendor feeds.

### Why institutions use it

Institutional systematic teams increasingly build large signal libraries from alternative data, text, and ML pipelines because edge now comes from information variety and disciplined signal curation rather than indicator novelty.

### Why it may fit EGX specifically

EGX has lower structured-data depth than developed markets, which makes public text and narrative information relatively more valuable. The market may underreact or misprice local disclosure tone, contract news, staffing signals, and thematic cost exposure.

### Universe and exclusions

- Start with EGX30 and liquid near-index names only.
- Exclude names with minimal news/disclosure footprint in both Arabic and English.
- Exclude names where attention proxies are too noisy or ambiguous by ticker/name overlap.

### Data inputs and exact public source candidates

| Dataset | Public source candidate |
| --- | --- |
| EGX disclosures | EGX disclosure archive |
| Company-specific headlines | Mubasher and ArabFinance |
| IR announcements and annual reports | Company IR pages |
| Search interest | Google Trends |
| Hiring intensity | Company career pages, public job boards such as Wuzzuf, optional LinkedIn/manual capture only if legally permitted |
| Commodity/input-cost maps | Public commodity series and manually curated issuer-exposure maps |

### Lag assumptions and survivorship controls

- Text is timestamped by publication time; enter only next trading day.
- Search and hiring data use weekly or monthly frequency only.
- No retroactive dictionary edits allowed after backtest inspection.
- Company-name ambiguity must be resolved with a maintained alias map before scoring.

### Signal definitions and rank formula

V0 uses a weekly cross-sectional score:

- Disclosure/news tone: 40%
  - dictionary-based positive vs negative score from disclosures and headlines
- Attention divergence: 20%
  - rising search/news attention with weak recent price reaction scores higher
- Hiring intensity: 20%
  - monthly hiring growth for strategic roles where public data is available
- Input-cost / thematic fit: 20%
  - issuer exposure map combined with current macro/commodity backdrop

Final score:

`alt_score = 0.40 * tone + 0.20 * attention_divergence + 0.20 * hiring + 0.20 * thematic_fit`

### Allocation and risk-budget rules

- This sleeve is phase-2 by default.
- When activated, hold top 5 names equal weight.
- Maximum name weight: 10%.
- Can also be used as a 15% overlay sleeve on top of Blueprint 1 rather than a standalone portfolio.

### Rebalance frequency and turnover controls

- Weekly score refresh.
- Entry threshold: top decile of available scores.
- Exit threshold: below median score or negative disclosure shock.
- Max one-way turnover target: 30% weekly equivalent.

### Benchmark set

- ETF monthly DCA.
- ETF buy-and-hold.
- Multi-Factor Stock Selection Core.
- Event and Disclosure Reaction.

### Acceptance criteria

- Adds incremental return or information coefficient beyond price/volume-only signals.
- Survives realistic lagging, sparse data, and alias-resolution tests.
- Improves portfolio-level results when layered on the Multi-Factor core or Event sleeve.

### Kill criteria

- Signal disappears once strict publication lags are applied.
- Results depend on one noisy source such as search interest alone.
- Coverage is too sparse to support a diversified sleeve.

### Public-feasible v0

- Use only public disclosures, headlines, deterministic text dictionaries, Google Trends, and manually curated thematic maps.
- Hiring signal is optional in v0 unless legal/technical collection is clearly feasible.

### Phase-2 upgrades

- Arabic/English transformer models for disclosure tone.
- Earnings-call or management-language embeddings if transcripts become accessible.
- Premium alternative data vendors for transactions, geolocation, web traffic, or broker revisions.

### References

- [Reimagining alpha with data and AI](https://www.blackrock.com/us/financial-professionals/insights/data-driven-investing)
- [Alternative data for systematic investing](https://www.blackrock.com/gls-download/literature/whitepaper/alt-data-for-systematic-investing.pdf)
- [The Rise of Machine Learning at Man AHL](https://www.man.com/insights/the-rise-of-machine-learning)

## Evaluation Framework

Every blueprint must be judged against the same benchmark and anti-overfitting stack.

### Mandatory benchmarks

- ETF monthly DCA
- ETF buy-and-hold
- current `dca_pullback_only`
- current stock-selection outputs
- current annual-top10 stock outputs where relevant

### Mandatory metrics

- ETF holdout excess vs DCA
- Median stock out-of-sample excess vs DCA
- Pass rate across stocks
- Turnover
- Fee sensitivity
- Start-date sensitivity
- Max drawdown
- Return/drawdown ratio

### Anti-overfitting rules

- No signal may use data before its public timestamp plus the specified lag.
- No retroactive dictionary or event taxonomy edits after observing backtest outcomes.
- Multiple-testing count must be logged for every blueprint family.
- Report Deflated Sharpe Ratio or a similar multiple-testing-adjusted metric before promoting any blueprint to implementation.

### Anti-overfitting references

- [Deflating the Sharpe Ratio](https://ssrn.com/abstract=2465675)
- [How backtest overfitting in finance leads to false discoveries](https://mathinvestor.org/2022/01/how-backtest-overfitting-in-finance-leads-to-false-discoveries/)
