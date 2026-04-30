# Institutional EGX Roadmap

This roadmap converts the institutional EGX blueprint pack into a staged execution order. The sequence is strict, and every stage has explicit stop/go criteria before moving to the next step.

## Stable Document Schemas

### Blueprint card schema

- Title
- Objective
- Thesis
- Why institutions use it
- Why it may fit EGX specifically
- Universe and exclusions
- Data inputs and exact public source candidates
- Lag assumptions and survivorship controls
- Signal definitions and rank formula
- Allocation and risk-budget rules
- Rebalance frequency and turnover controls
- Benchmark set
- Acceptance criteria
- Kill criteria
- Public-feasible v0
- Phase-2 upgrades
- References

### Source matrix schema

- Dataset
- Public source candidate
- Frequency
- Expected lag
- Historical backfill difficulty
- Coverage risk
- Legal / licensing note
- Needed for

### Roadmap entry schema

- Stage
- Objective
- Required inputs
- Outputs
- Entry criteria
- Exit criteria
- Stop / go decision

## Implementation Order

Priority order is fixed:

1. EGX Multi-Factor Stock Selection Core
2. EGX Event and Disclosure Reaction
3. EGX Macro-Regime Risk Budgeter
4. EGX Defensive Quality Compounder
5. EGX Alternative-Data / NLP Lite

## Stage 1 — Research

### Objective

Lock the design pack and source matrix before any implementation work begins.

### Required inputs

- `docs/institutional_egx_blueprints.md`
- `docs/institutional_egx_source_matrix.md`
- Existing repo benchmark outputs and run summaries

### Outputs

- Approved blueprint definitions
- Fixed benchmark set
- Fixed evaluation metrics
- Data-priority shortlist per blueprint

### Entry criteria

- None; this is the starting stage.

### Exit criteria

- All 5 blueprints exist and use the fixed schema.
- Every required signal in every v0 design has a named public source candidate or is explicitly marked phase-2.
- No blueprint requires shorting, leverage, or derivatives in core scope.

### Stop / go decision

- Go only if the design pack is decision-complete.
- Stop if any v0 blueprint still requires undefined data or discretionary judgment.

## Stage 2 — Data Acquisition

### Objective

Acquire and normalize only the public-feasible minimum data package needed for the first 3 blueprints.

### Required inputs

- Membership history
- Stock OHLCV
- ETF/index OHLCV
- Dividend and capital-event metadata
- Public disclosure/news metadata
- CBE FX, rates, and inflation data

### Outputs

- Versioned raw-source inventory
- Normalized event tables
- Lag-aware metadata tables
- Data-quality and coverage report

### Entry criteria

- Stage 1 complete.

### Exit criteria

- At least 95% of active historical EGX30 names have usable OHLCV.
- Dividend and capital-event history is consistent enough to support v0 proxies.
- Macro data series have unambiguous publication dates.
- Source lineage and versioning rules are implemented before model testing begins.

### Stop / go decision

- Go if coverage and lag discipline are acceptable.
- Stop if event or macro data cannot be timestamped reliably.

## Stage 3 — V0 Proxy Build

### Objective

Implement public-feasible v0 versions only, in strict order.

### Required inputs

- Stage 2 normalized datasets
- Existing benchmark logic

### Outputs

- V0 Multi-Factor Stock Selection Core
- V0 Event and Disclosure Reaction
- V0 Macro-Regime Risk Budgeter

### Entry criteria

- Stage 2 complete.

### Exit criteria

- Each v0 blueprint runs end-to-end with realistic data lags.
- Each v0 blueprint is benchmarked against:
  - ETF monthly DCA
  - ETF buy-and-hold
  - current `dca_pullback_only`
  - current stock-selection outputs
  - annual-top10 stock outputs where relevant
- Multiple-testing count is recorded for all parameter sweeps.

### Stop / go decision

- Promote only the best-performing blueprint that survives the evaluation framework.
- Stop any blueprint that fails its kill criteria before moving on.

## Stage 4 — V1 Implementation

### Objective

Deepen the highest-surviving blueprints with heavier but still public backfill work.

### Required inputs

- Surviving v0 blueprint(s)
- Public filing backfills
- Improved event taxonomies
- Better sector and issuer maps

### Outputs

- V1 Multi-Factor Core with real value and quality fields
- V1 Event sleeve with broader event classes and better text handling
- V1 Macro overlay with validated publication-lag controls
- Defensive Quality Compounder if data coverage permits

### Entry criteria

- At least one v0 blueprint shows a credible edge after fees and lag stress.

### Exit criteria

- Filing-based fields are timestamped and lagged correctly.
- V1 materially improves on v0 or meaningfully improves robustness.
- No improvement is driven only by one rebalance window or one symbol cluster.

### Stop / go decision

- Go only if v1 adds real incremental value net of added complexity.
- Stop if accounting-heavy backfill is incomplete or unstable.

## Stage 5 — V2 Enhancements

### Objective

Layer optional institutional-style upgrades only after v1 proves durable.

### Required inputs

- Stable v1 blueprint(s)
- Optional premium data vendors or richer public text/hiring/search pipelines

### Outputs

- Alternative-Data / NLP Lite
- Filing-NLP and bilingual text models
- Search/hiring overlays
- Better commodity and thematic exposure maps

### Entry criteria

- Stage 4 complete and durable.

### Exit criteria

- Added data improves out-of-sample performance or diversification net of cost and complexity.
- Coverage is broad enough to support a sleeve, not isolated anecdotes.

### Stop / go decision

- Go only if optional data proves additive beyond price, event, and macro information.
- Stop if coverage is sparse or legal/licensing risk is unresolved.

## Strategy-by-Strategy Build Sequence

### 1. EGX Multi-Factor Stock Selection Core

- v0:
  - Momentum, low-risk, liquidity, dividend continuity, capital-discipline proxies.
- v1:
  - Add public filing-derived value and quality.
- v2:
  - Sector neutrality, richer accounting quality, alternative-data overlay.

### 2. EGX Event and Disclosure Reaction

- v0:
  - Rules-based event inventory from Mubasher, EGX, ArabFinance, and company IR.
- v1:
  - Better event taxonomy, improved positive/negative classification.
- v2:
  - NLP and event-surprise normalization.

### 3. EGX Macro-Regime Risk Budgeter

- v0:
  - Breadth, FX, rates, inflation, ETF/index trend.
- v1:
  - Better commodity stress inputs and issuer sensitivity maps.
- v2:
  - Sector-specific risk budgets and optional sovereign-risk proxies.

### 4. EGX Defensive Quality Compounder

- v0:
  - Price-stability, beta, drawdown, liquidity, payout continuity, dilution penalties.
- v1:
  - Add profitability and leverage from public filings.
- v2:
  - Add deeper governance and cash-flow quality fields if feasible.

### 5. EGX Alternative-Data / NLP Lite

- v0:
  - Public disclosures, headlines, deterministic text dictionaries, search trends.
- v1:
  - Better alias maps, bilingual text scoring, richer thematic maps.
- v2:
  - Premium alt-data or advanced ML/NLP if truly additive.

## Evaluation Framework

Every implementation candidate must be scored on the same panel:

- ETF holdout excess vs DCA
- Median stock out-of-sample excess vs DCA
- Pass rate across stocks
- Turnover
- Fee sensitivity
- Start-date sensitivity
- Max drawdown
- Return/drawdown ratio

## Anti-Overfitting Rules

- Apply publication lag to every non-price field.
- Keep a model log of all tested variants, including rejected ones.
- Use a strict holdout and cross-symbol evaluation before promoting a blueprint.
- Report a multiple-testing-adjusted metric such as Deflated Sharpe Ratio before declaring a winner.
- Do not retune event dictionaries, macro thresholds, or factor weights after observing holdout performance without opening a new research cycle.

## Stop / Go Gates by Priority

### Before promoting Multi-Factor Core

- Must beat current stock-selection outputs on either median stock excess vs DCA or return/drawdown ratio.
- Must remain implementable under liquidity and turnover constraints.

### Before promoting Event and Disclosure Reaction

- Must show positive net expectancy after fees/slippage.
- Must not depend mostly on one event class.

### Before promoting Macro-Regime Risk Budgeter

- Must improve drawdown-adjusted performance of an underlying sleeve.
- Must not succeed only by permanently shrinking exposure.

### Before promoting Defensive Quality Compounder

- Must demonstrate lower drawdown with acceptable return sacrifice.
- Must not reduce to one sector or one issuer cluster.

### Before promoting Alternative-Data / NLP Lite

- Must add information beyond price/event/macro signals.
- Must have stable public coverage and legal collection path.

## Success Definition

The project succeeds when one or more institutional-style EGX blueprints:

- beats ETF monthly DCA on ETF holdout,
- improves median stock out-of-sample performance,
- survives lag and fee stress,
- and remains implementable with public-feasible long-only EGX data.

If none clear that bar, the correct outcome is to preserve the current repo winners and stop the blueprint from graduating.
