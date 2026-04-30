# Institutional EGX Source Matrix

This matrix lists the public-feasible datasets needed for the institutional EGX blueprint pack. The goal is to make later implementation decision-complete before any code is written.

| Dataset | Public source candidate | Frequency | Expected lag | Historical backfill difficulty | Coverage risk | Legal / licensing note | Needed for |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EGX30 ETF OHLCV | Existing repo file `data/normalized/EGX30_ETF.csv` | Daily | Next open after close | Low | Low | Local file maintained by repo | v1 |
| EGX30 index OHLCV | Existing repo file `data/normalized/EGX30_INDEX.csv` | Daily | Next open after close | Low | Low | Local file maintained by repo | v1 |
| Single-stock OHLCV | Mubasher historical CSV URLs already stored in `data/stock_rotation/universe.csv` | Daily | Next open after close | Low | Medium | Public pages; confirm scraping and redistribution terms before automation | v1 |
| Current EGX30 constituents | EGX30 ETF rebalancing workbook referenced in `config/stock_rotation.yaml` | Ad hoc / rebalance | 1 trading day after workbook publication | Low | Low | Public workbook; archive local copies for lineage | v1 |
| Historical EGX30 membership deltas | Mubasher and ArabFinance links already logged in `data/stock_rotation/historical_review_notes.md` | Semiannual / event driven | Effective date or article date + 1 trading day | Medium | Medium | Public articles; store links and raw snapshots, not just parsed output | v1 |
| Dividend actions | Mubasher stock pages, company IR pages, EGX disclosures | Event driven | 1-3 trading days after posting | Medium | Medium | Public pages; corporate IR pages can change without notice | v1 |
| Rights issues / capital increases | EGX disclosures, Mubasher, company IR pages | Event driven | 1-3 trading days after posting | Medium | Medium | Public pages; archive event metadata locally for auditability | v1 |
| Index review add/remove announcements | Mubasher and ArabFinance market news, EGX notices if accessible | Event driven | 1 trading day after publication | Medium | Medium | News pages may change URLs; archive article metadata | v1 |
| Corporate disclosure text | EGX disclosure archive, company IR pages | Event driven | 1 trading day after publication | High | Medium | Public text may have reuse restrictions; store links plus minimal derived metadata | v1 |
| Company headline text | Mubasher and ArabFinance company pages | Event driven | 1 trading day after publication | Medium | Medium | Public text; check TOS before large-scale scraping | v1 |
| Annual reports / statements | Company IR pages, EGX filing archive, Mubasher statement pages when available | Quarterly / annual | Filing date + 45 calendar days | High | High | Public filings; need consistent versioning and filing-date capture | v1 |
| Payout continuity history | Company dividend pages, annual reports, Mubasher cash distribution notices | Annual / event driven | Filing date + 3 trading days | Medium | Medium | Public filings and notices; archive dates and amounts | v1 |
| Beta and realized volatility | Derived from local OHLCV panel | Daily / monthly | Next open after close | Low | Low | Derived in-house from public price data | v1 |
| Liquidity / ADV | Derived from local OHLCV panel | Daily / monthly | Next open after close | Low | Low | Derived in-house from public price and volume data | v1 |
| Breadth metrics | Derived from membership history plus stock OHLCV | Daily / monthly | Next open after close | Medium | Medium | Derived in-house; requires verified membership timeline | v1 |
| Official FX rates | CBE exchange-rate historical data | Daily | End of day + 1 trading day | Low | Low | Official public data | v1 |
| Policy rates / MPC decisions | CBE MPC press releases | Event driven | 1 trading day after release | Low | Low | Official public releases | v1 |
| Headline and core inflation | CAPMAS CPI releases plus CBE inflation notes | Monthly | Release date + 1 trading day | Low | Low | Official public releases | v1 |
| Macro outlook context | CBE monetary policy reports / economic outlook | Quarterly | Report date + 1 trading day | Low | Low | Official public releases | optional |
| Brent / energy proxy | Public end-of-day market data source such as Stooq, ICE summary pages, or Investing.com if terms allow | Daily | End of day + 1 trading day | Medium | Medium | Prefer sources with clearly permitted download terms | v1 |
| Fertilizer / industrial commodity proxy | Public commodity market portals or exchange summary pages | Daily / weekly | End of day + 1 trading day | Medium | High | Source fragmentation is likely; verify continuity before use | optional |
| Search interest | Google Trends | Weekly | Weekly publication lag | Medium | Medium | Public interface; automate only within allowed usage limits | optional |
| Hiring / job-post activity | Company career pages, Wuzzuf, optional LinkedIn/manual capture | Weekly / monthly | 1-7 days after posting | High | High | Legal and scraping constraints must be validated before automation | optional |
| Broker sentiment / revisions | Public research snippets, media summaries, earnings-call commentary where accessible | Event driven | 1 trading day after publication | High | High | Usually incomplete publicly; keep optional unless broad coverage exists | optional |
| Thematic exposure map | Manually curated issuer classifications tied to commodity, rate, FX, and macro sensitivity | Manual refresh monthly | 1 trading day after curation | Medium | Medium | Internal derived dataset; maintain changelog and rationale | v1 |
| Disclosure / headline dictionaries | Internal lexicons derived from public text | Manual / versioned | Version effective date only | Medium | Medium | Internal derived dataset; must be version-controlled and never backfit silently | v1 |

## Minimum Data Package for First Implementation

The first implementation wave should require only:

- Stock OHLCV
- Verified membership history
- ETF and index OHLCV
- Liquidity metrics
- Beta / volatility metrics
- Dividend continuity
- Capital-event history
- CBE FX / rates / inflation
- Public disclosure and news metadata

## Optional / Phase-2 Data Package

Only promote these after the minimum package works:

- Backfilled financial statements
- Search-interest data
- Hiring intensity
- Commodity-input series beyond a simple Brent proxy
- Broker sentiment or revisions
- Premium alternative data

## Source Governance Rules

- Store raw source URLs, acquisition date, and effective date for every event-level dataset.
- Version every derived dictionary, alias map, and thematic exposure map.
- Never treat access difficulty as a modeling feature; if a dataset cannot be captured consistently, move it to optional.
- Prefer official sources when available; use media sources only for event discovery or gap-filling, never without source lineage.
