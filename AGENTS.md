# AGENTS.md

## Project

- Name: `egx-research`
- Purpose: local EGX30 ETF research, backtesting, optimization, and reporting
- Entry CLI: `egx`

## Key Paths

- Code: `egx_research/`
- Config: `config/default.yaml`
- Stock config: `config/stock_rotation.yaml`
- Raw data: `data/raw/`
- Normalized data: `data/normalized/`
- Run artifacts: `runs/`
- Tests: `tests/`
- TradingView Pine: `tradingview/`
- Stock rotation data: `data/stock_rotation/`
  - current universe: `data/stock_rotation/universe.csv`
  - current panel: `data/stock_rotation/panel.csv`
  - membership snapshots: `data/stock_rotation/membership_snapshots.csv`
  - verified partial history: `data/stock_rotation/membership_verified_partial.csv`
  - review ledger: `data/stock_rotation/historical_review_notes.md`

## Main Commands

- Install: `python3 -m pip install -e '.[dev]'`
- Tests: `pytest -q`
- Ingest: `egx ingest --input <csv-or-url> --config config/default.yaml`
- Optimize: `egx optimize --config config/default.yaml`
- Report: `egx report --run-id <run_id>`
- Stock sync: `egx stock-sync --config config/stock_rotation.yaml`
- Stock backtest: `egx stock-rotate-backtest --config config/stock_rotation.yaml --run-id <run_id>`
- Stock report: `egx stock-rotate-report --run-id <run_id>`
- Stock strategy v2: `egx stock-strategy-v2 --config config/stock_rotation.yaml --run-id <run_id>`
- Stock strategy v3: `egx stock-strategy-v3 --config config/stock_rotation_multifactor.yaml --run-id <run_id>`
- Stock strategy v4: `egx stock-strategy-v4 --config config/stock_rotation_multifactor.yaml --run-id <run_id>`
- Stock strategy robustness: `egx stock-strategy-robustness --config config/stock_rotation_multifactor.yaml --run-id <run_id>`
- Paper track: `egx paper-track --model-run-id <run_id> --start-date YYYY-MM-DD`

## Repo Rules

- Keep strategies `daily`, `long-only`, and `local-first` unless explicitly asked otherwise.
- Compare strategies against monthly DCA using the existing benchmark logic.
- Preserve the current directory contract:
  - source code in `egx_research/`
  - user data in `data/`
  - generated outputs in `runs/`
- Do not hand-edit generated artifacts in `runs/`, `egx_research.egg-info/`, or `__pycache__/`.
- When adding indicators or strategies, wire them through the existing optimizer/reporting path instead of adding one-off scripts where possible.
- Run `pytest -q` after code changes.

## Current Focus

- ETF dataset: `data/normalized/EGX30_ETF.csv`
- Index dataset: `data/normalized/EGX30_INDEX.csv`
- Current data coverage:
  - ETF: `2015-01-15` to `2026-04-05`
  - EGX index: `1998-01-04` to `2026-04-30`
  - stock panel: `2001-05-15` to `2026-04-09`
  - fundamentals: `733` rows
- Current research compares:
  - base families: `trend`, `mean_reversion`, `breakout`
  - Pine-derived family: `fdi_supertrend`
  - structured family: `hierarchy_combo`
  - allocation family: `kama_cci_atr_allocation`
  - overlay family: `dca_tactical_overlay`
  - pullback model: `dca_pullback_only`
  - stock rotation model: `stock_rotation`
  - event rebound max5 v2: `rebound_max5_v2`
  - event rebound max5 v3: `rebound_max5_v3`
  - paper track helper: `egx paper-track`
- Current best overlay default:
  - `80%` core
  - tactical sleeve `100 / 60 / 20`
  - `KAMA 35 / 5 / 51`
  - `CCI 38 / 20`
  - `ATR 7 / 4.4 / 2.6`
- Current best DCA-vs-DCA variant:
  - `dca_pullback_only`
  - `KAMA 26 / 3 / 52`
  - `CCI len 26`
  - buy only if `CCI <= -45`
  - trend intact within `1.7 ATR`
  - `ATR len 8`
  - latest run: `runs/dca-pullback-only-20260411/`
- TradingView implementation:
  - `tradingview/dca_tactical_overlay.pine`
  - `tradingview/dca_tactical_overlay_indicator.pine`
  - `tradingview/dca_pullback_only_indicator.pine`
- Stock rotation implementation:
  - `egx_research/stock_rotation_data.py`
  - `egx_research/stock_rotation.py`
  - `egx_research/stock_rotation_reporting.py`
  - current-basket optimistic run: `runs/stock-rotation-live-20260411/`
  - partial-historical-fee run: `runs/stock-rotation-historical-fee-20260411/`
  - verified-partial-membership run: `runs/stock-rotation-partial-membership-20260413/`
  - fixed buy fee: `5 EGP` per stock buy
  - current conclusion: stock rotation no longer beats ETF DCA once the verified partial membership timeline is used
- Event rebound implementation:
  - `egx_research/stock_strategy_research.py`
  - `egx_research/stock_strategy_validation.py`
  - v2 final run: `runs/rebound-max5-v2-final-20260430/`
  - v3 final run: `runs/rebound-max5-v3-final-20260430/`
  - v3 variant lab: `runs/rebound-max5-v3-variant-lab-20260430/`
  - v3 robustness pack: `runs/stock-strategy-robustness-v3-20260430/`
  - v4 candidate run: `runs/rebound-max5-v4-candidate-20260430/`
  - v4 labs:
    - `runs/rebound-max5-v4-variant-lab-20260430/`
    - `runs/rebound-max5-v4-exposure-lab-20260430/`
    - `runs/rebound-max5-v4-cost-stress-20260430/`
  - current preferred stock strategy candidate: `rebound_max5_v4`
  - v3 uses point-in-time sector multifactor rank + ETF/index/breadth regime filter
  - v3 OOS vs v2: higher TWR/CAGR/Sharpe and lower max DD, but lower full-period CAGR
  - v4 adds future-fundamental cleanup, cost/edge gate, and modest exposure scaling
  - v4 candidate OOS improves vs v3, but full-period CAGR is lower and x3 cost stress is near ETF
  - robustness caveats: ETF/panel data stale vs index; fundamentals include future filing dates and need QA before live use

## Notes For Agents

- `optimization.py` scores candidates on walk-forward, then filters on final holdout vs DCA.
- `reporting.py` expects completed run folders under `runs/<run_id>/`.
- `strategies.py` is the main place to add new indicator logic or hierarchy combinations.
- Stock history currently syncs from:
  - official ETF workbook for current constituents
  - Mubasher stock pages + embedded historical CSV URLs for current-EGX names
- Stock membership currently uses:
  - backfilled baseline snapshot
  - current official ETF workbook snapshot
  - verified partial review deltas from public articles
- Treat stock-rotation results before full historical membership reconstruction as provisional.
- Treat v3 as promising but not live-ready until stale ETF/panel data and future-dated fundamentals are fixed.
