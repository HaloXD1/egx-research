# BTC Institutional Ensemble

`crypto_institutional_ensemble` is a daily, long-only BTC allocation research family.
It combines eight independent sleeves—trend, derivatives, on-chain, liquidity, macro,
sentiment, mean reversion, and carry—rather than treating a larger indicator count as
evidence of alpha.

Each sleeve emits a causal score and coverage value. An expanding ridge model is
recalibrated every 30 days using only 90-day outcomes that had matured before the
decision date. Coefficients are non-negative, capped, and shrunk toward equal weights.
The model publishes its expected return, confidence, sleeve weights, contributions,
and disagreement on every date.

Five regime probabilities—bull, bear, range, crisis, and recovery—modify a continuous
risk/cost-aware allocation. Exposure is constrained to spot BTC without leverage.
Risk increases occur on the weekly schedule and respect the maximum target change;
risk reductions can occur immediately. Crisis conditions may reduce exposure below
the normal core.

Run a focused research campaign with:

```bash
egx crypto-research \
  --config config/crypto_btc.yaml \
  --family crypto_institutional_ensemble \
  --run-id institutional-btc
egx crypto-report --run-id institutional-btc
```

The run writes:

- `institutional_daily_attribution.csv`
- `institutional_current_state.csv`
- `institutional_sleeve_ablation.csv`
- normal nested outer-fold, cost-stress, holdout, and report artifacts

The family has additional acceptance gates: positive DCA excess in at least 70% of
outer folds, at least 3% mean one-year excess, no outer drawdown above 35%, positive
incremental value from every active sleeve, sufficient external coverage, and verified
point-in-time external vintages. Current provider histories are explicitly unverified,
so the model cannot be frozen or marked trade-allowed until live snapshots establish
that evidence.

Every `crypto-sync` stores content-addressed copies of the normalized, feature, and raw
CSV inputs under `data/crypto/vintages/` with an append-only manifest. New snapshots
prove what was available from their retrieval date forward; they do not retroactively
certify older provider history.
