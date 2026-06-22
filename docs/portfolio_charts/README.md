# Portfolio Chart Set

Recommended portfolio charts:

1. `01_membership_equity_curve.png` - data provenance changes the conclusion.
2. `02_holdout_validation.png` - holdout validation, with pullback rule vs DCA.
3. `07_holdout_excess_ladder.png` - holdout excess shown as percentage points, not ratio math.
4. `08_risk_adjusted_map.png` - return shown beside drawdown and Sharpe.

Supporting charts:

- `03_factor_ic_heatmap.png` - factor signal quality.
- `04_data_quality_flags.png` - data-quality flags.
- `05_topn_sensitivity.png` - selection-size sensitivity.
- `06_signal_consistency.png` - signal stability across time windows.

Regenerate with:

```bash
python3 scripts/generate_portfolio_charts.py
```
