from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from egx_research.indicators import atr, rsi, sma


INSTITUTIONAL_FAMILY = "crypto_institutional_ensemble"
INSTITUTIONAL_SLEEVES = (
    "trend",
    "derivatives",
    "onchain",
    "liquidity",
    "macro",
    "sentiment",
    "mean_reversion",
    "carry",
)
EXTERNAL_SLEEVES = frozenset(
    {"derivatives", "onchain", "liquidity", "macro", "sentiment", "carry"}
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _rolling_zscore(values: pd.Series, window: int = 180) -> pd.Series:
    minimum = max(20, window // 3)
    average = values.rolling(window, min_periods=minimum).mean()
    deviation = values.rolling(window, min_periods=minimum).std(ddof=0)
    return ((values - average) / deviation.replace(0.0, np.nan)).clip(-3.0, 3.0)


def _bounded(values: pd.Series | np.ndarray) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return np.tanh(pd.to_numeric(series, errors="coerce")).clip(-1.0, 1.0)


def _component_score(
    components: Iterable[pd.Series],
    index: pd.Index,
    minimum_coverage: float,
) -> tuple[pd.Series, pd.Series]:
    parts = pd.concat([part.reindex(index) for part in components], axis=1)
    coverage = parts.notna().mean(axis=1)
    score = parts.mean(axis=1, skipna=True).where(
        coverage >= minimum_coverage, 0.0
    )
    return score.clip(-1.0, 1.0).fillna(0.0), coverage.fillna(0.0)


def build_institutional_sleeves(
    data: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    daily_return = close.pct_change()
    daily_volatility = daily_return.rolling(90, min_periods=30).std(ddof=0)
    minimum_coverage = float(params["minimum_sleeve_coverage"])

    trend_parts: list[pd.Series] = []
    for lookback in (
        int(params["trend_short"]),
        int(params["trend_medium"]),
        int(params["trend_long"]),
    ):
        scaled_return = close.pct_change(lookback) / (
            daily_volatility * np.sqrt(lookback)
        ).replace(0.0, np.nan)
        trend_parts.append(_bounded(scaled_return))
    long_average = sma(close, int(params["trend_long"]))
    trend_parts.append(
        _bounded((close / long_average - 1.0) * 8.0)
    )
    trend, trend_coverage = _component_score(
        trend_parts, data.index, minimum_coverage
    )

    open_interest = _numeric(data, "derivatives_open_interest")
    oi_direction = _rolling_zscore(open_interest.pct_change(30)) * np.sign(
        close.pct_change(30)
    )
    derivatives, derivatives_coverage = _component_score(
        [
            _bounded(oi_direction),
            _bounded(-_rolling_zscore(_numeric(data, "derivatives_long_short_ratio"))),
            _bounded(-_rolling_zscore(_numeric(data, "derivatives_leverage_ratio"))),
            _bounded(-_rolling_zscore(_numeric(data, "derivatives_toptrader_position_ratio"))),
        ],
        data.index,
        minimum_coverage,
    )

    inflow = _numeric(data, "FlowInExUSD")
    outflow = _numeric(data, "FlowOutExUSD")
    flow_balance = (outflow - inflow) / (outflow.abs() + inflow.abs()).replace(
        0.0, np.nan
    )
    onchain, onchain_coverage = _component_score(
        [
            _bounded(-_rolling_zscore(_numeric(data, "CapMVRVCur"))),
            _bounded(_rolling_zscore(_numeric(data, "AdrActCnt").pct_change(30))),
            _bounded(_rolling_zscore(_numeric(data, "TxCnt").pct_change(30))),
            _bounded(_rolling_zscore(_numeric(data, "HashRate").pct_change(60))),
            _bounded(_rolling_zscore(flow_balance)),
        ],
        data.index,
        minimum_coverage,
    )

    etf_flow = _numeric(data, "etf_net_flow_usd").rolling(7, min_periods=2).sum()
    liquidity, liquidity_coverage = _component_score(
        [
            _bounded(
                _rolling_zscore(
                    _numeric(data, "liquidity_stablecoin_supply").pct_change(90)
                )
            ),
            _bounded(_rolling_zscore(_numeric(data, "liquidity_dry_powder_ratio"))),
            _bounded(_rolling_zscore(etf_flow)),
            _bounded(_rolling_zscore(_numeric(data, "spot_coinbase_premium"))),
        ],
        data.index,
        minimum_coverage,
    )

    macro, macro_coverage = _component_score(
        [
            _bounded(_rolling_zscore(_numeric(data, "macro_nasdaq").pct_change(60))),
            _bounded(-_rolling_zscore(_numeric(data, "macro_dollar").pct_change(60))),
            _bounded(-_rolling_zscore(_numeric(data, "macro_us10y").diff(30))),
            _bounded(
                _rolling_zscore(_numeric(data, "macro_fed_liquidity").pct_change(90))
            ),
            _bounded(-_rolling_zscore(_numeric(data, "macro_vix"))),
        ],
        data.index,
        minimum_coverage,
    )

    sentiment, sentiment_coverage = _component_score(
        [
            _bounded(-_rolling_zscore(_numeric(data, "fear_greed_value"))),
            _bounded(-_rolling_zscore(_numeric(data, "options_dvol"))),
            _bounded(-_rolling_zscore(_numeric(data, "options_put_call_ratio"))),
        ],
        data.index,
        minimum_coverage,
    )

    atr_value = atr(high, low, close, 14)
    distance = (close - sma(close, 20)) / atr_value.replace(0.0, np.nan)
    long_regime = (close > long_average).astype(float)
    mean_reversion, mean_reversion_coverage = _component_score(
        [
            _bounded(-distance),
            _bounded(-_rolling_zscore(close.pct_change(7))),
            ((50.0 - rsi(close, 14)) / 50.0).clip(-1.0, 1.0),
        ],
        data.index,
        minimum_coverage,
    )
    mean_reversion *= 0.25 + 0.75 * long_regime

    carry, carry_coverage = _component_score(
        [
            _bounded(-_rolling_zscore(_numeric(data, "funding_rate_mean"))),
            _bounded(-_rolling_zscore(_numeric(data, "derivatives_basis"))),
        ],
        data.index,
        minimum_coverage,
    )

    scores = pd.DataFrame(
        {
            "trend": trend,
            "derivatives": derivatives,
            "onchain": onchain,
            "liquidity": liquidity,
            "macro": macro,
            "sentiment": sentiment,
            "mean_reversion": mean_reversion,
            "carry": carry,
        },
        index=data.index,
    )
    coverage = pd.DataFrame(
        {
            "trend": trend_coverage,
            "derivatives": derivatives_coverage,
            "onchain": onchain_coverage,
            "liquidity": liquidity_coverage,
            "macro": macro_coverage,
            "sentiment": sentiment_coverage,
            "mean_reversion": mean_reversion_coverage,
            "carry": carry_coverage,
        },
        index=data.index,
    )
    return scores, coverage


def build_probabilistic_regimes(
    data: pd.DataFrame,
    trend_score: pd.Series,
    params: dict[str, Any],
) -> pd.DataFrame:
    close = pd.to_numeric(data["close"], errors="coerce")
    returns = close.pct_change()
    volatility = returns.rolling(30, min_periods=15).std(ddof=0) * np.sqrt(365.0)
    volatility_percentile = volatility.rolling(730, min_periods=180).rank(pct=True)
    drawdown = (1.0 - close / close.rolling(365, min_periods=90).max()).clip(0.0, 1.0)
    short_momentum = _bounded(
        close.pct_change(30)
        / (returns.rolling(60, min_periods=20).std(ddof=0) * np.sqrt(30)).replace(
            0.0, np.nan
        )
    ).fillna(0.0)
    trend = trend_score.fillna(0.0)
    vol_rank = volatility_percentile.fillna(0.5)
    dd = drawdown.fillna(0.0)

    raw = pd.DataFrame(
        {
            "bull": 1.8 * trend + 0.8 * short_momentum - 0.4 * vol_rank,
            "bear": -1.8 * trend - 0.7 * short_momentum + 0.8 * dd,
            "range": 1.2 * (1.0 - trend.abs()) - 0.5 * dd,
            "crisis": 2.5 * dd + 1.5 * vol_rank - 1.2 * trend,
            "recovery": 1.4 * dd + 1.1 * short_momentum.clip(lower=0.0) + 0.5 * trend,
        },
        index=data.index,
    )
    shifted = raw.sub(raw.max(axis=1), axis=0)
    exponent = np.exp(shifted)
    probability = exponent.div(exponent.sum(axis=1), axis=0).fillna(0.2)
    probability.columns = [f"regime_probability_{column}" for column in probability]
    probability["realized_volatility_30d"] = volatility
    probability["volatility_percentile"] = volatility_percentile
    probability["drawdown_365d"] = drawdown
    probability["institutional_regime"] = probability[
        [f"regime_probability_{column}" for column in raw]
    ].idxmax(axis=1).str.removeprefix("regime_probability_")
    return probability


def _capped_normalize(weights: np.ndarray, maximum: float) -> np.ndarray:
    values = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if values.sum() <= 0:
        values = np.ones_like(values)
    active_count = int((values > 0).sum())
    cap = max(float(maximum), 1.0 / active_count if active_count else 1.0)
    result = np.zeros_like(values)
    free = values > 0
    remaining = 1.0
    while free.any():
        proposal = remaining * values[free] / values[free].sum()
        over = proposal > cap
        free_indices = np.flatnonzero(free)
        if not over.any():
            result[free_indices] = proposal
            break
        capped_indices = free_indices[over]
        result[capped_indices] = cap
        free[capped_indices] = False
        remaining = 1.0 - float(result.sum())
    return result / result.sum()


def expanding_regularized_forecast(
    scores: pd.DataFrame,
    coverage: pd.DataFrame,
    close: pd.Series,
    params: dict[str, Any],
) -> pd.DataFrame:
    horizon = 90
    minimum_training = 730
    recalibration_days = 30
    lookback = int(params["calibration_lookback"])
    ridge_lambda = float(params["ridge_lambda"])
    shrinkage = float(params["equal_weight_shrinkage"])
    maximum_weight = float(params["maximum_sleeve_weight"])
    names = list(scores.columns)
    count = len(scores)
    forecast = np.zeros(count, dtype=float)
    confidence = np.zeros(count, dtype=float)
    ensemble_score = np.zeros(count, dtype=float)
    weights = np.full((count, len(names)), 1.0 / len(names), dtype=float)
    contributions = np.zeros((count, len(names)), dtype=float)
    forward_return = pd.to_numeric(close, errors="coerce").shift(-horizon) / close - 1.0

    first_refit = minimum_training + horizon
    for refit in range(first_refit, count, recalibration_days):
        train_end = refit - horizon - 1
        train_start = max(0, train_end - lookback + 1)
        train = scores.iloc[train_start : train_end + 1].copy()
        train["target"] = forward_return.iloc[train_start : train_end + 1]
        train["coverage"] = coverage.iloc[train_start : train_end + 1].mean(axis=1)
        train = train.iloc[::7].dropna(subset=["target"])
        train = train.loc[train["coverage"] >= float(params["minimum_sleeve_coverage"])]
        if len(train) < 40:
            continue
        x = train[names].to_numpy(dtype=float)
        y = train["target"].clip(-0.75, 1.5).to_numpy(dtype=float)
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_centered = x - x_mean
        y_centered = y - y_mean
        penalty = ridge_lambda * np.eye(len(names))
        beta = np.linalg.solve(x_centered.T @ x_centered + penalty, x_centered.T @ y_centered)
        positive = np.clip(beta, 0.0, None)
        available = (x.std(axis=0) > 1e-6).astype(float)
        equal = available / available.sum() if available.sum() else np.full(len(names), 1.0 / len(names))
        learned = positive / positive.sum() if positive.sum() else equal
        fitted_weights = _capped_normalize(
            shrinkage * equal + (1.0 - shrinkage) * learned,
            maximum_weight,
        )
        train_score = x @ fitted_weights
        score_variance = float(np.var(train_score))
        scale = 0.0
        if score_variance > 1e-12:
            scale = max(
                0.0,
                float(np.cov(train_score, y, ddof=0)[0, 1])
                / (score_variance + ridge_lambda / len(train)),
            )
        intercept = y_mean - scale * float(train_score.mean())
        residual = y - (intercept + scale * train_score)
        target_scale = max(float(np.std(y)), 1e-6)
        fit_confidence = 1.0 / (1.0 + float(np.sqrt(np.mean(residual**2))) / target_scale)

        block_end = min(count, refit + recalibration_days)
        for position in range(refit, block_end):
            current_coverage = coverage.iloc[position].to_numpy(dtype=float)
            effective = fitted_weights * current_coverage
            if effective.sum() <= 0:
                continue
            active_count = int((effective > 0).sum())
            dynamic_maximum = max(
                maximum_weight,
                1.0 / active_count if active_count else 1.0,
            )
            effective = _capped_normalize(effective, dynamic_maximum)
            current_scores = scores.iloc[position].to_numpy(dtype=float)
            current_score = float(current_scores @ effective)
            forecast[position] = float(
                np.clip(intercept + scale * current_score, -0.5, 1.5)
            )
            confidence[position] = float(
                np.clip(fit_confidence * current_coverage.mean(), 0.0, 1.0)
            )
            ensemble_score[position] = current_score
            weights[position] = effective
            contributions[position] = effective * current_scores

    result = pd.DataFrame(
        {
            "institutional_expected_return_90d": forecast,
            "institutional_model_confidence": confidence,
            "institutional_ensemble_score": ensemble_score,
        },
        index=scores.index,
    )
    for index, name in enumerate(names):
        result[f"institutional_weight_{name}"] = weights[:, index]
        result[f"institutional_contribution_{name}"] = contributions[:, index]
    return result


def _stateful_allocation(
    dates: pd.Series,
    proposed: pd.Series,
    params: dict[str, Any],
) -> pd.Series:
    core = float(params["core_weight"])
    maximum_change = float(params["maximum_target_change"])
    minimum_change = 0.05
    rebalance_days = 7
    target = np.full(len(proposed), core, dtype=float)
    current = core
    first_date = pd.Timestamp(dates.iloc[0])
    for index in range(len(proposed)):
        elapsed = (pd.Timestamp(dates.iloc[index]) - first_date).days
        desired = float(proposed.iloc[index])
        difference = desired - current
        if desired < core and difference < -minimum_change:
            current = desired
        elif difference < -minimum_change:
            current += float(np.clip(difference, -maximum_change, 0.0))
        elif elapsed % rebalance_days == 0 and difference >= minimum_change:
            current += float(np.clip(difference, 0.0, maximum_change))
        target[index] = current
    return pd.Series(target, index=proposed.index, dtype=float)


def build_institutional_ensemble_frame(
    data: pd.DataFrame,
    params: dict[str, Any],
    *,
    disabled_sleeves: Iterable[str] = (),
) -> pd.DataFrame:
    frame = data.copy()
    scores, coverage = build_institutional_sleeves(frame, params)
    disabled = set(disabled_sleeves)
    for sleeve in disabled:
        if sleeve not in INSTITUTIONAL_SLEEVES:
            raise ValueError(f"unknown institutional sleeve: {sleeve}")
        scores[sleeve] = 0.0
        coverage[sleeve] = 0.0
    regimes = build_probabilistic_regimes(frame, scores["trend"], params)
    calibration = expanding_regularized_forecast(
        scores,
        coverage,
        pd.to_numeric(frame["close"], errors="coerce"),
        params,
    )

    horizon = 90.0
    expected = calibration["institutional_expected_return_90d"].clip(-0.5, 1.5)
    expected_net = expected - 2.0 * float(params["execution_cost_bps"]) / 10_000.0
    annual_expected = pd.Series(
        np.where(
            expected_net > -1.0,
            np.power(1.0 + expected_net, 365.0 / horizon) - 1.0,
            -1.0,
        ),
        index=frame.index,
    ).clip(-1.0, 2.0)
    volatility = regimes["realized_volatility_30d"].replace(0.0, np.nan)
    drawdown_penalty = 0.5 * regimes["drawdown_365d"].fillna(0.0)
    risk_adjusted_edge = (
        annual_expected
        - float(params["minimum_annual_edge"])
        - drawdown_penalty
    )
    edge = risk_adjusted_edge.clip(lower=0.0)
    optimal_risk = (
        edge
        / (
            float(params["risk_aversion"])
            * volatility.pow(2).replace(0.0, np.nan)
        )
    ).clip(0.0, 1.0).fillna(0.0)
    regime_multiplier = (
        regimes["regime_probability_bull"]
        + 0.8 * regimes["regime_probability_recovery"]
        + 0.45 * regimes["regime_probability_range"]
        + 0.15 * regimes["regime_probability_bear"]
    ).clip(0.0, 1.0)
    volatility_scale = (
        float(params["volatility_target"]) / volatility
    ).clip(upper=1.0).fillna(0.0)
    weight_matrix = calibration[
        [f"institutional_weight_{sleeve}" for sleeve in INSTITUTIONAL_SLEEVES]
    ].to_numpy(dtype=float)
    score_matrix = scores[list(INSTITUTIONAL_SLEEVES)].to_numpy(dtype=float)
    score_mean = calibration["institutional_ensemble_score"].to_numpy(dtype=float)
    disagreement = pd.Series(
        np.sqrt(
            np.sum(
                weight_matrix * (score_matrix - score_mean[:, None]) ** 2,
                axis=1,
            )
        ),
        index=frame.index,
    ).clip(0.0, 1.0)
    disagreement_scale = (1.0 - disagreement).clip(0.25, 1.0)
    daily_return = pd.to_numeric(frame["close"], errors="coerce").pct_change()
    expected_shortfall = -daily_return.rolling(180, min_periods=60).apply(
        lambda values: float(
            np.mean(np.sort(values)[: max(1, int(np.ceil(len(values) * 0.05)))])
        ),
        raw=True,
    )
    expected_shortfall_scale = (
        float(params["expected_shortfall_limit"])
        / expected_shortfall.replace(0.0, np.nan)
    ).clip(upper=1.0).fillna(0.0)
    tactical = (
        optimal_risk
        * calibration["institutional_model_confidence"]
        * regime_multiplier
        * volatility_scale
        * disagreement_scale
        * expected_shortfall_scale
    ).clip(0.0, 1.0)
    core = float(params["core_weight"])
    proposed = core + (1.0 - core) * tactical
    crisis = (
        (
            regimes["regime_probability_crisis"]
            >= float(params["crisis_probability"])
        )
        | (
            (regimes["drawdown_365d"] >= float(params["crisis_drawdown"]))
            & (regimes["volatility_percentile"] >= 0.90)
            & (scores["trend"] < 0.0)
        )
    )
    proposed = proposed.where(~crisis, float(params["crisis_allocation"])).clip(0.0, 1.0)
    target = _stateful_allocation(frame["date"], proposed, params)

    frame["target_allocation"] = target
    frame["floor_allocation"] = float(params["crisis_allocation"])
    frame["institutional_proposed_allocation"] = proposed
    frame["institutional_annual_expected_return"] = annual_expected
    frame["institutional_risk_adjusted_edge"] = risk_adjusted_edge
    frame["institutional_sleeve_disagreement"] = disagreement
    frame["institutional_expected_shortfall_95_1d"] = expected_shortfall
    frame["institutional_external_coverage"] = coverage[
        list(EXTERNAL_SLEEVES)
    ].mean(axis=1)
    for sleeve in INSTITUTIONAL_SLEEVES:
        frame[f"institutional_sleeve_{sleeve}"] = scores[sleeve]
        frame[f"institutional_coverage_{sleeve}"] = coverage[sleeve]
    for column in regimes:
        frame[column] = regimes[column]
    for column in calibration:
        frame[column] = calibration[column]

    frame["entry_signal"] = target > target.shift(1).fillna(core) + 0.05
    frame["exit_signal"] = target < target.shift(1).fillna(core) - 0.05
    frame["atr"] = atr(
        pd.to_numeric(frame["high"], errors="coerce"),
        pd.to_numeric(frame["low"], errors="coerce"),
        pd.to_numeric(frame["close"], errors="coerce"),
        14,
    )
    frame["stop_mult"] = 0.0
    frame["trail_mult"] = 0.0
    return frame
