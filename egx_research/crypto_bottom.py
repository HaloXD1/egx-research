from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egx_research.crypto_config import CryptoConfig
from egx_research.crypto_research import load_crypto_feature_data
from egx_research.indicators import adx, atr, bollinger_bands, ema, macd, rsi, sma
from egx_research.utils import ensure_dir, to_native, write_json
from egx_research.crypto_sources import run_data_quality_checks



HORIZONS = (30, 60, 90)
TOLERANCES = (0.05, 0.10)
DEFAULT_PRIOR_WEIGHTS = {
    "price_structure": 0.25,
    "capitulation": 0.15,
    "derivatives": 0.15,
    "onchain": 0.20,
    "spot_demand": 0.10,
    "macro": 0.10,
    "sentiment": 0.05,
}


@dataclass(frozen=True)
class BottomScoreRun:
    run_id: str
    run_dir: Path
    report_path: Path
    summary: dict[str, Any]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _clip01(value: pd.Series | float) -> pd.Series | float:
    return np.clip(value, 0.0, 1.0)


def _scale_between(series: pd.Series, low: float, high: float) -> pd.Series:
    if high == low:
        return pd.Series(0.0, index=series.index)
    return pd.Series(_clip01((series - low) / (high - low)), index=series.index)


def _load_optional_source(raw_dir: Path, filename: str) -> pd.DataFrame | None:
    path = raw_dir / filename
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    if "date" not in frame.columns:
        return None
    return frame.sort_values("date").drop_duplicates("date", keep="last")


# Columns the component scoring reads via _optional_column.
_SIGNAL_COLUMNS = [
    "funding_rate_mean", "derivatives_open_interest", "derivatives_liquidations_long_usd",
    "CapMVRVCur", "FlowInExUSD", "FlowOutExUSD", "AdrActCnt", "TxCnt",
    "etf_net_flow_usd", "etf_net_flow_btc", "spot_coinbase_premium",
    "liquidity_stablecoin_supply", "options_options_skew", "options_put_call_ratio",
    "fear_greed_value", "macro_nasdaq", "macro_us10y", "macro_dollar",
    "macro_fed_liquidity", "macro_vix",
]

_SIGNAL_ALIASES = {
    "derivatives_open_interest": "open_interest",
    "derivatives_open_interest_value": "open_interest_value",
    "spot_coinbase_close": "coinbase_close",
    "spot_coinbase_premium": "coinbase_premium",
    "liquidity_stablecoin_supply": "stablecoin_supply",
    "options_options_skew": "options_skew",
    "options_put_call_ratio": "put_call_ratio",
    "options_dvol": "dvol",
}


def _apply_signal_aliases(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for canonical, alias in _SIGNAL_ALIASES.items():
        if alias not in panel.columns:
            continue
        if canonical not in panel.columns:
            panel[canonical] = panel[alias]
        else:
            panel[canonical] = panel[canonical].combine_first(panel[alias])
    return panel


def _merge_optional_sources(data: pd.DataFrame, config: CryptoConfig) -> tuple[pd.DataFrame, list[str]]:
    panel = data.copy()
    raw_dir = Path(config.data.raw_dir)
    loaded: list[str] = []
    optional_files = {
        "btc_etf_flows.csv": "etf",
        "open_interest.csv": "derivatives",
        "coinbase_premium.csv": "spot",
        "stablecoin_supply.csv": "liquidity",
        "options_skew.csv": "options",
    }
    for filename, prefix in optional_files.items():
        source = _load_optional_source(raw_dir, filename)
        if source is None:
            continue
        renamed = source.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in source.columns
                if column != "date" and not column.startswith(f"{prefix}_")
            }
        )
        keep_columns = ["date"] + [column for column in renamed.columns if column != "date" and column not in panel.columns]
        if len(keep_columns) == 1:
            continue
        renamed = renamed[keep_columns]
        panel = panel.merge(renamed, on="date", how="left")
        loaded.append(filename)

    base_columns = set(data.columns)
    optional_columns = [column for column in panel.columns if column not in base_columns and column != "date"]
    for column in optional_columns:
        panel[column] = _safe_numeric(panel[column]).shift(1)

    panel = _apply_signal_aliases(panel)

    # Report columns that actually have data (including those already in features)
    active = [col for col in _SIGNAL_COLUMNS if col in panel.columns and panel[col].notna().any()]
    loaded += [f"feature:{col}" for col in active]
    return panel, loaded


def _add_market_indicators(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = _safe_numeric(frame[column])

    frame["return_1d_calc"] = frame["close"].pct_change()
    frame["return_7d_calc"] = frame["close"].pct_change(7)
    frame["return_30d_calc"] = frame["close"].pct_change(30)
    frame["sma20"] = sma(frame["close"], 20)
    frame["sma50"] = sma(frame["close"], 50)
    frame["sma200"] = sma(frame["close"], 200)
    frame["ema20"] = ema(frame["close"], 20)
    frame["rsi14"] = rsi(frame["close"], 14)
    frame["adx14"] = adx(frame["high"], frame["low"], frame["close"], 14)
    frame["atr14"] = atr(frame["high"], frame["low"], frame["close"], 14)
    lower, middle, upper = bollinger_bands(frame["close"], 20, 2.0)
    frame["bb_lower"] = lower
    frame["bb_middle"] = middle
    frame["bb_upper"] = upper
    macd_line, macd_signal, macd_hist = macd(frame["close"])
    frame["macd"] = macd_line
    frame["macd_signal"] = macd_signal
    frame["macd_hist"] = macd_hist

    frame["rolling_low_21"] = frame["low"].rolling(21, min_periods=5).min()
    frame["rolling_low_30"] = frame["low"].rolling(30, min_periods=10).min()
    frame["rolling_low_90"] = frame["low"].rolling(90, min_periods=30).min()
    frame["rolling_high_20"] = frame["high"].rolling(20, min_periods=5).max()
    frame["rolling_high_365"] = frame["high"].rolling(365, min_periods=90).max()
    frame["drawdown_365"] = frame["close"] / frame["rolling_high_365"] - 1.0
    # Guard against partial-day volume: if the latest row's volume is < 25% of the
    # prior row's, treat it as partial and use the prior row's volume for the ratio.
    vol = frame["volume"].copy()
    if len(vol) >= 2:
        prev_vol = vol.iloc[-2]
        if prev_vol > 0 and vol.iloc[-1] < 0.25 * prev_vol:
            vol.iloc[-1] = prev_vol
    frame["volume_ratio_20"] = vol / vol.rolling(20, min_periods=10).mean()
    frame["down_wick_pct"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / frame["close"].replace(0, np.nan)
    frame["close_location"] = (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["vol30"] = frame["return_1d_calc"].rolling(30, min_periods=15).std(ddof=0) * np.sqrt(365)
    frame["vol90"] = frame["return_1d_calc"].rolling(90, min_periods=30).std(ddof=0) * np.sqrt(365)
    frame["vol_compression"] = 1.0 - frame["vol30"] / frame["vol90"].replace(0, np.nan)
    frame["bounce_from_21d_low"] = frame["close"] / frame["rolling_low_21"].replace(0, np.nan) - 1.0

    # Calculate derived technical indicators (Sharpe Ratios and STH MVRV)
    mean_ret_30 = frame["return_1d_calc"].rolling(30, min_periods=15).mean()
    std_ret_30 = frame["return_1d_calc"].rolling(30, min_periods=15).std(ddof=1)
    frame["sharpe_30d"] = (mean_ret_30 / std_ret_30.replace(0, np.nan)).fillna(0.0) * np.sqrt(365)

    mean_ret_90 = frame["return_1d_calc"].rolling(90, min_periods=45).mean()
    std_ret_90 = frame["return_1d_calc"].rolling(90, min_periods=45).std(ddof=1)
    frame["sharpe_90d"] = (mean_ret_90 / std_ret_90.replace(0, np.nan)).fillna(0.0) * np.sqrt(365)

    pv = frame["close"] * frame["volume"]
    rolling_pv = pv.rolling(155, min_periods=30).sum()
    rolling_v = frame["volume"].rolling(155, min_periods=30).sum()
    sth_realized_price = rolling_pv / rolling_v.replace(0, np.nan)
    sth_realized_price = sth_realized_price.fillna(frame["close"])
    frame["sth_realized_price"] = sth_realized_price
    frame["sth_mvrv"] = frame["close"] / sth_realized_price

    days_since = []
    lows = frame["low"].to_numpy(dtype=float)
    for index in range(len(frame)):
        start = max(0, index - 29)
        window = lows[start : index + 1]
        if np.isnan(window).all():
            days_since.append(np.nan)
            continue
        low_offset = int(np.nanargmin(window))
        days_since.append(index - (start + low_offset))
    frame["days_since_30d_low"] = days_since
    return frame


def _optional_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return _safe_numeric(frame[column])
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _component_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"date": frame["date"]})

    # Sharpe ratio price structure signal
    sharpe_30d = frame["sharpe_30d"] if "sharpe_30d" in frame.columns else pd.Series(0.0, index=frame.index)
    sharpe_recovery = _scale_between(sharpe_30d, -3.0, 1.5)

    reclaim20 = (frame["close"] > frame["sma20"]).astype(float)
    reclaim50 = (frame["close"] > frame["sma50"]).astype(float)
    reclaim200 = (frame["close"] > frame["sma200"]).astype(float)
    macd_improving = (frame["macd_hist"] > frame["macd_hist"].shift(3)).astype(float)
    rsi_recovery = (
        (frame["rsi14"].between(35, 58))
        & (frame["rsi14"].rolling(14, min_periods=5).min() < 35)
    ).astype(float)
    recent_low = _scale_between(14 - frame["days_since_30d_low"], 0, 14)
    bounce = _scale_between(frame["bounce_from_21d_low"], 0.02, 0.12)
    out["price_structure"] = pd.concat(
        [reclaim20, reclaim50 * 0.8, reclaim200 * 0.5, macd_improving, rsi_recovery, recent_low, bounce, sharpe_recovery],
        axis=1,
    ).mean(axis=1)

    # STH MVRV capitulation signal
    sth_mvrv = frame["sth_mvrv"] if "sth_mvrv" in frame.columns else pd.Series(1.0, index=frame.index)
    sth_mvrv_cap = 1.0 - _scale_between(sth_mvrv, 0.85, 1.05)

    drawdown = _scale_between(-frame["drawdown_365"], 0.18, 0.55)
    volume_climax = _scale_between(frame["volume_ratio_20"], 1.2, 3.0)
    wick_reversal = (_scale_between(frame["down_wick_pct"], 0.01, 0.08) * _scale_between(frame["close_location"], 0.45, 0.85))
    vol_flush = _scale_between(frame["vol30"], 0.45, 1.10)
    post_flush = _scale_between(frame["vol_compression"], -0.10, 0.35)
    out["capitulation"] = pd.concat([drawdown, volume_climax, wick_reversal, vol_flush, post_flush, sth_mvrv_cap], axis=1).mean(axis=1)

    funding = _optional_column(frame, "funding_rate_mean")
    funding_reset = _scale_between(-funding, -0.0002, 0.0015)
    funding_cooling = _scale_between(-(funding - funding.rolling(14, min_periods=5).mean()), -0.0002, 0.0008)
    oi = _optional_column(frame, "derivatives_open_interest")
    oi_flush = _scale_between(-(oi.pct_change(7)), 0.02, 0.20)
    long_liq = _optional_column(frame, "derivatives_liquidations_long_usd")
    long_liq_spike = _scale_between(long_liq / long_liq.rolling(30, min_periods=10).mean(), 1.5, 8.0)
    out["derivatives"] = pd.concat([funding_reset, funding_cooling, oi_flush, long_liq_spike], axis=1).mean(axis=1)

    mvrv = _optional_column(frame, "CapMVRVCur")
    mvrv_cheap = 1.0 - _scale_between(mvrv, 1.0, 2.7)
    mvrv_recovery = _scale_between(mvrv - mvrv.rolling(30, min_periods=10).min(), 0.0, 0.35)
    outflow = _optional_column(frame, "FlowOutExUSD")
    inflow = _optional_column(frame, "FlowInExUSD")
    exchange_outflow = _scale_between((outflow - inflow) / frame["close"].replace(0, np.nan), 1000, 50000)
    active_addresses = _optional_column(frame, "AdrActCnt")
    active_recovery = _scale_between(active_addresses.pct_change(30), -0.05, 0.20)
    tx_count = _optional_column(frame, "TxCnt")
    tx_recovery = _scale_between(tx_count.pct_change(30), -0.05, 0.20)

    # Derived MVRV Z-Score signal
    sply = _optional_column(frame, "SplyCur")
    mcap = frame["close"] * sply
    rcap = mcap / mvrv.replace(0.0, np.nan)
    mvrv_zscore = (mcap - rcap) / mcap.expanding(min_periods=30).std().replace(0.0, np.nan)
    zscore_cheapness = 1.0 - _scale_between(mvrv_zscore, 0.1, 5.0)

    out["onchain"] = pd.concat([mvrv_cheap, mvrv_recovery, exchange_outflow, active_recovery, tx_recovery, zscore_cheapness], axis=1).mean(axis=1)

    spot_signals: list[pd.Series] = []
    etf_flow = _optional_column(frame, "etf_net_flow_usd").fillna(_optional_column(frame, "etf_net_flow_btc") * frame["close"])
    if etf_flow.notna().sum() > 30:
        # Forward-fill over weekends/gaps (max 3 days) so rolling windows stay valid
        etf_filled = etf_flow.ffill(limit=3)
        etf_5d = etf_filled.rolling(5, min_periods=3).sum()
        etf_20d_abs = etf_filled.abs().rolling(20, min_periods=5).sum()
        spot_signals.append(_scale_between(etf_5d / etf_20d_abs.replace(0, np.nan), -0.05, 0.35))
    coinbase_premium = _optional_column(frame, "spot_coinbase_premium")
    if coinbase_premium.notna().sum() > 30:
        spot_signals.append(_scale_between(coinbase_premium, -0.002, 0.006))
    stable_supply = _optional_column(frame, "liquidity_stablecoin_supply")
    if stable_supply.notna().sum() > 30:
        spot_signals.append(_scale_between(stable_supply.pct_change(30), -0.02, 0.08))
    # Volume ratio: crypto trades 24/7 but weekends are naturally lower; 0.6 floor
    spot_signals.append(_scale_between(frame["volume_ratio_20"], 0.6, 2.0))
    out["spot_demand"] = pd.concat(spot_signals, axis=1).mean(axis=1) if spot_signals else pd.Series(0.5, index=frame.index)

    nasdaq = _optional_column(frame, "macro_nasdaq")
    dollar = _optional_column(frame, "macro_dollar")
    us10y = _optional_column(frame, "macro_us10y")
    vix = _optional_column(frame, "macro_vix")
    fed_liq = _optional_column(frame, "macro_fed_liquidity")
    macro_risk = _scale_between(nasdaq.pct_change(20), -0.06, 0.08)
    dollar_tailwind = _scale_between(-dollar.pct_change(20), -0.02, 0.05)
    yield_tailwind = _scale_between(-(us10y - us10y.shift(20)), -0.25, 0.60)
    vix_cooling = _scale_between(-(vix - vix.shift(20)) / vix.shift(20).replace(0, np.nan), -0.10, 0.35)
    liquidity = _scale_between(fed_liq.pct_change(60), -0.03, 0.06)
    out["macro"] = pd.concat([macro_risk, dollar_tailwind, yield_tailwind, vix_cooling, liquidity], axis=1).mean(axis=1)

    # Derived NUPL signal
    nupl = 1.0 - 1.0 / mvrv.replace(0.0, np.nan)
    nupl_washout = 1.0 - _scale_between(nupl, -0.15, 0.45)

    fear_greed = _optional_column(frame, "fear_greed_value")
    fear_washout = 1.0 - _scale_between(fear_greed, 18, 60)
    fear_recovery = _scale_between(fear_greed - fear_greed.rolling(14, min_periods=5).min(), 0, 25)
    options_skew = _optional_column(frame, "options_options_skew")
    put_call = _optional_column(frame, "options_put_call_ratio")
    hedge_washout = pd.concat(
        [
            _scale_between(options_skew, 0.0, 0.20),
            _scale_between(put_call, 0.8, 1.6),
        ],
        axis=1,
    ).mean(axis=1)
    out["sentiment"] = pd.concat([fear_washout, fear_recovery, hedge_washout, nupl_washout], axis=1).mean(axis=1)

    component_columns = [column for column in out.columns if column != "date"]
    out[component_columns] = out[component_columns].clip(lower=0.0, upper=1.0)
    return out


def _future_success_labels(frame: pd.DataFrame, horizon: int, tolerance: float, anchor_lookback: int = 21) -> pd.Series:
    lows = frame["low"].to_numpy(dtype=float)
    labels = np.full(len(frame), np.nan, dtype=float)
    for index in range(len(frame)):
        future_end = index + horizon
        if future_end >= len(frame):
            continue
        anchor_start = max(0, index - anchor_lookback + 1)
        anchor_low = np.nanmin(lows[anchor_start : index + 1])
        future_low = np.nanmin(lows[index + 1 : future_end + 1])
        if not np.isfinite(anchor_low) or not np.isfinite(future_low):
            continue
        labels[index] = float(future_low >= anchor_low * (1.0 - tolerance))
    return pd.Series(labels, index=frame.index)


def _train_mask(frame: pd.DataFrame, label: pd.Series, as_of_idx: int) -> pd.Series:
    mask = label.notna()
    mask &= frame.index < as_of_idx
    mask &= frame["days_since_30d_low"].fillna(999) <= 14
    mask &= frame["drawdown_365"].fillna(0.0) <= -0.08
    return mask


def _learn_component_weights(
    components: pd.DataFrame,
    labels: pd.Series,
    mask: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame, float]:
    component_columns = [column for column in components.columns if column != "date"]
    if int(mask.sum()) < 40:
        weights = dict(DEFAULT_PRIOR_WEIGHTS)
        hitrates = pd.DataFrame(
            [
                {
                    "component": key,
                    "weight": value,
                    "active_rows": 0,
                    "active_hitrate": np.nan,
                    "base_hitrate": np.nan,
                    "lift": 0.0,
                }
                for key, value in weights.items()
            ]
        )
        return weights, hitrates, float(labels[mask].mean()) if mask.any() else 0.5

    base = float(labels[mask].mean())
    raw_weights: dict[str, float] = {}
    rows = []
    for column in component_columns:
        values = components.loc[mask, column]
        active = values >= 0.55
        active_rows = int(active.sum())
        if active_rows >= 20:
            active_hitrate = float(labels.loc[mask].loc[active].mean())
            coverage_penalty = np.sqrt(active_rows / (active_rows + 75.0))
            lift = max(0.0, active_hitrate - base) * float(coverage_penalty)
        else:
            active_hitrate = np.nan
            lift = 0.0
        prior = DEFAULT_PRIOR_WEIGHTS.get(column, 0.05)
        raw = 0.35 * prior + 0.65 * lift
        raw_weights[column] = raw
        rows.append(
            {
                "component": column,
                "prior_weight": prior,
                "active_rows": active_rows,
                "active_hitrate": active_hitrate,
                "base_hitrate": base,
                "lift": lift,
                "raw_weight": raw,
            }
        )

    total = sum(raw_weights.values())
    weights = {key: value / total for key, value in raw_weights.items()} if total > 0 else dict(DEFAULT_PRIOR_WEIGHTS)
    for row in rows:
        row["weight"] = weights[row["component"]]
    return weights, pd.DataFrame(rows).sort_values("weight", ascending=False), base


def _weighted_score(components: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    values = pd.Series(0.0, index=components.index, dtype=float)
    weight_sum = pd.Series(0.0, index=components.index, dtype=float)
    for column, weight in weights.items():
        if column not in components.columns:
            continue
        valid = components[column].notna()
        values.loc[valid] += components.loc[valid, column] * weight
        weight_sum.loc[valid] += weight
    return values / weight_sum.replace(0.0, np.nan)


def _calibrate_from_bins(score: pd.Series, label: pd.Series, mask: pd.Series, latest_score: float, base: float) -> tuple[float, str]:
    train = pd.DataFrame({"score": score[mask], "label": label[mask]}).dropna()
    if len(train) < 40 or not np.isfinite(latest_score):
        fallback = float(_clip01(base * 0.45 + latest_score * 0.55 if np.isfinite(latest_score) else base))
        return fallback, "fallback"
    try:
        train["bin"] = pd.qcut(train["score"], q=min(6, max(3, len(train) // 35)), duplicates="drop")
    except ValueError:
        fallback = float(_clip01(base * 0.45 + latest_score * 0.55))
        return fallback, "fallback"
    grouped = train.groupby("bin", observed=True).agg(avg_score=("score", "mean"), hitrate=("label", "mean"), rows=("label", "size"))
    if grouped.empty:
        return float(_clip01(base * 0.45 + latest_score * 0.55)), "fallback"
    nearest_idx = (grouped["avg_score"] - latest_score).abs().idxmin()
    nearest = grouped.loc[nearest_idx]
    shrink = float(nearest["rows"] / (nearest["rows"] + 40.0))
    calibrated = float(base * (1.0 - shrink) + nearest["hitrate"] * shrink)
    return float(_clip01(calibrated)), "bin"


def _fit_logistic_probability(
    components: pd.DataFrame,
    labels: pd.Series,
    mask: pd.Series,
    latest_idx: int,
) -> tuple[float | None, dict[str, float] | None]:
    columns = [column for column in components.columns if column != "date"]
    train = components.loc[mask, columns].copy()
    y = labels.loc[mask].copy()
    train = train.fillna(train.median(numeric_only=True)).fillna(0.5)
    latest = components.loc[[latest_idx], columns].copy().fillna(train.median(numeric_only=True)).fillna(0.5)
    if len(train) < 80 or y.nunique(dropna=True) < 2:
        return None, None

    mean = train.mean()
    std = train.std(ddof=0).replace(0, 1.0)
    x = ((train - mean) / std).to_numpy(dtype=float)
    x_latest = ((latest - mean) / std).to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    x_latest = np.column_stack([np.ones(len(x_latest)), x_latest])
    y_values = y.to_numpy(dtype=float)
    beta = np.zeros(x.shape[1], dtype=float)
    learning_rate = 0.03
    ridge = 0.03
    for _ in range(900):
        z = np.clip(x @ beta, -35, 35)
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = (x.T @ (pred - y_values)) / len(y_values)
        grad[1:] += ridge * beta[1:] / len(y_values)
        beta -= learning_rate * grad
    latest_logit = float((x_latest @ beta)[0])
    prob = float(1.0 / (1.0 + np.exp(-np.clip(latest_logit, -35, 35))))
    coefs = {"intercept": float(beta[0])}
    coefs.update({column: float(value) for column, value in zip(columns, beta[1:], strict=False)})
    return prob, coefs


def _confidence_band(probability: float) -> str:
    if probability >= 0.90:
        return "extreme"
    if probability >= 0.80:
        return "very_high"
    if probability >= 0.70:
        return "high"
    if probability >= 0.60:
        return "medium"
    return "low"


def _recommendation(probability: float, strict_probability: float, latest: pd.Series) -> str:
    if probability >= 0.85 and strict_probability >= 0.75:
        return "Accumulation favored: staged buys are justified, but keep invalidation below the recent low."
    if probability >= 0.75:
        return "Start or add in tranches: evidence favors a local bottom, but keep cash for one retest."
    if probability >= 0.65:
        return "Probe only: use small size and require a higher low or reclaim before adding."
    if float(latest.get("days_since_30d_low", 999)) <= 3:
        return "Wait for confirmation: current setup can still be a first bounce after a fresh low."
    return "Defensive/neutral: bottom evidence is not strong enough for aggressive buying."


def _support_resistance(latest: pd.Series, tolerance: float) -> dict[str, float]:
    anchor = float(latest["rolling_low_21"])
    return {
        "recent_21d_low": anchor,
        "invalidation_low": anchor * (1.0 - tolerance),
        "sma20": float(latest["sma20"]) if pd.notna(latest["sma20"]) else np.nan,
        "sma50": float(latest["sma50"]) if pd.notna(latest["sma50"]) else np.nan,
        "sma200": float(latest["sma200"]) if pd.notna(latest["sma200"]) else np.nan,
        "prior_20d_high": float(latest["rolling_high_20"]) if pd.notna(latest["rolling_high_20"]) else np.nan,
    }


def _format_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:,.0f}"


def _report_html(summary: dict[str, Any], grid: pd.DataFrame, components: pd.DataFrame, hitrates: pd.DataFrame) -> str:
    latest = summary["latest"]
    best = summary["best_case"]
    component_rows = []
    for item in summary["component_snapshot"]:
        component_rows.append(
            "<tr>"
            f"<td>{html.escape(item['component'])}</td>"
            f"<td>{item['score']:.2f}</td>"
            f"<td>{item['weight']:.1%}</td>"
            f"<td>{html.escape(item['interpretation'])}</td>"
            "</tr>"
        )
    grid_html = grid.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    hitrate_html = hitrates.to_html(index=False, float_format=lambda x: f"{x:.3f}") if not hitrates.empty else "<p>No hitrate table.</p>"
    
    dq = summary.get("data_quality", {})
    dq_warnings_html = ""
    if dq.get("warnings"):
        dq_warnings_html = "<h3>Warnings:</h3><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in dq["warnings"]) + "</ul>"
    else:
        dq_warnings_html = "<p>No warnings. All critical data sources are fresh.</p>"

    rating_color = "green"
    if dq.get("reliability_rating") == "Warning":
        rating_color = "orange"
    elif dq.get("reliability_rating") == "Degraded":
        rating_color = "red"

    dq_details_rows = []
    for name, status in sorted(dq.get("source_statuses", {}).items()):
        status_color = "green"
        if status["status"] in ("stale", "partial"):
            status_color = "orange"
        elif status["status"] == "missing":
            status_color = "red"
            
        lag_val = status["lag_hours"]
        lag_str = f"{lag_val:.1f}" if lag_val is not None else "N/A"
            
        dq_details_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(status['provider'])}</td>"
            f"<td>{html.escape(status['category'])}</td>"
            f"<td style='color: {status_color}; font-weight: bold;'>{html.escape(status['status'])}</td>"
            f"<td>{lag_str}</td>"
            f"<td>{html.escape(status['latest_date'] or 'N/A')}</td>"
            f"<td>{status['coverage']:.1%}</td>"
            f"<td>{'Critical' if status['critical'] else 'Optional'}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BTC Bottom Confidence Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; color: #17202a; line-height: 1.45; }}
    h1, h2 {{ color: #101820; }}
    .summary {{ border-left: 5px solid #1f77b4; padding: 12px 18px; background: #f5f9fc; }}
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #e8f1fb; margin-right: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 26px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    th {{ background: #f2f5f8; }}
    .warn {{ color: #8a4b00; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>BTC Bottom Confidence Report</h1>
  <h2>Executive Summary</h2>
  <div class="summary">
    <p><strong>Current read:</strong> {best['confidence_pct']:.1f}% confidence that the recent BTC low holds over {best['horizon_days']} days with a {best['tolerance_pct']:.0f}% allowed breach. Band: <strong>{html.escape(best['confidence_band'])}</strong>.</p>
    <p><strong>Recommendation:</strong> {html.escape(summary['recommendation'])}</p>
    <p><strong>Important:</strong> this is a historical probability score, not a guarantee that BTC cannot print a lower wick.</p>
  </div>
  <p>
    <span class="pill">As of {html.escape(str(latest['date']))}</span>
    <span class="pill">Close ${_format_num(latest['close'])}</span>
    <span class="pill">Recent low ${_format_num(summary['levels']['recent_21d_low'])}</span>
    <span class="pill">Invalidation ${_format_num(summary['levels']['invalidation_low'])}</span>
  </p>

  <h2>Data Quality & Reliability Audit</h2>
  <div class="summary" style="border-left: 5px solid {rating_color}; background: #fafafa;">
    <p><strong>Reliability Rating:</strong> {dq.get('reliability_rating')} (Score: {dq.get('reliability_score', 1.0):.2f})</p>
    <p><strong>Status Note:</strong> {html.escape(dq.get('reliability_note', ''))}</p>
    {dq_warnings_html}
  </div>

  <h3>Source Quality Details</h3>
  <table>
    <thead>
      <tr>
        <th>Source</th>
        <th>Provider</th>
        <th>Category</th>
        <th>Status</th>
        <th>Lag (Hours)</th>
        <th>Latest Date</th>
        <th>Coverage</th>
        <th>Type</th>
      </tr>
    </thead>
    <tbody>
      {"".join(dq_details_rows)}
    </tbody>
  </table>

  <h2>What The Model Is Saying</h2>
  <p>The framework combines price structure, capitulation, derivatives, on-chain value, spot demand, macro, and sentiment. Weights are learned from historical hit rate lift for the selected horizon and tolerance, then blended with a small prior so scarce optional data cannot dominate.</p>
  <table>
    <thead><tr><th>Component</th><th>Score</th><th>Weight</th><th>Read</th></tr></thead>
    <tbody>{''.join(component_rows)}</tbody>
  </table>

  <h2>Confidence Grid</h2>
  <p>Use the 5% tolerance rows as the stricter clean-bottom view and 10% rows as the tactical bottom-zone view.</p>
  {grid_html}

  <h2>Recommended Action</h2>
  <p>{html.escape(summary['recommendation'])}</p>
  <ul>
    <li>Do not treat anything below 80% as all-in evidence.</li>
    <li>Add most aggressively only after a higher low or reclaim confirms the bounce.</li>
    <li>Reduce or pause if BTC closes below the invalidation area or if macro/derivatives turn hostile again.</li>
  </ul>

  <h2>Audit: Historical Hitrates And Weights</h2>
  {hitrate_html}

  <h2>Caveats And Assumptions</h2>
  <p class="warn">No model can prove the market will not go lower. This model estimates whether the recent low is likely to hold inside a defined horizon/tolerance.</p>
  <p>Optional paid/API exports are used when present in <code>data/crypto/raw/</code>: BTC ETF flows, open interest/liquidations, Coinbase premium, stablecoin supply, and options skew. Missing optional sources reduce coverage but do not block the local score.</p>
</body>
</html>"""



def _interpret_component(score: float) -> str:
    if score >= 0.75:
        return "strong support"
    if score >= 0.55:
        return "constructive"
    if score >= 0.35:
        return "mixed"
    return "weak"


def run_crypto_bottom_score(
    config: CryptoConfig,
    config_path: str | Path,
    run_id: str | None = None,
    as_of_date: str | None = None,
) -> BottomScoreRun:
    data = load_crypto_feature_data(config)
    data, optional_sources = _merge_optional_sources(data, config)
    frame = _add_market_indicators(data)
    components = _component_scores(frame)
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        eligible = frame.index[pd.to_datetime(frame["date"]) <= as_of]
        if len(eligible) == 0:
            raise ValueError(f"No BTC data on or before {as_of_date}.")
        latest_idx = int(eligible[-1])
    else:
        as_of = pd.Timestamp(frame["date"].max())
        latest_idx = int(frame.index[-1])
    latest = frame.loc[latest_idx]
    data_quality = run_data_quality_checks(config, frame, as_of)


    run_name = run_id or f"crypto-bottom-score-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / run_name)

    grid_rows = []
    hitrate_frames = []
    coef_payload: dict[str, Any] = {}
    latest_scores = {}
    best_row: dict[str, Any] | None = None
    strict_probability = 0.0

    for horizon in HORIZONS:
        for tolerance in TOLERANCES:
            labels = _future_success_labels(frame, horizon=horizon, tolerance=tolerance)
            mask = _train_mask(frame, labels, latest_idx)
            weights, hitrates, base_rate = _learn_component_weights(components, labels, mask)
            blended_score = _weighted_score(components, weights)
            latest_score = float(blended_score.iloc[latest_idx])
            rule_prob, calibration_method = _calibrate_from_bins(blended_score, labels, mask, latest_score, base_rate)
            ml_prob, coefs = _fit_logistic_probability(components, labels, mask, latest_idx)
            confidence = float(_clip01(rule_prob if ml_prob is None else 0.55 * rule_prob + 0.45 * ml_prob))
            if horizon == 60 and tolerance == 0.05:
                strict_probability = confidence
            row = {
                "horizon_days": horizon,
                "tolerance": tolerance,
                "tolerance_pct": tolerance * 100,
                "training_rows": int(mask.sum()),
                "base_hitrate": base_rate,
                "rule_probability": rule_prob,
                "ml_probability": ml_prob,
                "confidence": confidence,
                "confidence_pct": confidence * 100,
                "confidence_band": _confidence_band(confidence),
                "weighted_signal_score": latest_score,
                "calibration_method": calibration_method,
            }
            grid_rows.append(row)
            hitrates = hitrates.copy()
            hitrates["horizon_days"] = horizon
            hitrates["tolerance"] = tolerance
            hitrate_frames.append(hitrates)
            coef_payload[f"{horizon}d_{int(tolerance * 100)}pct"] = coefs
            latest_scores[f"{horizon}d_{int(tolerance * 100)}pct"] = {
                "weights": weights,
                "weighted_signal_score": latest_score,
            }
            if best_row is None or (confidence, horizon, tolerance) > (
                best_row["confidence"],
                best_row["horizon_days"],
                best_row["tolerance"],
            ):
                best_row = row

    assert best_row is not None
    grid = pd.DataFrame(grid_rows).sort_values(["horizon_days", "tolerance"]).reset_index(drop=True)
    hitrates_all = pd.concat(hitrate_frames, ignore_index=True) if hitrate_frames else pd.DataFrame()
    best_key = f"{best_row['horizon_days']}d_{int(best_row['tolerance'] * 100)}pct"
    best_weights = latest_scores[best_key]["weights"]
    levels = _support_resistance(latest, best_row["tolerance"])
    recommendation = _recommendation(float(best_row["confidence"]), strict_probability, latest)

    component_snapshot = [
        {
            "component": column,
            "score": float(components.loc[latest_idx, column]) if pd.notna(components.loc[latest_idx, column]) else np.nan,
            "weight": float(best_weights.get(column, 0.0)),
            "interpretation": _interpret_component(float(components.loc[latest_idx, column])) if pd.notna(components.loc[latest_idx, column]) else "missing",
        }
        for column in DEFAULT_PRIOR_WEIGHTS
    ]
    component_snapshot.sort(key=lambda item: item["weight"], reverse=True)

    summary = {
        "run_id": run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "features_path": config.data.features_path,
        "optional_sources_loaded": optional_sources,
        "optional_feature_columns_present": [
            column
            for column in _SIGNAL_COLUMNS
            if column in frame.columns
        ],
        "latest": {
            "date": str(pd.Timestamp(latest["date"]).date()),
            "close": float(latest["close"]),
            "rsi14": float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else None,
            "drawdown_365": float(latest["drawdown_365"]) if pd.notna(latest["drawdown_365"]) else None,
            "days_since_30d_low": float(latest["days_since_30d_low"]) if pd.notna(latest["days_since_30d_low"]) else None,
        },
        "best_case": best_row,
        "strict_60d_5pct_confidence": strict_probability,
        "recommendation": recommendation,
        "levels": levels,
        "component_snapshot": component_snapshot,
        "model_note": "Probability recent low holds inside horizon/tolerance; not a guarantee of no lower print.",
        "data_quality": data_quality,
        "reliability_rating": data_quality["reliability_rating"],
        "reliability_note": data_quality["reliability_note"],
    }

    grid.to_csv(run_dir / "bottom_probability_grid.csv", index=False)
    components.to_csv(run_dir / "bottom_component_scores.csv", index=False)
    hitrates_all.to_csv(run_dir / "bottom_feature_hitrates.csv", index=False)
    write_json(run_dir / "bottom_model_coefficients.json", coef_payload)
    write_json(run_dir / "bottom_score_summary.json", to_native(summary))
    report_path = run_dir / "bottom_report.html"
    report_path.write_text(_report_html(summary, grid, components, hitrates_all), encoding="utf-8")
    return BottomScoreRun(run_id=run_name, run_dir=run_dir, report_path=report_path, summary=to_native(summary))
