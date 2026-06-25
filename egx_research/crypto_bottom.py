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
    "derivatives_basis", "derivatives_taker_buy_sell_ratio",
    "derivatives_long_short_ratio", "derivatives_leverage_ratio",
    "derivatives_long_liq_usd", "derivatives_short_liq_usd", "derivatives_total_liq_usd",
    "derivatives_liq_imbalance", "derivatives_heatmap_nearest_down_liq", "derivatives_heatmap_nearest_up_liq",
    "CapMVRVCur", "FlowInExUSD", "FlowOutExUSD", "AdrActCnt", "TxCnt",
    "etf_net_flow_usd", "etf_net_flow_btc", "spot_coinbase_premium",
    "liquidity_stablecoin_supply", "liquidity_exchange_stablecoin_reserves", "liquidity_dry_powder_ratio",
    "options_options_skew", "options_put_call_ratio",
    "options_25d_skew", "options_put_call_oi", "options_put_call_volume",
    "options_iv_30d", "options_term_structure", "options_dvol",
    "fear_greed_value", "macro_nasdaq", "macro_us10y", "macro_dollar",
    "macro_fed_liquidity", "macro_vix",
    "etf_flow_ibit", "etf_flow_fbtc", "etf_flow_arkb", "etf_flow_bitb", "etf_flow_gbtc",
    "onchain_exchange_reserve_btc", "onchain_exchange_netflow_btc",
    "onchain_exchange_netflow_usd", "onchain_whale_inflow_usd",
    "onchain_realized_profit_loss_exchange",
    "onchain_sth_realized_price", "onchain_sth_mvrv", "onchain_sth_sopr",
    "onchain_sopr", "onchain_realized_loss_usd", "onchain_realized_profit_usd",
]

_SIGNAL_ALIASES = {
    "derivatives_open_interest": "open_interest",
    "derivatives_open_interest_value": "open_interest_value",
    "derivatives_basis": "basis",
    "derivatives_taker_buy_sell_ratio": "taker_buy_sell_ratio",
    "derivatives_long_short_ratio": "long_short_ratio",
    "derivatives_leverage_ratio": "leverage_ratio",
    "spot_coinbase_close": "coinbase_close",
    "spot_coinbase_premium": "coinbase_premium",
    "liquidity_stablecoin_supply": "stablecoin_supply",
    "liquidity_exchange_stablecoin_reserves": "exchange_stablecoin_reserves",
    "liquidity_dry_powder_ratio": "dry_powder_ratio",
    "options_options_skew": "options_skew",
    "options_put_call_ratio": "put_call_ratio",
    "options_dvol": "dvol",
    "options_25d_skew": "options_skew",
    "options_put_call_oi": "put_call_ratio",
    "derivatives_liquidations_long_usd": "derivatives_long_liq_usd",
    "derivatives_long_liq_usd": "long_liq_usd",
    "derivatives_short_liq_usd": "short_liq_usd",
    "derivatives_total_liq_usd": "total_liq_usd",
    "derivatives_liq_imbalance": "liq_imbalance",
    "derivatives_heatmap_nearest_down_liq": "heatmap_nearest_down_liq",
    "derivatives_heatmap_nearest_up_liq": "heatmap_nearest_up_liq",
    "onchain_sth_realized_price": "sth_realized_price",
    "onchain_sth_mvrv": "sth_mvrv",
    "onchain_sth_sopr": "sth_sopr",
    "onchain_sopr": "sopr",
    "onchain_realized_loss_usd": "realized_loss_usd",
    "onchain_realized_profit_usd": "realized_profit_usd",
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
    # Also handle fallback mapping if raw name is in panel
    if "long_liq_usd" in panel.columns and "derivatives_liquidations_long_usd" not in panel.columns:
        panel["derivatives_liquidations_long_usd"] = panel["long_liq_usd"]
    return panel


def _merge_optional_sources(data: pd.DataFrame, config: CryptoConfig) -> tuple[pd.DataFrame, list[str]]:
    panel = data.copy()
    raw_dir = Path(config.data.raw_dir)
    loaded: list[str] = []
    optional_files = {
        "btc_etf_flows.csv": "etf",
        "open_interest.csv": "derivatives",
        "futures_positioning.csv": "derivatives",
        "coinbase_premium.csv": "spot",
        "stablecoin_supply.csv": "liquidity",
        "exchange_stablecoin_reserves.csv": "liquidity",
        "options_skew.csv": "options",
        "liquidations.csv": "derivatives",
        "exchange_flows.csv": "onchain",
        "glassnode_sth_sopr.csv": "onchain",
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

    # Compute lag-safe dry powder ratio after shifting loop
    if "liquidity_dry_powder_ratio" not in panel.columns:
        stable_col = "liquidity_stablecoin_supply"
        if stable_col in panel.columns and "close" in panel.columns and panel[stable_col].notna().any():
            sply_col = "SplyCur" if "SplyCur" in panel.columns else ("onchain_SplyCur" if "onchain_SplyCur" in panel.columns else None)
            sply = panel[sply_col] if sply_col else pd.Series(19500000.0, index=panel.index)
            sply = sply.fillna(19500000.0).replace(0, 19500000.0)
            prev_close = panel["close"].shift(1)
            btc_mcap_lagged = prev_close * sply
            panel["liquidity_dry_powder_ratio"] = panel[stable_col] / btc_mcap_lagged.replace(0, np.nan)

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
    sth_realized_price_proxy = rolling_pv / rolling_v.replace(0, np.nan)
    sth_realized_price_proxy = sth_realized_price_proxy.fillna(frame["close"])
    frame["sth_realized_price_proxy"] = sth_realized_price_proxy
    frame["sth_mvrv_proxy"] = frame["close"] / sth_realized_price_proxy

    # Prefer real STH realized price, fall back to proxy
    if "onchain_sth_realized_price" in frame.columns:
        frame["sth_realized_price"] = _safe_numeric(frame["onchain_sth_realized_price"]).fillna(sth_realized_price_proxy)
    else:
        frame["sth_realized_price"] = sth_realized_price_proxy

    # Prefer real STH MVRV, fall back to proxy, set audit label
    if "onchain_sth_mvrv" in frame.columns:
        real_mvrv = _safe_numeric(frame["onchain_sth_mvrv"])
        frame["sth_mvrv"] = real_mvrv.fillna(frame["sth_mvrv_proxy"])
        frame["sth_mvrv_source"] = np.where(real_mvrv.notna(), "real", "proxy")
    else:
        frame["sth_mvrv"] = frame["sth_mvrv_proxy"]
        frame["sth_mvrv_source"] = "proxy"

    # STH SOPR fallback cascade: real STH SOPR -> real overall SOPR -> missing
    real_sth_sopr = _optional_column(frame, "onchain_sth_sopr")
    real_sopr = _optional_column(frame, "onchain_sopr")
    frame["sth_sopr"] = real_sth_sopr.fillna(real_sopr)
    frame["sth_sopr_source"] = np.where(
        real_sth_sopr.notna(),
        "real_sth",
        np.where(real_sopr.notna(), "real_sopr", "missing")
    )

    # Realized Loss Climax ratio (realized loss compared to 30d average)
    realized_loss = _optional_column(frame, "onchain_realized_loss_usd")
    frame["realized_loss_ratio"] = realized_loss / realized_loss.rolling(30, min_periods=10).mean().replace(0, np.nan)


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

    # ETF flow indicators
    etf_ibit = _optional_column(frame, "etf_flow_ibit")
    etf_fbtc = _optional_column(frame, "etf_flow_fbtc")
    etf_arkb = _optional_column(frame, "etf_flow_arkb")
    etf_bitb = _optional_column(frame, "etf_flow_bitb")
    etf_gbtc = _optional_column(frame, "etf_flow_gbtc")
    etf_total = _optional_column(frame, "etf_net_flow_usd")

    etf_ibit_filled = etf_ibit.fillna(0.0)
    etf_fbtc_filled = etf_fbtc.fillna(0.0)
    etf_arkb_filled = etf_arkb.fillna(0.0)
    etf_bitb_filled = etf_bitb.fillna(0.0)
    etf_gbtc_filled = etf_gbtc.fillna(0.0)
    etf_total_filled = etf_total.fillna(0.0)

    frame["etf_flow_5d_sum"] = etf_total_filled.rolling(5, min_periods=1).sum()
    frame["etf_flow_20d_sum"] = etf_total_filled.rolling(20, min_periods=1).sum()
    frame["etf_flow_acceleration"] = frame["etf_flow_5d_sum"] - frame["etf_flow_20d_sum"] / 4.0

    # Flow pct of BTC market cap
    sply = _optional_column(frame, "SplyCur").ffill().fillna(19.5e6)
    mcap = frame["close"] * sply
    frame["etf_flow_pct_mcap_5d"] = frame["etf_flow_5d_sum"] / mcap.replace(0, np.nan)

    # ex-GBTC flows
    ex_gbtc_flow = etf_total - etf_gbtc
    ex_gbtc_flow_filled = ex_gbtc_flow.fillna(0.0)
    frame["etf_ex_gbtc_flow_5d_sum"] = ex_gbtc_flow_filled.rolling(5, min_periods=1).sum()
    frame["etf_ex_gbtc_flow_20d_sum"] = ex_gbtc_flow_filled.rolling(20, min_periods=1).sum()

    # ex-GBTC persistence: fraction of trading days in the last 20 calendar days where ex_gbtc_flow > 0
    pos_ex_gbtc_days = (ex_gbtc_flow > 0).rolling(20, min_periods=1).sum().fillna(0.0)
    trade_days = (ex_gbtc_flow.notna() & (ex_gbtc_flow != 0)).rolling(20, min_periods=1).sum().fillna(0.0)
    frame["etf_ex_gbtc_persistence_20d"] = (pos_ex_gbtc_days / trade_days.replace(0, np.nan)).fillna(0.0)

    # Issuer breadth: non-GBTC major issuers with positive flows
    breadth = (
        (etf_ibit > 0).astype(int) +
        (etf_fbtc > 0).astype(int) +
        (etf_arkb > 0).astype(int) +
        (etf_bitb > 0).astype(int)
    )
    frame["etf_issuer_breadth"] = breadth / 4.0
    frame["etf_issuer_breadth_5d"] = frame["etf_issuer_breadth"].rolling(5, min_periods=1).mean().fillna(0.0)

    return frame


def _optional_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return _safe_numeric(frame[column])
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _derivatives_subsignals(frame: pd.DataFrame) -> pd.DataFrame:
    funding = _optional_column(frame, "funding_rate_mean")
    funding_reset = _scale_between(-funding, -0.0002, 0.0015)
    funding_cooling = _scale_between(-(funding - funding.rolling(14, min_periods=5).mean()), -0.0002, 0.0008)
    oi = _optional_column(frame, "derivatives_open_interest")
    oi_flush = _scale_between(-(oi.pct_change(7)), 0.02, 0.20)
    long_liq = _optional_column(frame, "derivatives_liquidations_long_usd")
    short_liq = _optional_column(frame, "derivatives_short_liq_usd")
    long_liq_spike = _scale_between(long_liq / long_liq.rolling(30, min_periods=10).mean(), 1.5, 8.0)
    liq_imbalance = _optional_column(frame, "derivatives_liq_imbalance")
    if liq_imbalance.isna().all() and not long_liq.isna().all() and not short_liq.isna().all():
        denom = long_liq + short_liq
        liq_imbalance = ((long_liq - short_liq) / denom.replace(0, np.nan)).fillna(0.0)
    imbalance_washout = _scale_between(liq_imbalance, 0.4, 0.9)
    heatmap_down = _optional_column(frame, "derivatives_heatmap_nearest_down_liq")
    heatmap_risk = _scale_between((frame["close"] - heatmap_down) / frame["close"].replace(0, np.nan), 0.0, 0.05)

    out = pd.DataFrame(
        {
            "sub_derivatives_funding_reset": funding_reset,
            "sub_derivatives_funding_cooling": funding_cooling,
            "sub_derivatives_oi_flush": oi_flush,
            "sub_derivatives_long_liq_spike": long_liq_spike,
            "sub_derivatives_imbalance_washout": imbalance_washout,
            "sub_derivatives_heatmap_risk": heatmap_risk,
        },
        index=frame.index,
    )

    has_new_positioning = any(
        _optional_column(frame, column).notna().any()
        for column in [
            "derivatives_basis",
            "derivatives_taker_buy_sell_ratio",
            "derivatives_long_short_ratio",
            "derivatives_leverage_ratio",
        ]
    )
    if not has_new_positioning:
        return out

    spot_led_bounce = pd.Series(np.nan, index=frame.index, dtype=float)
    if oi.notna().any():
        spot_led = frame["close"].pct_change(5) - oi.pct_change(5)
        spot_led_bounce = _scale_between(spot_led, -0.05, 0.15)

    lev = _optional_column(frame, "derivatives_leverage_ratio")
    lev_flush = _scale_between(-lev.pct_change(7), -0.10, 0.20)
    lev_mean_90 = lev.rolling(90, min_periods=30).mean()
    lev_std_90 = lev.rolling(90, min_periods=30).std()
    lev_z = (lev - lev_mean_90) / lev_std_90.replace(0, np.nan)
    lev_cheap = 1.0 - _scale_between(lev_z, -1.5, 1.5)
    leverage_reset = pd.concat([lev_flush, lev_cheap], axis=1).mean(axis=1)

    ls_ratio = _optional_column(frame, "derivatives_long_short_ratio")
    taker_ratio = _optional_column(frame, "derivatives_taker_buy_sell_ratio")
    ls_penalty = 1.0 - _scale_between(ls_ratio, 1.2, 2.2)
    taker_penalty = 1.0 - _scale_between(taker_ratio, 1.0, 1.3)
    overheated_long_short_penalty = pd.concat([ls_penalty, taker_penalty], axis=1).mean(axis=1)

    out["sub_derivatives_spot_led_bounce"] = spot_led_bounce
    out["sub_derivatives_leverage_reset"] = leverage_reset
    out["sub_derivatives_overheated_long_short_penalty"] = overheated_long_short_penalty
    return out


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
    cap_signals = [drawdown, volume_climax, wick_reversal, vol_flush, post_flush, sth_mvrv_cap]

    # STH SOPR capitulation (only if sth_sopr has valid data)
    sth_sopr = frame["sth_sopr"] if "sth_sopr" in frame.columns else pd.Series(np.nan, index=frame.index)
    if "sth_sopr_source" in frame.columns and (frame["sth_sopr_source"] != "missing").any():
        sth_sopr_cap = 1.0 - _scale_between(sth_sopr, 0.94, 1.02)
        cap_signals.append(sth_sopr_cap)

    # Realized Loss Climax (only if realized_loss_ratio has valid data)
    realized_loss_ratio = frame["realized_loss_ratio"] if "realized_loss_ratio" in frame.columns else pd.Series(np.nan, index=frame.index)
    if realized_loss_ratio.notna().any():
        realized_loss_climax = _scale_between(realized_loss_ratio, 1.5, 6.0)
        cap_signals.append(realized_loss_climax)

    # Liquidations and Heatmap features
    long_liq = _optional_column(frame, "derivatives_liquidations_long_usd")
    short_liq = _optional_column(frame, "derivatives_short_liq_usd")
    long_liq_spike = _scale_between(long_liq / long_liq.rolling(30, min_periods=10).mean(), 1.5, 8.0)

    liq_imbalance = _optional_column(frame, "derivatives_liq_imbalance")
    if liq_imbalance.isna().all() and not long_liq.isna().all() and not short_liq.isna().all():
        denom = long_liq + short_liq
        liq_imbalance = (long_liq - short_liq) / denom.replace(0, np.nan)
        liq_imbalance = liq_imbalance.fillna(0.0)
    imbalance_washout_score = _scale_between(liq_imbalance, 0.4, 0.9)

    heatmap_down = _optional_column(frame, "derivatives_heatmap_nearest_down_liq")
    heatmap_risk_score = _scale_between((frame["close"] - heatmap_down) / frame["close"].replace(0, np.nan), 0.0, 0.05)

    cap_signals.extend([long_liq_spike, imbalance_washout_score, heatmap_risk_score])
    out["capitulation"] = pd.concat(cap_signals, axis=1).mean(axis=1)

    derivative_signals = _derivatives_subsignals(frame)
    out["derivatives"] = derivative_signals.mean(axis=1)

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

    # New exchange flows indicators
    reserve = _optional_column(frame, "onchain_exchange_reserve_btc")
    reserve_change_90 = reserve.pct_change(90)
    reserve_downtrend = _scale_between(-reserve_change_90, -0.01, 0.05)

    netflow_btc = _optional_column(frame, "onchain_exchange_netflow_btc").fillna(
        _optional_column(frame, "onchain_exchange_netflow_usd") / frame["close"].replace(0, np.nan)
    )
    rolling_outflow_14d_btc = (-netflow_btc).rolling(14, min_periods=5).mean()
    net_outflow_cap = _scale_between(rolling_outflow_14d_btc, 100, 5000)

    whale_inflow = _optional_column(frame, "onchain_whale_inflow_usd")
    whale_ratio = whale_inflow / whale_inflow.rolling(30, min_periods=10).mean().replace(0, np.nan)
    whale_inflow_score = 1.0 - _scale_between(whale_ratio, 1.2, 2.5)

    realized_pl = _optional_column(frame, "onchain_realized_profit_loss_exchange")
    pl_mean = realized_pl.rolling(90, min_periods=30).mean()
    pl_std = realized_pl.rolling(90, min_periods=30).std().replace(0, np.nan)
    pl_zscore = (realized_pl - pl_mean) / pl_std
    realized_capitulation = _scale_between(-pl_zscore, 1.0, 2.5)

    # Precedence logic: dedicated net_outflow_cap replaces legacy exchange_outflow if data is present
    if net_outflow_cap.notna().any():
        exchange_flow_final = net_outflow_cap
    else:
        exchange_flow_final = exchange_outflow

    onchain_signals = [
        mvrv_cheap,
        mvrv_recovery,
        exchange_flow_final,
        active_recovery,
        tx_recovery,
        zscore_cheapness,
    ]

    if reserve_downtrend.notna().any():
        onchain_signals.append(reserve_downtrend)
    if whale_inflow_score.notna().any():
        onchain_signals.append(whale_inflow_score)
    if realized_capitulation.notna().any():
        onchain_signals.append(realized_capitulation)

    sth_mvrv_reclaim = _scale_between(sth_mvrv, 0.98, 1.05)
    onchain_signals.append(sth_mvrv_reclaim)
    if "sth_sopr_source" in frame.columns and (frame["sth_sopr_source"] != "missing").any():
        sth_sopr_reclaim = _scale_between(sth_sopr, 0.98, 1.04)
        onchain_signals.append(sth_sopr_reclaim)

    out["onchain"] = pd.concat(onchain_signals, axis=1).mean(axis=1)

    spot_signals: list[pd.Series] = []
    etf_flow = _optional_column(frame, "etf_net_flow_usd").fillna(_optional_column(frame, "etf_net_flow_btc") * frame["close"])
    if etf_flow.notna().sum() > 30:
        etf_5d = frame["etf_flow_5d_sum"]
        etf_20d_abs = etf_flow.abs().fillna(0.0).rolling(20, min_periods=5).sum()
        total_flow_ratio = etf_5d / etf_20d_abs.replace(0, np.nan)
        total_flow_score = _scale_between(total_flow_ratio, -0.05, 0.35).fillna(0.5)

        ex_gbtc_5d = frame["etf_ex_gbtc_flow_5d_sum"]
        gbtc_flow = _optional_column(frame, "etf_flow_gbtc")
        ex_gbtc_flow = etf_flow - gbtc_flow
        ex_gbtc_20d_abs = ex_gbtc_flow.abs().fillna(0.0).rolling(20, min_periods=5).sum()
        ex_gbtc_flow_ratio = ex_gbtc_5d / ex_gbtc_20d_abs.replace(0, np.nan)
        ex_gbtc_flow_score = _scale_between(ex_gbtc_flow_ratio, -0.05, 0.35).fillna(0.5)

        persistence = frame["etf_ex_gbtc_persistence_20d"]
        breadth_5d = frame["etf_issuer_breadth_5d"]

        etf_score = 0.4 * total_flow_score + 0.6 * (ex_gbtc_flow_score * (0.3 + 0.7 * (0.5 * persistence + 0.5 * breadth_5d)))
        spot_signals.append(etf_score)
    coinbase_premium = _optional_column(frame, "spot_coinbase_premium")
    if coinbase_premium.notna().sum() > 30:
        spot_signals.append(_scale_between(coinbase_premium, -0.002, 0.006))
    stable_supply = _optional_column(frame, "liquidity_stablecoin_supply")
    if stable_supply.notna().sum() > 30:
        spot_signals.append(_scale_between(stable_supply.pct_change(30), -0.02, 0.08))
        spot_signals.append(_scale_between(stable_supply.pct_change(90), -0.05, 0.15))
    exchange_reserves = _optional_column(frame, "liquidity_exchange_stablecoin_reserves")
    if exchange_reserves.notna().sum() > 30:
        spot_signals.append(_scale_between(exchange_reserves.pct_change(30), -0.05, 0.15))
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

    macro_signals = [macro_risk, dollar_tailwind, yield_tailwind, vix_cooling, liquidity]
    dry_powder = _optional_column(frame, "liquidity_dry_powder_ratio")
    if dry_powder.notna().sum() > 30:
        macro_signals.append(_scale_between(dry_powder, 0.05, 0.20))
    elif stable_supply is not None and stable_supply.notna().sum() > 30:
        sply = _optional_column(frame, "SplyCur").fillna(19500000.0)
        prev_close = frame["close"].shift(1)
        btc_mcap_lagged = prev_close * sply
        dry_powder_calc = stable_supply / btc_mcap_lagged.replace(0, np.nan)
        macro_signals.append(_scale_between(dry_powder_calc, 0.05, 0.20))

    out["macro"] = pd.concat(macro_signals, axis=1).mean(axis=1)

    # Derived NUPL signal
    nupl = 1.0 - 1.0 / mvrv.replace(0.0, np.nan)
    nupl_washout = 1.0 - _scale_between(nupl, -0.15, 0.45)

    fear_greed = _optional_column(frame, "fear_greed_value")
    fear_washout = 1.0 - _scale_between(fear_greed, 18, 60)
    fear_recovery = _scale_between(fear_greed - fear_greed.rolling(14, min_periods=5).min(), 0, 25)
    opt_skew = _optional_column(frame, "options_25d_skew").fillna(_optional_column(frame, "options_options_skew"))
    opt_pcr_oi = _optional_column(frame, "options_put_call_oi").fillna(_optional_column(frame, "options_put_call_ratio"))
    opt_pcr_vol = _optional_column(frame, "options_put_call_volume")

    options_skew_panic = _scale_between(opt_skew, 0.0, 0.20)
    options_oi_panic = _scale_between(opt_pcr_oi, 0.8, 1.6)
    options_vol_panic = _scale_between(opt_pcr_vol, 0.8, 1.6)

    hedge_washout = pd.concat(
        [options_skew_panic, options_oi_panic, options_vol_panic],
        axis=1,
    ).mean(axis=1)
    out["sentiment"] = pd.concat([fear_washout, fear_recovery, hedge_washout, nupl_washout], axis=1).mean(axis=1)

    component_columns = [column for column in out.columns if column != "date"]
    out[component_columns] = out[component_columns].clip(lower=0.0, upper=1.0)
    return out


def _weekly_rsi_from_daily(frame: pd.DataFrame) -> pd.Series:
    weekly = (
        frame[["date", "close"]]
        .assign(date=lambda data: pd.to_datetime(data["date"]))
        .set_index("date")["close"]
        .resample("W-SUN")
        .last()
    )
    weekly_rsi = rsi(weekly, 14)
    return weekly_rsi.reindex(pd.to_datetime(frame["date"]), method="ffill").reset_index(drop=True)


def _confirmation_audit(frame: pd.DataFrame) -> pd.DataFrame:
    weekly_rsi14 = _weekly_rsi_from_daily(frame)
    higher_low = frame["rolling_low_21"] > frame["rolling_low_21"].shift(10)
    funding = _optional_column(frame, "funding_rate_mean")
    funding_normalized = (funding.abs() <= 0.0005) & (funding >= funding.rolling(14, min_periods=5).min())
    sth_reclaim = (frame["close"] > frame["sth_realized_price"]) | (frame["sth_mvrv"] >= 0.98)
    above_sma20 = frame["close"] > frame["sma20"]
    above_sma50 = frame["close"] > frame["sma50"]
    weekly_rsi_reclaim = (weekly_rsi14 >= 40.0) & (weekly_rsi14 > weekly_rsi14.shift(1))

    confirmations = pd.concat(
        [
            above_sma20.astype(float),
            above_sma50.astype(float),
            higher_low.astype(float),
            weekly_rsi_reclaim.astype(float),
            sth_reclaim.astype(float),
            funding_normalized.astype(float),
        ],
        axis=1,
    ).sum(axis=1)
    state = np.full(len(frame), "none", dtype=object)
    state[confirmations >= 1] = "early"
    state[confirmations >= 3] = "confirmed"
    state[(above_sma20 & above_sma50 & higher_low & weekly_rsi_reclaim).fillna(False).to_numpy()] = "trend_reclaim"
    return pd.DataFrame(
        {
            "date": frame["date"],
            "close_above_sma20": above_sma20.astype(float),
            "close_above_sma50": above_sma50.astype(float),
            "higher_low_confirmed": higher_low.astype(float),
            "weekly_rsi14": weekly_rsi14,
            "weekly_rsi_reclaim": weekly_rsi_reclaim.astype(float),
            "sth_reclaim": sth_reclaim.astype(float),
            "funding_normalized": funding_normalized.astype(float),
            "confirmation_count": confirmations,
            "confirmation_state": state,
        }
    )


def _cycle_phase_audit(frame: pd.DataFrame) -> pd.DataFrame:
    date = pd.to_datetime(frame["date"])
    halvings = [pd.Timestamp("2012-11-28"), pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")]
    days_since_halving = pd.Series(np.nan, index=frame.index, dtype=float)
    for halving in halvings:
        days = (date - halving).dt.days
        days_since_halving = days_since_halving.where(days < 0, days)
    sma200w = frame["close"].rolling(1400, min_periods=700).mean()
    distance_200w = frame["close"] / sma200w.replace(0, np.nan) - 1.0
    mvrv = _optional_column(frame, "CapMVRVCur").fillna(frame["sth_mvrv"])
    stable_supply = _optional_column(frame, "liquidity_stablecoin_supply")
    liquidity_trend = stable_supply.pct_change(90)
    prior_ath = frame["close"].expanding(min_periods=30).max().shift(1)
    prior_ath_reclaim = frame["close"] >= prior_ath

    phase = np.full(len(frame), "mid_cycle", dtype=object)
    deep_bear = (frame["drawdown_365"] <= -0.55) & (frame["close"] < frame["sma200"]) & (mvrv < 1.15)
    late_bear = (frame["drawdown_365"] <= -0.35) & ((mvrv < 1.35) | (distance_200w < 0.35))
    early_recovery = (frame["close"] > frame["sma50"]) & (frame["drawdown_365"] <= -0.15) & (mvrv < 2.2)
    overheated = (mvrv > 3.5) | (frame["drawdown_365"] > -0.05) | prior_ath_reclaim.fillna(False)
    phase[late_bear.fillna(False).to_numpy()] = "late_bear"
    phase[deep_bear.fillna(False).to_numpy()] = "deep_bear"
    phase[early_recovery.fillna(False).to_numpy()] = "early_recovery"
    phase[overheated.fillna(False).to_numpy()] = "overheated"

    return pd.DataFrame(
        {
            "date": frame["date"],
            "days_since_halving": days_since_halving,
            "distance_200w_ma": distance_200w,
            "cycle_mvrv": mvrv,
            "liquidity_trend_90d": liquidity_trend,
            "prior_ath_reclaim": prior_ath_reclaim.astype(float),
            "cycle_phase": phase,
        }
    )


def _future_quality_labels(
    frame: pd.DataFrame,
    horizon: int,
    tolerance: float,
    max_drawdown: float,
) -> pd.DataFrame:
    hold = _future_success_labels(frame, horizon=horizon, tolerance=tolerance)
    rows = []
    for idx in frame.index:
        fwd_return, fwd_drawdown = _future_return_drawdown(frame, int(idx), horizon)
        if fwd_return is None or fwd_drawdown is None or pd.isna(hold.loc[idx]):
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        dd_penalty = float(_clip01((abs(min(0.0, fwd_drawdown)) - max_drawdown) / max_drawdown))
        return_ok = fwd_return >= -0.05
        quality = float(bool(hold.loc[idx]) and fwd_drawdown >= -max_drawdown and return_ok)
        risk_adjusted = float(_clip01(0.65 * hold.loc[idx] + 0.20 * max(0.0, fwd_return) - 0.55 * dd_penalty))
        rows.append((quality, dd_penalty, risk_adjusted, fwd_return))
    return pd.DataFrame(
        rows,
        columns=["quality_outcome", "forward_drawdown_penalty", "risk_adjusted_bottom_score", "forward_return_for_quality"],
        index=frame.index,
    )


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


def _add_regime_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    date = pd.to_datetime(frame["date"])
    nasdaq_ret_20 = _optional_column(frame, "macro_nasdaq").pct_change(20)
    vix = _optional_column(frame, "macro_vix")
    is_macro_stress = (vix > 28) | (nasdaq_ret_20 < -0.08)
    is_bear = (frame["close"] < frame["sma200"]) & (frame["drawdown_365"] <= -0.20)
    is_bull_correction = (frame["close"] >= frame["sma200"]) & (frame["drawdown_365"] <= -0.08)
    halving_dates = [pd.Timestamp("2012-11-28"), pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-19")]
    is_post_halving = pd.Series(False, index=frame.index)
    for halving in halving_dates:
        days = (date - halving).dt.days
        is_post_halving |= days.between(0, 365)
    is_etf_era = date >= pd.Timestamp("2024-01-11")

    primary = np.full(len(frame), "chop/range", dtype=object)
    primary[is_etf_era.to_numpy()] = "ETF-era"
    primary[is_post_halving.to_numpy()] = "post-halving expansion"
    primary[is_bull_correction.fillna(False).to_numpy()] = "bull correction"
    primary[is_bear.fillna(False).to_numpy()] = "bear market"
    primary[is_macro_stress.fillna(False).to_numpy()] = "macro stress"
    frame["primary_regime"] = primary

    tags: list[str] = []
    for idx in frame.index:
        row_tags = []
        if bool(is_macro_stress.loc[idx]):
            row_tags.append("macro_stress")
        if bool(is_bear.loc[idx]):
            row_tags.append("bear_market")
        if bool(is_bull_correction.loc[idx]):
            row_tags.append("bull_correction")
        if bool(is_post_halving.loc[idx]):
            row_tags.append("post_halving")
        if bool(is_etf_era.loc[idx]):
            row_tags.append("etf_era")
        if not row_tags:
            row_tags.append("chop_range")
        tags.append(",".join(row_tags))
    frame["regime_tags"] = tags
    return frame


def _calibrate_regime_probability(
    score: pd.Series,
    label: pd.Series,
    mask: pd.Series,
    regimes: pd.Series,
    current_regime: str,
    latest_score: float,
    global_base: float,
) -> tuple[float, str, int, float]:
    regime_mask = mask & (regimes == current_regime)
    regime_rows = int(regime_mask.sum())
    if regime_rows < 40:
        global_prob, global_method = _calibrate_from_bins(score, label, mask, latest_score, global_base)
        return global_prob, f"global_{global_method}_sparse_regime", regime_rows, 0.0
    regime_base = float(label[regime_mask].mean()) if regime_mask.any() else global_base
    regime_prob, regime_method = _calibrate_from_bins(score, label, regime_mask, latest_score, regime_base)
    global_prob, _ = _calibrate_from_bins(score, label, mask, latest_score, global_base)
    shrink = float(regime_rows / (regime_rows + 80.0))
    blended = float(_clip01(global_prob * (1.0 - shrink) + regime_prob * shrink))
    return blended, f"regime_{regime_method}", regime_rows, shrink


def _future_return_drawdown(frame: pd.DataFrame, index: int, horizon: int) -> tuple[float | None, float | None]:
    future_end = index + horizon
    if future_end >= len(frame):
        return None, None
    close = float(frame.iloc[index]["close"])
    if close <= 0:
        return None, None
    future = frame.iloc[index + 1 : future_end + 1]
    future_return = float(frame.iloc[future_end]["close"] / close - 1.0)
    future_drawdown = float(future["low"].min() / close - 1.0)
    return future_return, future_drawdown


def _bottom_type_label(frame: pd.DataFrame, index: int, horizon: int = 90) -> str | None:
    future_return, future_drawdown = _future_return_drawdown(frame, index, horizon)
    if future_return is None or future_drawdown is None:
        return None
    close = float(frame.iloc[index]["close"])
    future_end = index + horizon
    future_close = frame.iloc[index + 1 : future_end + 1]["close"]
    future_sma200 = frame.iloc[index + 1 : future_end + 1]["sma200"]
    reclaimed = bool((future_close > future_sma200).fillna(False).rolling(10, min_periods=3).sum().max() >= 5)
    if future_return >= 0.35 and future_drawdown >= -0.12 and reclaimed:
        return "cycle bottom"
    if future_return >= 0.15 and future_drawdown >= -0.15:
        return "tradable swing bottom"
    if future_return >= 0.05 and future_drawdown >= -0.10:
        return "local bounce"
    return "dead-cat bounce risk"


def _classify_current_bottom_type(latest: pd.Series, confidence: float, regime: str) -> dict[str, Any]:
    dd = float(latest.get("drawdown_365", 0.0)) if pd.notna(latest.get("drawdown_365", np.nan)) else 0.0
    rsi_val = float(latest.get("rsi14", 50.0)) if pd.notna(latest.get("rsi14", np.nan)) else 50.0
    days_low = float(latest.get("days_since_30d_low", 999.0)) if pd.notna(latest.get("days_since_30d_low", np.nan)) else 999.0
    close = float(latest["close"])
    sma200 = float(latest.get("sma200", np.nan))
    scores = {
        "cycle bottom": 0.15 + max(0.0, -dd - 0.45) * 1.2 + confidence * 0.35,
        "tradable swing bottom": 0.20 + max(0.0, -dd - 0.18) * 0.9 + confidence * 0.30,
        "local bounce": 0.35 + (1.0 if days_low <= 7 else 0.0) * 0.15 + max(0.0, 55 - rsi_val) / 100.0,
        "dead-cat bounce risk": 0.15 + (1.0 if pd.notna(sma200) and close < sma200 else 0.0) * 0.20 + max(0.0, 0.65 - confidence) * 0.5,
    }
    if regime == "macro stress":
        scores["dead-cat bounce risk"] += 0.20
    if regime == "bear market" and dd <= -0.35:
        scores["cycle bottom"] += 0.15
    total = sum(max(0.0, value) for value in scores.values()) or 1.0
    probs = {key: float(max(0.0, value) / total) for key, value in scores.items()}
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    return {"primary": ranked[0][0], "probabilities": probs, "ranked": [{"type": key, "probability": value} for key, value in ranked]}


def _recommendation_engine(confidence: float, bottom_type: str, regime: str, latest: pd.Series, levels: dict[str, float]) -> dict[str, Any]:
    band = _confidence_band(confidence)
    if regime == "macro stress" or bottom_type == "dead-cat bounce risk":
        action, sizing = ("defensive", "0-5%")
    elif band in {"extreme", "very_high"} and bottom_type in {"cycle bottom", "tradable swing bottom"}:
        action, sizing = ("aggressive accumulation", "35-60%")
    elif band == "high" and bottom_type in {"cycle bottom", "tradable swing bottom", "local bounce"}:
        action, sizing = ("tranche", "15-35%")
    elif band == "medium":
        action, sizing = ("probe", "5-15%")
    elif float(latest.get("days_since_30d_low", 999)) <= 3:
        action, sizing = ("wait", "0-5%")
    else:
        action, sizing = ("defensive", "0-5%")
    confirmation = levels.get("sma20")
    if confirmation is None or not np.isfinite(confirmation) or float(latest["close"]) > confirmation:
        confirmation = levels.get("prior_20d_high", float(latest["close"]) * 1.05)
    return {
        "action": action,
        "sizing_guidance": sizing,
        "confidence_band": band,
        "invalidation_level": levels.get("invalidation_low"),
        "confirmation_level": confirmation,
        "what_would_change_confidence": [
            "Higher score: regime-specific hitrates improve, spot demand broadens, or derivatives/on-chain reset strengthens.",
            "Lower score: invalidation low breaks, macro stress tag appears, or top positive drivers fade.",
        ],
    }


def _washout_evidence(frame: pd.DataFrame, idx: int) -> dict[str, Any]:
    window = slice(max(0, idx - 2), idx + 1)
    long_liq = _optional_column(frame, "derivatives_liquidations_long_usd")
    spike = 0.0
    if long_liq.notna().any():
        spike_series = long_liq / long_liq.rolling(30, min_periods=10).mean()
        spike = float(spike_series.iloc[window].max(skipna=True))
        if not np.isfinite(spike):
            spike = 0.0
    realized_loss = frame["realized_loss_ratio"] if "realized_loss_ratio" in frame.columns else pd.Series(np.nan, index=frame.index)
    realized_loss_climax = bool(realized_loss.iloc[window].max(skipna=True) >= 2.5) if realized_loss.notna().any() else False
    sth_capitulation = bool(float(frame.iloc[idx].get("sth_mvrv", np.nan)) <= 0.98)
    volume_climax = bool(float(frame.iloc[idx].get("volume_ratio_20", 0.0)) >= 1.8)
    vol_flush = bool(float(frame.iloc[idx].get("vol30", 0.0)) >= 0.8)
    detected = bool(spike >= 3.0 or realized_loss_climax or sth_capitulation or volume_climax or vol_flush)
    return {
        "washout_detected": detected,
        "recent_liquidation_spike": spike,
        "realized_loss_climax": realized_loss_climax,
        "sth_mvrv_capitulation": sth_capitulation,
        "volume_climax": volume_climax,
        "volatility_flush": vol_flush,
    }


def _false_bottom_penalty(
    latest: pd.Series,
    components: pd.DataFrame,
    latest_idx: int,
    confirmation_state: str,
    cycle_phase: str,
    washout: dict[str, Any],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"factor": 1.0, "reasons": [], "below_sma200": False, "macro_weak": False, "washout_detected": washout["washout_detected"]}
    close = float(latest["close"])
    sma200 = float(latest.get("sma200", np.nan))
    below_sma200 = bool(np.isfinite(sma200) and close < sma200)
    macro_score = float(components.loc[latest_idx, "macro"]) if "macro" in components.columns and pd.notna(components.loc[latest_idx, "macro"]) else 0.5
    macro_weak = macro_score < 0.40
    confirmed = confirmation_state in {"confirmed", "trend_reclaim"}
    factor = 1.0
    reasons: list[str] = []
    if below_sma200 and macro_weak and not washout["washout_detected"]:
        factor *= 0.82
        reasons.append("below_sma200_weak_macro_no_washout")
    if cycle_phase == "deep_bear" and not confirmed:
        factor *= 0.90
        reasons.append("deep_bear_without_confirmation")
    return {
        "factor": float(_clip01(factor)),
        "reasons": reasons,
        "below_sma200": below_sma200,
        "macro_weak": macro_weak,
        "macro_score": macro_score,
        "washout_detected": washout["washout_detected"],
    }


def _tranche_engine(
    confidence: float,
    confirmation_state: str,
    cycle_phase: str,
    levels: dict[str, float],
    require_confirmation: bool,
) -> dict[str, Any]:
    band = _confidence_band(confidence)
    eligible = 0
    reason = "confidence below medium"
    if band in {"medium", "high", "very_high", "extreme"}:
        eligible = 20
        reason = "medium setup permits probe"
    if eligible >= 20 and (not require_confirmation or confirmation_state in {"confirmed", "trend_reclaim"}):
        eligible = 50
        reason = "confirmation unlocks first add"
    if confirmation_state == "trend_reclaim" and band in {"high", "very_high", "extreme"}:
        eligible = 80
        reason = "trend reclaim unlocks second add"
    if cycle_phase == "deep_bear" and confirmation_state not in {"confirmed", "trend_reclaim"}:
        eligible = min(eligible, 20)
        reason = "deep bear keeps sizing capped before confirmation"
    reserve = max(0, 100 - eligible)
    next_locked = "reserve for undercut/retest" if eligible >= 80 else "needs confirmation or trend reclaim"
    return {
        "policy": "20/30/30/20",
        "eligible_allocation_pct": eligible,
        "reserve_allocation_pct": reserve,
        "current_tranche": reason,
        "next_tranche_locked_reason": next_locked,
        "confirmation_state": confirmation_state,
        "cycle_phase": cycle_phase,
        "invalidation_level": levels.get("invalidation_low"),
        "confirmation_level": levels.get("prior_20d_high"),
        "tranches": [
            {"name": "probe", "allocation_pct": 20, "unlocked": eligible >= 20},
            {"name": "confirmation_add", "allocation_pct": 30, "unlocked": eligible >= 50},
            {"name": "trend_reclaim_add", "allocation_pct": 30, "unlocked": eligible >= 80},
            {"name": "undercut_retest_reserve", "allocation_pct": 20, "unlocked": False},
        ],
    }


def _driver_attribution(components: pd.DataFrame, latest_idx: int, weights: dict[str, float], data_quality: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for component, weight in weights.items():
        if component not in components.columns:
            continue
        score = components.loc[latest_idx, component]
        if pd.isna(score):
            continue
        score = float(score)
        weight = float(weight)
        rows.append(
            {
                "driver": component,
                "score": score,
                "weight": weight,
                "positive_contribution": score * weight,
                "negative_contribution": (1.0 - score) * weight,
            }
        )
    positive = sorted(rows, key=lambda row: row["positive_contribution"], reverse=True)
    negative = sorted(rows, key=lambda row: row["negative_contribution"], reverse=True)
    source_penalties = []
    for warning in data_quality.get("warnings", []):
        source_penalties.append({"driver": "source_quality", "message": warning, "negative_contribution": 0.02})
    return {
        "positive_drivers": positive[:5],
        "negative_drivers": negative[:5],
        "waterfall": rows,
        "source_penalties": source_penalties,
    }


def _walk_forward_validation(
    frame: pd.DataFrame,
    components: pd.DataFrame,
    horizon: int,
    tolerance: float,
    step_days: int = 63,
    max_drawdown: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = _future_success_labels(frame, horizon=horizon, tolerance=tolerance)
    quality = _future_quality_labels(frame, horizon=horizon, tolerance=tolerance, max_drawdown=max_drawdown)
    rows = []
    start = min(max(500, horizon * 4), max(0, len(frame) - horizon - 1))
    for idx in range(start, len(frame) - horizon - 1, step_days):
        mask = _train_mask(frame, labels, idx)
        if int(mask.sum()) < 40:
            continue
        weights, _, base_rate = _learn_component_weights(components, labels, mask)
        blended = _weighted_score(components, weights)
        latest_score = float(blended.iloc[idx])
        prob, method, regime_rows, shrink = _calibrate_regime_probability(
            blended,
            labels,
            mask,
            frame["primary_regime"],
            str(frame.iloc[idx]["primary_regime"]),
            latest_score,
            base_rate,
        )
        outcome = labels.iloc[idx]
        fwd_return, fwd_drawdown = _future_return_drawdown(frame, idx, horizon)
        quality_outcome = quality.loc[idx, "quality_outcome"]
        rows.append(
            {
                "date": str(pd.Timestamp(frame.iloc[idx]["date"]).date()),
                "horizon_days": horizon,
                "tolerance": tolerance,
                "regime": str(frame.iloc[idx]["primary_regime"]),
                "confidence": prob,
                "outcome": int(outcome) if pd.notna(outcome) else np.nan,
                "quality_outcome": int(quality_outcome) if pd.notna(quality_outcome) else np.nan,
                "forward_return": fwd_return,
                "forward_drawdown": fwd_drawdown,
                "forward_drawdown_penalty": quality.loc[idx, "forward_drawdown_penalty"],
                "risk_adjusted_bottom_score": quality.loc[idx, "risk_adjusted_bottom_score"],
                "training_rows": int(mask.sum()),
                "regime_training_rows": regime_rows,
                "regime_shrinkage": shrink,
                "calibration_method": method,
            }
        )
    validation = pd.DataFrame(rows)
    if validation.empty:
        return validation, {"rows": 0}
    valid = validation.dropna(subset=["outcome", "confidence"])
    brier = float(((valid["confidence"] - valid["outcome"]) ** 2).mean()) if not valid.empty else None
    calibration_error = float(abs(valid["confidence"].mean() - valid["outcome"].mean())) if not valid.empty else None
    quality_valid = validation.dropna(subset=["quality_outcome"]) if "quality_outcome" in validation.columns else pd.DataFrame()
    summary = {
        "rows": int(len(valid)),
        "brier_score": brier,
        "calibration_error": calibration_error,
        "quality_hitrate": float(quality_valid["quality_outcome"].mean()) if not quality_valid.empty else None,
        "avg_forward_drawdown_penalty": float(validation["forward_drawdown_penalty"].mean()) if "forward_drawdown_penalty" in validation else None,
        "avg_risk_adjusted_bottom_score": float(validation["risk_adjusted_bottom_score"].mean()) if "risk_adjusted_bottom_score" in validation else None,
    }
    return validation, summary


def _confidence_bucket_summary(validation: pd.DataFrame) -> list[dict[str, Any]]:
    buckets = [
        ("low", 0.0, 0.60),
        ("medium", 0.60, 0.70),
        ("high", 0.70, 0.80),
        ("very_high", 0.80, 0.90),
        ("extreme", 0.90, 1.01),
    ]
    rows = []
    required = {"outcome", "confidence", "forward_drawdown", "forward_return"}
    valid = validation.dropna(subset=["outcome", "confidence"]) if required.issubset(validation.columns) else pd.DataFrame()
    for name, low, high in buckets:
        subset = valid[(valid["confidence"] >= low) & (valid["confidence"] < high)] if not valid.empty else pd.DataFrame()
        rows.append(
            {
                "bucket": name,
                "confidence_min": low,
                "confidence_max": min(high, 1.0),
                "sample_size": int(len(subset)),
                "observed_hitrate": float(subset["outcome"].mean()) if len(subset) else None,
                "quality_hitrate": float(subset["quality_outcome"].mean()) if len(subset) and "quality_outcome" in subset else None,
                "avg_forward_drawdown": float(subset["forward_drawdown"].mean()) if len(subset) and subset["forward_drawdown"].notna().any() else None,
                "avg_forward_return": float(subset["forward_return"].mean()) if len(subset) and subset["forward_return"].notna().any() else None,
                "avg_risk_adjusted_bottom_score": float(subset["risk_adjusted_bottom_score"].mean()) if len(subset) and "risk_adjusted_bottom_score" in subset else None,
                "is_sparse": bool(len(subset) < 5),
            }
        )
    return rows


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
    bucket_rows = []
    for row in summary.get("confidence_buckets", []):
        hitrate = _format_pct(row["observed_hitrate"]) if row.get("observed_hitrate") is not None else "n/a"
        quality_hitrate = _format_pct(row["quality_hitrate"]) if row.get("quality_hitrate") is not None else "n/a"
        fwd_dd = _format_pct(row["avg_forward_drawdown"]) if row.get("avg_forward_drawdown") is not None else "n/a"
        fwd_ret = _format_pct(row["avg_forward_return"]) if row.get("avg_forward_return") is not None else "n/a"
        sparse = "yes" if row.get("is_sparse") else "no"
        bucket_rows.append(
            "<tr>"
            f"<td>{html.escape(row['bucket'])}</td>"
            f"<td>{row['sample_size']}</td>"
            f"<td>{hitrate}</td>"
            f"<td>{quality_hitrate}</td>"
            f"<td>{fwd_dd}</td>"
            f"<td>{fwd_ret}</td>"
            f"<td>{sparse}</td>"
            "</tr>"
        )
    bucket_html = "".join(bucket_rows) or "<tr><td colspan='6'>No validation buckets available.</td></tr>"

    driver_pos = "".join(
        f"<li>{html.escape(row['driver'])}: {row['positive_contribution']:.3f}</li>"
        for row in summary.get("driver_attribution", {}).get("positive_drivers", [])[:3]
    )
    driver_neg = "".join(
        f"<li>{html.escape(row['driver'])}: {row['negative_contribution']:.3f}</li>"
        for row in summary.get("driver_attribution", {}).get("negative_drivers", [])[:3]
    )
    recommendation_details = summary.get("recommendation_details", {})
    bottom_type = summary.get("bottom_type", {})
    validation_summary = summary.get("walk_forward_validation", {})
    confirmation = summary.get("confirmation", {})
    false_penalty = summary.get("false_bottom_penalty", {})
    tranche_plan = summary.get("tranche_plan", {})
    penalty_reasons = ", ".join(false_penalty.get("reasons", [])) or "none"
    tranche_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('name', '')))}</td>"
        f"<td>{row.get('allocation_pct', 0)}%</td>"
        f"<td>{'yes' if row.get('unlocked') else 'no'}</td>"
        "</tr>"
        for row in tranche_plan.get("tranches", [])
    )
    
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
    <p><strong>Current read:</strong> {best['adjusted_confidence_pct']:.1f}% adjusted quality confidence that the recent BTC low holds over {best['horizon_days']} days with a {best['tolerance_pct']:.0f}% allowed breach. Band: <strong>{html.escape(best['adjusted_confidence_band'])}</strong>.</p>
    <p><strong>Recommendation:</strong> {html.escape(summary['recommendation'])}</p>
    <p><strong>Action:</strong> {html.escape(str(recommendation_details.get('action', 'n/a')))} | <strong>Sizing:</strong> {html.escape(str(recommendation_details.get('sizing_guidance', 'n/a')))}</p>
    <p><strong>Regime:</strong> {html.escape(str(summary.get('regime', {}).get('primary', 'n/a')))} | <strong>Bottom type:</strong> {html.escape(str(bottom_type.get('primary', 'n/a')))}</p>
    <p><strong>Confirmation:</strong> {html.escape(str(confirmation.get('confirmation_state', 'n/a')))} | <strong>Cycle phase:</strong> {html.escape(str(summary.get('cycle_phase', 'n/a')))}</p>
    <p><strong>Liquidation Driver:</strong> {html.escape(summary.get('washout_driver_text', ''))}</p>
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

  <h2>Regime, Bottom Type, And Recommendation</h2>
  <p><strong>Primary regime:</strong> {html.escape(str(summary.get('regime', {}).get('primary', 'n/a')))} ({html.escape(str(summary.get('regime', {}).get('tags', '')))}).</p>
  <p><strong>Bottom type:</strong> {html.escape(str(bottom_type.get('primary', 'n/a')))}. This augments the horizon/tolerance confidence and does not replace it.</p>
  <p><strong>False-bottom penalty factor:</strong> {false_penalty.get('factor', 1.0):.2f}. <strong>Reasons:</strong> {html.escape(penalty_reasons)}.</p>
  <p><strong>Action:</strong> {html.escape(str(recommendation_details.get('action', 'n/a')))}. <strong>Sizing guidance:</strong> {html.escape(str(recommendation_details.get('sizing_guidance', 'n/a')))}.</p>
  <p><strong>Confirmation:</strong> ${_format_num(recommendation_details.get('confirmation_level', np.nan))}. <strong>Invalidation:</strong> ${_format_num(recommendation_details.get('invalidation_level', np.nan))}.</p>

  <h2>Confirmation And Tranche Plan</h2>
  <p><strong>State:</strong> {html.escape(str(confirmation.get('confirmation_state', 'n/a')))}. <strong>Eligible allocation:</strong> {tranche_plan.get('eligible_allocation_pct', 0)}%. <strong>Reserve:</strong> {tranche_plan.get('reserve_allocation_pct', 0)}%.</p>
  <p><strong>Current tranche:</strong> {html.escape(str(tranche_plan.get('current_tranche', 'n/a')))}. <strong>Next locked because:</strong> {html.escape(str(tranche_plan.get('next_tranche_locked_reason', 'n/a')))}.</p>
  <table>
    <thead><tr><th>Tranche</th><th>Allocation</th><th>Unlocked</th></tr></thead>
    <tbody>{tranche_rows}</tbody>
  </table>

  <h2>Confidence Grid</h2>
  <p>Use the 5% tolerance rows as the stricter clean-bottom view and 10% rows as the tactical bottom-zone view.</p>
  {grid_html}

  <h2>Walk-Forward Reliability</h2>
  <p>Brier score: {validation_summary.get('brier_score', 'n/a')}. Calibration error: {validation_summary.get('calibration_error', 'n/a')}. Sparse buckets are flagged and should not be overclaimed.</p>
  <table>
    <thead><tr><th>Bucket</th><th>Samples</th><th>Observed hitrate</th><th>Quality hitrate</th><th>Avg forward drawdown</th><th>Avg forward return</th><th>Sparse</th></tr></thead>
    <tbody>{bucket_html}</tbody>
  </table>

  <h2>Driver Attribution</h2>
  <p>Top bullish and bearish drivers are based on latest component scores and selected calibration weights.</p>
  <h3>Bullish</h3>
  <ul>{driver_pos or '<li>No positive drivers available.</li>'}</ul>
  <h3>Bearish / Missing</h3>
  <ul>{driver_neg or '<li>No bearish drivers available.</li>'}</ul>

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
    frame = _add_regime_labels(frame)
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
    current_regime = str(latest["primary_regime"])
    data_quality = run_data_quality_checks(config, frame, as_of)
    quality_config = config.bottom_quality
    confirmation_audit = _confirmation_audit(frame)
    cycle_phase_audit = _cycle_phase_audit(frame) if quality_config.cycle_phase_enabled else pd.DataFrame({"date": frame["date"], "cycle_phase": "mid_cycle"})
    latest_confirmation = confirmation_audit.loc[latest_idx].to_dict()
    latest_cycle_phase = str(cycle_phase_audit.loc[latest_idx, "cycle_phase"])

    # Heatmap downside penalty calculation
    close_val = float(latest["close"])
    down_liq = float(latest.get("derivatives_heatmap_nearest_down_liq", np.nan))
    if pd.isna(down_liq):
        down_liq = float(latest.get("heatmap_nearest_down_liq", np.nan))

    penalty_factor = 1.0
    if not pd.isna(down_liq) and down_liq > 0 and close_val > 0:
        distance_pct = (close_val - down_liq) / close_val
        if 0 < distance_pct < 0.03:
            penalty_factor = 1.0 - 0.15 * (1.0 - distance_pct / 0.03)
    heatmap_penalty_factor = penalty_factor

    # Liquidation washout check
    long_liq_series = _optional_column(frame, "derivatives_liquidations_long_usd")
    if long_liq_series.notna().any():
        # rolling 3-day max of the spike
        spike_series = long_liq_series / long_liq_series.rolling(30, min_periods=10).mean()
        recent_spike = float(spike_series.iloc[max(0, latest_idx - 2):latest_idx + 1].max())
    else:
        recent_spike = 0.0

    imbalance_series = _optional_column(frame, "derivatives_liq_imbalance")
    if imbalance_series.isna().all():
        short_liq_series = _optional_column(frame, "derivatives_short_liq_usd")
        if long_liq_series.notna().any() and short_liq_series.notna().any():
            denom = long_liq_series + short_liq_series
            imbalance_series = (long_liq_series - short_liq_series) / denom.replace(0, np.nan)
            imbalance_series = imbalance_series.fillna(0.0)

    if imbalance_series.notna().any():
        recent_imbalance = float(imbalance_series.iloc[max(0, latest_idx - 2):latest_idx + 1].max())
    else:
        recent_imbalance = 0.0

    washout_detected = (recent_spike >= 3.0) or (recent_imbalance >= 0.7)

    if washout_detected:
        washout_text = f"Washout detected (recent spike: {recent_spike:.1f}x, imbalance: {recent_imbalance:.1%}). Recent leverage flush cleared out buyers, supporting bottom formation."
    else:
        washout_text = "No washout: lack of severe liquidation spikes suggests leverage is still building or calm."
    washout_evidence = _washout_evidence(frame, latest_idx)
    if washout_detected:
        washout_evidence["washout_detected"] = True
    false_bottom_penalty = _false_bottom_penalty(
        latest,
        components,
        latest_idx,
        str(latest_confirmation["confirmation_state"]),
        latest_cycle_phase,
        washout_evidence,
        quality_config.false_bottom_penalty_enabled,
    )
    penalty_factor *= float(false_bottom_penalty["factor"])
    combined_penalty_factor = penalty_factor

    run_name = run_id or f"crypto-bottom-score-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = ensure_dir(Path("runs") / run_name)

    grid_rows = []
    hitrate_frames = []
    coef_payload: dict[str, Any] = {}
    latest_scores = {}
    best_row: dict[str, Any] | None = None
    primary_row: dict[str, Any] | None = None
    strict_probability = 0.0
    strict_adjusted_probability = 0.0

    for horizon in HORIZONS:
        for tolerance in TOLERANCES:
            hold_labels = _future_success_labels(frame, horizon=horizon, tolerance=tolerance)
            quality_labels = _future_quality_labels(
                frame,
                horizon=horizon,
                tolerance=tolerance,
                max_drawdown=quality_config.max_forward_drawdown,
            )
            is_primary_quality = (
                horizon == quality_config.primary_horizon_days
                and abs(tolerance - quality_config.primary_tolerance) < 1e-9
            )
            labels = quality_labels["quality_outcome"] if is_primary_quality else hold_labels
            mask = _train_mask(frame, labels, latest_idx)
            weights, hitrates, global_base_rate = _learn_component_weights(components, labels, mask)
            base_rate = global_base_rate
            regime_mask = mask & (frame["primary_regime"] == current_regime)
            if int(regime_mask.sum()) >= 40:
                weights, hitrates, base_rate = _learn_component_weights(components, labels, regime_mask)
            blended_score = _weighted_score(components, weights)
            latest_score = float(blended_score.iloc[latest_idx])
            rule_prob, calibration_method, regime_rows, regime_shrinkage = _calibrate_regime_probability(
                blended_score,
                labels,
                mask,
                frame["primary_regime"],
                current_regime,
                latest_score,
                global_base_rate,
            )
            ml_prob, coefs = _fit_logistic_probability(components, labels, mask, latest_idx)
            confidence = float(_clip01(rule_prob if ml_prob is None else 0.55 * rule_prob + 0.45 * ml_prob))
            adjusted_confidence = confidence * penalty_factor
            if horizon == 60 and tolerance == 0.05:
                strict_probability = confidence
                strict_adjusted_probability = adjusted_confidence
            row = {
                "horizon_days": horizon,
                "tolerance": tolerance,
                "tolerance_pct": tolerance * 100,
                "training_rows": int(mask.sum()),
                "regime": current_regime,
                "regime_training_rows": regime_rows,
                "regime_shrinkage": regime_shrinkage,
                "base_hitrate": base_rate,
                "global_base_hitrate": global_base_rate,
                "rule_probability": rule_prob,
                "ml_probability": ml_prob,
                "confidence": confidence,
                "confidence_pct": confidence * 100,
                "adjusted_confidence": adjusted_confidence,
                "adjusted_confidence_pct": adjusted_confidence * 100,
                "confidence_band": _confidence_band(confidence),
                "adjusted_confidence_band": _confidence_band(adjusted_confidence),
                "weighted_signal_score": latest_score,
                "calibration_method": calibration_method,
                "quality_calibrated": is_primary_quality,
                "latest_quality_outcome": quality_labels.loc[latest_idx, "quality_outcome"],
                "latest_forward_drawdown_penalty": quality_labels.loc[latest_idx, "forward_drawdown_penalty"],
                "latest_risk_adjusted_bottom_score": quality_labels.loc[latest_idx, "risk_adjusted_bottom_score"],
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
            if is_primary_quality:
                primary_row = row

    assert best_row is not None
    best_row = primary_row or best_row
    grid = pd.DataFrame(grid_rows).sort_values(["horizon_days", "tolerance"]).reset_index(drop=True)
    hitrates_all = pd.concat(hitrate_frames, ignore_index=True) if hitrate_frames else pd.DataFrame()
    best_key = f"{best_row['horizon_days']}d_{int(best_row['tolerance'] * 100)}pct"
    best_weights = latest_scores[best_key]["weights"]
    levels = _support_resistance(latest, best_row["tolerance"])
    recommendation = _recommendation(float(best_row["confidence"]), strict_probability, latest)
    adjusted_recommendation = _recommendation(float(best_row["adjusted_confidence"]), strict_adjusted_probability, latest)
    bottom_type = _classify_current_bottom_type(latest, float(best_row["adjusted_confidence"]), current_regime)
    recommendation_details = _recommendation_engine(
        float(best_row["adjusted_confidence"]),
        str(bottom_type["primary"]),
        current_regime,
        latest,
        levels,
    )
    tranche_plan = _tranche_engine(
        float(best_row["adjusted_confidence"]),
        str(latest_confirmation["confirmation_state"]),
        latest_cycle_phase,
        levels,
        quality_config.confirmation_required_for_adds,
    )
    recommendation_details["sizing_guidance"] = f"{tranche_plan['eligible_allocation_pct']}% eligible / {tranche_plan['reserve_allocation_pct']}% reserve"
    recommendation_details["confirmation_state"] = str(latest_confirmation["confirmation_state"])
    recommendation_details["cycle_phase"] = latest_cycle_phase

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
    driver_attribution = _driver_attribution(components, latest_idx, best_weights, data_quality)
    validation, validation_summary = _walk_forward_validation(
        frame,
        components,
        int(best_row["horizon_days"]),
        float(best_row["tolerance"]),
        max_drawdown=quality_config.max_forward_drawdown,
    )
    confidence_buckets = _confidence_bucket_summary(validation)

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
            "primary_regime": current_regime,
            "regime_tags": str(latest["regime_tags"]),
            "rsi14": float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else None,
            "drawdown_365": float(latest["drawdown_365"]) if pd.notna(latest["drawdown_365"]) else None,
            "days_since_30d_low": float(latest["days_since_30d_low"]) if pd.notna(latest["days_since_30d_low"]) else None,
        },
        "best_case": best_row,
        "strict_60d_5pct_confidence": strict_probability,
        "strict_60d_5pct_adjusted_confidence": strict_adjusted_probability,
        "heatmap_penalty_factor": heatmap_penalty_factor,
        "combined_penalty_factor": combined_penalty_factor,
        "false_bottom_penalty": false_bottom_penalty,
        "washout_driver_text": washout_text,
        "washout_evidence": washout_evidence,
        "recommendation": recommendation,
        "adjusted_recommendation": adjusted_recommendation,
        "recommendation_details": recommendation_details,
        "confirmation": to_native(latest_confirmation),
        "cycle_phase": latest_cycle_phase,
        "tranche_plan": tranche_plan,
        "bottom_quality_config": to_native(quality_config.__dict__),
        "bottom_type": bottom_type,
        "regime": {"primary": current_regime, "tags": str(latest["regime_tags"])},
        "levels": levels,
        "component_snapshot": component_snapshot,
        "driver_attribution": driver_attribution,
        "walk_forward_validation": validation_summary,
        "confidence_buckets": confidence_buckets,
        "model_note": "Probability recent low holds inside horizon/tolerance; not a guarantee of no lower print.",
        "data_quality": data_quality,
        "reliability_rating": data_quality["reliability_rating"],
        "reliability_note": data_quality["reliability_note"],
        "options_audit": {
            "options_25d_skew": float(latest["options_25d_skew"]) if "options_25d_skew" in latest and pd.notna(latest["options_25d_skew"]) else None,
            "options_put_call_oi": float(latest["options_put_call_oi"]) if "options_put_call_oi" in latest and pd.notna(latest["options_put_call_oi"]) else None,
            "options_put_call_volume": float(latest["options_put_call_volume"]) if "options_put_call_volume" in latest and pd.notna(latest["options_put_call_volume"]) else None,
            "options_iv_30d": float(latest["options_iv_30d"]) if "options_iv_30d" in latest and pd.notna(latest["options_iv_30d"]) else None,
            "options_term_structure": float(latest["options_term_structure"]) if "options_term_structure" in latest and pd.notna(latest["options_term_structure"]) else None,
            "options_dvol": float(latest["options_dvol"]) if "options_dvol" in latest and pd.notna(latest["options_dvol"]) else None,
        },
    }

    derivatives_audit = _derivatives_subsignals(frame).copy()
    derivatives_audit.insert(0, "date", frame["date"])
    derivatives_audit["derivatives_overall"] = components["derivatives"]
    derivatives_audit.to_csv(run_dir / "bottom_derivatives_audit.csv", index=False)
    confirmation_audit.to_csv(run_dir / "bottom_confirmation_audit.csv", index=False)
    cycle_phase_audit.to_csv(run_dir / "bottom_cycle_phase_audit.csv", index=False)

    type_audit_rows = []
    for idx in frame.index:
        label = _bottom_type_label(frame, int(idx))
        if label is None:
            continue
        type_audit_rows.append(
            {
                "date": frame.loc[idx, "date"],
                "primary_regime": frame.loc[idx, "primary_regime"],
                "bottom_type_label": label,
            }
        )
    pd.DataFrame(type_audit_rows).to_csv(run_dir / "bottom_type_audit.csv", index=False)

    audit_components = components.copy()
    audit_components["primary_regime"] = frame["primary_regime"]
    audit_components["regime_tags"] = frame["regime_tags"]
    for column in derivatives_audit.columns:
        if column != "date":
            audit_components[column] = derivatives_audit[column]

    grid.to_csv(run_dir / "bottom_probability_grid.csv", index=False)
    audit_components.to_csv(run_dir / "bottom_component_scores.csv", index=False)
    hitrates_all.to_csv(run_dir / "bottom_feature_hitrates.csv", index=False)
    if not validation.empty:
        validation.to_csv(run_dir / "bottom_walkforward_validation.csv", index=False)
    pd.DataFrame(confidence_buckets).to_csv(run_dir / "bottom_confidence_buckets.csv", index=False)
    write_json(run_dir / "bottom_model_coefficients.json", coef_payload)
    write_json(run_dir / "bottom_driver_attribution.json", to_native(driver_attribution))
    write_json(run_dir / "bottom_validation_summary.json", to_native({"summary": validation_summary, "confidence_buckets": confidence_buckets}))
    write_json(
        run_dir / "bottom_quality_summary.json",
        to_native(
            {
                "bottom_quality_config": quality_config.__dict__,
                "false_bottom_penalty": false_bottom_penalty,
                "heatmap_penalty_factor": heatmap_penalty_factor,
                "combined_penalty_factor": combined_penalty_factor,
                "confirmation": latest_confirmation,
                "cycle_phase": latest_cycle_phase,
                "tranche_plan": tranche_plan,
                "walk_forward_validation": validation_summary,
            }
        ),
    )
    write_json(run_dir / "bottom_score_summary.json", to_native(summary))
    report_path = run_dir / "bottom_report.html"
    report_path.write_text(_report_html(summary, grid, components, hitrates_all), encoding="utf-8")
    return BottomScoreRun(run_id=run_name, run_dir=run_dir, report_path=report_path, summary=to_native(summary))
