from __future__ import annotations

from egx_research.backtest import run_strategy_backtest
from egx_research.config import BacktestConfig
from egx_research.strategies import build_strategy_frame


def test_each_family_produces_trades_on_synthetic_data(synthetic_ohlcv) -> None:
    params = {
        "trend": {"fast_ma": 20, "slow_ma": 80, "ma_type": "EMA", "adx_len": 14, "adx_threshold": 18.0, "atr_stop": 2.0, "atr_trail": 1.5},
        "mean_reversion": {"regime_ma": 120, "rsi_len": 5, "rsi_entry": 30.0, "rsi_exit": 55.0, "bb_len": 20, "bb_std": 2.0, "atr_stop": 2.0},
        "breakout": {"entry_lookback": 40, "exit_lookback": 20, "regime_ma": 120, "atr_stop": 2.5, "atr_trail": 1.5},
        "fdi_supertrend": {"per": 30, "speed": 20, "mult": 3.0, "adapt": True},
        "kama_cci_atr_allocation": {"kama_len": 20, "kama_fast": 2, "kama_slow": 30, "cci_len": 20, "cci_threshold": 10.0, "allocation_ladder": "100_50_0", "atr_len": 14, "atr_stop": 2.0, "atr_trail": 1.5},
        "dca_tactical_overlay": {"core_weight": 0.75, "kama_len": 20, "kama_fast": 2, "kama_slow": 30, "cci_len": 20, "cci_threshold": 10.0, "sleeve_ladder": "100_60_20", "atr_len": 14, "atr_stop": 2.0, "atr_trail": 1.5},
        "dca_zone_overlay": {"core_weight": 0.8, "kama_len": 20, "kama_fast": 2, "kama_slow": 30, "cci_len": 20, "buy_mild": -20.0, "buy_deep": -80.0, "trim_mild": 80.0, "trim_hard": 140.0, "trend_buffer_atr": 0.5, "trim_buffer_atr": 1.0, "zone_profile": "100_75_50_25_0", "atr_len": 14, "atr_stop": 2.0, "atr_trail": 1.5},
        "dca_pullback_topup": {"core_weight": 0.8, "reserve_weight": 0.2, "kama_len": 20, "kama_fast": 2, "kama_slow": 30, "cci_len": 20, "buy_mild": -20.0, "buy_deep": -80.0, "trend_buffer_atr": 0.5, "topup_profile": "25_50_100", "atr_len": 14, "atr_stop": 2.0, "atr_trail": 1.5},
        "dca_pullback_only": {"kama_len": 20, "kama_fast": 2, "kama_slow": 30, "cci_len": 20, "buy_threshold": -40.0, "trend_buffer_atr": 0.5, "atr_len": 14},
        "hierarchy_combo": {
            "hierarchy": "trend_momentum_volume",
            "trend_indicator": "supertrend",
            "st_len": 10,
            "st_mult": 2.5,
            "momentum_indicator": "rsi",
            "mom_rsi_len": 14,
            "mom_rsi_threshold": 52.0,
            "volume_indicator": "cmf",
            "cmf_len": 20,
            "cmf_threshold": -0.02,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_dynamic_momentum": {
            "stoch_len": 55,
            "kd_smooth": 3,
            "mom_fast": 13,
            "mom_slow": 34,
            "mom_signal": 5,
            "trend_len": 80,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_multi_bbands": {
            "bb_len": 20,
            "inner_mult": 1.4,
            "outer_mult": 2.6,
            "trend_ma": 80,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_zlema_band": {
            "zlema_len": 21,
            "band_mult": 1.0,
            "trend_ma": 80,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_ichimoku": {
            "tenkan_len": 9,
            "kijun_len": 26,
            "senkou_b_len": 52,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_ravi": {
            "fast_len": 7,
            "slow_len": 28,
            "bias_len": 120,
            "ravi_entry": 0.8,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_cci_rsi": {
            "cci_len": 20,
            "rsi_len": 10,
            "rsi_signal": 5,
            "cci_entry": -40.0,
            "cci_exit": 80.0,
            "trend_ma": 80,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
        "blackcat_superj": {
            "stoch_len": 14,
            "kd_smooth": 3,
            "j_smooth": 5,
            "trigger_len": 4,
            "trend_ma": 80,
            "oversold": 25.0,
            "atr_stop": 2.0,
            "atr_trail": 1.5,
        },
    }
    config = BacktestConfig(initial_cash=50000.0, monthly_contribution=1000.0, fee_bps=0.0, slippage_bps=0.0)
    for family, family_params in params.items():
        frame = build_strategy_frame(synthetic_ohlcv, family, family_params)
        result = run_strategy_backtest(frame, 0, len(frame) - 1, config)
        assert result.metrics["final_equity"] > 0.0
        if family == "dca_pullback_only":
            assert result.trades["shares"].sum() > 0.0
        else:
            assert result.metrics["closed_trades"] > 0.0
