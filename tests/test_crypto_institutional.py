from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from egx_research.crypto_config import CryptoConfig, save_crypto_config
from egx_research.crypto_institutional import (
    INSTITUTIONAL_FAMILY,
    INSTITUTIONAL_SLEEVES,
    build_institutional_ensemble_frame,
)
from egx_research.crypto_research import _institutional_objective, run_crypto_research
from egx_research.crypto_strategies import normalize_crypto_params
from egx_research.crypto_paper_tracking import (
    build_current_crypto_signal,
    paper_track_crypto_strategy,
)


def _institutional_features(rows: int = 1400, crash: bool = False) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 * np.exp(index * 0.0007 + 0.08 * np.sin(index / 45.0))
    if crash:
        close[-90:] *= np.linspace(1.0, 0.45, 90)
        close[-30:] *= 1.0 + 0.08 * np.sin(np.arange(30) * 2.0)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2019-01-01", periods=rows, freq="D"),
            "open": close * 0.998,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1000.0 + index,
            "CapMVRVCur": 1.5 + 0.4 * np.sin(index / 80.0),
            "AdrActCnt": 500_000 + index * 100,
            "TxCnt": 250_000 + index * 50,
            "HashRate": 1_000_000 + index * 200,
            "FlowInExUSD": 1_000_000 + 50_000 * np.sin(index / 20.0),
            "FlowOutExUSD": 1_050_000 + 50_000 * np.cos(index / 20.0),
            "derivatives_open_interest": 1_000_000 + index * 500,
            "derivatives_long_short_ratio": 1.0 + 0.2 * np.sin(index / 18.0),
            "derivatives_leverage_ratio": 0.15 + 0.02 * np.sin(index / 25.0),
            "derivatives_toptrader_position_ratio": 1.0 + 0.1 * np.cos(index / 22.0),
            "funding_rate_mean": 0.0001 + 0.00005 * np.sin(index / 14.0),
            "derivatives_basis": 0.05 + 0.02 * np.sin(index / 30.0),
            "liquidity_stablecoin_supply": 100_000_000_000 + index * 20_000_000,
            "liquidity_dry_powder_ratio": 0.12 + 0.01 * np.sin(index / 70.0),
            "etf_net_flow_usd": 10_000_000 * np.sin(index / 12.0),
            "spot_coinbase_premium": 0.001 * np.sin(index / 10.0),
            "macro_nasdaq": 10_000 + index * 3,
            "macro_us10y": 4.0 - index * 0.0003,
            "macro_dollar": 105.0 - index * 0.001,
            "macro_fed_liquidity": 8_000 + index,
            "macro_vix": 20.0 + 3.0 * np.sin(index / 17.0),
            "fear_greed_value": 50.0 + 30.0 * np.sin(index / 25.0),
            "options_dvol": 60.0 + 10.0 * np.sin(index / 19.0),
            "options_put_call_ratio": 0.8 + 0.1 * np.cos(index / 21.0),
        }
    )
    return frame


def _params() -> dict[str, object]:
    return normalize_crypto_params(INSTITUTIONAL_FAMILY, {})


def test_institutional_objective_prioritizes_dca_excess_and_drawdown_limit() -> None:
    good = {
        "excess_return_vs_dca": 0.08,
        "max_drawdown": 0.30,
        "return_dd": 0.8,
        "sharpe": 1.0,
    }
    bad = {
        "excess_return_vs_dca": 0.02,
        "max_drawdown": 0.50,
        "return_dd": 1.2,
        "sharpe": 1.5,
    }
    assert _institutional_objective(good) > _institutional_objective(bad)


def test_institutional_frame_is_bounded_probabilistic_and_attributed() -> None:
    frame = build_institutional_ensemble_frame(
        _institutional_features(),
        _params(),
    )
    probabilities = frame[
        [
            f"regime_probability_{name}"
            for name in ("bull", "bear", "range", "crisis", "recovery")
        ]
    ]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert frame["target_allocation"].between(0.0, 1.0).all()
    assert frame["target_allocation"].notna().all()
    weight_columns = [f"institutional_weight_{name}" for name in INSTITUTIONAL_SLEEVES]
    assert np.allclose(frame[weight_columns].sum(axis=1), 1.0)
    calibrated = frame["institutional_model_confidence"] > 0
    assert (
        frame.loc[calibrated, weight_columns].max(axis=1)
        <= float(_params()["maximum_sleeve_weight"]) + 1e-12
    ).all()
    contribution_columns = [
        f"institutional_contribution_{name}" for name in INSTITUTIONAL_SLEEVES
    ]
    assert np.allclose(
        frame.loc[calibrated, contribution_columns].sum(axis=1),
        frame.loc[calibrated, "institutional_ensemble_score"],
    )
    increases = frame["target_allocation"].diff().dropna().clip(lower=0.0)
    assert increases.max() <= float(_params()["maximum_target_change"]) + 1e-12


def test_institutional_decisions_do_not_change_when_future_data_changes() -> None:
    original = _institutional_features()
    changed = original.copy()
    cutoff = 1150
    changed.loc[cutoff:, "close"] *= 3.0
    changed.loc[cutoff:, "open"] *= 3.0
    changed.loc[cutoff:, "high"] *= 3.0
    changed.loc[cutoff:, "low"] *= 3.0
    changed.loc[cutoff:, "funding_rate_mean"] = 1.0
    first = build_institutional_ensemble_frame(original, _params())
    second = build_institutional_ensemble_frame(changed, _params())
    compared = [
        "target_allocation",
        "institutional_expected_return_90d",
        "institutional_model_confidence",
        "institutional_regime",
    ]
    pd.testing.assert_frame_equal(
        first.loc[: cutoff - 1, compared],
        second.loc[: cutoff - 1, compared],
    )


def test_institutional_frame_fails_safe_when_external_features_are_missing() -> None:
    data = _institutional_features()[
        ["date", "open", "high", "low", "close", "volume"]
    ]
    frame = build_institutional_ensemble_frame(data, _params())
    assert frame["institutional_external_coverage"].eq(0.0).all()
    assert frame["target_allocation"].between(0.0, 1.0).all()
    assert frame["target_allocation"].notna().all()


def test_crisis_state_can_reduce_below_normal_core() -> None:
    params = _params()
    params["crisis_probability"] = 0.35
    params["crisis_allocation"] = 0.0
    frame = build_institutional_ensemble_frame(
        _institutional_features(crash=True),
        params,
    )
    assert frame["regime_probability_crisis"].tail(90).max() >= 0.35
    assert frame["target_allocation"].tail(90).min() < float(params["core_weight"])


def test_current_signal_explains_regime_forecast_and_sleeves() -> None:
    params = _params()
    frame = build_institutional_ensemble_frame(
        _institutional_features(),
        params,
    )
    signal = build_current_crypto_signal(frame, INSTITUTIONAL_FAMILY, params)
    assert signal["execution_timing"] == "next_verified_execution_window"
    assert "institutional_regime" in signal["diagnostics"]
    assert "institutional_expected_return_90d" in signal["diagnostics"]
    assert any("largest sleeve contributions" in reason for reason in signal["reasons"])


def test_paper_track_blocks_institutional_model_without_verified_vintages(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "data/crypto/features").mkdir(parents=True)
    (tmp_path / "runs/model").mkdir(parents=True)
    config = CryptoConfig()
    config.data.features_dir = "data/crypto/features"
    config.data.normalized_dir = "data/crypto/normalized"
    config.data.raw_dir = "data/crypto/raw"
    save_crypto_config("config/crypto_btc.yaml", config)
    features = _institutional_features(rows=1000)
    features.to_csv(config.data.features_path, index=False)
    Path(config.data.features_dir, "BTCUSDT_feature_quality.json").write_text(
        json.dumps({"point_in_time_vintage_verified": False}),
        encoding="utf-8",
    )
    candidate = {
        "family": INSTITUTIONAL_FAMILY,
        "params": _params(),
        "passed_filters": True,
        "rank_score": 1.0,
        "neighbor_pass_rate": 1.0,
        "wf_metrics": {
            "score": 1.0,
            "excess_return_vs_dca": 0.1,
            "sharpe": 1.0,
            "max_drawdown": 0.2,
        },
    }
    Path("runs/model/candidates.json").write_text(
        json.dumps({"candidates": [candidate]}),
        encoding="utf-8",
    )
    run_dir = paper_track_crypto_strategy(
        model_run_id="model",
        start_date="2021-01-01",
        config_path="config/crypto_btc.yaml",
        out_run_id="paper",
        max_data_stale_days=99999,
    )
    summary = json.loads(
        (run_dir / "paper_track_summary.json").read_text(encoding="utf-8")
    )
    assert summary["trade_allowed"] is False
    assert summary["model_accepted"] is False
    assert "model_not_accepted" in summary["block_reasons"]


def test_institutional_research_writes_attribution_and_blocks_unverified_vintages(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "data/crypto/features").mkdir(parents=True)
    config = CryptoConfig()
    config.data.features_dir = "data/crypto/features"
    config.data.normalized_dir = "data/crypto/normalized"
    config.data.raw_dir = "data/crypto/raw"
    config.search.families = [INSTITUTIONAL_FAMILY]
    config.search.trials_per_family = 1
    config.validation.holdout_ratio = 0.2
    config.validation.primary_train_bars = 400
    config.validation.primary_test_bars = 120
    config.validation.primary_step_bars = 120
    config.validation.outer_test_bars = 120
    config.validation.outer_step_bars = 120
    config.validation.purge_bars = 5
    config.validation.embargo_bars = 5
    save_crypto_config("config/crypto_btc.yaml", config)
    features = _institutional_features(rows=1200)
    features.to_csv(config.data.features_path, index=False)
    Path(config.data.features_dir, "BTCUSDT_feature_quality.json").write_text(
        json.dumps({"point_in_time_vintage_verified": False}),
        encoding="utf-8",
    )

    run_id = run_crypto_research(
        config,
        "config/crypto_btc.yaml",
        trials_override=1,
        family_override=INSTITUTIONAL_FAMILY,
        run_id="institutional-smoke",
    )
    run_dir = Path("runs") / run_id
    summary = json.loads(
        (run_dir / "crypto_research_summary.json").read_text(encoding="utf-8")
    )
    assert summary["top_family"] == INSTITUTIONAL_FAMILY
    assert summary["external_data_required"] is True
    assert summary["external_vintages_verified"] is False
    assert summary["production_eligible"] is False
    assert (run_dir / "institutional_daily_attribution.csv").exists()
    assert (run_dir / "institutional_current_state.csv").exists()
    assert (run_dir / "institutional_sleeve_ablation.csv").exists()
