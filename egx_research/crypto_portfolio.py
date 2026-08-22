from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioPolicy:
    maximum_asset_weight: float = 0.30
    maximum_venue_weight: float = 0.80
    maximum_quote_asset_weight: float = 1.00
    target_annualized_volatility: float = 0.40
    maximum_one_way_turnover: float = 0.25
    maximum_adv_participation: float = 0.01
    maximum_gross_exposure: float = 1.00
    cash_asset: str = "USDT"
    annualization_periods: float = 365.0


@dataclass(frozen=True)
class PortfolioAllocation:
    weights: dict[str, float]
    cash_weight: float
    expected_annualized_volatility: float
    one_way_turnover: float
    venue_weights: dict[str, float]
    quote_asset_weights: dict[str, float]
    capacity_limited_assets: tuple[str, ...]
    correlation_adjustments: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_policy(policy: PortfolioPolicy) -> None:
    bounded = (
        policy.maximum_asset_weight,
        policy.maximum_venue_weight,
        policy.maximum_quote_asset_weight,
        policy.maximum_one_way_turnover,
        policy.maximum_adv_participation,
        policy.maximum_gross_exposure,
    )
    if any(value < 0 for value in bounded):
        raise ValueError("portfolio limits cannot be negative")
    if policy.target_annualized_volatility <= 0 or policy.annualization_periods <= 0:
        raise ValueError("portfolio volatility settings must be positive")


def _group_cap(
    weights: pd.Series,
    candidates: pd.DataFrame,
    column: str,
    limit: float,
) -> pd.Series:
    result = weights.copy()
    for _, indices in candidates.groupby(column).groups.items():
        total = float(result.loc[list(indices)].sum())
        if total > limit and total > 0:
            result.loc[list(indices)] *= limit / total
    return result


def construct_portfolio(
    candidates: pd.DataFrame,
    returns_history: pd.DataFrame,
    current_weights: dict[str, float],
    *,
    portfolio_equity: float,
    policy: PortfolioPolicy | None = None,
) -> PortfolioAllocation:
    selected_policy = policy or PortfolioPolicy()
    _validate_policy(selected_policy)
    required = {
        "asset_id",
        "venue",
        "quote_asset",
        "score",
        "realized_volatility",
        "adv_quote",
        "eligible",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"portfolio candidates missing columns: {', '.join(missing)}")
    active = candidates.loc[candidates["eligible"].astype(bool)].copy()
    active = active.loc[
        (pd.to_numeric(active["score"], errors="coerce") > 0)
        & (pd.to_numeric(active["realized_volatility"], errors="coerce") > 0)
    ].reset_index(drop=True)
    if active["asset_id"].duplicated().any():
        raise ValueError("portfolio candidates require one execution venue per asset")
    if active.empty:
        turnover = 0.5 * sum(abs(value) for value in current_weights.values())
        return PortfolioAllocation(
            {}, 1.0, 0.0, turnover, {}, {}, (), {}
        )

    asset_ids = active["asset_id"].astype(str).tolist()
    available_returns = returns_history.reindex(columns=asset_ids)
    correlation = available_returns.corr(min_periods=20).fillna(0.0)
    adjustments: dict[str, float] = {}
    for asset in asset_ids:
        peers = correlation.loc[asset].drop(labels=[asset], errors="ignore").clip(lower=0)
        average_positive = float(peers.mean()) if len(peers) else 0.0
        adjustments[asset] = 1.0 / (1.0 + average_positive)

    scores = pd.to_numeric(active["score"], errors="coerce").clip(lower=0)
    volatility = pd.to_numeric(active["realized_volatility"], errors="coerce")
    raw = pd.Series(
        [scores.iloc[index] / volatility.iloc[index] * adjustments[asset]
         for index, asset in enumerate(asset_ids)],
        index=active.index,
        dtype=float,
    )
    weights = raw / raw.sum() * selected_policy.maximum_gross_exposure
    weights = weights.clip(upper=selected_policy.maximum_asset_weight)
    weights = _group_cap(
        weights, active, "venue", selected_policy.maximum_venue_weight
    )
    weights = _group_cap(
        weights,
        active,
        "quote_asset",
        selected_policy.maximum_quote_asset_weight,
    )

    covariance = available_returns.cov(min_periods=20).fillna(0.0).to_numpy(dtype=float)
    vector = weights.to_numpy(dtype=float)
    variance = float(vector @ covariance @ vector) * selected_policy.annualization_periods
    annualized_volatility = sqrt(max(0.0, variance))
    if annualized_volatility > selected_policy.target_annualized_volatility:
        scale = selected_policy.target_annualized_volatility / annualized_volatility
        weights *= scale

    capacity_limited: list[str] = []
    if portfolio_equity <= 0:
        raise ValueError("portfolio equity must be positive")
    for index, row in active.iterrows():
        asset = str(row["asset_id"])
        current = max(0.0, float(current_weights.get(asset, 0.0)))
        maximum_delta = (
            max(0.0, float(row["adv_quote"]))
            * selected_policy.maximum_adv_participation
            / portfolio_equity
        )
        desired = float(weights.loc[index])
        lower = max(0.0, current - maximum_delta)
        upper = current + maximum_delta
        constrained = min(max(desired, lower), upper)
        if not np.isclose(constrained, desired):
            capacity_limited.append(asset)
        weights.loc[index] = constrained

    desired_map = {
        asset: float(weights.iloc[index]) for index, asset in enumerate(asset_ids)
    }
    all_assets = set(current_weights) | set(desired_map)
    one_way_turnover = 0.5 * sum(
        abs(desired_map.get(asset, 0.0) - current_weights.get(asset, 0.0))
        for asset in all_assets
    )
    if one_way_turnover > selected_policy.maximum_one_way_turnover and one_way_turnover > 0:
        blend = selected_policy.maximum_one_way_turnover / one_way_turnover
        desired_map = {
            asset: current_weights.get(asset, 0.0)
            + blend * (desired_map.get(asset, 0.0) - current_weights.get(asset, 0.0))
            for asset in all_assets
        }
        desired_map = {asset: max(0.0, value) for asset, value in desired_map.items() if value > 1e-12}

    active_assets = set(asset_ids)
    desired_map = {
        asset: min(value, selected_policy.maximum_asset_weight)
        for asset, value in desired_map.items()
        if asset in active_assets
    }
    for column, limit in (
        ("venue", selected_policy.maximum_venue_weight),
        ("quote_asset", selected_policy.maximum_quote_asset_weight),
    ):
        for group, rows in active.groupby(column):
            members = set(rows["asset_id"].astype(str))
            total = sum(desired_map.get(asset, 0.0) for asset in members)
            if total > limit and total > 0:
                scale = limit / total
                for asset in members:
                    if asset in desired_map:
                        desired_map[asset] *= scale
    gross = sum(desired_map.values())
    if gross > selected_policy.maximum_gross_exposure and gross > 0:
        scale = selected_policy.maximum_gross_exposure / gross
        desired_map = {asset: value * scale for asset, value in desired_map.items()}
    all_assets = set(current_weights) | set(desired_map)
    one_way_turnover = 0.5 * sum(
        abs(desired_map.get(asset, 0.0) - current_weights.get(asset, 0.0))
        for asset in all_assets
    )

    active_lookup = active.set_index("asset_id")
    venue_weights: dict[str, float] = {}
    quote_weights: dict[str, float] = {}
    for asset, weight in desired_map.items():
        if asset not in active_lookup.index:
            continue
        row = active_lookup.loc[asset]
        venue_weights[str(row["venue"])] = venue_weights.get(str(row["venue"]), 0.0) + weight
        quote_weights[str(row["quote_asset"])] = quote_weights.get(str(row["quote_asset"]), 0.0) + weight

    final_vector = np.array([desired_map.get(asset, 0.0) for asset in asset_ids])
    final_variance = float(final_vector @ covariance @ final_vector) * selected_policy.annualization_periods
    gross = sum(desired_map.values())
    return PortfolioAllocation(
        weights=desired_map,
        cash_weight=max(0.0, 1.0 - gross),
        expected_annualized_volatility=sqrt(max(0.0, final_variance)),
        one_way_turnover=float(one_way_turnover),
        venue_weights=venue_weights,
        quote_asset_weights=quote_weights,
        capacity_limited_assets=tuple(sorted(capacity_limited)),
        correlation_adjustments=adjustments,
    )


def portfolio_return_series(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    aligned = returns.reindex(columns=list(weights)).fillna(0.0)
    vector = np.array([weights[column] for column in aligned.columns], dtype=float)
    return pd.Series(aligned.to_numpy(dtype=float) @ vector, index=returns.index)
