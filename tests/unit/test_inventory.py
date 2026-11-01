"""Inventory simulation engine tests — the shared baseline/counterfactual engine."""

from __future__ import annotations

import numpy as np
import pytest

from mro_intelligence.simulation import PolicyParams, simulate_sku_inventory


def _flat_policy(
    mu: float, sigma: float = 1.0, lead: float = 1.0, z: float = 1.0, coverage: float = 1.5
):
    return PolicyParams(
        mu=np.full(12, mu),
        sigma=np.full(12, sigma),
        lead_time_months=lead,
        z=z,
        coverage_months=coverage,
    )


def test_generous_stock_covers_the_horizon() -> None:
    """Opening stock above horizon demand means zero lost sales."""
    demand = np.full(12, 2.0)
    path = simulate_sku_inventory(demand, opening_stock=30, policy=_flat_policy(mu=2, sigma=1.0))
    assert path.units_lost.sum() == 0
    assert path.stockout_months == 0
    assert path.fill_rate_over(demand) == pytest.approx(1.0)


def test_zero_opening_stock_and_slow_lead_stockouts_early() -> None:
    demand = np.full(12, 10.0)
    path = simulate_sku_inventory(demand, opening_stock=0, policy=_flat_policy(mu=10, lead=2.0))
    assert path.stockout_months >= 2  # first arrivals cannot land before lead time
    assert path.units_lost[:2].sum() > 0
    assert path.fill_rate_over(demand) < 1.0


def test_higher_safety_factor_reduces_lost_sales() -> None:
    rng = np.random.default_rng(11)
    demand = rng.poisson(8.0, size=24).astype(float)
    low = simulate_sku_inventory(
        demand, opening_stock=20, policy=_flat_policy(mu=8, sigma=3.0, z=0.0, coverage=1.5)
    )
    high = simulate_sku_inventory(
        demand, opening_stock=20, policy=_flat_policy(mu=8, sigma=3.0, z=2.5, coverage=1.5)
    )
    assert high.units_lost.sum() <= low.units_lost.sum()


def test_fill_rate_with_zero_demand_is_one() -> None:
    path = simulate_sku_inventory(
        np.zeros(6), opening_stock=5, policy=_flat_policy(mu=0.0, sigma=0.0)
    )
    assert path.fill_rate_over(np.zeros(6)) == 1.0


def test_path_averages_and_arrays_shapes() -> None:
    demand = np.full(12, 4.0)
    path = simulate_sku_inventory(demand, opening_stock=10, policy=_flat_policy(mu=4, sigma=1.0))
    assert path.on_hand_end.shape == (12,)
    assert path.units_ordered.shape == (12,)
    assert path.units_received.shape == (12,)
    assert path.units_lost.shape == (12,)
    assert path.avg_inventory >= 0


def test_engine_is_deterministic() -> None:
    demand = np.array([5.0, 0.0, 12.0, 3.0, 0.0, 0.0, 9.0, 1.0, 4.0, 2.0, 0.0, 7.0])
    a = simulate_sku_inventory(demand, opening_stock=8, policy=_flat_policy(mu=4, sigma=2.0, z=1.0))
    b = simulate_sku_inventory(demand, opening_stock=8, policy=_flat_policy(mu=4, sigma=2.0, z=1.0))
    assert np.array_equal(a.on_hand_end, b.on_hand_end)
    assert np.array_equal(a.units_lost, b.units_lost)
