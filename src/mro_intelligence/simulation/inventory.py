"""Counterfactual simulation — shared inventory engine.

One replenishment engine powers both:

1. the synthetic world's "ERP as-is" baseline (flat reorder rules, no
   seasonality awareness), and
2. the ROI simulator's optimized counterfactual (seasonal forecast-driven
   reorder points, service-level safety stock).

Same engine + different policy parameters = an apples-to-apples comparison.
The engine is a pure function of its inputs (no I/O), so it is directly
unit-testable and safe to reuse in counterfactuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyParams:
    """Replenishment policy parameters (monthly review)."""

    mu: np.ndarray  # policy's belief about mean monthly demand (per month)
    sigma: np.ndarray  # policy's belief about demand std (per month)
    lead_time_months: float
    z: float  # safety factor on demand variability
    coverage_months: float  # order-up-to horizon in months of mu


@dataclass(frozen=True)
class InventoryPath:
    on_hand_end: np.ndarray  # end-of-month on-hand units
    units_ordered: np.ndarray  # ordered during month (arrive per lead time)
    units_received: np.ndarray  # received during month
    units_lost: np.ndarray  # unmet demand (stockout) during month
    stockout_months: int

    @property
    def avg_inventory(self) -> float:
        return float(np.mean(self.on_hand_end))

    def fill_rate_over(self, demand: np.ndarray) -> float:
        """Unit fill rate: share of demand served without a stockout."""
        denom = float(np.sum(demand))
        if denom <= 0:
            return 1.0
        served = denom - float(np.sum(self.units_lost))
        return served / denom


def simulate_sku_inventory(
    demand: np.ndarray,
    opening_stock: float,
    policy: PolicyParams,
    rng: np.random.Generator | None = None,
) -> InventoryPath:
    """Simulate monthly-review (s, S) replenishment for one SKU at one branch.

    Deterministic given the demand series: lead-time variability is folded into
    ``lead_time_months`` (fractional arrival months). ``rng`` is unused today and
    accepted for future stochastic-lead-time variants so callers can pass it
    symmetrically.
    """
    n_months = len(demand)
    horizon = max(1, round(policy.coverage_months))  # order-up-to horizon, in months
    # Pipeline: pipeline[m] = units scheduled to arrive in month m.
    pipeline = np.zeros(n_months + int(np.ceil(policy.lead_time_months)) + 2)

    on_hand = float(opening_stock)
    on_hand_end = np.zeros(n_months)
    units_ordered = np.zeros(n_months)
    units_received = np.zeros(n_months)
    units_lost = np.zeros(n_months)
    stockout_months = 0

    lt = max(1, round(policy.lead_time_months))

    for m in range(n_months):
        # Receive everything scheduled for this month.
        arriving = pipeline[m]
        if arriving > 0:
            on_hand += arriving
            units_received[m] = arriving

        # Serve demand; unmet demand is lost (walk-away), tracked explicitly.
        served = min(on_hand, demand[m])
        lost = demand[m] - served
        on_hand -= served
        units_lost[m] = lost
        if lost > 0:
            stockout_months += 1

        # Monthly review: reorder if position (on hand + pipeline) <= reorder point.
        position = on_hand + pipeline[m + 1 : m + lt + 1].sum()
        mu_m = float(policy.mu[m]) if len(policy.mu) > m else float(policy.mu[-1])
        sigma_m = float(policy.sigma[m]) if len(policy.sigma) > m else float(policy.sigma[-1])
        reorder_point = mu_m * policy.lead_time_months + policy.z * sigma_m * np.sqrt(
            policy.lead_time_months
        )
        order_up_to = mu_m * horizon + policy.z * sigma_m * np.sqrt(horizon)
        if position <= reorder_point:
            qty = max(0.0, order_up_to - position)
            units_ordered[m] = qty
            arrival = m + lt + 1  # lead time later (+1 for review at month start)
            if arrival < len(pipeline):
                pipeline[arrival] += qty

        on_hand_end[m] = on_hand

    return InventoryPath(
        on_hand_end=on_hand_end,
        units_ordered=units_ordered,
        units_received=units_received,
        units_lost=units_lost,
        stockout_months=stockout_months,
    )
