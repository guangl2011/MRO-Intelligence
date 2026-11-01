"""Counterfactual inventory simulation: the SDD's "what if 12 months ago" walk.

For each treated SKU we replay monthly demand against a policy-driven
inventory trajectory (order-up-to a target cover with a replenishment lead
time) and compare it with the as-is path. The as-is path uses observed
on-hand and observed lost sales from the ERP snapshots; nothing about the
policy is claimed for the control group.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

MIN_REORDER = 0.5  # reorder only when the gap exceeds half a month of demand


@dataclass(frozen=True)
class WalkResult:
    """One SKU-month trajectory under a replenishment policy."""

    sales: list[float]
    lost: list[float]
    ending_on_hand: list[float]

    @property
    def fill_rate(self) -> float:
        demanded = sum(self.sales) + sum(self.lost)
        if demanded <= 0:
            return 1.0
        return sum(self.sales) / demanded

    @property
    def avg_on_hand(self) -> float:
        if not self.ending_on_hand:
            return 0.0
        return sum(self.ending_on_hand) / len(self.ending_on_hand)


def inventory_walk(
    demand: Sequence[float],
    start_on_hand: float,
    target_cover: float,
    rate_fn: Callable[[int], float],
    lead_months: int,
) -> WalkResult:
    """Replay `demand` under an order-up-to policy.

    `rate_fn(month_index)` returns the demand rate the policy may plan with at
    that month (past data only — no peeking at the month being served).
    `lead_months` is the replenishment lead time; orders arrive that many
    months later.
    """
    pipeline: dict[int, float] = {}
    on_hand = start_on_hand
    sales: list[float] = []
    lost: list[float] = []
    ending: list[float] = []

    for month_index, demand_m in enumerate(demand):
        arrival = pipeline.pop(month_index, 0.0)
        available = on_hand + arrival
        sold = min(demand_m, available)
        sales.append(sold)
        lost.append(max(demand_m - sold, 0.0))
        on_hand = available - sold

        # Plan from past months only: the rate uses months before this one.
        planned_rate = rate_fn(month_index)
        target = target_cover * planned_rate
        gap = target - on_hand
        if gap > MIN_REORDER:
            pipeline[month_index + max(lead_months, 1)] = gap
        ending.append(on_hand)

    return WalkResult(sales=sales, lost=lost, ending_on_hand=ending)


def trailing_rate(past: Sequence[float]) -> float:
    """Mean monthly demand over `past`; the policy's planning rate."""
    if not past:
        return 0.0
    return sum(past) / len(past)
