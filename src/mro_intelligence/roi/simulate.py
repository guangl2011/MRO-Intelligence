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
PROTECTION_MONTHS = 12  # recent window for the SKU's demand quantiles
COVER_MIN, COVER_MAX = 0.5, 4.0  # frontier sweep bounds the level interpolates over


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


def _quantile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated quantile (numpy's default method)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def protected_level(
    target_cover: float,
    trailing: float,
    lead_months: int,
    recent: Sequence[float],
) -> float:
    """Order-up-to level: mean-cover target, floored by a demand quantile.

    A trailing mean starves lumpy SKUs — the exact ERP failure the product
    exists to fix — so the level never drops below a service quantile of
    lead+review demand, interpolated between the recent 50th and 95th
    percentile monthly demand as the cover knob widens. `recent` must be
    pre-window history only (no leakage into the replayed months).
    """
    base = target_cover * trailing
    if not recent:
        return base
    service_month = _quantile(recent[-PROTECTION_MONTHS:], 0.50) + (
        _quantile(recent[-PROTECTION_MONTHS:], 0.95) - _quantile(recent[-PROTECTION_MONTHS:], 0.50)
    ) * min(max((target_cover - COVER_MIN) / (COVER_MAX - COVER_MIN), 0.0), 1.0)
    return max(base, (max(lead_months, 1) + 1) * service_month)


def sizing_rate_fn(
    full: Sequence[float], window_start: int, lead_months: int, target_cover: float
) -> Callable[[int], float]:
    """The replay's sizing rule as a walk rate function.

    One sizing policy for both simulation paths: the counterfactual replay
    and the frontier sweep share this, so the report's per-SKU rows and the
    swept frontier can never disagree about what the policy does. The walk
    multiplies the returned rate by target_cover, so the protected level is
    divided back out; history ends at the window boundary (no leakage).
    """
    pre_window = full[:window_start]

    def rate(idx: int) -> float:
        level = protected_level(
            target_cover,
            trailing_rate(full[: window_start + idx]),
            lead_months,
            pre_window,
        )
        return level / target_cover

    return rate
