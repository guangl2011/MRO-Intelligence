"""Inventory-vs-fill-rate frontier: sweep the policy's target cover.

Each grid point re-runs the counterfactual walk across the treated SKUs at a
different target cover and records the aggregate tradeoff — average inventory
value against realized fill rate. The SDD reports this frontier instead of a
single promised operating point.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from mro_intelligence.roi import simulate

FRONTIER_GRID_MONTHS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
# Fraction of the best fill rate a point must reach to be the recommended
# operating point (the smallest inventory that stays within the band).
OPERATING_BAND = 0.98


@dataclass(frozen=True)
class FrontierPoint:
    target_cover: float
    avg_inventory_value: float
    fill_rate: float


def frontier(
    demand_by_sku: dict[str, list[float]],
    unit_cost_by_sku: dict[str, float],
    start_on_hand_by_sku: dict[str, float],
    lead_months: int,
    grid: Sequence[float] = FRONTIER_GRID_MONTHS,
    window_start: int = 0,
) -> pd.DataFrame:
    """Simulate each grid point; one row per (target_cover, inventory, fill).

    The walk replays the window months with the same protected sizing rule
    as the counterfactual (shared `sizing_rate_fn`), grounded in pre-window
    history — not a cold start from month zero, which would understate the
    policy's service for reasons the ERP never faced.
    """
    rows = []
    for target_cover in grid:
        inventory_total = 0.0
        fill = 0.0
        skus = sorted(demand_by_sku)
        for sku in skus:
            series = demand_by_sku[sku]
            rate_fn = simulate.sizing_rate_fn(series, window_start, lead_months, target_cover)
            result = simulate.inventory_walk(
                series[window_start:],
                start_on_hand=start_on_hand_by_sku.get(sku, 0.0),
                target_cover=target_cover,
                rate_fn=rate_fn,
                lead_months=lead_months,
            )
            unit_cost = unit_cost_by_sku.get(sku, 0.0)
            inventory_total += result.avg_on_hand * unit_cost
            fill += result.fill_rate
        rows.append(
            FrontierPoint(
                target_cover=float(target_cover),
                avg_inventory_value=inventory_total,
                fill_rate=fill / max(len(skus), 1),
            )
        )
    return pd.DataFrame(
        {
            "target_cover_months": [p.target_cover for p in rows],
            "avg_inventory_value": [round(p.avg_inventory_value, 2) for p in rows],
            "fill_rate": [round(p.fill_rate, 4) for p in rows],
        }
    )


def operating_point(
    frontier_df: pd.DataFrame, min_fill: float | None = None
) -> FrontierPoint | None:
    """Smallest-inventory point meeting the service bar.

    With `min_fill` (the ERP's own recorded fill) the recommendation is a
    Pareto claim — match the service the distributor gets today at the least
    inventory the policy can hold. Without one, fall back to the smallest
    inventory within OPERATING_BAND of the frontier's best fill.
    """
    if frontier_df.empty:
        return None
    best_fill = float(frontier_df["fill_rate"].max())
    bar = best_fill * OPERATING_BAND if min_fill is None else min_fill
    eligible = frontier_df[frontier_df["fill_rate"] >= bar]
    if eligible.empty:
        # No grid point matches today's service — the frontier cannot make
        # the Pareto claim this sweep. Recommend the best the policy can do
        # instead; the frontier table itself shows the service gap.
        eligible = frontier_df[frontier_df["fill_rate"] >= best_fill * OPERATING_BAND]
    # Among in-band points prefer the smallest inventory; ties break to the
    # higher fill rate.
    eligible = eligible.sort_values(["avg_inventory_value", "fill_rate"], ascending=[True, False])
    row = eligible.iloc[0]
    return FrontierPoint(
        target_cover=float(row["target_cover_months"]),
        avg_inventory_value=float(row["avg_inventory_value"]),
        fill_rate=float(row["fill_rate"]),
    )
