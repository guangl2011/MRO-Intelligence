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
) -> pd.DataFrame:
    """Simulate each grid point; one row per (target_cover, inventory, fill)."""
    rows = []
    for target_cover in grid:
        inventory_total = 0.0
        fill = 0.0
        skus = sorted(demand_by_sku)
        for sku in skus:
            series = demand_by_sku[sku]

            def rate_fn(idx: int, series: list[float] = series) -> float:
                return simulate.trailing_rate(series[:idx])

            result = simulate.inventory_walk(
                series,
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


def operating_point(frontier_df: pd.DataFrame) -> FrontierPoint | None:
    """Smallest-inventory point whose fill rate stays in the operating band."""
    if frontier_df.empty:
        return None
    best_fill = float(frontier_df["fill_rate"].max())
    band = best_fill * OPERATING_BAND
    eligible = frontier_df[frontier_df["fill_rate"] >= band]
    # Among in-band points prefer the smallest inventory; ties break to the
    # higher fill rate.
    eligible = eligible.sort_values(["avg_inventory_value", "fill_rate"], ascending=[True, False])
    row = eligible.iloc[0]
    return FrontierPoint(
        target_cover=float(row["target_cover_months"]),
        avg_inventory_value=float(row["avg_inventory_value"]),
        fill_rate=float(row["fill_rate"]),
    )
