"""Partner ROI baseline: the SDD Step-5 benefit model.

Seven annual benefit categories (AXON-plan structure: carrying cost, fill-rate
revenue, long-tail margin, returns, labor automation, expedited freight,
purchasing cost) plus the one-time dead-stock cash release, recomputed from
the partner's own extracts. Every economic parameter lives in `RoiParams`
with an explicit default and a basis string that lands in the report row —
the report states its assumptions, it does not assert vendor numbers. Tiers
scale each effect's realization: conservative 60%, baseline 80%,
aggressive 100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TIER_FACTORS: dict[str, float] = {
    "conservative": 0.60,
    "baseline": 0.80,
    "aggressive": 1.00,
}


@dataclass(frozen=True)
class RoiParams:
    """Economic assumptions; every default is a parameter, not a fact."""

    carrying_rate_annual: float = 0.20  # holding cost per inventory dollar per year
    labor_rate_hour: float = 45.0
    dedup_hours_per_merge: float = 1.5  # analyst hours per duplicate merge avoided
    enrichment_minutes_per_sku: float = 2.0  # attribute completion per SKU
    returns_handling_cost: float = 35.0  # per wrong-part return processed
    expedite_premium_per_po: float = 250.0  # average premium on a rush PO
    rush_lead_days: int = 7  # POs with lead time at or below this are "rush"
    purchasing_savings_rate: float = 0.015  # recoverable share of annual PO spend
    sim_months: int = 12  # replay window for annualized effects


@dataclass(frozen=True)
class BenefitRow:
    category: str
    basis: str
    annual_value: float
    one_time_value: float = 0.0


def dead_stock_cash_release(
    health_report: pd.DataFrame, unit_cost_by_sku: dict[str, float]
) -> float:
    """One-time cash release: dead SKUs still holding stock."""
    dead = health_report[health_report["dead_stock"]]
    return float(
        sum(
            unit_cost_by_sku.get(str(sku), 0.0) * float(on_hand)
            for sku, on_hand in zip(dead["sku"], dead["on_hand"], strict=True)
        )
    )


def carrying_savings(inventory_reduction_value: float, params: RoiParams) -> float:
    """Annual carrying cost avoided on simulated inventory reduction."""
    return inventory_reduction_value * params.carrying_rate_annual


def fill_rate_revenue(recovered_margin: float) -> float:
    """Margin recovered when the policy walk serves demand the ERP lost."""
    return max(recovered_margin, 0.0)


def labor_automation(n_duplicates_merged: int, n_skus_enriched: int, params: RoiParams) -> float:
    """Analyst hours avoided by auto-merge and attribute completion."""
    hours = n_duplicates_merged * params.dedup_hours_per_merge
    hours += n_skus_enriched * params.enrichment_minutes_per_sku / 60.0
    return hours * params.labor_rate_hour


def returns_savings(wrong_part_return_count: int, params: RoiParams) -> float:
    """Handling cost avoided when fitment prevents wrong-part orders."""
    return wrong_part_return_count * params.returns_handling_cost


def expedited_freight(rush_po_count: int, params: RoiParams) -> float:
    """Premium avoided when replenishment planning replaces rush POs."""
    return rush_po_count * params.expedite_premium_per_po


def purchasing_cost(annual_po_spend: float, params: RoiParams) -> float:
    """Savings from supplier cross-reference and price-file parsing."""
    return annual_po_spend * params.purchasing_savings_rate


def long_tail_margin(z_margin: float, z_fill_gain: float) -> float:
    """Long-tail (Z-quadrant) margin made reliably servable by the policy."""
    return max(z_margin * z_fill_gain, 0.0)


@dataclass
class RoiReport:
    """Assembles benefit rows and renders the tiered report frame."""

    rows: list[BenefitRow] = field(default_factory=list)

    def add(self, row: BenefitRow) -> None:
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        records = [
            {
                "category": row.category,
                "basis": row.basis,
                "tier": tier,
                "annual_value": round(row.annual_value * factor, 2),
                "one_time_value": round(row.one_time_value * factor, 2),
            }
            for row in self.rows
            for tier, factor in TIER_FACTORS.items()
        ]
        return pd.DataFrame(records)

    def totals(self) -> pd.DataFrame:
        """Per-tier annual and one-time totals."""
        records = []
        for tier, factor in TIER_FACTORS.items():
            records.append(
                {
                    "tier": tier,
                    "annual_value_total": round(sum(r.annual_value for r in self.rows) * factor, 2),
                    "one_time_value_total": round(
                        sum(r.one_time_value for r in self.rows) * factor, 2
                    ),
                }
            )
        return pd.DataFrame(records)
