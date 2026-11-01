"""ABC-XYZ segmentation (SDD Step 3).

ABC by revenue contribution (cumulative revenue shares), XYZ by demand
variability/sparsity — the operational proxy the SDD uses for stability (X),
moderate/seasonal (Y), and intermittent/sparse (Z) demand.
"""

from __future__ import annotations

import pandas as pd

A_SHARE = 0.65  # SKUs covering the first 65% of revenue are A (AXON §3.2)
B_SHARE = 0.85  # next 20% are B; the remainder are C
ZERO_SHARE_Z = 0.5  # >=50% zero-demand months makes a SKU intermittent
CV_Y = 0.5  # coefficient of variation above this is not "stable"


def abc_classify(
    revenue_by_sku: pd.Series, a_share: float = A_SHARE, b_share: float = B_SHARE
) -> pd.Series:
    """A/B/C by cumulative revenue share (descending revenue)."""
    revenue = revenue_by_sku.sort_values(ascending=False)
    total = float(revenue.sum())
    if total <= 0:
        return pd.Series("C", index=revenue_by_sku.index, name="abc_class")
    cum_share = revenue.cumsum() / total
    classes = pd.Series("C", index=revenue.index, name="abc_class")
    classes[cum_share <= b_share] = "B"  # B first, then A re-tightens the top
    classes[cum_share <= a_share] = "A"
    return classes.reindex(revenue_by_sku.index)


def xyz_classify(
    monthly_units: pd.DataFrame, zero_share_z: float = ZERO_SHARE_Z, cv_y: float = CV_Y
) -> pd.Series:
    """X/Y/Z per SKU from a sku x month unit-demand frame.

    Z: intermittent/sparse (zero-month share >= threshold or CV above Y band).
    Y: moderate variability. X: stable (low CV, low sparsity).
    """
    units = monthly_units.astype(float)
    zero_share = (units == 0).mean(axis=1)
    mean = units.mean(axis=1).replace(0.0, float("nan"))
    cv = units.std(axis=1, ddof=0) / mean

    classes = pd.Series("Y", index=units.index, name="xyz_class")
    classes[(zero_share >= zero_share_z) | (cv > 1.0)] = "Z"
    classes[(zero_share < zero_share_z) & (cv <= cv_y)] = "X"
    return classes


def segment(items: pd.DataFrame, order_lines: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU segmentation with revenue and demand-mix facts attached."""
    revenue_by_sku = (
        order_lines.assign(
            revenue=lambda df: df["qty"].astype(float) * df["unit_price"].astype(float)
        )
        .groupby("sku")["revenue"]
        .sum()
        # SKUs with no order lines still classify (zero revenue -> C class).
        .reindex(items["sku"].to_numpy(), fill_value=0.0)
    )
    months = sorted(order_lines["order_date"].astype(str).str.slice(0, 7).unique())
    monthly = (
        order_lines.assign(month=order_lines["order_date"].astype(str).str.slice(0, 7))
        .groupby(["sku", "month"])["qty"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=months, fill_value=0.0)
    )

    out = items[["sku"]].copy()
    out["annual_revenue"] = revenue_by_sku.to_numpy()
    out["abc_class"] = abc_classify(revenue_by_sku).to_numpy()
    monthly = monthly.reindex(items["sku"].to_numpy(), fill_value=0.0)
    out["xyz_class"] = xyz_classify(monthly).to_numpy()
    return out
