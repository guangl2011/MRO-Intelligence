"""Monthly demand series construction and SKU health-card measures.

The SDD spec (art_tvoLR1rG §4 step 2) defines the health card as the atomic
unit every later analysis consumes: recent demand, cover, stockouts, dead
stock. These builders produce exactly that, per SKU, from canonical extracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Health-card windows, mirroring the SDD SKU card definition.
DEAD_STOCK_MONTHS = 12
EVAL_HORIZON_MONTHS = 3


def monthly_demand(order_lines: pd.DataFrame) -> pd.DataFrame:
    """Order lines -> (sku, month) demand quantities over a complete month range.

    Months with no sales appear as explicit zeros: intermittent-demand models
    need the zeros, and the ERP baseline's flat rate only makes sense against
    the same calendar.
    """
    lines = order_lines.copy()
    lines["month"] = lines["order_date"].astype(str).str.slice(0, 7)
    # The loader hands back strings; demand math needs numbers.
    lines["qty"] = pd.to_numeric(lines["qty"], errors="raise")
    totals = lines.groupby(["sku", "month"], as_index=False)["qty"].sum()
    all_months = pd.period_range(totals["month"].min(), totals["month"].max(), freq="M")
    month_index = pd.DataFrame({"month": [str(m) for m in all_months]})
    skus = pd.DataFrame({"sku": sorted(totals["sku"].unique())})
    grid = skus.merge(month_index, how="cross")
    return grid.merge(totals, on=["sku", "month"], how="left").fillna({"qty": 0.0})


def horizon_split(
    demand: pd.DataFrame, horizon_months: int = EVAL_HORIZON_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a (sku, month, qty) frame into train and held-out months.

    The split is chronological (latest months held out) and identical for
    every SKU: no SKU trains on the months its peers are evaluated on.
    """
    months = sorted(demand["month"].unique())
    if len(months) <= horizon_months:
        raise ValueError(
            f"need more than {horizon_months} months of demand history, got {len(months)}"
        )
    eval_months = set(months[-horizon_months:])
    train = demand[~demand["month"].isin(eval_months)]
    eval_df = demand[demand["month"].isin(eval_months)]
    return train.reset_index(drop=True), eval_df.reset_index(drop=True)


def series_for(train: pd.DataFrame, sku: str) -> np.ndarray:
    """Train-month demand vector for one SKU, ordered by month."""
    one = train[train["sku"] == sku].sort_values("month")
    return one["qty"].to_numpy(dtype=float)


def series_by_sku(train: pd.DataFrame) -> dict[str, np.ndarray]:
    """Train-month demand vector for every SKU, in one pass.

    Equivalent to calling series_for per SKU, but scans the frame once
    instead of once per SKU — the per-SKU scan is what made large tenants
    quadratic. `train` holds one row per (sku, month), so sorting by month
    then grouping yields the same month-ordered vectors series_for returns.
    """
    ordered = train.sort_values("month")
    return {
        str(sku): group["qty"].to_numpy(dtype=float)
        for sku, group in ordered.groupby("sku", sort=True)
    }


def latest_on_hand(inventory_monthly: pd.DataFrame) -> pd.DataFrame:
    """Network-wide on-hand per SKU from the latest snapshot month."""
    snap = inventory_monthly.copy()
    snap["month"] = snap["month"].astype(str).str.slice(0, 7)
    snap["on_hand"] = pd.to_numeric(snap["on_hand"], errors="raise")
    latest = snap.sort_values("month").groupby("sku", as_index=False).tail(1)
    return latest.groupby("sku", as_index=False)["on_hand"].sum()


def dead_stock_flag(last12: float, on_hand: float) -> bool:
    """SDD dead stock: zero sales in the trailing 12 months while still stocked."""
    return bool(last12 <= 0.0 and on_hand > 0.0)


def cover_months(on_hand: float, monthly_forecast: float) -> float:
    """Months of cover at the forecast rate; inf when forecast is zero."""
    if monthly_forecast <= 0:
        return float("inf")
    return on_hand / monthly_forecast


def stockout_risk(on_hand: float, q25: float, q75: float, q95: float) -> str:
    """Risk tier from where on-hand sits in the forecast demand distribution.

    On-hand at or below the 25th-percentile monthly demand empties fastest
    (high); the 75th and 95th percentiles bound the medium and low tiers.
    """
    if on_hand <= q25:
        return "high"
    if on_hand <= q75:
        return "medium"
    if on_hand <= q95:
        return "low"
    return "none"
