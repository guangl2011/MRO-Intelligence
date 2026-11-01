"""ROI pipeline: counterfactual simulation, frontier, and the SDD Step-5 report.

Integration point with M3: the simulator consumes the SKU health cards
(`sku_health_report.csv`) to pick treated SKUs — dead stock, high stockout
risk, over-covered positions — and the canonical catalog for duplicate-labor
counts. Everything else comes from the tenant's own extracts; no ERP writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mro_intelligence.demand import series as srs
from mro_intelligence.roi import frontier as ftr
from mro_intelligence.roi import roi, simulate

OVERCOVER_MONTHS = 6.0  # a position above this many months of cover is treated
BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 7

TREATED = "treated"
CONTROL = "control"


@dataclass(frozen=True)
class RoiOutputs:
    tenant_id: str
    report: pd.DataFrame
    totals: pd.DataFrame
    frontier: pd.DataFrame
    counterfactual: pd.DataFrame


@dataclass(frozen=True)
class SimInputs:
    """Per-SKU series and prices the counterfactual replays against."""

    demand_by_sku: dict[str, list[float]]  # full monthly history, chronological
    window_start: int  # first index of the replay window in the full series
    actual_lost: dict[str, float]  # ERP-recorded lost units in the window
    actual_sold: dict[str, float]  # realized units in the window
    on_hand_by_sku: dict[str, list[float]]  # observed on-hand in the window, monthly
    start_on_hand: dict[str, float]  # on-hand at the window start
    unit_cost: dict[str, float]
    margin: dict[str, float]


def assign_groups(health_report: pd.DataFrame) -> pd.DataFrame:
    """Policy-eligible SKUs are treated; the rest are the control comparator."""
    report = health_report.copy()
    report["months_of_cover"] = pd.to_numeric(report["months_of_cover"], errors="coerce")
    eligible = (
        report["dead_stock"].astype(str).str.lower().isin(("true", "1"))
        | (report["stockout_risk"] == "high")
        | (report["months_of_cover"] > OVERCOVER_MONTHS)
    )
    report["group"] = np.where(eligible, TREATED, CONTROL)
    return report[["sku", "abc_class", "xyz_class", "group", "dead_stock", "stockout_risk"]]


def _margin_by_sku(order_lines: pd.DataFrame, item_master: pd.DataFrame) -> dict[str, float]:
    """Per-SKU average unit margin from realized lines; unknown margin earns
    no claim (0.0) rather than an invented rate."""
    lines = order_lines.copy()
    lines["unit_price"] = pd.to_numeric(lines["unit_price"], errors="coerce")
    lines["unit_cost"] = pd.to_numeric(lines["unit_cost"], errors="coerce")
    lines["margin"] = (lines["unit_price"] - lines["unit_cost"]).clip(lower=0.0)
    margins = lines.groupby("sku")["margin"].mean()
    return {str(sku): float(value) for sku, value in margins.items()}


def prepare_inputs(
    extracts: dict[str, pd.DataFrame], health_report: pd.DataFrame, params: roi.RoiParams
) -> SimInputs:
    """Build the per-SKU series, prices, and stock positions for simulation."""
    demand = srs.monthly_demand(extracts["order_lines"])
    if "item_master" in extracts:
        demand = demand.merge(extracts["item_master"][["sku", "category"]], on="sku", how="left")
    months = sorted(demand["month"].unique())
    if len(months) < params.sim_months + 3:
        raise ValueError(
            f"need at least {params.sim_months + 3} months of history for the "
            f"counterfactual window; got {len(months)}"
        )
    window_start = len(months) - params.sim_months

    demand_by_sku = {
        str(sku): [float(q) for q in grp.sort_values("month")["qty"]]
        for sku, grp in demand.groupby("sku")
    }

    # Lost and sold units inside the replay window, branch-summed per SKU.
    window_months = set(months[window_start:])
    inv = extracts["inventory_monthly"].copy()
    inv["units_lost"] = pd.to_numeric(inv["units_lost"], errors="coerce").fillna(0.0)
    inv["on_hand"] = pd.to_numeric(inv["on_hand"], errors="coerce").fillna(0.0)
    inv_window = inv[inv["month"].isin(window_months)]
    actual_lost = inv_window.groupby("sku")["units_lost"].sum().to_dict()
    sold = demand[demand["month"].isin(window_months)].groupby("sku")["qty"].sum().to_dict()

    on_hand_by_sku: dict[str, list[float]] = {}
    for sku, grp in inv_window.groupby("sku"):
        on_hand_by_sku[str(sku)] = [float(o) for o in grp["on_hand"]]

    start_snap = inv[inv["month"] == months[window_start]]
    start_on_hand = start_snap.groupby("sku")["on_hand"].sum().to_dict()

    item = extracts.get("item_master")
    unit_cost: dict[str, float] = {}
    if item is not None and "unit_cost" in item.columns:
        costs = item.copy()
        costs["unit_cost"] = pd.to_numeric(costs["unit_cost"], errors="coerce")
        unit_cost = dict(zip(costs["sku"].astype(str), costs["unit_cost"].fillna(0.0), strict=True))

    return SimInputs(
        demand_by_sku=demand_by_sku,
        window_start=window_start,
        actual_lost={str(k): float(v) for k, v in actual_lost.items()},
        actual_sold={str(k): float(v) for k, v in sold.items()},
        on_hand_by_sku=on_hand_by_sku,
        start_on_hand={str(k): float(v) for k, v in start_on_hand.items()},
        unit_cost=unit_cost,
        margin=_margin_by_sku(
            extracts["order_lines"], item if item is not None else pd.DataFrame()
        ),
    )


def _replay_one(
    inputs: SimInputs, sku: str, target_cover: float, lead_months: int
) -> tuple[float, float, float]:
    """Per-SKU (inventory value delta, recovered margin, fill-rate gain).

    Positive inventory delta = the policy holds less value than the ERP did.
    """
    full = inputs.demand_by_sku[sku]
    window = full[inputs.window_start :]

    def rate_fn(idx: int) -> float:
        return simulate.trailing_rate(full[: inputs.window_start + idx])

    walk = simulate.inventory_walk(
        window,
        start_on_hand=inputs.start_on_hand.get(sku, 0.0),
        target_cover=target_cover,
        rate_fn=rate_fn,
        lead_months=lead_months,
    )
    cost = inputs.unit_cost.get(sku, 0.0)
    observed = inputs.on_hand_by_sku.get(sku, [])
    actual_avg_value = (float(np.mean(observed)) * cost) if observed else 0.0
    inventory_delta = actual_avg_value - walk.avg_on_hand * cost
    margin = inputs.margin.get(sku, 0.0)
    actual_lost = inputs.actual_lost.get(sku, 0.0)
    policy_lost = sum(walk.lost)
    recovered = max(actual_lost - policy_lost, 0.0) * margin
    demanded = inputs.actual_sold.get(sku, 0.0) + actual_lost
    actual_fill = 1.0 - actual_lost / demanded if demanded > 0 else 1.0
    fill_gain = walk.fill_rate - actual_fill
    return inventory_delta, recovered, fill_gain


def run_counterfactual(
    inputs: SimInputs,
    groups: pd.DataFrame,
    target_cover: float,
    lead_months: int,
    carrying_rate_annual: float,
) -> pd.DataFrame:
    """Per-SKU effects plus a seeded bootstrap interval on the treated total."""
    treated = groups[groups["group"] == TREATED]
    rows = []
    deltas: list[float] = []
    for _, row in treated.iterrows():
        sku = str(row["sku"])
        if sku not in inputs.demand_by_sku:
            continue
        inv_delta, recovered, fill_gain = _replay_one(inputs, sku, target_cover, lead_months)
        annual_benefit = inv_delta * carrying_rate_annual + recovered
        deltas.append(annual_benefit)
        rows.append(
            {
                "sku": sku,
                "group": TREATED,
                "xyz_class": row["xyz_class"],
                "inventory_value_delta": round(inv_delta, 2),
                "recovered_margin": round(recovered, 2),
                "fill_rate_gain": round(fill_gain, 4),
                "annual_benefit": round(annual_benefit, 2),
            }
        )
    for _, row in groups[groups["group"] == CONTROL].iterrows():
        sku = str(row["sku"])
        if sku not in inputs.demand_by_sku:
            continue
        # Controls are observed only: no policy is claimed for them.
        rows.append(
            {
                "sku": sku,
                "group": CONTROL,
                "xyz_class": row["xyz_class"],
                "inventory_value_delta": 0.0,
                "recovered_margin": 0.0,
                "fill_rate_gain": 0.0,
                "annual_benefit": 0.0,
            }
        )

    frame = pd.DataFrame(rows)
    if deltas:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = rng.choice(np.array(deltas), size=(BOOTSTRAP_DRAWS, len(deltas)), replace=True)
        totals = draws.sum(axis=1)
        lo, mid, hi = np.percentile(totals, [5, 50, 95])
        summary = pd.DataFrame(
            [
                {
                    "segment": "treated-total",
                    "point_estimate": round(float(np.sum(deltas)), 2),
                    "p05": round(float(lo), 2),
                    "p50": round(float(mid), 2),
                    "p95": round(float(hi), 2),
                }
            ]
        )
        return pd.concat([frame, summary], ignore_index=True)
    return frame


def lead_time_months(purchase_orders: pd.DataFrame | None) -> int:
    """Median PO lead time in whole months; 1 when no PO data exists."""
    if purchase_orders is None or purchase_orders.empty:
        return 1
    po = purchase_orders.copy()
    po["order_date"] = pd.to_datetime(po["order_date"], errors="coerce")
    po["receipt_date"] = pd.to_datetime(po["receipt_date"], errors="coerce")
    lag_days = (po["receipt_date"] - po["order_date"]).dt.days.dropna()
    lag_days = lag_days[lag_days >= 0]
    if lag_days.empty:
        return 1
    return max(int(np.ceil(float(lag_days.median()) / 30.0)), 1)


def _duplicate_count(canonical_catalog: pd.DataFrame | None) -> int:
    """Duplicate-SKU merges implied by the canonical catalog's part mapping."""
    if canonical_catalog is None or "proposed_global_part_id" not in canonical_catalog.columns:
        return 0
    parts = canonical_catalog["proposed_global_part_id"].dropna()
    return max(len(parts) - parts.nunique(), 0)


def run_roi(
    extracts: dict[str, pd.DataFrame],
    health_report: pd.DataFrame,
    tenant_id: str,
    *,
    canonical_catalog: pd.DataFrame | None = None,
    params: roi.RoiParams | None = None,
) -> RoiOutputs:
    """Counterfactual simulation, frontier sweep, and the tiered SDD report."""
    params = params or roi.RoiParams()
    groups = assign_groups(health_report)
    inputs = prepare_inputs(extracts, health_report, params)
    lead = lead_time_months(extracts.get("purchase_orders"))

    front = ftr.frontier(inputs.demand_by_sku, inputs.unit_cost, inputs.start_on_hand, lead)
    point = ftr.operating_point(front)
    target_cover = point.target_cover if point is not None else 2.0
    counterfactual = run_counterfactual(
        inputs, groups, target_cover, lead, params.carrying_rate_annual
    )

    treated_rows = counterfactual[
        (counterfactual["group"] == TREATED) & (counterfactual["sku"].notna())
    ]
    inventory_reduction = float(treated_rows["inventory_value_delta"].sum())
    recovered_margin = float(treated_rows["recovered_margin"].sum())
    long_tail = treated_rows[treated_rows["xyz_class"] == "Z"]
    z_fill_gain = float(long_tail["fill_rate_gain"].mean()) if not long_tail.empty else 0.0

    # Annual Z-quadrant margin: window sold + lost units valued at margin.
    z_skus = set(long_tail["sku"].astype(str))
    z_annual_margin = sum(
        inputs.margin.get(sku, 0.0)
        * (inputs.actual_sold.get(sku, 0.0) + inputs.actual_lost.get(sku, 0.0))
        for sku in z_skus
    )

    report = roi.RoiReport()
    dead_value = roi.dead_stock_cash_release(health_report, inputs.unit_cost)
    report.add(
        roi.BenefitRow(
            category="dead_stock_cash_release",
            basis=(
                "dead SKUs' on-hand units at item-master unit cost "
                f"({int(health_report['dead_stock'].sum())} SKUs); one-time"
            ),
            annual_value=0.0,
            one_time_value=dead_value,
        )
    )
    report.add(
        roi.BenefitRow(
            category="inventory_carrying",
            basis=(
                f"net of counterfactual inventory change (${inventory_reduction:,.0f} "
                f"avg value; includes stock BUILT for stockout-risk SKUs) x "
                f"{params.carrying_rate_annual:.0%} annual carrying rate"
            ),
            annual_value=roi.carrying_savings(inventory_reduction, params),
        )
    )
    report.add(
        roi.BenefitRow(
            category="fill_rate_revenue",
            basis=(
                "margin on demand the policy serves but the ERP lost "
                f"(${recovered_margin:,.0f} in the {params.sim_months}-month window)"
            ),
            annual_value=roi.fill_rate_revenue(recovered_margin),
        )
    )
    report.add(
        roi.BenefitRow(
            category="long_tail_margin",
            basis=(
                f"Z-quadrant annual margin (${z_annual_margin:,.0f}) x simulated "
                f"fill-rate gain ({z_fill_gain:.3f}) on treated Z SKUs"
            ),
            annual_value=roi.long_tail_margin(z_annual_margin, z_fill_gain),
        )
    )

    returns = extracts.get("returns")
    wrong_part = int((returns["reason"] == "WRONG_PART").sum()) if returns is not None else 0
    report.add(
        roi.BenefitRow(
            category="returns_savings",
            basis=(
                f"{wrong_part} wrong-part returns x "
                f"${params.returns_handling_cost:,.0f} handling cost"
            ),
            annual_value=roi.returns_savings(wrong_part, params),
        )
    )
    dupes = _duplicate_count(canonical_catalog)
    report.add(
        roi.BenefitRow(
            category="labor_automation",
            basis=(
                f"{dupes} duplicate merges x {params.dedup_hours_per_merge}h + "
                f"{len(health_report)} SKU enrichments x "
                f"{params.enrichment_minutes_per_sku}min at "
                f"${params.labor_rate_hour:,.0f}/h"
            ),
            annual_value=roi.labor_automation(dupes, len(health_report), params),
        )
    )

    po = extracts.get("purchase_orders")
    annual_po_spend = 0.0
    rush_count = 0
    if po is not None and not po.empty:
        costs = pd.to_numeric(po["unit_cost"], errors="coerce")
        qtys = pd.to_numeric(po["qty_ordered"], errors="coerce").fillna(0.0)
        annual_po_spend = float((costs.fillna(0.0) * qtys).sum()) * (12.0 / params.sim_months)
        po_dated = po.copy()
        po_dated["order_date"] = pd.to_datetime(po_dated["order_date"], errors="coerce")
        po_dated["receipt_date"] = pd.to_datetime(po_dated["receipt_date"], errors="coerce")
        lag = (po_dated["receipt_date"] - po_dated["order_date"]).dt.days
        rush_count = int((lag <= params.rush_lead_days).sum())
    report.add(
        roi.BenefitRow(
            category="expedited_freight",
            basis=(
                f"{rush_count} rush POs (lead <= {params.rush_lead_days} days) x "
                f"${params.expedite_premium_per_po:,.0f} premium"
            ),
            annual_value=roi.expedited_freight(rush_count, params),
        )
    )
    report.add(
        roi.BenefitRow(
            category="purchasing_cost",
            basis=(
                f"annual PO spend ${annual_po_spend:,.0f} x "
                f"{params.purchasing_savings_rate:.1%} cross-reference savings"
            ),
            annual_value=roi.purchasing_cost(annual_po_spend, params),
        )
    )

    return RoiOutputs(
        tenant_id=tenant_id,
        report=report.frame(),
        totals=report.totals(),
        frontier=front,
        counterfactual=counterfactual,
    )


def write_roi(outputs: RoiOutputs, out_dir: Path | str) -> dict[str, object]:
    """Write the tiered report, frontier, and counterfactual detail."""
    tenant_dir = Path(out_dir) / outputs.tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    report_path = tenant_dir / "roi_baseline_report.csv"
    outputs.report.to_csv(report_path, index=False)
    frontier_path = tenant_dir / "inventory_frontier.csv"
    outputs.frontier.to_csv(frontier_path, index=False)
    counterfactual_path = tenant_dir / "counterfactual_simulation.csv"
    outputs.counterfactual.to_csv(counterfactual_path, index=False)
    totals = outputs.totals.set_index("tier")
    return {
        "report": report_path,
        "frontier": frontier_path,
        "counterfactual": counterfactual_path,
        "baseline_annual": float(totals.loc["baseline", "annual_value_total"]),
        "baseline_one_time": float(totals.loc["baseline", "one_time_value_total"]),
    }
