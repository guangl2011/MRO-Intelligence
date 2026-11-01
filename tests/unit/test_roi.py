"""Unit and integration tests for the partner ROI simulator."""

from __future__ import annotations

import pandas as pd
import pytest

from mro_intelligence.roi import frontier as ftr
from mro_intelligence.roi import pipeline as rpipe
from mro_intelligence.roi import roi, simulate


def test_inventory_walk_replays_order_up_to() -> None:
    # Constant demand 10/month, start 25 units, target 2 months of cover,
    # one-month lead: the first plan sees no history (rate 0), then the
    # order-up-to arithmetic is exact.
    result = simulate.inventory_walk(
        [10.0] * 6,
        start_on_hand=25.0,
        target_cover=2.0,
        rate_fn=lambda idx: simulate.trailing_rate([10.0] * idx),
        lead_months=1,
    )
    assert result.sales == [10.0] * 6
    assert result.lost == [0.0] * 6
    assert result.ending_on_hand == [15.0, 5.0, 10.0, 10.0, 10.0, 10.0]
    assert result.fill_rate == pytest.approx(1.0)
    assert result.avg_on_hand == pytest.approx(10.0)


def test_inventory_walk_records_lost_sales_when_starved() -> None:
    # No replenishment at all (rate 0): initial 3 units then permanent stockout.
    result = simulate.inventory_walk(
        [10.0] * 4,
        start_on_hand=3.0,
        target_cover=0.0,
        rate_fn=lambda _idx: 0.0,
        lead_months=1,
    )
    assert result.sales == [3.0, 0.0, 0.0, 0.0]
    assert result.lost == [7.0, 10.0, 10.0, 10.0]
    assert result.fill_rate == pytest.approx(3.0 / 40.0)  # 3 sold of 40 demanded


def test_assign_groups_treats_dead_high_risk_and_overcover() -> None:
    health = pd.DataFrame(
        {
            "sku": ["DEAD", "RISKY", "HOARD", "STEADY"],
            "abc_class": ["C", "A", "B", "A"],
            "xyz_class": ["Z", "X", "Y", "X"],
            "dead_stock": [True, False, False, False],
            "stockout_risk": ["low", "high", "low", "low"],
            "months_of_cover": [99.0, 1.0, 7.5, 3.0],
        }
    )
    groups = rpipe.assign_groups(health)
    assigned = dict(zip(groups["sku"], groups["group"], strict=True))
    assert assigned["DEAD"] == "treated"
    assert assigned["RISKY"] == "treated"
    assert assigned["HOARD"] == "treated"  # 7.5 months of cover > 6
    assert assigned["STEADY"] == "control"


def test_lead_time_months_from_po_lags() -> None:
    po = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "receipt_date": ["2024-01-20", "2024-03-12", "2024-05-30"],
        }
    )
    # Lags: 19, 40, 89 days -> median 40 -> ceil(40/30) = 2 months.
    assert rpipe.lead_time_months(po) == 2
    assert rpipe.lead_time_months(None) == 1
    assert rpipe.lead_time_months(pd.DataFrame({"order_date": [], "receipt_date": []})) == 1


def test_dead_stock_release_and_tiered_report() -> None:
    health = pd.DataFrame(
        {
            "sku": ["A", "B", "C"],
            "dead_stock": [True, True, False],
            "on_hand": [2.0, 3.0, 9.0],
        }
    )
    costs = {"A": 10.0, "B": 5.0}
    assert roi.dead_stock_cash_release(health, costs) == pytest.approx(35.0)

    report = roi.RoiReport()
    report.add(
        roi.BenefitRow(
            category="inventory_carrying",
            basis="test",
            annual_value=100.0,
            one_time_value=35.0,
        )
    )
    frame = report.frame()
    assert set(frame["tier"]) == {"conservative", "baseline", "aggressive"}
    baseline = frame[frame["tier"] == "baseline"].iloc[0]
    assert baseline["annual_value"] == pytest.approx(80.0)
    conservative = frame[frame["tier"] == "conservative"].iloc[0]
    assert conservative["annual_value"] == pytest.approx(60.0)
    totals = report.totals()
    assert totals.set_index("tier").loc["aggressive", "one_time_value_total"] == pytest.approx(35.0)


def test_frontier_inventory_monotone_in_target_cover() -> None:
    demand_by_sku = {
        "S1": [10.0] * 14,
        "S2": [0.0, 4.0, 0.0, 8.0] * 3 + [0.0, 4.0],
    }
    unit_cost = {"S1": 5.0, "S2": 20.0}
    start = {"S1": 10.0, "S2": 0.0}
    front = ftr.frontier(demand_by_sku, unit_cost, start, lead_months=1)
    assert len(front) == len(ftr.FRONTIER_GRID_MONTHS)
    values = front["avg_inventory_value"].tolist()
    assert values == sorted(values)  # more target cover never holds less stock
    fills = front["fill_rate"].tolist()
    assert fills == sorted(fills)  # ...and never serves less demand

    point = ftr.operating_point(front)
    assert point is not None
    assert point.fill_rate >= ftr.OPERATING_BAND * front["fill_rate"].max()


def _synthetic_extracts() -> dict[str, pd.DataFrame]:
    """Three SKUs over 18 months: one dead, one starving high-risk, one steady."""
    months = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 7)]
    lines = []
    for i, month in enumerate(months):
        lines.append(
            {
                "order_date": f"{month}-05",
                "sku": "STEADY",
                "qty": 10.0,
                "unit_price": 100.0,
                "unit_cost": 60.0,
            }
        )
        if i < 6:  # RISKY sells early in history, then stock vanishes
            lines.append(
                {
                    "order_date": f"{month}-05",
                    "sku": "RISKY",
                    "qty": 5.0,
                    "unit_price": 200.0,
                    "unit_cost": 120.0,
                }
            )
    order_lines = pd.DataFrame(lines)

    inv_rows = []
    for month in months:
        inv_rows.append({"month": month, "sku": "STEADY", "on_hand": 30.0, "units_lost": 0.0})
        inv_rows.append(
            {
                "month": month,
                "sku": "RISKY",
                "on_hand": 0.0 if month >= "2024-07" else 2.0,
                "units_lost": 5.0 if month >= "2024-07" else 0.0,
            }
        )
        inv_rows.append({"month": month, "sku": "DEAD", "on_hand": 4.0, "units_lost": 0.0})
    inventory = pd.DataFrame(inv_rows)

    item = pd.DataFrame(
        {
            "sku": ["STEADY", "RISKY", "DEAD"],
            "unit_cost": [60.0, 120.0, 50.0],
            "category": ["HVAC_COOLING"] * 3,
        }
    )
    po = pd.DataFrame(
        {
            "order_date": ["2024-01-01", "2024-02-01"],
            "receipt_date": ["2024-02-01", "2024-03-11"],
            "unit_cost": [60.0, 120.0],
            "qty_ordered": [10.0, 5.0],
        }
    )
    returns = pd.DataFrame(
        {"reason": ["WRONG_PART", "DAMAGED", "WRONG_PART"], "sku": ["RISKY"] * 3}
    )
    return {
        "order_lines": order_lines,
        "inventory_monthly": inventory,
        "item_master": item,
        "purchase_orders": po,
        "returns": returns,
    }


def _synthetic_health() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["DEAD", "RISKY", "STEADY"],
            "abc_class": ["C", "B", "A"],
            "xyz_class": ["Z", "Z", "X"],
            "dead_stock": [True, False, False],
            "stockout_risk": ["low", "high", "low"],
            "months_of_cover": [float("inf"), 0.0, 3.0],
            "on_hand": [4.0, 0.0, 30.0],
        }
    )


def test_run_roi_end_to_end_report_frontier_and_intervals() -> None:
    extracts = _synthetic_extracts()
    outputs = rpipe.run_roi(extracts, _synthetic_health(), tenant_id="t1")

    # Eight benefit rows (seven annual + one-time cash) x three tiers.
    assert set(outputs.report["category"]) == {
        "dead_stock_cash_release",
        "inventory_carrying",
        "fill_rate_revenue",
        "long_tail_margin",
        "returns_savings",
        "labor_automation",
        "expedited_freight",
        "purchasing_cost",
    }
    assert len(outputs.report) == 8 * 3
    rows = outputs.report.set_index(["category", "tier"])
    # Dead stock is pure cash; the carrying row is NET — the policy builds
    # stock for the starving SKU (RISKY, $120/unit) that outweighs the dead-
    # stock reduction, so carrying goes negative here while fill-rate revenue
    # more than pays for it. That tradeoff is the simulator's honest output.
    assert rows.loc[("inventory_carrying", "aggressive"), "annual_value"] < 0
    assert rows.loc[("fill_rate_revenue", "aggressive"), "annual_value"] > 0
    assert rows.loc[("dead_stock_cash_release", "aggressive"), "one_time_value"] == pytest.approx(
        200.0
    )

    totals = outputs.totals.set_index("tier")
    assert totals.loc["baseline", "annual_value_total"] > 0
    # Tier scaling: conservative = 0.6 x aggressive (per-row rounding drifts
    # the totals by fractions of a cent, hence the tolerance).
    assert totals.loc["conservative", "annual_value_total"] == pytest.approx(
        0.6 * totals.loc["aggressive", "annual_value_total"], rel=1e-3
    )
    # Dead stock: 4 units x $50 = $200 one-time (asserted on the row above).
    assert totals.loc["aggressive", "one_time_value_total"] == pytest.approx(200.0)

    assert len(outputs.frontier) == len(ftr.FRONTIER_GRID_MONTHS)
    treated = outputs.counterfactual[outputs.counterfactual["group"] == "treated"]
    summary = outputs.counterfactual[outputs.counterfactual["segment"] == "treated-total"]
    assert not treated.empty
    assert not summary.empty
    row = summary.iloc[0]
    assert row["p05"] <= row["point_estimate"] <= row["p95"]
    assert row["p05"] <= row["p50"] <= row["p95"]
    # RISKY loses 5 units/month in the window; the policy walk serves them.
    risky = treated[treated["sku"] == "RISKY"].iloc[0]
    assert risky["recovered_margin"] > 0
    assert risky["fill_rate_gain"] > 0

    paths = rpipe.write_roi(outputs, "/tmp/roi-test")
    assert paths["baseline_annual"] == pytest.approx(totals.loc["baseline", "annual_value_total"])
    for key in ("report", "frontier", "counterfactual"):
        assert paths[key].exists()


def test_run_roi_rejects_short_history() -> None:
    extracts = _synthetic_extracts()
    short = extracts["order_lines"][extracts["order_lines"]["order_date"] < "2025-03"]
    params = roi.RoiParams(sim_months=12)
    with pytest.raises(ValueError, match="months of history"):
        rpipe.prepare_inputs({**extracts, "order_lines": short}, _synthetic_health(), params)
