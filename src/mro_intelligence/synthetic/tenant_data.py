"""Tenant-level dataset generation for one synthetic distributor.

Produces every extract on the SDD §3 checklist plus the ground-truth bundle:
tenant SKUs (with deliberate duplicates), monthly demand with the long-tail
structure (A ~10% SKUs ≈ 65% revenue, C ~70% ≈ 10%), an "ERP as-is" inventory
world (flat reorder rules → seasonal stockouts + dead stock), purchase orders,
returns with reasons, work orders, and noisy OEM fitment evidence.

The generator is the sole holder of ground truth; the platform pipeline must
re-derive everything from the messy extracts alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from mro_intelligence.simulation import PolicyParams, simulate_sku_inventory
from mro_intelligence.synthetic.messy import corrupt_description, corrupt_mpn
from mro_intelligence.synthetic.profile import ProfileConfig, TenantProfile
from mro_intelligence.synthetic.world import (
    CATEGORY_FAILURES,
    GENERIC_BRAND,
    SEASON_COOLING,
    SEASON_FLAT,
    SEASON_HEATING,
    World,
)

# Monthly seasonality factors by group (Jan..Dec). Calibrated to US HVAC
# (cooling peaks May-Aug, heating Oct-Feb) and flat commodity-MRO demand.
SEASON_FACTORS: dict[str, tuple[float, ...]] = {
    SEASON_COOLING: (0.55, 0.55, 0.80, 1.10, 1.45, 1.70, 1.80, 1.60, 1.10, 0.80, 0.60, 0.55),
    SEASON_HEATING: (1.50, 1.40, 1.10, 0.80, 0.60, 0.50, 0.50, 0.60, 1.00, 1.30, 1.60, 1.70),
    SEASON_FLAT: (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
}

DUP_SHARE = 0.06  # extra duplicate SKUs as a share of the catalog
DEAD_C_SHARE = 0.32  # share of C-class SKUs whose sales stopped >12 months ago
RETURN_RATE = 0.04
WRONG_PART_WO_RATE = 0.06  # work orders that used an incompatible part

# Wholesale order-size targets per class (units per order line).
ORDER_SIZE = {"A": 30.0, "B": 12.0, "C": 3.0}
# Distributor margin per class (long tail carries a flatter realized markup).
MARGIN = {"A": 1.28, "B": 1.34, "C": 1.42}
TIER_FACTOR = {"CONTRACTOR": 1.00, "PROPERTY": 1.07, "FACILITY": 1.14}

MAX_ORDERS_PER_SKU_MONTH = 8  # volume cap keeping extract size realistic

OEM_RETURN_WEIGHTS = (0.08, 0.45, 0.30, 0.17)  # WRONG_PART, DAMAGED, TOO_MANY, OTHER
GENERIC_RETURN_WEIGHTS = (0.40, 0.30, 0.20, 0.10)


@dataclass
class TenantData:
    """One tenant's synthetic ERP extracts and ground truth."""

    tenant_id: str
    erp_layout: str
    item_master: pd.DataFrame
    order_lines: pd.DataFrame
    inventory_monthly: pd.DataFrame
    purchase_orders: pd.DataFrame
    price_file: pd.DataFrame
    returns: pd.DataFrame
    work_orders: pd.DataFrame
    oem_fitment_catalog: pd.DataFrame
    branches: pd.DataFrame
    customers: pd.DataFrame
    gt_sku_mapping: pd.DataFrame
    gt_fitment: pd.DataFrame
    gt_xref: pd.DataFrame
    gt_work_orders: pd.DataFrame
    manifest: dict = field(default_factory=dict)


def month_range(end_month: date, months: int) -> list[date]:
    """First-of-month dates for the history window (inclusive of end_month)."""
    out: list[date] = []
    for k in range(months - 1, -1, -1):
        total0 = end_month.month - 1 - k
        year = end_month.year + (total0 // 12)
        month = (total0 % 12) + 1
        out.append(date(year, month, 1))
    return out


def seasonality_series(season: str, months: list[date], rng: np.random.Generator) -> np.ndarray:
    """Monthly seasonal multipliers plus one hot-summer weather spike."""
    base = np.array([SEASON_FACTORS[season][d.month - 1] for d in months])
    midyear = months[len(months) // 2].year
    spike = np.array([1.30 if d.month in (6, 7, 8) and d.year == midyear else 1.0 for d in months])
    noise = rng.normal(1.0, 0.05, size=len(months))
    return base * spike * noise


def _draw_demand(
    mu: float, season: np.ndarray, dead_after: int | None, rng: np.random.Generator
) -> np.ndarray:
    """Monthly demand for one SKU-branch: NB noise for movers, intermittent
    zero-inflation for the slow tail, dead series cut off at ``dead_after``."""
    season = season.copy()
    if dead_after is not None:
        season[dead_after:] = 0.0
    if mu < 1.5:
        # Intermittent: demand occurs in ~35% of months, then small counts.
        p_active = 0.35
        lam = mu / p_active
        active = rng.random(len(season)) < p_active
        qty = rng.poisson(np.maximum(lam * season, 0.0))
        return np.where(active, qty, 0).astype(float)
    # Overdispersed Poisson via negative binomial (dispersion phi=8).
    phi = 8.0
    mean = np.maximum(mu * season, 0.0)
    n_shape = np.where(mean > 0, phi, 1.0)
    p_param = np.where(mean > 0, phi / (phi + mean), 1.0)
    return rng.negative_binomial(n_shape, p_param).astype(float)


def _erp_policy(
    mu_scalar: float, sigma_scalar: float, lead_time_months: float, n_months: int
) -> PolicyParams:
    """The "ERP as-is" policy: flat trailing-mean belief, z=1, no seasonality."""
    mu = np.full(n_months, max(mu_scalar, 0.0))
    sigma = np.full(n_months, max(sigma_scalar, 0.0))
    return PolicyParams(
        mu=mu, sigma=sigma, lead_time_months=lead_time_months, z=1.0, coverage_months=1.5
    )


def generate_tenant(
    world: World, config: ProfileConfig, tenant: TenantProfile, seed_offset: int
) -> TenantData:
    rng = np.random.default_rng(seed_offset)
    months = month_range(config.end_month, config.months)
    n_months = len(months)
    parts = world.parts
    part_pos = {p.global_part_id: i for i, p in enumerate(parts)}

    # --- Suppliers per part (category affinity drives lead times) ---------------
    part_supplier: dict[str, int] = {}
    for part in parts:
        aff = [i for i, s in enumerate(world.suppliers) if part.category in s.categories]
        idx = (
            aff[int(rng.integers(0, len(aff)))]
            if aff
            else int(rng.integers(0, len(world.suppliers)))
        )
        part_supplier[part.global_part_id] = idx

    # --- Branch / customer masters ----------------------------------------------
    branch_codes = [f"BR{i + 1}" for i in range(tenant.n_branches)]
    branches = pd.DataFrame(
        {
            "branch_code": branch_codes,
            "name": [f"Branch {b}" for b in branch_codes],
            "region": rng.choice(
                ("NORTHEAST", "SOUTHEAST", "MIDWEST", "MOUNTAIN"), size=len(branch_codes)
            ),
        }
    )
    seg_names = ("CONTRACTOR", "PROPERTY", "FACILITY")
    n_customers = tenant.n_customers
    cust_segments = rng.choice(seg_names, size=n_customers, p=(0.60, 0.25, 0.15))
    customers = pd.DataFrame(
        {
            "customer_id": [f"C-{i + 1:05d}" for i in range(n_customers)],
            "segment": cust_segments,
        }
    )
    cust_ids = customers["customer_id"].to_numpy()
    cust_segs = customers["segment"].to_numpy()

    # --- Tenant SKUs (+ deliberate duplicates) ------------------------------------
    n_dup = int(DUP_SHARE * len(parts))
    dup_targets = set(int(x) for x in rng.choice(len(parts), size=n_dup, replace=False))

    item_rows: list[dict[str, object]] = []
    sku_to_part: list[str] = []
    sku_seq = 0

    def add_sku(part_idx: int, is_dup: bool) -> None:
        nonlocal sku_seq
        sku_seq += 1
        part = parts[part_idx]
        # Long-tail neglect: C-class descriptions carry more corruption.
        level = {"A": 0.15, "B": 0.35, "C": 0.70}[part.abc_class]
        desc = corrupt_description(part.description, rng, level)
        brand = part.brand if rng.random() > 0.2 * level else part.brand.lower()
        mpn = part.oem_part_number
        if is_dup and rng.random() >= 0.3:
            mpn = corrupt_mpn(mpn, rng)
        cost = round(part.unit_cost * rng.uniform(0.98, 1.02), 2)
        sku = f"SKU-{sku_seq:07d}"
        item_rows.append(
            {
                "sku": sku,
                "description": desc,
                "brand": brand,
                "manufacturer": part.brand,
                "manufacturer_part_number": mpn,
                "category": part.category.upper(),
                "subcategory": part.subtype.upper(),
                "uom": "EA",
                "pack_qty": int(rng.choice((1, 1, 1, 2, 4))),
                "unit_cost": cost,
                "list_price": round(cost * rng.uniform(1.30, 1.60), 2),
                "status": "ACTIVE",
            }
        )
        sku_to_part.append(part.global_part_id)

    for i in range(len(parts)):
        add_sku(i, is_dup=False)
    for j in sorted(dup_targets):
        add_sku(j, is_dup=True)

    item_master = pd.DataFrame(item_rows)

    # Duplicate ground truth: any SKU whose global part already appeared under an
    # earlier SKU (the second and later occurrences are the duplicates).
    seen_parts: set[str] = set()
    is_dup_flags: list[bool] = []
    for part_id in sku_to_part:
        is_dup_flags.append(part_id in seen_parts)
        seen_parts.add(part_id)
    gt_sku_mapping = pd.DataFrame(
        {
            "sku": [str(r["sku"]) for r in item_rows],
            "global_part_id": sku_to_part,
            "is_duplicate": is_dup_flags,
        }
    )

    # --- Demand calibration ---------------------------------------------------------
    # Class revenue shares: A 65%, B 25%, C 10% of the annual target (AXON §3.2).
    part_class = np.array([p.abc_class for p in parts])
    class_counts = {c: max(1, int((part_class == c).sum())) for c in "ABC"}
    class_revenue = {
        "A": 0.65 * tenant.annual_revenue_target,
        "B": 0.25 * tenant.annual_revenue_target,
        "C": 0.10 * tenant.annual_revenue_target,
    }
    mu_by_part = np.zeros(len(parts))
    # Expected selling price basis: extracts price = cost x class margin x tier
    # factor (segment mix 60/25/15) x unit jitter (mean 1.0).
    expected_tier = (
        0.60 * TIER_FACTOR["CONTRACTOR"]
        + 0.25 * TIER_FACTOR["PROPERTY"]
        + 0.15 * TIER_FACTOR["FACILITY"]
    )
    for c in "ABC":
        idxs = np.where(part_class == c)[0]
        for i in idxs:
            expected_price = parts[i].unit_cost * MARGIN[c] * expected_tier
            annual_units = class_revenue[c] / class_counts[c] / expected_price
            mu_by_part[i] = annual_units / 12.0

    # Duplicate SKUs split the part's demand — the fragmentation that makes
    # duplicates costly (fragmented signal + duplicated safety stock).
    sku_mu = np.zeros(len(sku_to_part))
    by_part: dict[str, list[int]] = {}
    for sku_i, part_id in enumerate(sku_to_part):
        by_part.setdefault(part_id, []).append(sku_i)
    for part_id, sku_idxs in by_part.items():
        shares = (
            rng.dirichlet(np.full(len(sku_idxs), 4.0)) if len(sku_idxs) > 1 else np.array([1.0])
        )
        for k, sku_i in enumerate(sku_idxs):
            sku_mu[sku_i] = mu_by_part[part_pos[part_id]] * shares[k]

    n_stock_branches = {
        "A": tenant.n_branches,
        "B": max(2, tenant.n_branches - 1),
        "C": max(1, tenant.n_branches - 2),
    }
    dead_cutoff: dict[str, int] = {}
    # Dead stock is only observable with >=12 trailing zero-sales months; shorter
    # histories (unit-test profiles) carry no dead parts by construction.
    if n_months > 15:
        for part in parts:
            if part.abc_class == "C" and rng.random() < DEAD_C_SHARE:
                dead_cutoff[part.global_part_id] = int(rng.integers(3, n_months - 12))

    # --- Per (sku, branch): demand series + ERP inventory world ----------------------
    order_rows: list[dict[str, object]] = []
    inv_rows: list[dict[str, object]] = []
    po_rows: list[dict[str, object]] = []
    po_seq = 0
    order_seq = 0
    season_cache: dict[str, np.ndarray] = {}

    for sku_i, part_id in enumerate(sku_to_part):
        sku = str(item_master.at[sku_i, "sku"])
        p = parts[part_pos[part_id]]
        if p.seasonality not in season_cache:
            season_cache[p.seasonality] = seasonality_series(p.seasonality, months, rng)
        season = season_cache[p.seasonality]
        sup = world.suppliers[part_supplier[part_id]]
        lead_days = int(rng.integers(*sup.lead_time_days))
        lead_months = lead_days / 30.0

        nb = n_stock_branches[p.abc_class]
        stock_branches = rng.choice(tenant.n_branches, size=nb, replace=False) + 1
        weights = rng.dirichlet(np.full(nb, 3.0))
        cost_i = float(item_master.at[sku_i, "unit_cost"])

        for b_pos, b in enumerate(stock_branches):
            branch = f"BR{b}"
            mu_b = float(sku_mu[sku_i]) * float(weights[b_pos])
            demand = _draw_demand(mu_b, season, dead_cutoff.get(part_id), rng)
            opening = float(np.ceil(mu_b * 2)) + 1
            sigma = float(np.std(demand))
            path = simulate_sku_inventory(
                demand,
                opening_stock=opening,
                policy=_erp_policy(mu_b, sigma, lead_months, n_months),
            )
            for m in range(n_months):
                inv_rows.append(
                    {
                        "month": months[m].isoformat(),
                        "branch_code": branch,
                        "sku": sku,
                        "on_hand": int(path.on_hand_end[m]),
                        "units_lost": int(path.units_lost[m]),
                    }
                )
                if demand[m] <= 0:
                    continue
                target = ORDER_SIZE[p.abc_class]
                n_ord = int(min(MAX_ORDERS_PER_SKU_MONTH, max(1, np.ceil(demand[m] / target))))
                if n_ord > 1:
                    shares = rng.dirichlet(np.full(n_ord, 2.0))
                    splits = np.floor(demand[m] * shares).astype(int)
                    splits[0] += int(demand[m] - splits.sum())
                else:
                    splits = np.array([int(demand[m])])
                for q in splits:
                    if q <= 0:
                        continue
                    order_seq += 1
                    c_i = int(rng.integers(0, n_customers))
                    seg = str(cust_segs[c_i])
                    odate = months[m] + timedelta(days=int(rng.integers(0, 28)))
                    price = (
                        cost_i * MARGIN[p.abc_class] * TIER_FACTOR[seg] * rng.uniform(0.97, 1.03)
                    )
                    order_rows.append(
                        {
                            "order_id": f"OR-{order_seq:07d}",
                            "line_id": 1,
                            "order_date": odate.isoformat(),
                            "branch_code": branch,
                            "customer_id": str(cust_ids[c_i]),
                            "customer_segment": seg,
                            "sku": sku,
                            "qty": int(q),
                            "unit_price": round(float(price), 2),
                            "unit_cost": round(cost_i * rng.uniform(0.98, 1.02), 2),
                        }
                    )
                if path.units_ordered[m] > 0:
                    po_seq += 1
                    po_date = months[m] + timedelta(days=int(rng.integers(0, 5)))
                    po_rows.append(
                        {
                            "po_id": f"PO-{po_seq:07d}",
                            "supplier_id": sup.supplier_id,
                            "sku": sku,
                            "order_date": po_date.isoformat(),
                            "receipt_date": (po_date + timedelta(days=lead_days)).isoformat(),
                            "qty_ordered": int(path.units_ordered[m]),
                            "qty_received": int(path.units_ordered[m]),
                            "unit_cost": round(cost_i * rng.uniform(0.90, 1.00), 2),
                        }
                    )

    order_lines = pd.DataFrame(order_rows)
    inventory_monthly = pd.DataFrame(inv_rows)
    purchase_orders = pd.DataFrame(po_rows)

    # --- Returns ----------------------------------------------------------------------
    brand_by_sku = {
        str(s): parts[part_pos[sku_to_part[i]]].brand for i, s in enumerate(item_master["sku"])
    }
    returns = _build_returns(order_lines, brand_by_sku, rng)

    # --- Work orders (fault→repair ground truth + dispatcher extract) ------------------
    n_wos = max(200, tenant.n_work_orders)
    gt_work_orders, work_orders = _build_work_orders(
        world, item_master, sku_to_part, months, n_wos, rng
    )

    # --- Fitment evidence + price file / cross references -------------------------------
    gt_fitment = _fitment_table(world)
    oem_fitment_catalog = _noisy_fitment_evidence(world, rng)
    price_file, gt_xref = _build_price_file(world, part_supplier, rng)

    realized = float((order_lines["qty"] * order_lines["unit_price"]).sum())
    realized_annual = realized / n_months * 12
    manifest = {
        "profile": config.name,
        "tenant_id": tenant.tenant_id,
        "erp_layout": tenant.erp_layout,
        "months": n_months,
        "history_start": months[0].isoformat(),
        "history_end": months[-1].isoformat(),
        "n_skus": len(item_master),
        "n_global_parts": len(parts),
        "n_duplicates": int(gt_sku_mapping["is_duplicate"].sum()),
        "n_branches": tenant.n_branches,
        "n_order_lines": len(order_lines),
        "n_purchase_orders": len(purchase_orders),
        "n_work_orders": int(gt_work_orders["wo_id"].nunique()) if not gt_work_orders.empty else 0,
        "n_fitment_evidence": len(oem_fitment_catalog),
        "realized_annual_revenue": round(realized_annual, 0),
        "annual_revenue_target": tenant.annual_revenue_target,
        "seed_offset": seed_offset,
    }
    return TenantData(
        tenant_id=tenant.tenant_id,
        erp_layout=tenant.erp_layout,
        item_master=item_master,
        order_lines=order_lines,
        inventory_monthly=inventory_monthly,
        purchase_orders=purchase_orders,
        price_file=price_file,
        returns=returns,
        work_orders=work_orders,
        oem_fitment_catalog=oem_fitment_catalog,
        branches=branches,
        customers=customers,
        gt_sku_mapping=gt_sku_mapping,
        gt_fitment=gt_fitment,
        gt_xref=gt_xref,
        gt_work_orders=gt_work_orders,
        manifest=manifest,
    )


def _build_returns(
    order_lines: pd.DataFrame,
    brand_by_sku: dict[str, str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample returns with reason-code mix conditioned on OEM vs generic part."""
    if order_lines.empty:
        return pd.DataFrame(
            columns=["return_id", "order_id", "sku", "return_date", "qty", "reason"]
        )
    n = len(order_lines)
    picked = rng.random(n) < RETURN_RATE
    ret = order_lines[picked].reset_index(drop=True)

    reasons: list[str] = []
    for sku in ret["sku"]:
        weights = (
            GENERIC_RETURN_WEIGHTS
            if brand_by_sku[str(sku)] == GENERIC_BRAND
            else OEM_RETURN_WEIGHTS
        )
        reasons.append(str(rng.choice(("WRONG_PART", "DAMAGED", "TOO_MANY", "OTHER"), p=weights)))

    return pd.DataFrame(
        {
            "return_id": [f"RT-{i + 1:07d}" for i in range(len(ret))],
            "order_id": ret["order_id"],
            "sku": ret["sku"],
            "return_date": [
                str(date.fromisoformat(d) + timedelta(days=int(rng.integers(7, 31))))
                for d in ret["order_date"]
            ],
            "qty": np.maximum(1, (ret["qty"] * rng.uniform(0.3, 1.0, size=len(ret))).astype(int)),
            "reason": reasons,
        }
    )


def _build_work_orders(
    world: World,
    item_master: pd.DataFrame,
    sku_to_part: list[str],
    months: list[date],
    n_wos: int,
    rng: np.random.Generator,
):
    """Ground-truth WOs (fault→action→parts) plus a dispatcher-style extract
    that references tenant SKUs and free-text failure reports."""
    n_months = len(months)
    primary_sku: dict[str, str] = {}
    for sku, part_id in zip(item_master["sku"], sku_to_part, strict=True):
        primary_sku.setdefault(part_id, str(sku))

    n_wos = max(200, int(n_wos))
    gt_rows: list[dict[str, object]] = []
    wo_rows: list[dict[str, object]] = []
    model_by_cat: dict[str, list] = {}
    for model in world.models:
        model_by_cat.setdefault(model.category, []).append(model)

    for w in range(n_wos):
        cat = str(rng.choice(list(model_by_cat.keys())))
        model = model_by_cat[cat][int(rng.integers(0, len(model_by_cat[cat])))]
        failures = CATEGORY_FAILURES[cat]
        probs = np.array([f.prob for f in failures])
        fdef = failures[int(rng.choice(len(failures), p=probs / probs.sum()))]
        action_i = int(rng.integers(0, len(fdef.actions)))
        action_name, subtype_keys = fdef.actions[action_i]
        truth = set(world.fitment.get((model.model_id, fdef.code), ()))
        compatible = sorted(pid for pid in truth if world.part(pid).subtype in subtype_keys)
        if not compatible:
            continue
        k = int(rng.integers(1, min(3, len(compatible)) + 1))
        used = list(rng.choice(compatible, size=k, replace=False))
        if rng.random() < WRONG_PART_WO_RATE:
            # Swap one part for an incompatible part of the same subtype — the
            # real-world wrong-part event that returns and diagnostics surface.
            bad_pool = [
                p.global_part_id
                for p in world.parts
                if p.subtype == world.part(used[0]).subtype and p.global_part_id not in truth
            ]
            if bad_pool:
                used[int(rng.integers(0, len(used)))] = bad_pool[
                    int(rng.integers(0, len(bad_pool)))
                ]
        wo_id = f"WO-{w + 1:06d}"
        wo_date = months[int(rng.integers(0, n_months))] + timedelta(days=int(rng.integers(0, 28)))
        for pid in used:
            is_wrong = pid not in truth
            gt_rows.append(
                {
                    "wo_id": wo_id,
                    "model_id": model.model_id,
                    "failure_code": fdef.code,
                    "failure_name": fdef.name,
                    "repair_action": action_name,
                    "part_global_id": pid,
                    "used_wrong_part": is_wrong,
                }
            )
            wo_rows.append(
                {
                    "wo_id": wo_id,
                    "equipment_model": f"{model.make} {model.model_no}",
                    "model_id": model.model_id,
                    "failure_reported": fdef.name,
                    "repair_action": action_name,
                    "part_sku": primary_sku.get(pid, ""),
                    "confirmed_date": wo_date.isoformat(),
                    "technician_id": f"T-{int(rng.integers(1, 60)):03d}",
                }
            )
    return pd.DataFrame(gt_rows), pd.DataFrame(wo_rows)


def _fitment_table(world: World) -> pd.DataFrame:
    rows = [
        {"model_id": model_id, "failure_code": failure_code, "part_global_id": pid}
        for (model_id, failure_code), part_ids in sorted(world.fitment.items())
        for pid in part_ids
    ]
    return pd.DataFrame(rows, columns=["model_id", "failure_code", "part_global_id"])


def _noisy_fitment_evidence(world: World, rng: np.random.Generator) -> pd.DataFrame:
    """OEM catalog evidence: 85% of true (model, failure, OEM part) edges get
    1-2 evidence rows; 4% of rows name a wrong part of the same subtype."""
    model_by_id = {m.model_id: m for m in world.models}
    rows: list[dict[str, object]] = []
    seq = 0
    for (model_id, failure_code), part_ids in sorted(world.fitment.items()):
        model = model_by_id[model_id]
        oem_parts = [pid for pid in part_ids if not world.part(pid).replaces]
        same_subtype_pool: dict[str, list[str]] = {}
        for pid in part_ids:
            sub = world.part(pid).subtype
            same_subtype_pool.setdefault(sub, []).append(pid)
        for pid in oem_parts:
            if rng.random() > 0.85:
                continue
            for _ in range(int(rng.integers(1, 3))):
                seq += 1
                claim = pid
                if rng.random() < 0.04:
                    pool = [
                        q for q in same_subtype_pool.get(world.part(pid).subtype, []) if q != pid
                    ]
                    if pool:
                        claim = pool[int(rng.integers(0, len(pool)))]
                rows.append(
                    {
                        "evidence_id": f"EV-{seq:07d}",
                        "equipment_make": model.make,
                        "equipment_model_no": model.model_no,
                        "model_id": model_id,
                        "failure_code": failure_code,
                        "oem_part_number": world.part(claim).oem_part_number,
                        "note": f"Compatibility listing for {model.make} {model.model_no}",
                        "source": "OEM_CATALOG",
                    }
                )
    return pd.DataFrame(rows)


def _build_price_file(world: World, part_supplier: dict[str, int], rng: np.random.Generator):
    """Supplier price list with embedded OEM↔generic cross references.

    Generic parts appear once per replaced OEM part (the shape of real supplier
    cross-reference tables) — that is the price-file signal the platform mines.
    """
    rows: list[dict[str, object]] = []
    for part in world.parts:
        sup = world.suppliers[part_supplier[part.global_part_id]]
        base = {
            "supplier_id": sup.supplier_id,
            "unit_cost": round(part.unit_cost * rng.uniform(0.92, 1.05), 2),
            "lead_time_days": int(rng.integers(*sup.lead_time_days)),
            "discount_pct": round(float(rng.uniform(0, 12)), 1),
        }
        if part.replaces:
            spn = corrupt_mpn(part.oem_part_number, rng)
            for oem_id in part.replaces:
                rows.append(
                    {
                        **base,
                        "supplier_part_number": spn,
                        "oem_part_number": world.part(oem_id).oem_part_number,
                    }
                )
        else:
            rows.append(
                {
                    **base,
                    "supplier_part_number": part.oem_part_number,
                    "oem_part_number": part.oem_part_number,
                }
            )
    gt_xref = pd.DataFrame(
        [
            {"generic_global_id": g.global_part_id, "replaced_oem_global_id": o}
            for g in world.parts
            for o in g.replaces
        ]
    )
    return pd.DataFrame(rows), gt_xref
