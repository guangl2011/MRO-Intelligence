"""Synthetic tenant generator invariants — data + hidden truth must stay consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mro_intelligence.synthetic.profile import ProfileConfig
from mro_intelligence.synthetic.tenant_data import DUP_SHARE, TenantData, generate_tenant
from mro_intelligence.synthetic.world import build_world, count_equipment_models

RETURN_REASONS = {"WRONG_PART", "DAMAGED", "TOO_MANY", "OTHER"}


@pytest.fixture(scope="session")
def tenant_data(tiny_profile: ProfileConfig) -> TenantData:
    tenant = tiny_profile.tenants[0]
    rng = np.random.default_rng(9)
    n_oem = int(tenant.n_skus / (1 + DUP_SHARE))
    world = build_world(rng, n_oem, count_equipment_models(tenant.n_skus))
    return generate_tenant(world, tiny_profile, tenant, tiny_profile.seed)


def test_item_master_has_duplicate_sku_entries(tenant_data: TenantData) -> None:
    dup_skus = tenant_data.gt_sku_mapping["is_duplicate"].sum()
    assert dup_skus > 0, "tenant item master must contain duplicate-SKU entries"
    # Duplicates = two tenant SKUs resolving to the same global part.
    part_counts = tenant_data.gt_sku_mapping["global_part_id"].value_counts()
    assert (part_counts > 1).sum() >= dup_skus - 1  # conservative consistency check


def test_all_extract_skus_exist_in_item_master(tenant_data: TenantData) -> None:
    skus = set(tenant_data.item_master["sku"].astype(str))
    frames = {
        "order_lines": tenant_data.order_lines,
        "inventory_monthly": tenant_data.inventory_monthly,
        "purchase_orders": tenant_data.purchase_orders,
        "returns": tenant_data.returns,
    }
    for name, frame in frames.items():
        if len(frame):
            assert set(frame["sku"].astype(str)) <= skus, f"{name} references unknown SKUs"
    # Work orders name the part through the dispatcher-facing SKU column.
    if len(tenant_data.work_orders):
        assert set(tenant_data.work_orders["part_sku"].astype(str)) <= skus, (
            "work orders reference unknown SKUs"
        )


def test_return_reasons_are_in_the_expected_set(tenant_data: TenantData) -> None:
    if len(tenant_data.returns):
        assert set(tenant_data.returns["reason"]) <= RETURN_REASONS
        assert (tenant_data.returns["qty"] > 0).all()


def test_gt_work_orders_reference_ground_truth_parts(tenant_data: TenantData) -> None:
    gt_parts = set(tenant_data.gt_fitment["part_global_id"])
    wo_parts = set(tenant_data.gt_work_orders["part_global_id"])
    assert wo_parts <= gt_parts, "ground-truth WOs use parts outside the fitment truth"


def test_work_order_extract_carries_free_text_and_tenant_skus(tenant_data: TenantData) -> None:
    wo = tenant_data.work_orders
    assert len(wo) == len(tenant_data.gt_work_orders)
    assert wo["failure_reported"].str.len().gt(0).all()
    assert wo["repair_action"].str.len().gt(0).all()


def test_order_lines_reference_valid_customers_and_branches(tenant_data: TenantData) -> None:
    customers = set(tenant_data.customers["customer_id"].astype(str))
    branches = set(tenant_data.branches["branch_code"].astype(str))
    assert set(tenant_data.order_lines["customer_id"].astype(str)) <= customers
    assert set(tenant_data.order_lines["branch_code"].astype(str)) <= branches


def test_inventory_covers_all_months_for_stocked_skus(tenant_data: TenantData) -> None:
    inv = tenant_data.inventory_monthly
    assert len(inv) > 0
    months_per_sku = inv.groupby("sku")["month"].nunique()
    assert months_per_sku.nunique() == 1, "stocked SKUs must have a complete monthly series"


def test_manifest_reports_realized_revenue(tenant_data: TenantData) -> None:
    assert tenant_data.manifest["realized_annual_revenue"] > 0
    assert tenant_data.manifest["tenant_id"] == "tinyco"
    assert tenant_data.manifest["profile"] == "test-tiny"


def test_generation_is_deterministic(tiny_profile: ProfileConfig) -> None:
    tenant = tiny_profile.tenants[0]
    rng = np.random.default_rng(9)
    n_oem = int(tenant.n_skus / (1 + DUP_SHARE))
    world = build_world(rng, n_oem, count_equipment_models(tenant.n_skus))
    a = generate_tenant(world, tiny_profile, tenant, tiny_profile.seed)
    b = generate_tenant(world, tiny_profile, tenant, tiny_profile.seed)
    pd.testing.assert_frame_equal(a.order_lines, b.order_lines)
