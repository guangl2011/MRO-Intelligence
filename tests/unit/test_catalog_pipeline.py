"""End-to-end catalog-stage tests on the tiny synthetic profile.

The pipeline runs on emitted ERP extracts exactly as a partner bundle would
arrive; ground truth stays hidden until the evaluation call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mro_intelligence.catalog.evaluate import evaluate_dedup
from mro_intelligence.catalog.pipeline import load_extracts, run_catalog, write_catalog
from mro_intelligence.synthetic.emit import emit_tenant
from mro_intelligence.synthetic.profile import ProfileConfig
from mro_intelligence.synthetic.tenant_data import DUP_SHARE, generate_tenant
from mro_intelligence.synthetic.world import build_world, count_equipment_models


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory: pytest.TempPathFactory, tiny_profile: ProfileConfig) -> Path:
    """Emit the tiny tenant's extracts once for the whole module."""
    tenant = tiny_profile.tenants[0]
    world = build_world(
        np.random.default_rng(7),
        int(tenant.n_skus / (1 + DUP_SHARE)),
        count_equipment_models(tenant.n_skus),
    )
    data = generate_tenant(world, tiny_profile, tenant, tiny_profile.seed)
    root = tmp_path_factory.mktemp("bundle")
    emit_tenant(data, root)
    return root / data.tenant_id


def test_catalog_pipeline_runs_end_to_end(tiny_bundle: Path) -> None:
    extracts = load_extracts(tiny_bundle, "p21")
    outputs = run_catalog(extracts, tenant_id="tinyco")
    assert outputs.stats["n_skus"] == len(extracts["item_master"])
    assert outputs.stats["n_parts"] <= outputs.stats["n_skus"]


def test_catalog_pipeline_finds_duplicates_above_recall_floor(tiny_bundle: Path) -> None:
    extracts = load_extracts(tiny_bundle, "p21")
    outputs = run_catalog(extracts, tenant_id="tinyco")
    gt = pd.read_csv(tiny_bundle / "ground_truth" / "sku_mapping.csv", dtype=str)
    metrics = evaluate_dedup(outputs.merge_plan.assignments, gt)
    # The generator plants DUP_SHARE duplicates with corrupted descriptions;
    # MPN blocking alone should recover most. A low floor keeps the test
    # robust to corruption draws while still catching blocking regressions.
    assert metrics["recall"] >= 0.5, metrics
    assert metrics["precision"] >= 0.5, metrics


def test_review_queue_is_populated_for_ambiguous_pairs(tiny_bundle: Path) -> None:
    extracts = load_extracts(tiny_bundle, "p21")
    outputs = run_catalog(extracts, tenant_id="tinyco")
    # The demo catalog's same-brand same-category neighbors produce review-band
    # candidates; an empty queue would mean the thresholds are mis-calibrated.
    assert outputs.stats["n_review"] > 0


def test_write_catalog_outputs_round_trip(tiny_bundle: Path, tmp_path: Path) -> None:
    extracts = load_extracts(tiny_bundle, "p21")
    outputs = run_catalog(extracts, tenant_id="tinyco")
    paths = write_catalog(outputs, tmp_path)
    catalog = pd.read_csv(paths["catalog"], dtype=str)
    review = pd.read_csv(paths["review_queue"], dtype=str)
    seg = pd.read_csv(paths["segmentation"], dtype=str)
    assert len(catalog) == outputs.stats["n_skus"]
    assert {"proposed_global_part_id", "safety_flags"} <= set(catalog.columns)
    assert {"task_id", "score", "reason"} <= set(review.columns)
    assert {"sku", "abc_class", "xyz_class", "annual_revenue"} <= set(seg.columns)
    assert set(seg["abc_class"]) <= {"A", "B", "C"}


def test_load_extracts_preserves_all_tenants_rows(tiny_bundle: Path) -> None:
    """Every extract row survives the canonical reverse mapping."""
    extracts = load_extracts(tiny_bundle, "p21")
    raw_orders = pd.read_csv(tiny_bundle / "extracts" / "P21_ORDER_LINES.csv", dtype=str)
    assert len(extracts["order_lines"]) == len(raw_orders)


def test_run_catalog_rejects_invalid_tenant_id(tiny_bundle: Path) -> None:
    extracts = load_extracts(tiny_bundle, "p21")
    with pytest.raises(ValueError, match="invalid tenant id"):
        run_catalog(extracts, tenant_id="bad id with spaces")


def test_load_extracts_fails_loudly_on_missing_column(tiny_bundle: Path, tmp_path: Path) -> None:
    """A truncated partner file surfaces as an error, not a silent NaN column."""
    import shutil

    copied = tmp_path / "copy"
    shutil.copytree(tiny_bundle, copied)
    item_path = copied / "extracts" / "P21_ITEM_MASTER.csv"
    frame = pd.read_csv(item_path, dtype=str)
    frame.drop(columns=["MFR_PART_NO"]).to_csv(item_path, index=False)
    with pytest.raises(ValueError, match="manufacturer_part_number"):
        load_extracts(copied, "p21")
