"""Emit round-trip tests — extracts in ERP layout, truth in canonical form."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mro_intelligence.ingest.layout import load_template_for_layout
from mro_intelligence.synthetic.emit import emit_tenant
from mro_intelligence.synthetic.profile import ProfileConfig
from mro_intelligence.synthetic.tenant_data import DUP_SHARE, generate_tenant
from mro_intelligence.synthetic.world import build_world, count_equipment_models


@pytest.fixture(scope="session")
def emitted(tiny_profile: ProfileConfig, tmp_path_factory: pytest.TempPathFactory) -> dict:
    tenant = tiny_profile.tenants[0]
    rng = np.random.default_rng(9)
    n_oem = int(tenant.n_skus / (1 + DUP_SHARE))
    world = build_world(rng, n_oem, count_equipment_models(tenant.n_skus))
    data = generate_tenant(world, tiny_profile, tenant, tiny_profile.seed)
    out_dir = tmp_path_factory.mktemp("bundle")
    return emit_tenant(data, out_dir)


def test_bundle_layout(emitted: dict) -> None:
    tenant_root = Path(emitted["files"]["order_lines"]).parent.parent
    assert (tenant_root / "extracts").is_dir()
    assert (tenant_root / "ground_truth").is_dir()
    assert (tenant_root / "manifest.json").is_file()


def test_all_extract_files_written(emitted: dict) -> None:
    template = load_template_for_layout("p21")
    for extract_name, spec in template.extracts.items():
        assert extract_name in emitted["files"], f"missing extract {extract_name}"
        assert Path(emitted["files"][extract_name]).name == spec.file


def test_ground_truth_uses_canonical_names(emitted: dict) -> None:
    gt = pd.read_csv(emitted["files"]["gt_sku_mapping"], dtype=str)
    assert list(gt.columns) == ["sku", "global_part_id", "is_duplicate"]
    wo = pd.read_csv(emitted["files"]["gt_work_orders"], dtype=str)
    assert list(wo.columns) == [
        "wo_id",
        "model_id",
        "failure_code",
        "failure_name",
        "repair_action",
        "part_global_id",
        "used_wrong_part",
    ]


def test_extract_dates_use_the_template_format(emitted: dict) -> None:
    """P21 displays dates as MM/DD/YYYY; ground truth keeps ISO dates."""
    orders = pd.read_csv(emitted["files"]["order_lines"], dtype=str)
    assert orders["ORD_DT"].str.match(r"\d{2}/\d{2}/\d{4}").all()
    wos = pd.read_csv(emitted["files"]["work_orders"], dtype=str)
    assert wos["COMPLETE_DT"].str.match(r"\d{2}/\d{2}/\d{4}").all()


def test_extract_headers_match_template_source_columns(emitted: dict) -> None:
    template = load_template_for_layout("p21")
    for extract_name, spec in template.extracts.items():
        header = pd.read_csv(emitted["files"][extract_name], nrows=0).columns.tolist()
        assert header == list(spec.columns.values()), f"{extract_name} header mismatch"


def test_manifest_round_trips(emitted: dict) -> None:
    manifest_path = Path(emitted["files"]["order_lines"]).parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["tenant_id"] == emitted["tenant_id"]
    assert manifest["n_skus"] == manifest["n_global_parts"] + manifest["n_duplicates"]


def test_p21_date_regex_has_two_digits(emitted: dict) -> None:
    """Guard against zero-stripped dates (e.g. '6/24/2024')."""
    orders = pd.read_csv(emitted["files"]["order_lines"], dtype=str)
    assert orders["ORD_DT"].str.match(re.compile(r"^\d{2}/\d{2}/\d{4}$")).all()
