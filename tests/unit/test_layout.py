"""ERP layout template tests — the shared contract between emit and ingest."""

from __future__ import annotations

import pytest

from mro_intelligence.ingest.layout import (
    DATE_COLUMNS,
    TEMPLATE_BY_LAYOUT,
    canonical_columns,
    load_template,
    load_template_for_layout,
    source_column_map,
)

EXPECTED_LAYOUTS = ("p21", "eclipse", "infor")
EXPECTED_EXTRACTS = (
    "item_master",
    "order_lines",
    "inventory_monthly",
    "purchase_orders",
    "returns",
    "work_orders",
    "oem_fitment_catalog",
    "price_file",
    "branches",
    "customers",
)


@pytest.mark.parametrize("layout", EXPECTED_LAYOUTS)
def test_all_layouts_load(layout: str) -> None:
    template = load_template_for_layout(layout)
    assert template.erp == TEMPLATE_BY_LAYOUT[layout]
    assert set(template.extracts) == set(EXPECTED_EXTRACTS)


@pytest.mark.parametrize("layout", EXPECTED_LAYOUTS)
def test_column_maps_are_bijective(layout: str) -> None:
    template = load_template_for_layout(layout)
    for spec in template.extracts.values():
        canon = canonical_columns(spec)
        source = list(source_column_map(spec).values())
        assert len(canon) == len(set(canon)), f"{layout}/{spec.name}: duplicate canonical columns"
        assert len(source) == len(set(source)), f"{layout}/{spec.name}: duplicate source columns"


@pytest.mark.parametrize("layout", EXPECTED_LAYOUTS)
def test_source_file_names_are_declared(layout: str) -> None:
    template = load_template_for_layout(layout)
    for spec in template.extracts.values():
        assert spec.file and (spec.file.endswith(".csv") or spec.file.endswith(".txt"))


def test_date_columns_are_recognized() -> None:
    template = load_template_for_layout("p21")
    order_spec = template.extracts["order_lines"]
    assert "order_date" in DATE_COLUMNS
    assert order_spec.date_format == "%m/%d/%Y"
    assert template.extracts["returns"].columns.get("return_date")


def test_unknown_layout_raises() -> None:
    with pytest.raises(KeyError, match="unknown ERP layout"):
        load_template_for_layout("sap")


def test_direct_template_load_by_erp_id() -> None:
    template = load_template("epicor_prophet21")
    assert template.extracts["item_master"].columns["sku"] == "ITEM_ID"
