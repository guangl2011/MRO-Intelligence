"""Unit tests for catalog normalization primitives."""

from __future__ import annotations

import pandas as pd
import pytest

from mro_intelligence.catalog.normalize import (
    clean_text,
    extract_attributes,
    infer_safety_flags,
    normalize_item_master,
    normalize_mpn,
)


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  MOTOR   1/3 HP \t 460V  ") == "MOTOR 1/3 HP 460V"


def test_normalize_mpn_strips_supplier_styles() -> None:
    assert normalize_mpn("ABC-123") == "ABC123"
    assert normalize_mpn("abc-123") == "ABC123"
    assert normalize_mpn(" ABC 123 ") == "ABC123"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("T&P RELIEF VALVE 3/4 IN", {"tp_relief"}),
        ("NATURAL GAS FURNACE IGNITOR", {"gas"}),
        ("R-32 REFRIGERANT SENSOR", {"a2l_refrigerant"}),
        ("BACKFLOW PREVENTER 1 IN RPZ", {"backflow"}),
        ("PLEATED FILTER MERV 11", set()),
        ("GAS T&P BACKFLOW COMBO", {"gas", "tp_relief", "backflow"}),
    ],
)
def test_infer_safety_flags(text: str, expected: set[str]) -> None:
    assert infer_safety_flags(text) == frozenset(expected)


def test_extract_attributes_reads_distributor_vocabulary() -> None:
    attrs = extract_attributes("PLEATED FILTER 20x25x2 MERV 11")
    assert attrs["size"] == "20X25X2"
    assert attrs["merv"] == "11"


def test_extract_attributes_ignores_absent_labels() -> None:
    assert extract_attributes("6-ROUND BAG FILTER 45PCT") == {
        "efficiency_pct": "45",
        "diameter_in": "6",
    }


def test_normalize_item_master_adds_columns_without_mutation() -> None:
    frame = pd.DataFrame(
        {
            "sku": ["SKU-1", "SKU-2"],
            "description": ["  Brand X   GAS VALVE 370V ", "brand x gas valve 370v"],
            "brand": ["acme", "ACME "],
            "manufacturer_part_number": ["abc-123", "ABC123"],
            "category": ["hvac_cooling", "HVAC_COOLING"],
        }
    )
    normalized = normalize_item_master(frame)
    assert list(normalized["sku"]) == ["SKU-1", "SKU-2"]
    assert list(normalized["clean_description"]) == [
        "Brand X GAS VALVE 370V",
        "brand x gas valve 370v",
    ]
    assert list(normalized["normalized_mpn"]) == ["ABC123", "ABC123"]
    assert list(normalized["normalized_brand"]) == ["ACME", "ACME"]
    assert (normalized["safety_flags"] == "gas").all()
    # The input frame is untouched.
    assert "clean_description" not in frame.columns
