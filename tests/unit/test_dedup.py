"""Unit tests for duplicate detection: scoring, safety gate, clustering."""

from __future__ import annotations

import pandas as pd

from mro_intelligence.catalog.dedup import propose_merges, score_pair
from mro_intelligence.catalog.normalize import normalize_item_master


def _items(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "sku": "",
        "description": "",
        "brand": "ACME",
        "manufacturer": "ACME",
        "manufacturer_part_number": "",
        "category": "FILTERS",
        "subcategory": "-",
        "uom": "EA",
        "pack_qty": 1,
        "unit_cost": 10.0,
        "list_price": 14.0,
        "status": "ACTIVE",
    }
    return normalize_item_master(pd.DataFrame([{**base, **row} for row in rows]))


def test_identical_mpn_and_brand_auto_merges() -> None:
    items = _items(
        [
            {
                "sku": "A",
                "manufacturer_part_number": "X1-100",
                "description": "FILTER 20x25x2 MERV 11",
            },
            {
                "sku": "B",
                "manufacturer_part_number": "X1100",
                "description": "filter 20x25x2 merv 11",
            },
        ]
    )
    plan = propose_merges(items)
    pair = score_pair(items, 0, 1)
    assert pair.mpn_match is True
    grouped = plan.assignments.groupby("proposed_global_part_id")["sku"].apply(list)
    assert sorted(grouped) == [["A", "B"]]


def test_different_mpn_with_same_attributes_is_not_auto_merged() -> None:
    """Regression: MPN disagreement is counter-evidence, even with identical attrs."""
    items = _items(
        [
            {
                "sku": "A",
                "manufacturer_part_number": "ZA-1502",
                "description": "PLEATED FILTER 20x25x2 MERV 8",
            },
            {
                "sku": "B",
                "manufacturer_part_number": "KU-6289",
                "description": "FILTER PLEATED 20x25x2 MERV 8",
            },
        ]
    )
    pair = score_pair(items, 0, 1)
    assert pair.mpn_match is False
    plan = propose_merges(items)
    # The high-similarity variant lands in review, never in an auto cluster.
    assert plan.assignments["proposed_global_part_id"].nunique() == 2
    assert any(t.sku_a == "A" and t.sku_b == "B" for t in plan.review_queue)


def test_same_brand_category_different_parts_stay_separate() -> None:
    items = _items(
        [
            {
                "sku": "A",
                "manufacturer_part_number": "X1-100",
                "description": "PLEATED FILTER 20x25x2 MERV 11",
            },
            {
                "sku": "B",
                "manufacturer_part_number": "Y9-999",
                "description": "CONTACTOR 40A 24V 2POLE",
            },
        ]
    )
    plan = propose_merges(items)
    parts = plan.assignments["proposed_global_part_id"].nunique()
    assert parts == 2


def test_safety_conflicting_pair_is_never_merged() -> None:
    items = _items(
        [
            {"sku": "SAFE-A", "manufacturer_part_number": "V-1", "description": "GAS VALVE 24V"},
            {"sku": "SAFE-B", "manufacturer_part_number": "V-1", "description": "WATER VALVE 24V"},
        ]
    )
    pair = score_pair(items, 0, 1)
    assert pair.mpn_match is True  # the risky candidate exists...
    plan = propose_merges(items)
    assert plan.assignments["proposed_global_part_id"].nunique() == 2  # ...but is not merged
    assert ("SAFE-A", "SAFE-B", "safety_conflict") in plan.rejected_pairs


def test_review_band_task_is_queued_not_merged() -> None:
    items = _items(
        [
            {
                "sku": "A",
                "manufacturer_part_number": "P-1",
                "description": "PLEATED FILTER 20x25x2 MERV 11",
            },
            # Same brand/category, similar wording, different MPN and one attr.
            {
                "sku": "B",
                "manufacturer_part_number": "P-2",
                "description": "PLEATED FILTER 20x25x1 MERV 11",
            },
        ]
    )
    plan = propose_merges(items, auto_threshold=0.99, review_threshold=0.10)
    assert plan.assignments["proposed_global_part_id"].nunique() == 2
    assert len(plan.review_queue) == 1
    task = plan.review_queue[0]
    assert task.sku_a == "A" and task.sku_b == "B"
    assert task.kind == "duplicate_merge"


def test_propose_merges_is_deterministic() -> None:
    items = _items(
        [
            {"sku": "A", "manufacturer_part_number": "X1", "description": "MOTOR 1/3 HP 460V"},
            {"sku": "B", "manufacturer_part_number": "X1", "description": "MOTOR 1/3 HP 460V"},
            {"sku": "C", "manufacturer_part_number": "Z9", "description": "CAPACITOR 35+5 440V"},
        ]
    )
    first, second = propose_merges(items), propose_merges(items)
    assert first.assignments.equals(second.assignments)
    assert first.stats == second.stats


def test_every_sku_receives_a_part_id() -> None:
    items = _items(
        [
            {"sku": "A", "manufacturer_part_number": "X1", "description": "VALVE 1 IN"},
            {"sku": "B", "manufacturer_part_number": "X1", "description": "VALVE 1 IN"},
            {"sku": "C", "manufacturer_part_number": "X2", "description": "VALVE 2 IN"},
        ]
    )
    plan = propose_merges(items)
    assert set(plan.assignments["sku"]) == {"A", "B", "C"}
    assert plan.assignments["proposed_global_part_id"].notna().all()
