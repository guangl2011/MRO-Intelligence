"""Unit tests for ABC-XYZ segmentation."""

from __future__ import annotations

import pandas as pd

from mro_intelligence.catalog.segmentation import abc_classify, segment, xyz_classify


def test_abc_classes_follow_cumulative_revenue() -> None:
    revenue = pd.Series({"s1": 500.0, "s2": 300.0, "s3": 150.0, "s4": 50.0})
    classes = abc_classify(revenue, a_share=0.65, b_share=0.85)
    # Cumulative shares: s1 0.50 (A band <=0.65), s2 0.80 (B band <=0.85),
    # s3 0.95 and s4 1.00 (C).
    assert list(classes.reindex(["s1", "s2", "s3", "s4"])) == ["A", "B", "C", "C"]


def test_abc_zero_revenue_skus_are_c() -> None:
    revenue = pd.Series({"s1": 0.0, "s2": 0.0})
    assert list(abc_classify(revenue)) == ["C", "C"]


def test_xyz_sparse_demand_is_z() -> None:
    units = pd.DataFrame(
        {
            "sparse": [0, 0, 0, 5, 0, 0],  # zero share 4/6, spiky
            "steady": [10, 11, 10, 9, 10, 10],  # low CV, no zero months
            "jumpy": [2, 14, 3, 15, 2, 12],  # CV ~0.7: Y band
        }
    ).T  # rows are SKUs, columns are months
    classes = xyz_classify(units)
    assert classes["sparse"] == "Z"
    assert classes["steady"] == "X"
    assert classes["jumpy"] == "Y"


def test_segment_attaches_revenue_and_classes() -> None:
    items = pd.DataFrame({"sku": ["A", "B", "C"]})
    orders = pd.DataFrame(
        {
            "sku": ["A", "A", "B"],
            "order_date": ["2026-01-15", "2026-02-15", "2026-03-15"],
            "qty": [4, 6, 1],
            "unit_price": [100.0, 100.0, 50.0],
        }
    )
    seg = segment(items, orders)
    assert set(seg.columns) >= {"sku", "annual_revenue", "abc_class", "xyz_class"}
    assert float(seg.loc[seg["sku"] == "A", "annual_revenue"].iloc[0]) == 1000.0
    assert float(seg.loc[seg["sku"] == "C", "annual_revenue"].iloc[0]) == 0.0
    assert seg["abc_class"].isin(["A", "B", "C"]).all()
    assert seg["xyz_class"].isin(["X", "Y", "Z"]).all()


def test_segment_counts_zero_months_from_other_skus_sales() -> None:
    """A SKU absent from a month that exists in the extract gets a zero month."""
    items = pd.DataFrame({"sku": ["A", "B", "C"]})
    orders = pd.DataFrame(
        {
            # Months in the extract: Jan, Feb, Mar (C sells only in Feb).
            # A: demand [1, 0, 1]; B: [1, 0, 0]; C: [0, 1, 0].
            "sku": ["A", "A", "B", "C"],
            "order_date": ["2026-01-15", "2026-03-15", "2026-01-20", "2026-02-10"],
            "qty": [1, 1, 1, 1],
            "unit_price": [10.0, 10.0, 10.0, 10.0],
        }
    )
    seg = segment(items, orders)
    # A: CV ~0.7 with zero share 1/3 -> Y (not X).
    assert seg.loc[seg["sku"] == "A", "xyz_class"].iloc[0] == "Y"
    # B and C: zero share 2/3 -> Z.
    assert seg.loc[seg["sku"] == "B", "xyz_class"].iloc[0] == "Z"
    assert seg.loc[seg["sku"] == "C", "xyz_class"].iloc[0] == "Z"
