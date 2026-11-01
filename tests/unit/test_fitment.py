"""Unit tests for the M2 fitment graph: builder, bridge, prediction, harness."""

from __future__ import annotations

import pandas as pd
import pytest

from mro_intelligence.fitment.builder import bridge_vocabulary, build_edges, build_oem_index
from mro_intelligence.fitment.evaluate import (
    evaluate_fitment,
    resolve_lines,
    split_work_orders,
)
from mro_intelligence.fitment.graph import (
    SAFETY_CLASS_PROHIBITED_AUTO,
    SOURCE_OEM_CATALOG,
    SOURCE_WORK_ORDER,
    safety_class_for_flags,
)
from mro_intelligence.fitment.normalize import failure_key, normalize_phrase, repair_key
from mro_intelligence.fitment.predict import build_kit, rank_parts, rank_repairs


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["S1", "S2", "S3"],
            "normalized_mpn": ["MPN1", "MPN2", "MPN3"],
            "proposed_global_part_id": ["GP1", "GP2", "GP3"],
            "safety_flags": ["none", "none", "gas"],
        }
    )


def _wo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wo_id": ["W1", "W2", "W3"],
            "model_id": ["M1", "M1", "M2"],
            "equipment_model": ["KELVAIR KE-1", "KELVAIR KE-1", "AEROVUE AE-7"],
            "failure_reported": ["Fan motor seized", "fan motor seized", "Gas valve stuck"],
            "repair_action": ["Replace fan motor", "Replace fan motor", "Replace gas valve"],
            "part_sku": ["S1", "S1", "S3"],
            "confirmed_date": ["2026-01-01", "2026-01-05", "2026-01-09"],
            "technician_id": ["T1", "T2", "T3"],
        }
    )


def _oem() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "evidence_id": ["E1", "E2", "E3"],
            "model_id": ["M1", "M1", "M2"],
            "failure_code": ["F-FAN", "F-GAS", "F-GV"],
            "oem_part_number": ["MPN2", "MPN3", "MPN3"],
            "equipment_make": ["KELVAIR", "KELVAIR", "AEROVUE"],
            "equipment_model_no": ["KE-1", "KE-1", "AE-7"],
        }
    )


def test_normalize_phrase_strips_punctuation_and_case() -> None:
    assert normalize_phrase("  T&P valve WEEPING ") == "t p valve weeping"
    assert failure_key("Ignitor worn / no glow") == "ignitor worn no glow"
    assert repair_key("Replace TXV and line set") == "replace txv and line set"


def test_safety_flags_map_to_prohibited_auto() -> None:
    assert safety_class_for_flags(set()) == "none"
    assert safety_class_for_flags({"gas"}) == SAFETY_CLASS_PROHIBITED_AUTO
    assert safety_class_for_flags({"a2l_refrigerant", "backflow"}) == SAFETY_CLASS_PROHIBITED_AUTO


def test_builder_aggregates_support_and_evidence() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    wo_edges = [e for e in edges if e.source == SOURCE_WORK_ORDER]
    # W1 and W2 are the same (model, failure, repair, part) claim -> support 2.
    fan = [e for e in wo_edges if e.model_id == "M1"]
    assert len(fan) == 1
    assert fan[0].support == 2
    assert fan[0].part_ref == "GP1"
    assert fan[0].evidence_refs == ("W1", "W2")
    assert fan[0].failure_key == "fan motor seized"
    # The gas-valve edge inherits the part's safety class.
    gas = next(e for e in wo_edges if e.model_id == "M2")
    assert gas.safety_class == SAFETY_CLASS_PROHIBITED_AUTO
    # OEM listing rows resolve MPNs into parts and keep the supplier code.
    oem_edges = [e for e in edges if e.source == SOURCE_OEM_CATALOG]
    assert {e.failure_code for e in oem_edges} == {"F-FAN", "F-GAS", "F-GV"}
    assert {e.part_ref for e in oem_edges} == {"GP2", "GP3"}


def test_oem_index_drops_unresolvable_mpn() -> None:
    oem = pd.concat(
        [
            _oem(),
            pd.DataFrame(
                [
                    {
                        "evidence_id": "E9",
                        "model_id": "M9",
                        "failure_code": "F-X",
                        "oem_part_number": "GHOST-MPN",
                        "equipment_make": "X",
                        "equipment_model_no": "X-1",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    mpn_to_part = {
        str(m): p
        for m, p in _catalog()[["normalized_mpn", "proposed_global_part_id"]].itertuples(
            index=False, name=None
        )
    }
    index = build_oem_index(oem, mpn_to_part)
    assert all(model != "M9" for model, _ in index)


def test_bridge_maps_code_to_text_by_part_overlap() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    wo_edges = [e for e in edges if e.source == SOURCE_WORK_ORDER]
    mpn_to_part = {
        str(m): p
        for m, p in _catalog()[["normalized_mpn", "proposed_global_part_id"]].itertuples(
            index=False, name=None
        )
    }
    index = build_oem_index(_oem(), mpn_to_part)
    bridge = bridge_vocabulary(wo_edges, index)
    # F-GV lists MPN3->GP3; M2's work orders used exactly GP3 -> 1/1 overlap,
    # bridged to the full text key (model-scoped, like every TextKey).
    assert bridge[("M2", "F-GV")] == ("M2", "gas valve stuck")
    # F-FAN lists GP2 but M1's work orders only used GP1 -> 0 overlap.
    assert ("M1", "F-FAN") not in bridge


def test_rank_parts_prefers_empirical_and_flags_safety() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    ranked = rank_parts(edges, "M2", "gas valve stuck", top_k=3)
    assert isinstance(ranked, list)
    assert ranked[0].part_ref == "GP3"
    assert ranked[0].requires_technician is True


def test_rank_parts_abstains_without_evidence() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    result = rank_parts(edges, "M9", "unknown failure", top_k=3)
    assert not isinstance(result, list)
    assert "no evidence" in result.reason


def test_rank_repairs_orders_by_support() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    ranked = rank_repairs(edges, "M1", "fan motor seized", top_k=3)
    assert isinstance(ranked, list)
    assert ranked[0].repair_key == "replace fan motor"
    assert ranked[0].support == 2


def test_build_kit_returns_up_to_size() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    kit = build_kit(edges, "M1", "fan motor seized", size=5)
    assert isinstance(kit, list)
    assert 1 <= len(kit) <= 5


def test_resolve_lines_maps_skus_and_flags_wrong_parts() -> None:
    flags = pd.DataFrame({"wo_id": ["W2"], "used_wrong_part": ["True"]})
    lines = resolve_lines(_wo(), _catalog(), flags)
    assert len(lines) == 3
    wrong = [ln for ln in lines if ln.is_wrong_part]
    assert len(wrong) == 1
    assert wrong[0].part_ref == "GP1"


def test_split_is_chronological() -> None:
    train, eval_df = split_work_orders(_wo(), train_share=2 / 3)
    assert list(train["wo_id"]) == ["W1", "W2"]
    assert list(eval_df["wo_id"]) == ["W3"]


def test_compatibility_universe_translates_truth_into_catalog_namespace() -> None:
    from mro_intelligence.fitment.pipeline import _compatibility_universe

    truth = pd.DataFrame(
        {"model_id": ["M2"], "failure_code": ["F-GV"], "part_global_id": ["GP-TRUE-9"]}
    )
    sku_mapping = pd.DataFrame(
        {"sku": ["S3", "S2"], "global_part_id": ["GP-TRUE-9", "GP-TRUE-OTHER"]}
    )
    catalog = _catalog()
    universe = _compatibility_universe(truth, sku_mapping, catalog)
    # S3 resolves to GP3 in the catalog; S2 belongs to another true part.
    assert universe == {("M2", "F-GV"): {"GP3"}}
    # A true part with no resolvable SKUs contributes nothing.
    orphan = _compatibility_universe(
        pd.DataFrame({"model_id": ["M9"], "failure_code": ["F-X"], "part_global_id": ["GP-GHOST"]}),
        sku_mapping,
        catalog,
    )
    assert orphan == {}


def test_evaluate_counts_hits_against_held_out_lines() -> None:
    edges = build_edges(_wo(), _catalog(), _oem())
    truth = pd.DataFrame(
        {
            "wo_id": ["W3"],
            "used_wrong_part": ["False"],
            "failure_code": ["F-GV"],
        }
    )
    lines = resolve_lines(_wo().iloc[2:3], _catalog(), truth)
    universe = {("M2", "F-GV"): {"GP1", "GP3"}}
    metrics = evaluate_fitment(edges, edges, lines, universe)
    row = metrics.to_row()
    # W3 used GP3 for "gas valve stuck": the graph's top-3 contains it, and
    # every recommended part is in the true compatible set for (M2, F-GV).
    assert row["top3_compatibility"] == 1.0
    assert row["part_top3_recall"] == 1.0
    assert row["repair_top3"] == 1.0


def test_evaluate_zero_queries_reports_zero_safely() -> None:
    metrics = evaluate_fitment([], [], [])
    assert metrics.to_row()["top3_compatibility"] == 0.0
    assert metrics.to_row()["kit_uplift"] == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Condenser fan motor seized", "condenser fan motor seized"),
        ("Blower motor failure", "blower motor failure"),
    ],
)
def test_failure_keys_are_stable(text: str, expected: str) -> None:
    assert failure_key(text) == expected
