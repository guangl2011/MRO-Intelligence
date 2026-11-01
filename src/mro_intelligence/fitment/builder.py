"""Build fitment edges from work-order history and OEM compatibility listings.

Two evidence sources feed the graph (blueprint §6 evidence classes):
- work orders: empirical claims -- what a tech actually used on this model
  for this failure (noisy: a small share of wrong parts, diluted by support).
- OEM compatibility listings: documented claims -- what the manufacturer's
  cross-reference says fits (sparse: a subset of compatible parts).

The two sources use different failure vocabularies: work orders carry the
technician's problem text, OEM listings carry supplier failure codes. The
bridge resolves them per model by shared part co-occurrence -- if most of a
code's listed parts show up in work orders under one problem text, the code
and the text describe the same failure. Unbridged listing edges stay in the
graph under their code key, flagged for review, and never feed predictions.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from mro_intelligence.catalog.normalize import normalize_mpn
from mro_intelligence.fitment.graph import (
    SOURCE_OEM_CATALOG,
    SOURCE_WORK_ORDER,
    FitmentEdge,
    safety_map_by_part,
)
from mro_intelligence.fitment.normalize import failure_key, repair_key

CodeKey = tuple[str, str]
"""(model_id, supplier failure code)."""

TextKey = tuple[str, str]
"""(model_id, normalized failure text)."""

WoAggKey = tuple[str, str, str, str]
"""(model_id, failure text key, repair text key, tenant sku)."""


def _aggregate_work_orders(
    work_orders: pd.DataFrame,
) -> tuple[dict[WoAggKey, set[str]], dict[TextKey, str]]:
    agg: dict[WoAggKey, set[str]] = defaultdict(set)
    names: dict[TextKey, str] = {}
    for wo_no, model_id, problem, repair, part in work_orders[
        ["wo_id", "model_id", "failure_reported", "repair_action", "part_sku"]
    ].itertuples(index=False, name=None):
        fkey = failure_key(str(problem))
        agg[(str(model_id), fkey, repair_key(str(repair)), str(part))].add(str(wo_no))
        names[(str(model_id), fkey)] = str(problem)
    return agg, names


def _model_name(work_orders: pd.DataFrame, model_id: str) -> str:
    rows = work_orders.loc[work_orders["model_id"] == model_id, "equipment_model"]
    return str(rows.iloc[0]) if len(rows) else model_id


def _wo_edges(
    work_orders: pd.DataFrame,
    sku_to_part: dict[str, str],
    safety_by_part: dict[str, str],
    wo_confidence_k: int,
) -> list[FitmentEdge]:
    agg, failure_names = _aggregate_work_orders(work_orders)
    edges: list[FitmentEdge] = []
    for (model_id, fkey, rkey, sku), wo_ids in sorted(agg.items()):
        part_ref = sku_to_part.get(sku)
        if part_ref is None:
            # Work-order SKU missing from the canonical catalog: the catalog
            # stage owns that record; the graph does not invent a part id.
            continue
        edges.append(
            FitmentEdge(
                model_id=model_id,
                model_name=_model_name(work_orders, model_id),
                failure_key=fkey,
                failure_name=failure_names.get((model_id, fkey), fkey),
                repair_key=rkey,
                repair_name=rkey,
                part_ref=part_ref,
                source=SOURCE_WORK_ORDER,
                support=len(wo_ids),
                confidence=len(wo_ids) / (len(wo_ids) + wo_confidence_k),
                safety_class=safety_by_part.get(part_ref, "none"),
                evidence_refs=tuple(sorted(wo_ids)),
            )
        )
    return edges


def build_oem_index(
    oem_compat: pd.DataFrame, mpn_to_part: dict[str, str]
) -> dict[CodeKey, list[tuple[str, tuple[str, ...]]]]:
    """Group listing rows into (model, code) -> [(part_ref, ev_ids)].

    Rows whose MPN does not resolve into the tenant's catalog are dropped:
    the listing references the OEM part universe, of which the tenant stocks
    a subset -- unresolvable rows carry no actionable claim for this tenant.
    """
    index: dict[CodeKey, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for (model_id, failure_code, mpn), group in oem_compat.groupby(
        ["model_id", "failure_code", "oem_part_number"], dropna=False
    ):
        part_ref = mpn_to_part.get(normalize_mpn(str(mpn)))
        if part_ref is None:
            continue
        index[(str(model_id), str(failure_code))].append(
            (part_ref, tuple(group["evidence_id"].astype(str)))
        )
    return index


def bridge_vocabulary(
    wo_edges: list[FitmentEdge], oem_index: dict[CodeKey, list[tuple[str, tuple[str, ...]]]]
) -> dict[CodeKey, TextKey]:
    """Resolve supplier failure codes onto work-order failure text per model.

    Overlap score for (code -> text) is the share of the code's listed parts
    observed in work orders under that text. A code maps to its best text
    when at least ``min_overlap`` of its listed parts co-occur; otherwise the
    code stays unbridged (its edges remain review-only).
    """
    wo_parts: dict[TextKey, set[str]] = defaultdict(set)
    for edge in wo_edges:
        wo_parts[(edge.model_id, edge.failure_key)].add(edge.part_ref)

    bridge: dict[CodeKey, TextKey] = {}
    for code_key, part_entries in oem_index.items():
        listed = {part for part, _ in part_entries}
        model_id, _code = code_key
        best_key, best_score = None, 0.0
        for (wo_model, text_key), observed in wo_parts.items():
            if wo_model != model_id:
                continue
            score = len(listed & observed) / len(listed) if listed else 0.0
            if score > best_score:
                best_key, best_score = (model_id, text_key), score
        if best_key is not None and best_score >= _MIN_BRIDGE_OVERLAP:
            bridge[code_key] = best_key
    return bridge


_MIN_BRIDGE_OVERLAP = 0.5


def _oem_edges(
    oem_index: dict[CodeKey, list[tuple[str, tuple[str, ...]]]],
    bridge: dict[CodeKey, TextKey],
    text_names: dict[TextKey, str],
    model_names: dict[str, str],
    safety_by_part: dict[str, str],
    oem_confidence_k: int,
) -> list[FitmentEdge]:
    edges: list[FitmentEdge] = []
    for code_key, part_entries in sorted(oem_index.items()):
        model_id, code = code_key
        text_key = bridge.get(code_key)
        support = sum(len(ev) for _, ev in part_entries)
        for part_ref, ev_ids in part_entries:
            if text_key is None:
                # Unbridged listing: keep the claim reviewable under the code
                # vocabulary, but predictions never surface it (abstention).
                fkey, fname = failure_key(f"code {code}"), str(code)
                state = "needs-review"
            else:
                fkey, fname = text_key[1], text_names.get(text_key, text_key[1])
                state = "inferred"
            edges.append(
                FitmentEdge(
                    model_id=model_id,
                    model_name=model_names.get(model_id, model_id),
                    failure_key=fkey,
                    failure_name=fname,
                    repair_key="oem-listing",
                    repair_name="OEM compatibility listing",
                    part_ref=part_ref,
                    source=SOURCE_OEM_CATALOG,
                    support=support,
                    confidence=support / (support + oem_confidence_k),
                    safety_class=safety_by_part.get(part_ref, "none"),
                    review_state=state,
                    failure_code=code,
                    evidence_refs=ev_ids,
                )
            )
    return edges


def build_edges(
    work_orders: pd.DataFrame,
    canonical_catalog: pd.DataFrame,
    oem_compat: pd.DataFrame,
    *,
    wo_confidence_k: int = 2,
    oem_confidence_k: int = 1,
) -> list[FitmentEdge]:
    """Aggregate raw extracts into evidence-carrying fitment edges."""
    sku_to_part = {
        str(sku): str(part)
        for sku, part in canonical_catalog[["sku", "proposed_global_part_id"]].itertuples(
            index=False, name=None
        )
    }
    safety_by_part = safety_map_by_part(canonical_catalog)
    mpn_to_part = {
        str(mpn).strip().upper(): str(part)
        for mpn, part in canonical_catalog[["normalized_mpn", "proposed_global_part_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
        if str(mpn).strip()
    }

    wo_edges = _wo_edges(work_orders, sku_to_part, safety_by_part, wo_confidence_k)
    oem_index = build_oem_index(oem_compat, mpn_to_part)
    _, text_names = _aggregate_work_orders(work_orders)
    model_names = {
        str(mid): _model_name(work_orders, str(mid)) for mid in work_orders["model_id"].unique()
    }
    bridge = bridge_vocabulary(wo_edges, oem_index)
    return [
        *wo_edges,
        *_oem_edges(
            oem_index,
            bridge,
            text_names,
            model_names,
            safety_by_part,
            oem_confidence_k,
        ),
    ]
