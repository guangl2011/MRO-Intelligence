"""Fitment pipeline stage: work orders + OEM listings -> versioned graph + metrics.

Held-out protocol (S1 §5): build the graph on the earliest 75% of work
orders plus the full OEM listing; score on the remaining 25%. The OEM-only
variant of the graph is the baseline for the kit-coverage uplift metric --
identical queries and parts, only the evidence differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mro_intelligence.catalog.normalize import normalize_mpn
from mro_intelligence.core.tenancy import TenantContext
from mro_intelligence.fitment.builder import build_edges, build_oem_index
from mro_intelligence.fitment.evaluate import (
    FitmentMetrics,
    evaluate_fitment,
    resolve_lines,
    split_work_orders,
)
from mro_intelligence.fitment.graph import (
    EDGE_COLUMNS,
    SOURCE_OEM_CATALOG,
    FitmentEdge,
    safety_map_by_part,
)
from mro_intelligence.ingest.loader import load_bundle_extracts

EXTRACTS = ("work_orders", "oem_fitment_catalog")


@dataclass(frozen=True)
class FitmentOutputs:
    tenant_id: str
    edges: list[FitmentEdge]
    metrics: FitmentMetrics | None


def run_fitment(
    extracts: dict[str, pd.DataFrame],
    canonical_catalog: pd.DataFrame,
    tenant_id: str,
    *,
    wrong_part_flags: pd.DataFrame | None = None,
    compatibility_truth: pd.DataFrame | None = None,
    sku_mapping: pd.DataFrame | None = None,
) -> FitmentOutputs:
    """Build the fitment graph for one tenant and, when truth is available,
    score the S1 §5 acceptance metrics on held-out work orders."""
    context = TenantContext.of(tenant_id)
    work_orders = extracts["work_orders"]
    oem_compat = extracts["oem_fitment_catalog"]

    train_wos, eval_wos = split_work_orders(work_orders)
    edges = build_edges(train_wos, canonical_catalog, oem_compat)

    # The baseline graph drops failure reasoning entirely: same listings,
    # same catalog resolution, answered at equipment level.
    mpn_to_part = _mpn_to_part(canonical_catalog)
    oem_only_edges = _oem_only_edges(oem_compat, mpn_to_part, canonical_catalog)

    metrics: FitmentMetrics | None = None
    if wrong_part_flags is not None:
        lines = resolve_lines(eval_wos, canonical_catalog, wrong_part_flags)
        compatibility = (
            _compatibility_universe(compatibility_truth, sku_mapping, canonical_catalog)
            if compatibility_truth is not None and sku_mapping is not None
            else None
        )
        metrics = evaluate_fitment(edges, oem_only_edges, lines, compatibility)
    return FitmentOutputs(tenant_id=context.tenant_id, edges=edges, metrics=metrics)


def _compatibility_universe(
    truth: pd.DataFrame,
    sku_mapping: pd.DataFrame,
    canonical_catalog: pd.DataFrame,
) -> dict[tuple[str, str], set[str]]:
    """True (model, failure_code) -> compatible parts, in the catalog's namespace.

    Hidden truth speaks the generator's global part ids; the graph speaks M1's
    proposed part ids. The bridge runs each true part's tenant SKUs through
    the catalog mapping -- a recommendation counts as compatible when it
    resolves to any cluster those SKUs landed in.
    """
    sku_to_prop = dict(
        zip(
            canonical_catalog["sku"].astype(str),
            canonical_catalog["proposed_global_part_id"].astype(str),
            strict=True,
        )
    )
    true_to_props: dict[str, set[str]] = {}
    for sku, true_id in sku_mapping[["sku", "global_part_id"]].itertuples(index=False, name=None):
        prop = sku_to_prop.get(str(sku))
        if prop is not None:
            true_to_props.setdefault(str(true_id), set()).add(prop)

    universe: dict[tuple[str, str], set[str]] = {}
    for model_id, failure_code, part in truth[
        ["model_id", "failure_code", "part_global_id"]
    ].itertuples(index=False, name=None):
        props = true_to_props.get(str(part), set())
        if props:
            universe.setdefault((str(model_id), str(failure_code)), set()).update(props)
    return universe


def _mpn_to_part(canonical_catalog: pd.DataFrame) -> dict[str, str]:
    """Canonical MPN -> part id, using the catalog's own MPN normalization so
    dashed OEM listings resolve against punctuation-stripped catalog MPNs."""
    return {
        normalize_mpn(str(mpn)): str(part)
        for mpn, part in canonical_catalog[["normalized_mpn", "proposed_global_part_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
        if str(mpn).strip()
    }


def _oem_only_edges(
    oem_compat: pd.DataFrame,
    mpn_to_part: dict[str, str],
    canonical_catalog: pd.DataFrame,
) -> list[FitmentEdge]:
    """Catalog-only baseline: what the OEM listing says fits the EQUIPMENT.

    The baseline has no failure reasoning -- it knows which parts fit the
    model, not which failure needs which. Its edges live at model level
    (empty failure key), so the same prediction code answers the same
    queries; the uplift measures exactly what failure-specific evidence adds.
    """
    index = build_oem_index(oem_compat, mpn_to_part)
    safety_by_part = safety_map_by_part(canonical_catalog)
    model_names = _model_names(oem_compat)

    support_by_part: dict[tuple[str, str], int] = {}
    evidence: dict[tuple[str, str], list[str]] = {}
    for (model_id, _code), part_entries in index.items():
        for part_ref, ev_ids in part_entries:
            key = (model_id, part_ref)
            support_by_part[key] = support_by_part.get(key, 0) + len(ev_ids)
            evidence.setdefault(key, []).extend(ev_ids)

    edges: list[FitmentEdge] = []
    for (model_id, part_ref), support in sorted(support_by_part.items()):
        edges.append(
            FitmentEdge(
                model_id=model_id,
                model_name=model_names.get(model_id, model_id),
                failure_key="",
                failure_name="catalog-only baseline (equipment level)",
                repair_key="oem-listing",
                repair_name="OEM compatibility listing",
                part_ref=part_ref,
                source=SOURCE_OEM_CATALOG,
                support=support,
                confidence=support / (support + 1),
                safety_class=safety_by_part.get(part_ref, "none"),
                evidence_refs=tuple(evidence[(model_id, part_ref)]),
            )
        )
    return edges


def _model_names(oem_compat: pd.DataFrame) -> dict[str, str]:
    listing_cols = ["model_id", "equipment_make", "equipment_model_no"]
    names: dict[str, str] = {}
    for model_id, make, model_no in (
        oem_compat[listing_cols].drop_duplicates().itertuples(index=False, name=None)
    ):
        names[str(model_id)] = f"{make} {model_no}"
    return names


def write_fitment(outputs: FitmentOutputs, out_dir: Path) -> dict[str, str]:
    """Persist edges and (when present) metrics as CSVs."""
    tenant_dir = out_dir / outputs.tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)
    edges_path = tenant_dir / "fitment_edges.csv"
    pd.DataFrame([edge.to_row() for edge in outputs.edges], columns=EDGE_COLUMNS).to_csv(
        edges_path, index=False
    )
    paths = {"edges": str(edges_path)}
    if outputs.metrics is not None:
        metrics_path = tenant_dir / "fitment_metrics.csv"
        pd.DataFrame([outputs.metrics.to_row()]).to_csv(metrics_path, index=False)
        paths["metrics"] = str(metrics_path)
    return paths


def load_fitment_extracts(bundle_root: Path, layout: str) -> dict[str, pd.DataFrame]:
    """Read the fitment stage's extracts canonically (dates in ISO)."""
    return load_bundle_extracts(bundle_root, layout, EXTRACTS)
