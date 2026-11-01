"""Catalog pipeline stage: extracts -> normalized catalog + merge plan + segmentation.

Single-tenant by construction: one bundle directory is one tenant; the
pipeline validates the tenant context and stamps outputs with the tenant id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mro_intelligence.catalog.dedup import MergePlan, propose_merges
from mro_intelligence.catalog.normalize import normalize_item_master
from mro_intelligence.catalog.segmentation import segment
from mro_intelligence.core.tenancy import TenantContext
from mro_intelligence.ingest.layout import load_template_for_layout

EXTRACTS = ("item_master", "order_lines", "inventory_monthly", "purchase_orders")


@dataclass(frozen=True)
class CatalogOutputs:
    tenant_id: str
    normalized_items: pd.DataFrame
    merge_plan: MergePlan
    segmentation: pd.DataFrame

    @property
    def stats(self) -> dict[str, float]:
        return dict(self.merge_plan.stats)


def load_extracts(bundle_root: Path, layout: str) -> dict[str, pd.DataFrame]:
    """Read one tenant's ERP extract bundle, reverse-mapped to canonical names."""
    template = load_template_for_layout(layout)
    erp_dir = bundle_root / "extracts"
    extracts: dict[str, pd.DataFrame] = {}
    for extract_name, spec in template.extracts.items():
        if extract_name not in EXTRACTS:
            continue
        source_to_canonical = {src: canon for canon, src in spec.columns.items()}
        frame = pd.read_csv(erp_dir / spec.file, dtype=str)
        frame = frame.rename(columns=source_to_canonical)
        for canon in spec.columns:
            if canon not in frame.columns:
                raise ValueError(f"{extract_name}: extract missing canonical column {canon}")
        extracts[extract_name] = frame
    return extracts


def run_catalog(
    extracts: dict[str, pd.DataFrame],
    tenant_id: str,
    *,
    auto_threshold: float | None = None,
    review_threshold: float | None = None,
) -> CatalogOutputs:
    """Normalize, detect duplicates, and segment one tenant's catalog."""
    context = TenantContext.of(tenant_id)
    items = normalize_item_master(extracts["item_master"])
    merge_kwargs: dict[str, float] = {}
    if auto_threshold is not None:
        merge_kwargs["auto_threshold"] = auto_threshold
    if review_threshold is not None:
        merge_kwargs["review_threshold"] = review_threshold
    merge_plan = propose_merges(items, **merge_kwargs)
    segmentation = segment(items, extracts["order_lines"].copy())
    return CatalogOutputs(
        tenant_id=context.tenant_id,
        normalized_items=items,
        merge_plan=merge_plan,
        segmentation=segmentation,
    )


def write_catalog(outputs: CatalogOutputs, out_dir: Path) -> dict[str, str]:
    """Persist canonical catalog, review queue, and segmentation as CSVs."""
    tenant_dir = out_dir / outputs.tenant_id
    tenant_dir.mkdir(parents=True, exist_ok=True)

    catalog = outputs.normalized_items[
        [
            "sku",
            "clean_description",
            "normalized_mpn",
            "normalized_brand",
            "category",
            "safety_flags",
            "attributes",
        ]
    ].merge(outputs.merge_plan.assignments, on="sku")
    catalog_path = tenant_dir / "canonical_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    review_path = tenant_dir / "review_queue.csv"
    pd.DataFrame(
        [
            {
                "task_id": t.task_id,
                "kind": t.kind,
                "sku_a": t.sku_a,
                "sku_b": t.sku_b,
                "score": t.score,
                "reason": t.reason,
            }
            for t in outputs.merge_plan.review_queue
        ]
    ).to_csv(review_path, index=False)

    seg_path = tenant_dir / "segmentation.csv"
    outputs.segmentation.to_csv(seg_path, index=False)

    return {
        "catalog": str(catalog_path),
        "review_queue": str(review_path),
        "segmentation": str(seg_path),
    }
