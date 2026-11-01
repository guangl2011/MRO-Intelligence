"""Write a generated tenant bundle to disk.

Extracts are shaped by the shared ERP template (canonical -> source column
names + display date format); ground truth is written in canonical names with
ISO dates. The result is the standalone artifact a partner would hand over:
an extracts folder indistinguishable from a real SFTP drop, plus the hidden
ground truth the evaluation harness scores against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mro_intelligence.ingest.layout import DATE_COLUMNS, load_template_for_layout
from mro_intelligence.synthetic.tenant_data import TenantData

GROUND_TRUTH_ATTRS: dict[str, str] = {
    "sku_mapping": "gt_sku_mapping",
    "fitment": "gt_fitment",
    "xref": "gt_xref",
    "work_orders": "gt_work_orders",
}


def _format_dates(df: pd.DataFrame, spec_columns: dict[str, str], date_format: str) -> pd.DataFrame:
    out = df.copy()
    for canon, source in spec_columns.items():
        if canon in DATE_COLUMNS and source in out.columns and len(out) > 0:
            out[source] = pd.to_datetime(out[source]).dt.strftime(date_format)
    return out


def emit_tenant(data: TenantData, out_dir: Path) -> dict:
    """Write extracts + ground truth + manifest; return the manifest with paths."""
    template = load_template_for_layout(data.erp_layout)
    tenant_dir = out_dir / data.tenant_id
    erp_dir = tenant_dir / "extracts"
    gt_dir = tenant_dir / "ground_truth"
    erp_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for extract_name, spec in template.extracts.items():
        frame = getattr(data, extract_name)
        # Contract enforcement at the boundary: generator and template must agree
        # on the canonical column set, and file column order follows the template.
        if set(frame.columns) != set(spec.columns):
            raise ValueError(
                f"extract {extract_name}: frame columns {sorted(frame.columns)} "
                f"do not match template {sorted(spec.columns)}"
            )
        renamed = frame.rename(columns={c: s for c, s in spec.columns.items()})[
            list(spec.columns.values())
        ]
        renamed = _format_dates(renamed, spec.columns, spec.date_format)
        path = erp_dir / spec.file
        renamed.to_csv(path, index=False)
        files[extract_name] = str(path)

    for name, attr in GROUND_TRUTH_ATTRS.items():
        frame = getattr(data, attr)
        path = gt_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        files[f"gt_{name}"] = str(path)

    manifest = {**data.manifest, "files": files}
    manifest_path = tenant_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
