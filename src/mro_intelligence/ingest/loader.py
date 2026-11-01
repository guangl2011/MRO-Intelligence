"""Bundle loading: one tenant's ERP extract directory -> canonical frames.

The loader is shared by every pipeline stage. It reverse-maps the template's
ERP-native column names onto canonical ones and translates the template's
display dates to ISO -- a P21 export writes 07/28/2024; every downstream
stage sees 2024-07-28.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mro_intelligence.ingest.layout import (
    DATE_COLUMNS,
    load_template_for_layout,
)


def load_bundle_extracts(
    bundle_root: Path, layout: str, extract_names: tuple[str, ...]
) -> dict[str, pd.DataFrame]:
    """Read the requested extracts from ``bundle_root/extracts`` canonically.

    Raises ValueError when an extract or a declared canonical column is
    missing: a truncated export must fail loudly, not degrade silently.
    """
    template = load_template_for_layout(layout)
    erp_dir = bundle_root / "extracts"
    extracts: dict[str, pd.DataFrame] = {}
    for extract_name in extract_names:
        spec = template.extracts.get(extract_name)
        if spec is None:
            raise ValueError(f"layout {layout}: no extract named {extract_name}")
        source_to_canonical = {src: canon for canon, src in spec.columns.items()}
        frame = pd.read_csv(erp_dir / spec.file, dtype=str)
        frame = frame.rename(columns=source_to_canonical)
        missing = [canon for canon in spec.columns if canon not in frame.columns]
        if missing:
            raise ValueError(f"{extract_name}: extract missing canonical columns {missing}")
        for date_col in DATE_COLUMNS:
            if date_col in frame.columns:
                frame[date_col] = pd.to_datetime(
                    frame[date_col], format=template.date_format
                ).dt.strftime("%Y-%m-%d")
        extracts[extract_name] = frame
    return extracts
