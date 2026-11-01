"""ERP extract layout templates.

One YAML template per ERP (Epicor Prophet 21, Epicor Eclipse, Infor) maps
canonical extract columns onto source-file column names. Both sides of the
boundary share these files:

- the synthetic generator writes extracts shaped by the template, and
- the ingest layer reads customer extracts shaped by the same template.

A new ERP deployment means adding one template file — no code.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

import yaml

# Canonical columns that carry dates and therefore need format translation
# between the template's display format and canonical ISO dates.
DATE_COLUMNS: frozenset[str] = frozenset(
    {
        "order_date",
        "return_date",
        "confirmed_date",
        "receipt_date",
        "month",
    }
)


@dataclass(frozen=True)
class ExtractSpec:
    """One extract file in an ERP layout."""

    name: str
    file: str
    columns: dict[str, str]  # canonical -> source column name
    date_format: str


@dataclass(frozen=True)
class Template:
    erp: str
    date_format: str
    extracts: dict[str, ExtractSpec]


def load_template(erp: str) -> Template:
    """Load the extract template for an ERP identifier (file name without .yaml)."""
    base = resources.files("mro_intelligence.ingest.templates") / f"{erp}.yaml"
    raw: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8"))
    date_format = raw.get("date_format", "%Y-%m-%d")
    extracts = {
        name: ExtractSpec(
            name=name,
            file=spec["file"],
            columns=dict(spec["columns"]),
            date_format=str(spec.get("date_format", date_format)),
        )
        for name, spec in raw["extracts"].items()
    }
    return Template(erp=raw.get("erp", erp), date_format=date_format, extracts=extracts)


def canonical_columns(spec: ExtractSpec) -> list[str]:
    """Canonical column order for an extract."""
    return list(spec.columns.keys())


# Dataset profiles refer to ERP layouts by short id; template files use the
# full ERP identifier. New ERP = new template file + one entry here.
TEMPLATE_BY_LAYOUT: dict[str, str] = {
    "p21": "epicor_prophet21",
    "eclipse": "epicor_eclipse",
    "infor": "infor_distribution",
}


def load_template_for_layout(layout: str) -> Template:
    """Resolve a profile's short layout id (p21/eclipse/infor) to its template."""
    try:
        return load_template(TEMPLATE_BY_LAYOUT[layout])
    except KeyError:
        known = ", ".join(sorted(TEMPLATE_BY_LAYOUT))
        raise KeyError(f"unknown ERP layout {layout!r}; known layouts: {known}") from None


def source_column_map(spec: ExtractSpec) -> dict[str, str]:
    """Mapping canonical -> source column for rename operations."""
    return dict(spec.columns)
