"""Versioned fitment edge schema.

Every edge is a proposition with evidence, never a bare link (blueprint §6):
an equipment model + failure can be repaired with a part, and the edge carries
where that claim came from, how often it was observed, a calibrated
confidence, a safety class, and a review state that a human can change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Safety classes mirror the canonical catalog's inferred flags. Any part with
# a restricted flag can never drive auto-substitution (blueprint §6/§8); the
# edge stays usable for guided (human-confirmed) repair.
SAFETY_CLASS_NONE = "none"
SAFETY_CLASS_RESTRICTED = "restricted"
SAFETY_CLASS_PROHIBITED_AUTO = "prohibited-auto"

REVIEW_INFERRED = "inferred"
REVIEW_HUMAN_APPROVED = "human-approved"
REVIEW_REJECTED = "rejected"

SOURCE_WORK_ORDER = "work-order"
SOURCE_OEM_CATALOG = "oem-catalog"

EDGE_COLUMNS = (
    "edge_id",
    "model_id",
    "model_name",
    "failure_key",
    "failure_name",
    "failure_code",
    "repair_key",
    "repair_name",
    "part_ref",
    "source",
    "support",
    "confidence",
    "safety_class",
    "review_state",
    "version",
    "evidence_refs",
)


@dataclass(frozen=True)
class FitmentEdge:
    """One equipment + failure + repair + part claim with evidence."""

    model_id: str
    model_name: str
    failure_key: str
    failure_name: str
    repair_key: str
    repair_name: str
    part_ref: str
    source: str  # SOURCE_WORK_ORDER or SOURCE_OEM_CATALOG
    support: int  # distinct work orders (or listing rows) backing the claim
    confidence: float
    safety_class: str = SAFETY_CLASS_NONE
    review_state: str = REVIEW_INFERRED
    version: int = 1
    failure_code: str = ""  # supplier code for OEM-listing edges, else empty
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def edge_id(self) -> str:
        """Stable identity of the claim (version lives in the version field)."""
        return f"{self.model_id}|{self.failure_key}|{self.repair_key}|{self.part_ref}|{self.source}"

    def to_row(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "failure_key": self.failure_key,
            "failure_name": self.failure_name,
            "failure_code": self.failure_code,
            "repair_key": self.repair_key,
            "repair_name": self.repair_name,
            "part_ref": self.part_ref,
            "source": self.source,
            "support": self.support,
            "confidence": round(self.confidence, 4),
            "safety_class": self.safety_class,
            "review_state": self.review_state,
            "version": self.version,
            "evidence_refs": ";".join(self.evidence_refs[:10]),
        }


def safety_map_by_part(canonical_catalog: pd.DataFrame) -> dict[str, str]:
    """Map canonical part id -> safety class derived from catalog safety flags."""
    return {
        str(part): safety_class_for_flags({f for f in _flag_tokens(flags)})
        for part, flags in canonical_catalog[["proposed_global_part_id", "safety_flags"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }


def _flag_tokens(flags: object) -> list[str]:
    """Safety tokens from a catalog cell; NaN/blank cells mean no flags."""
    if not isinstance(flags, str):
        return []
    return [f for f in flags.split(",") if f and f != "none"]


def safety_class_for_flags(flags: set[str]) -> str:
    """Map inferred safety flags to an edge safety class."""
    if not flags:
        return SAFETY_CLASS_NONE
    return SAFETY_CLASS_PROHIBITED_AUTO
