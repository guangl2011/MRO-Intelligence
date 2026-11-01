"""Catalog normalization primitives for M1 Catalog Intelligence.

Pure text and attribute transformations: the messy-distributor mess (case
noise, abbreviation drift, supplier MPN styles, missing attributes) goes in,
a canonical form comes out. No I/O at this layer — callers own DataFrames.
"""

from __future__ import annotations

import json
import re
from typing import Final

import pandas as pd

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

# Safety-relevant tokens mirror the blueprint's prohibited-auto-substitution
# classes (gas-fired, A2L refrigerant, T&P relief, backflow prevention).
_SAFETY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "gas",
        re.compile(r"\bGAS\b|\bNATURAL\s+GAS\b|\bLP\b|\bPROPANE\b|\bFURNACE\b|\bBOILER\b", re.I),
    ),
    ("a2l_refrigerant", re.compile(r"\bA2L\b|\bR-?32\b|\bR-?454[ABC]?\b|\bREFRIGERANT\b", re.I)),
    (
        "tp_relief",
        re.compile(r"\bT\s*&\s*P\b|\bTEMP(?:ERATURE)?\s*&?\s*PRESSURE\b|\bRELIEF\b", re.I),
    ),
    ("backflow", re.compile(r"\bBACKFLOW\b|\bVAC(?:UUM)?\s*BREAKER\b|\bRPZ\b", re.I)),
)

# Labeled attribute extraction over the distributor description vocabulary.
_ATTRIBUTE_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("size", re.compile(r"\b(\d{1,2}x\d{1,2}x?\d{0,2})\b")),
    ("merv", re.compile(r"\bMERV\s*(\d{1,2})\b", re.I)),
    ("efficiency_pct", re.compile(r"\b(\d{2})PCT\b", re.I)),
    ("watts", re.compile(r"\b(\d{1,3})W\b")),
    ("voltage_v", re.compile(r"\b(\d{2,3})V\b")),
    ("amps", re.compile(r"\b(\d{1,3})A\b")),
    ("mfd", re.compile(r"\b(\d{1,2}\+\d{1,2})\b")),
    ("btu_k", re.compile(r"\b(\d{2,3})KBTU\b", re.I)),
    ("psi", re.compile(r"\b(\d{2,4})PSI\b", re.I)),
    ("pole", re.compile(r"\b(\d)POLE\b", re.I)),
    ("diameter_in", re.compile(r"\b(\d{1,2})-?ROUND\b", re.I)),
    ("length_in", re.compile(r"\b(\d{1,2})IN\b", re.I)),
)


def clean_text(text: str) -> str:
    """Collapse whitespace and strip stray padding."""
    return _WHITESPACE.sub(" ", text).strip()


def normalize_mpn(mpn: str) -> str:
    """Canonical MPN: uppercase, punctuation-stripped (supplier code styles)."""
    return _NON_ALNUM.sub("", mpn.upper())


def infer_safety_flags(text: str) -> frozenset[str]:
    """Safety-relevant flags present in free text (blueprint safety classes)."""
    return frozenset(name for name, pattern in _SAFETY_PATTERNS if pattern.search(text))


def extract_attributes(text: str) -> dict[str, str]:
    """Labeled (label, value) attributes found in a description."""
    attrs: dict[str, str] = {}
    for label, pattern in _ATTRIBUTE_PATTERNS:
        match = pattern.search(text)
        if match:
            attrs[label] = match.group(1).upper()
    return attrs


def normalize_item_master(item_master: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns; the input frame is not mutated."""
    out = item_master.copy()
    descriptions = out["description"].astype(str)
    out["clean_description"] = descriptions.map(clean_text)
    out["normalized_mpn"] = out["manufacturer_part_number"].astype(str).map(normalize_mpn)
    out["normalized_brand"] = out["brand"].astype(str).str.upper().str.strip()
    out["safety_flags"] = out["clean_description"].map(
        lambda t: ",".join(sorted(infer_safety_flags(t)))
    )
    out["attributes"] = out["clean_description"].map(
        lambda t: json.dumps(extract_attributes(t), sort_keys=True)
    )
    return out
