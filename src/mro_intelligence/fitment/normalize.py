"""Text normalization for failure and repair identity.

Work-order free text is the only failure vocabulary a distributor's ERP
carries (there is no failure-code column in a P21 export), so the graph keys
failures on normalized problem text. The evaluation harness applies the same
functions to ground-truth failure names, keeping both namespaces joinable.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_phrase(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _NON_ALNUM.sub(" ", text.strip().lower()).strip()


def failure_key(problem_desc: str) -> str:
    """Graph key for a failure, derived from the tech's problem description."""
    return normalize_phrase(problem_desc)


def repair_key(repair_desc: str) -> str:
    """Graph key for a repair action, derived from the repair description."""
    return normalize_phrase(repair_desc)
