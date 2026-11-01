"""Duplicate-SKU detection for M1 Catalog Intelligence.

Design: cheap blocking, then pairwise scoring, then union-find clustering —
with a hard safety gate. SKUs whose inferred safety flags conflict are never
merged, automatically or into a review queue, because a cross-safety merge is
a prohibited-substitution hazard, not a judgment call (blueprint: safety
constraints live in the data model).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

# Blocks larger than this skip description-blocking (quadratic blowup on
# generic-brand catalogs); exact-MPN blocking still covers them.
MAX_BLOCK = 400
AUTO_MERGE_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.78


@dataclass(frozen=True)
class PairScore:
    sku_a: str
    sku_b: str
    score: float
    mpn_match: bool
    attribute_jaccard: float
    description_jaccard: float


@dataclass(frozen=True)
class ReviewTask:
    task_id: str
    kind: str
    sku_a: str
    sku_b: str
    score: float
    reason: str


@dataclass(frozen=True)
class MergePlan:
    """Merge decisions: sku -> proposed global part id, plus audit trails."""

    assignments: pd.DataFrame  # sku, proposed_global_part_id, decision
    review_queue: tuple[ReviewTask, ...]
    rejected_pairs: tuple[tuple[str, str, str], ...]
    stats: dict[str, float] = field(default_factory=dict)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(text.upper().split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        # Absence of evidence is not evidence of sameness: two records with
        # no extracted attributes are not scored as identical.
        return 0.0
    return len(a & b) / len(a | b)


def _attribute_pairs(attrs_json: str) -> frozenset[str]:
    attrs = json.loads(attrs_json)
    return frozenset(f"{k}={v}" for k, v in attrs.items())


def block_candidates(items: pd.DataFrame) -> list[tuple[int, int]]:
    """Candidate index pairs: exact normalized-MPN groups plus small
    (brand, category) blocks for MPN-corrupted duplicates."""
    candidates: set[tuple[int, int]] = set()

    by_mpn: dict[str, list[int]] = defaultdict(list)
    for idx, mpn in enumerate(items["normalized_mpn"]):
        by_mpn[str(mpn)].append(idx)
    for indexes in by_mpn.values():
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                candidates.add((indexes[i], indexes[j]))

    by_block: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(items.itertuples(index=False)):
        by_block[(row.normalized_brand, str(row.category).upper())].append(idx)
    for indexes in by_block.values():
        if len(indexes) > MAX_BLOCK:
            continue
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                candidates.add((indexes[i], indexes[j]))

    return sorted(candidates)


def score_pair(items: pd.DataFrame, i: int, j: int) -> PairScore:
    """Deterministic similarity for one candidate pair."""
    row_a, row_b = items.iloc[i], items.iloc[j]
    mpn_a, mpn_b = str(row_a["normalized_mpn"]), str(row_b["normalized_mpn"])
    mpn_match = bool(mpn_a) and mpn_a == mpn_b
    # Two present-but-different MPNs are counter-evidence: the same attributes
    # with different part numbers is a variant, not a duplicate.
    mpn_conflict = bool(mpn_a) and bool(mpn_b) and mpn_a != mpn_b
    brand_match = row_a["normalized_brand"] == row_b["normalized_brand"]
    attr_j = _jaccard(
        _attribute_pairs(str(row_a["attributes"])), _attribute_pairs(str(row_b["attributes"]))
    )
    desc_j = _jaccard(
        _tokens(str(row_a["clean_description"])), _tokens(str(row_b["clean_description"]))
    )
    base = 0.5 * attr_j + 0.5 * desc_j
    if mpn_match and brand_match:
        score = max(base, 0.93)
    elif mpn_match:
        score = max(base, 0.85)
    elif mpn_conflict:
        # Cap just under the auto-merge threshold: strongly similar pairs stay
        # in the review band, where a human decides with the evidence in front
        # of them; weak pairs keep their true base score and stay out.
        score = min(base, AUTO_MERGE_THRESHOLD - 0.01)
    else:
        score = base
    return PairScore(
        sku_a=str(row_a["sku"]),
        sku_b=str(row_b["sku"]),
        score=round(score, 4),
        mpn_match=mpn_match,
        attribute_jaccard=round(attr_j, 4),
        description_jaccard=round(desc_j, 4),
    )


def _safety_conflict(items: pd.DataFrame, i: int, j: int) -> bool:
    """Conflicting safety flags are a prohibited-merge hazard."""
    flags_a = set(str(items.iloc[i]["safety_flags"]).split(",")) - {""}
    flags_b = set(str(items.iloc[j]["safety_flags"]).split(",")) - {""}
    # Any difference in safety-relevant classification blocks the merge.
    return flags_a != flags_b


def propose_merges(
    items: pd.DataFrame,
    auto_threshold: float = AUTO_MERGE_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> MergePlan:
    """Cluster candidate pairs into merge decisions.

    - score >= auto_threshold: auto-merge edge (still recorded as a decision)
    - review_threshold <= score < auto_threshold: review queue (human approves)
    - conflicting safety flags: rejected outright, never merged
    """
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    review_queue: list[ReviewTask] = []
    rejected: list[tuple[str, str, str]] = []
    for i, j in block_candidates(items):
        pair = score_pair(items, i, j)
        if _safety_conflict(items, i, j):
            if pair.score >= review_threshold:
                rejected.append((pair.sku_a, pair.sku_b, "safety_conflict"))
            continue
        if pair.score < review_threshold:
            continue
        if pair.score < auto_threshold:
            review_queue.append(
                ReviewTask(
                    task_id=f"RV-{len(review_queue) + 1:05d}",
                    kind="duplicate_merge",
                    sku_a=pair.sku_a,
                    sku_b=pair.sku_b,
                    score=pair.score,
                    reason="similarity in review band",
                )
            )
            continue
        union(i, j)

    # Singletons keep their own part id; clusters share the root SKU's id.
    root_to_part: dict[int, str] = {}
    assignments: list[dict[str, str]] = []
    for idx, sku in enumerate(items["sku"]):
        root = find(idx)
        part_id = root_to_part.setdefault(root, f"GP-{sku!s}")
        assignments.append(
            {"sku": str(sku), "proposed_global_part_id": part_id, "decision": "cluster"}
        )
    assignments_df = pd.DataFrame(assignments)

    n_clustered = int(assignments_df["proposed_global_part_id"].duplicated().sum())
    stats = {
        "n_skus": float(len(items)),
        "n_parts": float(assignments_df["proposed_global_part_id"].nunique()),
        "n_merged_into_clusters": float(n_clustered),
        "n_review": float(len(review_queue)),
        "n_rejected_safety": float(len(rejected)),
    }
    return MergePlan(
        assignments=assignments_df,
        review_queue=tuple(review_queue),
        rejected_pairs=tuple(rejected),
        stats=stats,
    )
