"""Catalog-stage evaluation against synthetic ground truth.

The generator hides truth (gt_sku_mapping) the pipeline never sees; the
evaluation harness compares predicted duplicate clusters against it. Metrics
are SKU-pair precision/recall/F1 — the same pair-set framing the NSF
evaluation harness uses for fitment.
"""

from __future__ import annotations

import pandas as pd


def _truth_pairs(gt_sku_mapping: pd.DataFrame) -> set[tuple[str, str]]:
    """Ground-truth duplicate pairs: SKUs sharing a global part id."""
    by_part = gt_sku_mapping.groupby("global_part_id")["sku"].apply(list)
    pairs: set[tuple[str, str]] = set()
    for skus in by_part:
        if len(skus) < 2:
            continue
        ordered = sorted(skus)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.add((ordered[i], ordered[j]))
    return pairs


def _predicted_pairs(assignments: pd.DataFrame) -> set[tuple[str, str]]:
    by_part = assignments.groupby("proposed_global_part_id")["sku"].apply(list)
    pairs: set[tuple[str, str]] = set()
    for skus in by_part:
        if len(skus) < 2:
            continue
        ordered = sorted(skus)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.add((ordered[i], ordered[j]))
    return pairs


def evaluate_dedup(assignments: pd.DataFrame, gt_sku_mapping: pd.DataFrame) -> dict[str, float]:
    """Pair precision/recall/F1 of predicted duplicates vs ground truth."""
    truth = _truth_pairs(gt_sku_mapping)
    predicted = _predicted_pairs(assignments)
    tp = len(truth & predicted)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pair_tp": float(tp),
        "pair_fp": float(len(predicted - truth)),
        "pair_fn": float(len(truth - predicted)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
