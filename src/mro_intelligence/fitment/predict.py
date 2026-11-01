"""Prediction: rank parts and repair actions from graph evidence.

Ranking blends empirical work-order support (what techs actually used) with
documented OEM listing support (what the manufacturer says fits), weighted
toward the empirical source. Abstention is first-class: a query with no
usable evidence returns a reason instead of a guess (blueprint §6/§8), and
safety-classified parts surface only as technician-confirmed guidance, never
as auto-substitutions.
"""

from __future__ import annotations

from dataclasses import dataclass

from mro_intelligence.fitment.graph import (
    SAFETY_CLASS_NONE,
    SOURCE_WORK_ORDER,
    FitmentEdge,
)

# Empirical evidence outweighs documentation; constants keep the blend
# inspectable and testable (weights are a product decision, not hidden math).
WO_WEIGHT = 0.7
OEM_WEIGHT = 0.3
MIN_TOP_CONFIDENCE = 0.05


@dataclass(frozen=True)
class PartPrediction:
    part_ref: str
    score: float
    wo_support: int
    oem_support: int
    requires_technician: bool


@dataclass(frozen=True)
class RepairPrediction:
    repair_key: str
    repair_name: str
    support: int


@dataclass(frozen=True)
class Abstention:
    reason: str


def _usable(edges: list[FitmentEdge]) -> list[FitmentEdge]:
    """Drop rejected and unbridged edges from the prediction pool."""
    return [e for e in edges if e.review_state in ("inferred", "human-approved")]


def rank_parts(
    edges: list[FitmentEdge], model_id: str, failure_key_text: str, top_k: int = 3
) -> list[PartPrediction] | Abstention:
    """Top parts for a (model, failure), evidence-weighted, abstaining cleanly."""
    pool = [
        e for e in _usable(edges) if e.model_id == model_id and e.failure_key == failure_key_text
    ]
    if not pool:
        return Abstention("no evidence for this model + failure")
    scores: dict[str, list[FitmentEdge]] = {}
    for edge in pool:
        scores.setdefault(edge.part_ref, []).append(edge)
    predictions: list[PartPrediction] = []
    for part_ref, part_edges in scores.items():
        wo_support = max(
            (e.support for e in part_edges if e.source == SOURCE_WORK_ORDER), default=0
        )
        oem_support = max(
            (e.support for e in part_edges if e.source != SOURCE_WORK_ORDER), default=0
        )
        if wo_support == 0 and oem_support == 0:
            continue
        # Empirical strength saturates with observation (support 2 -> 0.5,
        # support 8 -> 0.8); documentation is binary. The 0.7/0.3 blend ranks
        # frequent-empirical > rare-empirical > documented-only.
        empirical_strength = wo_support / (wo_support + 2)
        score = WO_WEIGHT * empirical_strength + OEM_WEIGHT * (1.0 if oem_support else 0.0)
        safety = {e.safety_class for e in part_edges}
        # Any restricted edge on the part demands a technician -- a "none"
        # edge in the mix must not wash out the stricter class.
        requires_technician = any(c != SAFETY_CLASS_NONE for c in safety)
        predictions.append(
            PartPrediction(
                part_ref=part_ref,
                score=round(score, 4),
                wo_support=wo_support,
                oem_support=oem_support,
                requires_technician=requires_technician,
            )
        )
    predictions.sort(key=lambda p: (-p.score, -p.wo_support, -p.oem_support, p.part_ref))
    if not predictions or predictions[0].score < MIN_TOP_CONFIDENCE:
        return Abstention("insufficient evidence confidence")
    return predictions[:top_k]


def rank_repairs(
    edges: list[FitmentEdge], model_id: str, failure_key_text: str, top_k: int = 3
) -> list[RepairPrediction] | Abstention:
    """Top repair actions for a (model, failure), by observed support."""
    pool = [
        e
        for e in _usable(edges)
        if e.model_id == model_id
        and e.failure_key == failure_key_text
        and e.source == SOURCE_WORK_ORDER
    ]
    if not pool:
        return Abstention("no work-order evidence for this model + failure")
    support: dict[str, tuple[int, str]] = {}
    for edge in pool:
        seen, name = support.get(edge.repair_key, (0, edge.repair_name))
        support[edge.repair_key] = (seen + edge.support, name)
    ranked = sorted(support.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [RepairPrediction(k, name, s) for k, (s, name) in ranked[:top_k]]


def build_kit(
    edges: list[FitmentEdge], model_id: str, failure_key_text: str, size: int = 5
) -> list[PartPrediction] | Abstention:
    """A pre-visit kit: the top parts for the failure across repair actions."""
    return rank_parts(edges, model_id, failure_key_text, top_k=size)
