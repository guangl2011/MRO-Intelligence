"""NSF success-metric harness for the fitment graph (S1 §5).

Held-out protocol: the graph builds on the earliest 75% of work orders (by
completion date) plus the full OEM listing; metrics run on the remaining 25%.
Metric units are work-order part lines: one line is one (model, failure,
repair, part-used) decision a tech made.

- top3 compatibility: share of eligible lines where the part actually used
  appears in the graph's top-3 part recommendations (target >= 0.90).
- fault-to-repair top-3: share of eligible lines where the executed repair
  action appears in the graph's top-3 repair actions (target >= 0.80).
- kit coverage: share of eligible lines whose used part is in the pre-visit
  kit (top-5 parts for the failure). Reported for the full graph and for an
  OEM-listing-only graph; the uplift between them is the acceptance metric
  (target >= 0.15). Same lines, same parts, only the evidence differs --
  that is what makes the uplift a causal comparison.
- unsafe exposure: share of eligible lines where the recommended set matches
  a work order flagged as wrong-part usage (synthetic truth only).
- abstention rate: share of (model, failure) queries the graph declines to
  answer for lack of evidence. Reported separately, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mro_intelligence.fitment.graph import FitmentEdge
from mro_intelligence.fitment.normalize import failure_key, repair_key
from mro_intelligence.fitment.predict import build_kit, rank_parts, rank_repairs

TRAIN_SHARE = 0.75
KIT_SIZE = 5


@dataclass(frozen=True)
class RepairLine:
    """One held-out part decision to score against the graph."""

    model_id: str
    failure_key: str
    repair_key: str
    part_ref: str
    is_wrong_part: bool
    failure_code: str | None = None  # supplier code, known only from truth


@dataclass(frozen=True)
class FitmentMetrics:
    top3_compatibility: float
    part_top3_recall: float
    repair_top3: float
    kit_coverage_full: float
    kit_coverage_oem_only: float
    kit_uplift: float
    unsafe_exposure: float
    abstention_rate: float
    eval_lines: int

    def to_row(self) -> dict[str, float | int]:
        return {
            "top3_compatibility": round(self.top3_compatibility, 4),
            "part_top3_recall": round(self.part_top3_recall, 4),
            "repair_top3": round(self.repair_top3, 4),
            "kit_coverage_full": round(self.kit_coverage_full, 4),
            "kit_coverage_oem_only": round(self.kit_coverage_oem_only, 4),
            "kit_uplift": round(self.kit_uplift, 4),
            "unsafe_exposure": round(self.unsafe_exposure, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "eval_lines": self.eval_lines,
        }


def split_work_orders(
    work_orders: pd.DataFrame, train_share: float = TRAIN_SHARE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: earliest share for graph building, rest for eval."""
    ordered = work_orders.sort_values("confirmed_date")
    cut = int(len(ordered) * train_share)
    return ordered.iloc[:cut], ordered.iloc[cut:]


def resolve_lines(
    eval_work_orders: pd.DataFrame,
    canonical_catalog: pd.DataFrame,
    wrong_part_flags: pd.DataFrame | None = None,
) -> list[RepairLine]:
    """Turn held-out work-order rows into scoreable lines.

    Lines whose SKU is missing from the canonical catalog are dropped: the
    catalog stage owns that mapping, and the graph scores part identity, not
    tenant SKUs. Wrong-part flags (synthetic truth only) mark unsafe-usage
    lines; when absent every line counts as a legitimate repair.
    """
    sku_to_part = {
        str(sku): str(part)
        for sku, part in canonical_catalog[["sku", "proposed_global_part_id"]].itertuples(
            index=False, name=None
        )
    }
    wrong: dict[str, bool] = {}
    codes: dict[str, str] = {}
    if wrong_part_flags is not None:
        wrong = {
            str(row.wo_id): str(row.used_wrong_part).lower() == "true"
            for row in wrong_part_flags[["wo_id", "used_wrong_part"]].itertuples(index=False)
        }
        if "failure_code" in wrong_part_flags.columns:
            codes = {
                str(row.wo_id): str(row.failure_code)
                for row in wrong_part_flags[["wo_id", "failure_code"]].itertuples(index=False)
            }
    lines: list[RepairLine] = []
    for row in eval_work_orders.itertuples(index=False):
        part_ref = sku_to_part.get(str(row.part_sku))
        if part_ref is None:
            continue
        lines.append(
            RepairLine(
                model_id=str(row.model_id),
                failure_key=failure_key(str(row.failure_reported)),
                repair_key=repair_key(str(row.repair_action)),
                part_ref=part_ref,
                is_wrong_part=wrong.get(str(row.wo_id), False),
                failure_code=codes.get(str(row.wo_id)),
            )
        )
    return lines


def _score_top3(
    graph: list[FitmentEdge],
    lines: list[RepairLine],
    compatibility: dict[tuple[str, str], set[str]] | None = None,
) -> tuple[int, int, int, int, int, int, int]:
    """Returns (compat_recs, total_recs, part_hits, repair_hits,
    part_eligible, repair_eligible, unsafe).

    ``compat_recs``/``total_recs`` give precision@3 against the true
    (model, failure_code) compatible-part universe -- the S1 acceptance
    metric -- computable only when both truth inputs are supplied.
    ``part_hits`` recalls the actually-used part.
    """
    compat_recs = total_recs = 0
    part_hits = repair_hits = part_eligible = repair_eligible = unsafe = 0
    for line in lines:
        parts = rank_parts(graph, line.model_id, line.failure_key, top_k=3)
        if isinstance(parts, list):
            part_eligible += 1
            refs = [p.part_ref for p in parts]
            if compatibility is not None and line.failure_code is not None:
                universe = compatibility.get((line.model_id, line.failure_code), set())
                compat_recs += sum(1 for ref in refs if ref in universe)
                total_recs += len(refs)
            if line.part_ref in refs:
                if line.is_wrong_part:
                    unsafe += 1
                else:
                    part_hits += 1
        repairs = rank_repairs(graph, line.model_id, line.failure_key, top_k=3)
        if isinstance(repairs, list):
            repair_eligible += 1
            if any(r.repair_key == line.repair_key for r in repairs):
                repair_hits += 1
    return compat_recs, total_recs, part_hits, repair_hits, part_eligible, repair_eligible, unsafe


def _kit_coverage(
    graph: list[FitmentEdge],
    lines: list[RepairLine],
    kit_size: int,
    failure_override: str | None = None,
) -> tuple[int, int, int]:
    """Returns (hits, eligible, unsafe) for pre-visit kit coverage.

    ``failure_override`` re-targets every query (e.g. the catalog-only
    baseline answers at equipment level, ignoring the line's failure text).
    """
    hits = eligible = unsafe = 0
    for line in lines:
        fkey = failure_override if failure_override is not None else line.failure_key
        kit = build_kit(graph, line.model_id, fkey, size=kit_size)
        if isinstance(kit, list):
            eligible += 1
            covered = line.part_ref in {p.part_ref for p in kit}
            if line.is_wrong_part:
                if covered:
                    unsafe += 1
            elif covered:
                hits += 1
    return hits, eligible, unsafe


def evaluate_fitment(
    full_graph: list[FitmentEdge],
    oem_only_graph: list[FitmentEdge],
    lines: list[RepairLine],
    compatibility: dict[tuple[str, str], set[str]] | None = None,
) -> FitmentMetrics:
    """Score the acceptance metrics against held-out repair lines.

    All rates are per evaluation line so abstention, hits, and unsafe
    exposure share one denominator. Precision@3 (``top3_compatibility``)
    additionally needs the hidden compatible-part universe.
    """
    (
        compat_recs,
        total_recs,
        part_hits,
        repair_hits,
        part_eligible,
        repair_eligible,
        unsafe,
    ) = _score_top3(full_graph, lines, compatibility)
    kit_hits, kit_eligible, kit_unsafe = _kit_coverage(full_graph, lines, KIT_SIZE)
    oem_hits, oem_eligible, _ = _kit_coverage(oem_only_graph, lines, KIT_SIZE, failure_override="")

    kit_cov = kit_hits / kit_eligible if kit_eligible else 0.0
    oem_cov = oem_hits / oem_eligible if oem_eligible else 0.0
    n = len(lines)
    return FitmentMetrics(
        top3_compatibility=(compat_recs / total_recs) if total_recs else 0.0,
        part_top3_recall=part_hits / part_eligible if part_eligible else 0.0,
        repair_top3=repair_hits / repair_eligible if repair_eligible else 0.0,
        kit_coverage_full=kit_cov,
        kit_coverage_oem_only=oem_cov,
        kit_uplift=(kit_cov - oem_cov) / oem_cov if oem_cov > 0 else 0.0,
        unsafe_exposure=(unsafe + kit_unsafe) / n if n else 0.0,
        abstention_rate=1.0 - part_eligible / n if n else 0.0,
        eval_lines=n,
    )
