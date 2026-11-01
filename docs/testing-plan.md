# MRO-Intelligence Testing Plan

How this repo is tested: four layers, each answering a different question.
Run the fast layers locally before every push; CI re-runs them on every PR.

| Layer | Question it answers | Where | Runtime |
|---|---|---|---|
| Unit | Does each pure function behave on known inputs? | `tests/unit/` | seconds |
| Integration | Do modules compose correctly on synthetic extracts? | `tests/unit/test_catalog_pipeline.py`, `test_roi.py` | seconds |
| Evaluation harness | Do the NSF success metrics pass on simulated ground truth? | `--evaluate` flags on the pipeline CLIs | ~1–5 min (demo bundle) |
| Scale smoke | Does the whole chain hold up at AXON-like scale ($40M / 14.5K SKUs)? | commands below | ~15–25 min |

## Running the gate locally

```bash
pip install -e ".[dev]"
python -m ruff format src tests
python -m ruff check src tests
python -m mypy src
python -m pytest tests -q
```

CI (GitHub Actions) runs ruff format check, ruff check, mypy, and the unit
suite on every push and PR. Local runs are the primary gate; CI is the final
gate.

## Layer 1 — Unit tests

114 tests across 14 files protect the invariants that matter most:

| File | Protects |
|---|---|
| `test_tenancy.py` | tenant isolation invariants (tenant-keyed data, no cross-tenant leakage) |
| `test_layout.py` | P21 / Eclipse / Infor column mapping and date formats |
| `test_emit.py`, `test_world.py` | synthetic extract emission round-trips; catalog world invariants |
| `test_tenant_data.py` | tenant data consistency and generator determinism |
| `test_normalize.py` | SKU/MPN normalization, attribute and safety extraction |
| `test_dedup.py` | blocking, similarity scoring, safety-conflict merge protection |
| `test_segmentation.py` | ABC-XYZ classing (revenue cumulative thresholds, sparsity) |
| `test_catalog_pipeline.py` | catalog pipeline composition on synthetic frames |
| `test_fitment.py` | edge building, vocabulary bridge, ranking, abstention, safety escalation |
| `test_demand.py` | monthly series construction, Croston/SBA/TSB models, quantile bands, seasonal indices |
| `test_inventory.py` | order-up-to walk arithmetic, fill rate, lost sales |
| `test_roi.py` | ROI benefit model, tiering, frontier monotonicity, bootstrap interval ordering |

## Layer 2 — Integration on synthetic data

Integration tests compose real pipeline stages over synthetic extract
DataFrames (no network, no database): catalog loading → normalization →
segmentation, and ROI `prepare_inputs` → `run_roi` → `write_roi` on a
three-SKU, 18-month synthetic bundle with known dead-stock, stockout, and
steady SKUs. The ROI integration test asserts the tiered report structure,
the negative-carrying tradeoff (stock built for starving SKUs), dead-stock
cash value, frontier length and monotonicity, counterfactual interval
ordering, and the written CSV artifacts.

## Layer 3 — Evaluation harness (NSF success metrics)

The generator embeds hidden ground truth per tenant; the `--evaluate` flags
score real pipeline outputs against it. Metrics and thresholds come from the
NSF SBIR Phase I draft (S1 §5):

| Metric | Threshold | Measured by | Enforcement |
|---|---|---|---|
| Dedup precision / recall / F1 | reported (no numeric gate) | `run-catalog --evaluate` | printed report |
| Top-3 compatibility | ≥ 0.90 | `run-fitment` (held-out work orders) | printed report |
| Fault-to-repair Top-3 | ≥ 0.80 | `run-fitment` | printed report |
| Unsafe false matches | must be 0 | `run-fitment` | printed report |
| Kit-coverage uplift vs catalog-only baseline | ≥ 1.15x | `run-fitment` | printed report |
| Long-tail WMAPE improvement vs ERP-style trailing-mean baseline | ≥ 15% | `run-demand --evaluate` | **hard gate**: `WMAPE_GATE = 0.15`, `passes_gate()` prints PASS/FAIL |
| ROI tier totals, frontier, counterfactual intervals | structural | `run-roi` + `test_roi.py` | test assertions |

The demand gate is the only hard numeric gate in code; the fitment metrics
are printed against their S1 §5 thresholds for human judgement because the
correct action on a miss is a research decision (model or bridge changes),
not a mechanical failure. All evaluation numbers are produced on synthetic
ground truth — they validate the plumbing, not the research hypotheses;
real-distributor validation is the 4-week diagnostic (SDD).

## Layer 4 — Scale smoke (AXON-like profile)

The full chain at the AXON-like scale the product targets: $40M revenue,
~14.5K SKUs, HVAC-heavy long tail, all three ERP layouts (p21 / eclipse /
infor). Run before releases and after cross-cutting changes:

```bash
mro generate-synthetic --profile axontwin --out data/synthetic-axontwin
mro run-catalog --bundle data/synthetic-axontwin --tenant axontwin --layout p21 --out data/out-axontwin --evaluate
mro run-demand  --bundle data/synthetic-axontwin --tenant axontwin --layout p21 --out data/out-axontwin --evaluate   # >5 min; run detached
mro run-roi     --bundle data/synthetic-axontwin --tenant axontwin --layout p21 --out data/out-axontwin
```

Reference results (2026-11-01, `axontwin` p21 tenant): 14,477 SKUs → 13,864
canonical parts, dedup precision 1.000 / recall 0.749 / F1 0.856, 39,061
pairs queued for human review, 15 safety-rejected; realized annual revenue
$41.4M against the $40M target.

This is deliberately a smoke command, not a CI job: the demand stage fits
per-SKU models over 24 months of order lines and exceeds CI time budgets.

## Acceptance criteria per PR

A PR is merge-ready only when all of the following hold:

1. Local gate green (format, ruff, mypy, full unit suite) — CI mirrors this.
2. New behavior ships with tests in the same PR; bug fixes first reproduce
   the failure as a test.
3. Touched pipelines re-run with `--evaluate` on the demo bundle and their
   metric report is pasted into the PR description (values may regress only
   with an explicit explanation — never adjust tests or thresholds to make a
   number pass).
4. Safety-related changes (safety classes, abstention, review queue) state
   explicitly in the PR body that `unsafe` exposure remains 0 on the demo
   evaluation.
5. No ERP write path, no new runtime dependency without justification, and
   no scope beyond the module under change.

## Known gaps

- Tenancy is enforced today as the pure `TenantContext` primitive (id
  validation, Postgres GUC value, per-tenant warehouse paths); the PostgreSQL
  RLS layer and its tenant-leakage CI suite are designed but not yet built —
  they arrive with the first managed-Postgres deployment component.
- `run-fitment` does not fail the process when an S1 §5 threshold is missed;
  the demand gate does. Unifying gate enforcement is a deliberate open
  choice pending the first real-distributor baselines.
- End-to-end CI jobs for the synthetic bundle remain deferred (Layer 4 is
  manual) until a faster demand-stage path or CI budget justifies them.
