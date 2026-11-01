# MRO-Intelligence

Read-only intelligence layer over mid-size distributor ERPs: Catalog
Intelligence (M1), a Fitment & Substitution Graph (M2), and Long-tail Demand
Sensing (M3). Full architecture and partner-diagnostic documentation ship with
the MVP's final PR; today this file tracks setup and the stages that exist.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
mro --help
```

## Pipeline stages

Every stage runs on synthetic bundles so the whole system works end-to-end
without customer data.

```bash
# 1. Generate a synthetic distributor bundle (ERP extracts + hidden truth).
mro generate-synthetic --profile demo --out data/synthetic

# 2. M1 catalog stage: normalize, detect duplicate SKUs, segment ABC-XYZ.
mro run-catalog --bundle data/synthetic --tenant demo --layout p21 \
  --out data/processed --evaluate

# 3. M2 fitment stage: build the versioned fitment graph from work-order
#    history + OEM compatibility listings, then score NSF acceptance metrics.
mro run-fitment --bundle data/synthetic --tenant demo --layout p21 \
  --catalog data/processed/demo/canonical_catalog.csv \
  --out data/processed --evaluate
```

`--evaluate` scores each stage against the bundle's hidden ground truth
(synthetic bundles only — real partner data has none).

### Catalog stage outputs

- `canonical_catalog.csv` — normalized records with a proposed global part id
  per SKU (`proposed_global_part_id`), inferred safety flags, and extracted
  attributes.
- `review_queue.csv` — ambiguous duplicate pairs awaiting human approval. Merges
  are decisions with evidence, never silent: pairs whose only disagreement is
  the part number stay in review even when attributes match exactly.
- `segmentation.csv` — ABC (cumulative revenue) and XYZ (variability/sparsity)
  classes per SKU; the C-Z quadrant is the long-tail optimization target.

### Fitment stage outputs

- `fitment_edges.csv` — one row per versioned, evidence-carrying edge:
  equipment model → failure → repair → part, with source (work-order history
  vs OEM listing), support, confidence, safety class, evidence references, and
  review state. Failure vocabularies differ between work orders (technician
  text) and OEM listings (supplier codes); the bridge links them per model by
  shared part co-occurrence, and unbridged listing claims stay review-only —
  they never feed predictions.
- `fitment_metrics.csv` — acceptance metrics on held-out work orders (latest
  25%, chronologically split):

  | Metric | Meaning | NSF gate |
  |---|---|---|
  | `top3_compatibility` | Precision@3 against the true compatible-part universe | ≥ 0.90 |
  | `part_top3_recall` | The actually-used part appears in the top-3 | informational |
  | `repair_top3` | The performed repair appears in the top-3 repair actions | ≥ 0.80 |
  | `kit_coverage_full` vs `kit_coverage_oem_only` | Pre-visit kit (top-5) hit rate, failure-aware graph vs equipment-level catalog-only baseline | uplift ≥ 0.15 |
  | `unsafe_exposure` | Wrong-part lines whose recommended kit contains the wrong part | ≈ 0 |
  | `abstention_rate` | Share of held-out lines the graph declines to answer | visible, non-degenerate |

  Safety-constrained parts (gas, A2L refrigerant, T&P relief, backflow) are
  flagged on their edges and always route to a technician; abstention is a
  reported outcome, never hidden.

## Development

```bash
make check   # ruff format + ruff + mypy + unit tests
```

CI runs the same gate on every PR.
