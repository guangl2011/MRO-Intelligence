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
```

`--evaluate` scores duplicate detection against the bundle's hidden ground
truth (synthetic bundles only — real partner data has none).

### Catalog stage outputs

- `canonical_catalog.csv` — normalized records with a proposed global part id
  per SKU (`proposed_global_part_id`), inferred safety flags, and extracted
  attributes.
- `review_queue.csv` — ambiguous duplicate pairs awaiting human approval. Merges
  are decisions with evidence, never silent: pairs whose only disagreement is
  the part number stay in review even when attributes match exactly.
- `segmentation.csv` — ABC (cumulative revenue) and XYZ (variability/sparsity)
  classes per SKU; the C-Z quadrant is the long-tail optimization target.

## Development

```bash
make check   # ruff format + ruff + mypy + unit tests
```

CI runs the same gate on every PR.
