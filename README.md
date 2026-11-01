# MRO-Intelligence

Read-only intelligence layer over mid-size distributor ERPs: Catalog
Intelligence (M1), a Fitment & Substitution Graph (M2), Long-tail Demand
Sensing (M3), and the partner ROI simulator that turns their outputs into a
signed baseline report. Full architecture and partner-diagnostic
documentation ship with the MVP's final PR; today this file tracks setup and
the stages that exist.

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

# 4. M3 demand stage: intermittent-demand forecasts, SKU health cards, and
#    the long-tail WMAPE gate against the ERP-style baseline.
mro run-demand --bundle data/synthetic --tenant demo --layout p21 \
  --out data/processed --evaluate

# 5. ROI stage: replay a target inventory policy against the ERP's recorded
#    outcomes, sweep the inventory-vs-fill-rate frontier, and emit the tiered
#    ROI baseline report (SDD Step 5).
mro run-roi --bundle data/synthetic --tenant demo --layout p21 \
  --out data/processed
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

### Demand stage outputs

- `demand_forecasts.csv` — one row per SKU: model choice (SBA for smooth/
  erratic series, TSB for intermittent/lumpy), occurrence probability,
  mean size, next-month point forecast, and the 5–95% quantile band of the
  monthly demand distribution (composition of occurrence probability and
  recent size history).
- `sku_health_report.csv` — the SDD health card: SBC demand class, trailing
  12-month demand, months of cover, dead-stock flag (zero sales for 12
  months while still stocked), and stockout-risk tier (`high`/`medium`/`low`)
  from the on-hand position vs the forecast quantiles.
- `demand_metrics.csv` — held-out WMAPE (latest 3 months) against the
  ERP-style baseline (flat trailing-12-month mean, no seasonality):

  | Metric | Meaning | NSF gate |
  |---|---|---|
  | `improvement_longtail` | WMAPE improvement on the Z-quadrant (intermittent long tail) | ≥ 0.15 |
  | `improvement` | WMAPE improvement over the whole catalog | informational |

  The model multiplies the per-SKU intermittent rate (Croston/SBA/TSB
  family) by a category-level month-of-year seasonal index — pooled across
  a category's SKUs because a per-SKU index sees only ~2 samples per
  calendar month. The baseline is the ERP-as-is flat rate: seasonality-blind
  by construction, which is exactly the product claim.

  The health card's point forecast is the un-seasonalized rate (deliberately
  conservative for risk tiers); horizon forecasts used for the WMAPE gate
  apply the seasonal index.

### ROI stage outputs

The ROI stage reads the demand stage's `sku_health_report.csv` plus the raw
extracts and answers the SDD's signature question: *what would have happened
if the distributor had run this inventory policy for the last 12 months?*
Every number is a counterfactual replay with recorded basis — never a
vendor-asserted ROI.

- `roi_baseline_report.csv` — the seven AXON benefit categories plus the
  one-time dead-stock cash release, each row carrying its basis string and
  source columns, priced at three tiers (conservative 0.6x / baseline 0.8x /
  aggressive 1.0x of each driver). The carrying row is **net**: it credits
  overstock reduction and debits stock built for stockout-risk SKUs, so it
  can be negative — the report shows the tradeoff, not a sales pitch.
- `inventory_frontier.csv` — average inventory value and fill rate for each
  target-cover setting (0.5–4 months). The recommended operating point is the
  smallest-inventory sweep point that matches the ERP's own recorded fill
  rate (a same-service, less-inventory claim); when no sweep point clears
  that bar, it falls back to the smallest inventory within 98% of the
  frontier's best fill.
- `counterfactual_simulation.csv` — per treated SKU: ERP-recorded vs policy
  sold/lost units, recovered margin, fill-rate gain, and inventory change,
  with a bootstrap 90% interval on aggregate recovered margin (2,000
  resamples, fixed seed).

Treatment assignment follows the health card: dead-stock and high-risk SKUs
and B-class over-cover (≥ 6 months) are treated; A/B-class healthy SKUs are
controls. Lead times come from the median PO lag (order → receipt); unknown
leads default to one month.

## Development

```bash
make check   # ruff format + ruff + mypy + unit tests
```

CI runs the same gate on every PR. The testing strategy — unit layers,
synthetic-data integration, the NSF metric harness, and the AXON-scale smoke
run with acceptance criteria per PR — is documented in
[docs/testing-plan.md](docs/testing-plan.md).
