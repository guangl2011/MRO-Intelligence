"""MRO-Intelligence command line interface.

The ``mro`` entry point grows with the platform; it starts here with the
synthetic dataset generator so every downstream stage has data to run on.
"""

from __future__ import annotations

from pathlib import Path

import typer

from mro_intelligence.catalog.evaluate import evaluate_dedup
from mro_intelligence.catalog.pipeline import load_extracts, run_catalog, write_catalog
from mro_intelligence.demand.pipeline import run_demand, write_demand
from mro_intelligence.fitment.pipeline import load_fitment_extracts, run_fitment, write_fitment
from mro_intelligence.ingest.loader import load_bundle_extracts
from mro_intelligence.synthetic.generate import generate_dataset

app = typer.Typer(help="MRO-Intelligence MVP tooling.", no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Keep subcommand form even while a single command is registered."""


@app.command("generate-synthetic")
def generate_synthetic(
    profile: str = typer.Option("demo", "--profile", help="Dataset profile (demo | axontwin)."),
    out: str = typer.Option("data/synthetic", "--out", help="Output directory for the bundle."),
    seed: int | None = typer.Option(None, "--seed", help="Override the profile seed."),
) -> None:
    """Generate a synthetic distributor bundle: ERP extracts + ground truth."""
    manifests = generate_dataset(profile, out, seed=seed)
    for m in manifests:
        typer.echo(
            f"generated {m['tenant_id']} [{m['erp_layout']}]: "
            f"{m['n_skus']} SKUs, {m['n_order_lines']} order lines, "
            f"realized annual revenue ${m['realized_annual_revenue']:,.0f}"
        )


@app.command("run-catalog")
def run_catalog_command(
    bundle: str = typer.Option(
        ..., "--bundle", help="Synthetic bundle root (contains tenant dirs)."
    ),
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant directory under the bundle root."),
    layout: str = typer.Option(..., "--layout", help="ERP layout id (p21 | eclipse | infor)."),
    out: str = typer.Option(
        "data/processed", "--out", help="Output directory for processed outputs."
    ),
    evaluate: bool = typer.Option(
        False, "--evaluate", help="Score dedup against hidden ground truth."
    ),
) -> None:
    """Run the M1 catalog stage on a tenant's extract bundle."""
    tenant_dir = Path(bundle) / tenant_id
    extracts = load_extracts(tenant_dir, layout)
    outputs = run_catalog(extracts, tenant_id=tenant_id)
    paths = write_catalog(outputs, Path(out))
    stats = outputs.stats
    typer.echo(
        f"catalog: {stats['n_skus']:.0f} SKUs -> {stats['n_parts']:.0f} parts; "
        f"{stats['n_review']:.0f} queued for review, "
        f"{stats['n_rejected_safety']:.0f} safety-rejected"
    )
    typer.echo(f"wrote {paths['catalog']}")
    if evaluate:
        gt = Path(tenant_dir) / "ground_truth" / "sku_mapping.csv"
        if not gt.exists():
            typer.echo(f"no ground truth at {gt}; skipping evaluation")
            return
        import pandas as pd

        gt_frame = pd.read_csv(gt, dtype=str)
        metrics = evaluate_dedup(outputs.merge_plan.assignments, gt_frame)
        typer.echo(
            "dedup vs ground truth: "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}"
        )


@app.command("run-fitment")
def run_fitment_command(
    bundle: str = typer.Option(
        ..., "--bundle", help="Synthetic bundle root (contains tenant dirs)."
    ),
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant directory under the bundle root."),
    layout: str = typer.Option(..., "--layout", help="ERP layout id (p21 | eclipse | infor)."),
    catalog: str = typer.Option(..., "--catalog", help="Canonical catalog CSV from run-catalog."),
    out: str = typer.Option(
        "data/processed", "--out", help="Output directory for processed outputs."
    ),
    evaluate: bool = typer.Option(
        False,
        "--evaluate",
        help="Score S1 acceptance metrics against held-out ground truth.",
    ),
) -> None:
    """Run the M2 fitment stage: build the graph, optionally score NSF metrics."""
    import pandas as pd

    tenant_dir = Path(bundle) / tenant_id
    extracts = load_fitment_extracts(tenant_dir, layout)
    catalog_frame = pd.read_csv(catalog, dtype=str)
    wrong_part_flags: pd.DataFrame | None = None
    compatibility_truth: pd.DataFrame | None = None
    sku_mapping: pd.DataFrame | None = None
    if evaluate:
        gt = tenant_dir / "ground_truth" / "work_orders.csv"
        if not gt.exists():
            typer.echo(f"no work-order ground truth at {gt}; skipping evaluation")
        else:
            wrong_part_flags = pd.read_csv(gt, dtype=str)
            fit_truth = tenant_dir / "ground_truth" / "fitment.csv"
            if fit_truth.exists():
                compatibility_truth = pd.read_csv(fit_truth, dtype=str)
            sku_truth = tenant_dir / "ground_truth" / "sku_mapping.csv"
            if sku_truth.exists():
                sku_mapping = pd.read_csv(sku_truth, dtype=str)
    outputs = run_fitment(
        extracts,
        catalog_frame,
        tenant_id=tenant_id,
        wrong_part_flags=wrong_part_flags,
        compatibility_truth=compatibility_truth,
        sku_mapping=sku_mapping,
    )
    paths = write_fitment(outputs, Path(out))
    n_inferred = sum(1 for e in outputs.edges if e.review_state == "inferred")
    n_review = sum(1 for e in outputs.edges if e.review_state == "needs-review")
    n_safety = sum(1 for e in outputs.edges if e.safety_class != "none")
    typer.echo(
        f"fitment: {len(outputs.edges)} edges "
        f"({n_inferred} inferred, {n_review} review-only, {n_safety} safety-classified)"
    )
    typer.echo(f"wrote {paths['edges']}")
    if outputs.metrics is not None:
        m = outputs.metrics.to_row()
        typer.echo(
            "fitment vs held-out truth: "
            f"top3_compatibility={m['top3_compatibility']} "
            f"part_top3_recall={m['part_top3_recall']} "
            f"repair_top3={m['repair_top3']} "
            f"kit_coverage={m['kit_coverage_full']} "
            f"(baseline {m['kit_coverage_oem_only']}, "
            f"uplift {m['kit_uplift']}) "
            f"unsafe={m['unsafe_exposure']} abstention={m['abstention_rate']}"
        )
        typer.echo(f"wrote {paths['metrics']}")


@app.command("run-demand")
def run_demand_command(
    bundle: str = typer.Option(
        ..., "--bundle", help="Synthetic bundle root (contains tenant dirs)."
    ),
    tenant_id: str = typer.Option(..., "--tenant", help="Tenant directory under the bundle root."),
    layout: str = typer.Option(..., "--layout", help="ERP layout id (p21 | eclipse | infor)."),
    out: str = typer.Option(
        "data/processed", "--out", help="Output directory holding the catalog stage outputs."
    ),
    evaluate: bool = typer.Option(
        False,
        "--evaluate",
        help="Score WMAPE vs the ERP baseline on the held-out months.",
    ),
) -> None:
    """Run the M3 demand-sensing stage: forecasts, health cards, WMAPE."""
    import pandas as pd

    tenant_dir = Path(bundle) / tenant_id
    extracts = load_bundle_extracts(
        tenant_dir, layout, ("order_lines", "inventory_monthly", "item_master")
    )
    segmentation_path = Path(out) / tenant_id / "segmentation.csv"
    if not segmentation_path.exists():
        raise typer.BadParameter(
            f"no segmentation at {segmentation_path}; run the catalog stage first"
        )
    segmentation = pd.read_csv(segmentation_path, dtype=str)
    outputs = run_demand(extracts, segmentation, tenant_id=tenant_id, evaluate=evaluate)
    paths = write_demand(outputs, Path(out))
    n_dead = int(outputs.health_report["dead_stock"].sum())
    n_risk = int((outputs.health_report["stockout_risk"] == "high").sum())
    typer.echo(
        f"demand: {len(outputs.forecasts)} SKU forecasts "
        f"({n_dead} dead-stock, {n_risk} high stockout-risk)"
    )
    typer.echo(f"wrote {paths['health']}")
    if outputs.metrics is not None:
        typer.echo(
            "demand vs held-out months: "
            f"wmape_model={outputs.metrics.wmape_model:.3f} "
            f"wmape_baseline={outputs.metrics.wmape_baseline:.3f} "
            f"improvement={outputs.metrics.improvement:.3f} | long-tail "
            f"model={outputs.metrics.wmape_model_longtail:.3f} "
            f"baseline={outputs.metrics.wmape_baseline_longtail:.3f} "
            f"improvement={outputs.metrics.improvement_longtail:.3f} "
            f"(gate >= 0.15: {'PASS' if outputs.metrics.passes_gate() else 'FAIL'})"
        )
        typer.echo(f"wrote {paths['metrics']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
