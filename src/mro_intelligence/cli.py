"""MRO-Intelligence command line interface.

The ``mro`` entry point grows with the platform; it starts here with the
synthetic dataset generator so every downstream stage has data to run on.
"""

from __future__ import annotations

from pathlib import Path

import typer

from mro_intelligence.catalog.evaluate import evaluate_dedup
from mro_intelligence.catalog.pipeline import load_extracts, run_catalog, write_catalog
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
