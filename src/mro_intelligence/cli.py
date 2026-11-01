"""MRO-Intelligence command line interface.

The ``mro`` entry point grows with the platform; it starts here with the
synthetic dataset generator so every downstream stage has data to run on.
"""

from __future__ import annotations

import typer

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
