"""Synthetic dataset generation CLI.

Generates standalone synthetic distributor bundles (ERP extracts in each ERP's
own layout + hidden ground truth + manifest) that the pipeline and the
evaluation harness run against. No customer data is ever required.

Usage:
    python -m mro_intelligence.synthetic.generate --profile demo --out data/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mro_intelligence.synthetic.emit import emit_tenant
from mro_intelligence.synthetic.profile import ProfileConfig, TenantProfile, get_profile
from mro_intelligence.synthetic.tenant_data import DUP_SHARE, generate_tenant
from mro_intelligence.synthetic.world import World, build_world, count_equipment_models

DEFAULT_SEED = 20261101


def _build_world(config: ProfileConfig, tenant: TenantProfile, tenant_index: int) -> World:
    """Each tenant gets its own ground-truth world (independent distributor)."""
    rng = np.random.default_rng(config.seed + tenant_index)
    n_oem = int(tenant.n_skus / (1 + DUP_SHARE))
    return build_world(rng, n_oem, count_equipment_models(tenant.n_skus))


def generate_dataset(profile_name: str, out_dir: str | Path, seed: int | None = None) -> list[dict]:
    """Build the full synthetic bundle for a profile; returns per-tenant manifests."""
    config = get_profile(profile_name)
    if seed is not None:
        config = config.model_copy(update={"seed": seed})
    manifests: list[dict] = []
    for tenant_index, tenant in enumerate(config.tenants):
        world = _build_world(config, tenant, tenant_index)
        data = generate_tenant(world, config, tenant, config.seed + tenant_index)
        manifests.append(emit_tenant(data, Path(out_dir)))
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="demo", help="profile name (demo | axontwin)")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    manifests = generate_dataset(args.profile, args.out, seed=args.seed)
    for manifest in manifests:
        print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
