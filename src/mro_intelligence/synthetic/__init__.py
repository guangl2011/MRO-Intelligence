"""Synthetic AXON-like distributor data — the world, its extracts, and truth."""

from mro_intelligence.synthetic.profile import (
    ProfileConfig,
    TenantProfile,
    get_profile,
)
from mro_intelligence.synthetic.tenant_data import TenantData, generate_tenant
from mro_intelligence.synthetic.world import World, build_world

__all__ = [
    "ProfileConfig",
    "TenantData",
    "TenantProfile",
    "World",
    "build_world",
    "generate_tenant",
    "get_profile",
]
