"""Dataset profiles.

Two profiles ship with the MVP:

- ``demo`` — small single-tenant dataset (CI + local runs, ~1 minute).
- ``axontwin`` — full AXON-like scale: $40M revenue, ~12K SKUs, three tenants
  so the three ERP export templates (P21, Eclipse, Infor) are all exercised.

Sizes are the profile's *inputs*; realized revenue is an emergent output the
generator reports in the manifest (never a claim).
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict


class TenantProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    erp_layout: str  # p21 | eclipse | infor — which export template shapes the CSVs
    n_skus: int
    n_branches: int
    n_customers: int
    n_work_orders: int
    annual_revenue_target: float  # used to calibrate demand volumes, not promised


class ProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    seed: int
    months: int  # months of order history
    end_month: date  # last month of history (inclusive)
    tenants: tuple[TenantProfile, ...]


def _tenant(**kwargs: object) -> TenantProfile:
    return TenantProfile(**kwargs)  # type: ignore[arg-type]


_DEMO = ProfileConfig(
    name="demo",
    seed=20261101,
    months=30,
    end_month=date(2026, 9, 1),
    tenants=(
        _tenant(
            tenant_id="demo",
            erp_layout="p21",
            n_skus=1_500,
            n_branches=3,
            n_customers=400,
            n_work_orders=900,
            annual_revenue_target=5_000_000.0,
        ),
    ),
)

_AXON_TWIN = ProfileConfig(
    name="axontwin",
    seed=40_000_000,
    months=36,
    end_month=date(2026, 9, 1),
    tenants=(
        _tenant(
            tenant_id="axontwin",
            erp_layout="p21",
            n_skus=12_000,
            n_branches=5,
            n_customers=1_800,
            n_work_orders=6_000,
            annual_revenue_target=40_000_000.0,
        ),
        _tenant(
            tenant_id="northline",
            erp_layout="eclipse",
            n_skus=9_000,
            n_branches=4,
            n_customers=1_200,
            n_work_orders=4_000,
            annual_revenue_target=26_000_000.0,
        ),
        _tenant(
            tenant_id="valveflow",
            erp_layout="infor",
            n_skus=7_000,
            n_branches=3,
            n_customers=900,
            n_work_orders=3_000,
            annual_revenue_target=18_000_000.0,
        ),
    ),
)

_PROFILES: dict[str, ProfileConfig] = {"demo": _DEMO, "axontwin": _AXON_TWIN}


def get_profile(name: str) -> ProfileConfig:
    try:
        return _PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(_PROFILES))
        raise KeyError(f"unknown profile {name!r}; available: {known}") from None
