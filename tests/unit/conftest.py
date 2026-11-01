"""Shared test fixtures: a small, fast tenant profile for generator tests."""

from __future__ import annotations

from datetime import date

import pytest

from mro_intelligence.synthetic.profile import ProfileConfig, TenantProfile


@pytest.fixture(scope="session")
def tiny_profile() -> ProfileConfig:
    """A deliberately small single-tenant profile so generator tests stay fast."""
    return ProfileConfig(
        name="test-tiny",
        seed=42,
        months=14,
        end_month=date(2026, 9, 1),
        tenants=(
            TenantProfile(
                tenant_id="tinyco",
                erp_layout="p21",
                n_skus=220,
                n_branches=3,
                n_customers=60,
                n_work_orders=80,
                annual_revenue_target=600_000.0,
            ),
        ),
    )
