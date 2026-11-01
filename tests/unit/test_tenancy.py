"""Tenant-context primitive tests — the tenancy invariant's first line of defense."""

from __future__ import annotations

import pytest

from mro_intelligence.core.tenancy import TenantContext, TenantScopeError, assert_single_tenant


@pytest.mark.parametrize("tenant_id", ["demo", "axontwin", "tenant-01", "a" * 40])
def test_valid_tenant_ids_accepted(tenant_id: str) -> None:
    ctx = TenantContext.of(tenant_id)
    assert ctx.tenant_id == tenant_id
    assert ctx.guc_value() == tenant_id


@pytest.mark.parametrize(
    "tenant_id", ["", "Demo", "has space", "-leadingdash", "a" * 42, "dot.dot"]
)
def test_invalid_tenant_ids_rejected(tenant_id: str) -> None:
    with pytest.raises(ValueError, match="invalid tenant id"):
        TenantContext.of(tenant_id)


def test_warehouse_dir_is_prefixed_per_tenant() -> None:
    ctx = TenantContext.of("demo")
    assert ctx.warehouse_dir("data/warehouse") == "data/warehouse/demo"


def test_assert_single_tenant_returns_context() -> None:
    ctx = assert_single_tenant({"demo"})
    assert ctx.tenant_id == "demo"


@pytest.mark.parametrize("tenants", [set(), {"a", "b"}])
def test_assert_single_tenant_fails_loudly(tenants: set[str]) -> None:
    with pytest.raises(TenantScopeError, match="expected exactly 1 tenant scope"):
        assert_single_tenant(tenants)


def test_tenant_context_is_frozen() -> None:
    ctx = TenantContext.of("demo")
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        ctx.tenant_id = "other"  # type: ignore[misc]
