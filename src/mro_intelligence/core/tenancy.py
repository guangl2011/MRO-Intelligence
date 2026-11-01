"""Tenant context primitives.

Tenancy invariant (blueprint §5): shared infrastructure, logically isolated
tenants. Every tenant-keyed table carries ``tenant_id`` and is protected by
PostgreSQL row-level security; every application query path must scope to
exactly one tenant context. This module is the single place that constructs
and validates tenant identity — nothing else may pass a raw tenant id into a
query or a warehouse path.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

# Tenant ids become path segments (warehouse dirs) and RLS session-variable
# values; keep them boring and injective.
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,40}$")

# The RLS session GUC that policies key on. Set per connection/transaction.
TENANT_GUC = "app.tenant_id"


class TenantContext(BaseModel):
    """An authenticated tenant scope. Construct through :meth:`of` only."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str

    @field_validator("tenant_id")
    @classmethod
    def _validate_tenant_id(cls, value: str) -> str:
        if not _TENANT_ID_RE.match(value):
            raise ValueError(f"invalid tenant id {value!r}: must match {_TENANT_ID_RE.pattern}")
        return value

    @staticmethod
    def of(tenant_id: str) -> TenantContext:
        return TenantContext(tenant_id=tenant_id)

    def guc_value(self) -> str:
        """Value for ``SET app.tenant_id`` on a PostgreSQL session."""
        return self.tenant_id

    def warehouse_dir(self, root: str) -> str:
        """Per-tenant directory under a warehouse root (isolation by prefix)."""
        import os

        return os.path.join(root, self.tenant_id)


def assert_single_tenant(tenants: set[str]) -> TenantContext:
    """Fail loudly when a query path would touch more than one tenant."""
    if len(tenants) != 1:
        raise TenantScopeError(f"expected exactly 1 tenant scope, got {sorted(tenants)}")
    return TenantContext.of(tenants.pop())


class TenantScopeError(RuntimeError):
    """Raised when a query path is not scoped to exactly one tenant."""
