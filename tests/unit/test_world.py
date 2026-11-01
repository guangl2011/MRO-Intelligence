"""Catalog-world generator invariants — the ground truth must be well-formed."""

from __future__ import annotations

import numpy as np
import pytest

from mro_intelligence.synthetic.profile import ProfileConfig
from mro_intelligence.synthetic.tenant_data import DUP_SHARE
from mro_intelligence.synthetic.world import (
    CATEGORY_FAILURES,
    GENERIC_BRAND,
    SAFETY_PROHIBITED,
    World,
    _subtypes,
    build_world,
    count_equipment_models,
)


@pytest.fixture(scope="session")
def world(tiny_profile: ProfileConfig) -> World:
    tenant = tiny_profile.tenants[0]
    rng = np.random.default_rng(7)
    n_oem = int(tenant.n_skus / (1 + DUP_SHARE))
    return build_world(rng, n_oem, count_equipment_models(tenant.n_skus))


def test_prohibited_auto_subtypes_are_safety_flagged() -> None:
    """Gas, A2L refrigerant, T&P, and backflow parts must be prohibited-auto."""
    prohibited = {s.key for s in _subtypes() if s.safety == SAFETY_PROHIBITED}
    assert {"gas_valve", "txv_a2l", "line_set_a2l", "tp_valve", "backflow_rpz"} <= prohibited


def test_world_parts_have_unique_ids(world: World) -> None:
    ids = [p.global_part_id for p in world.parts]
    assert len(ids) == len(set(ids))


def test_generic_parts_replace_at_least_one_oem(world: World) -> None:
    generics = [p for p in world.parts if p.brand == GENERIC_BRAND]
    assert generics, "world should include generic replacement parts"
    for p in generics:
        assert p.replaces, f"generic part {p.global_part_id} replaces nothing"


def test_oem_parts_have_no_replacement_links(world: World) -> None:
    for p in world.parts:
        if p.brand != GENERIC_BRAND:
            assert p.replaces == ()


def test_failure_probabilities_sum_to_one() -> None:
    for category, failures in CATEGORY_FAILURES.items():
        total = sum(f.prob for f in failures)
        assert total == pytest.approx(1.0, abs=1e-6), f"{category} failure probs sum to {total}"


def test_actions_reference_known_subtypes(world: World) -> None:
    known = {s.key for s in _subtypes()}
    for category, failures in CATEGORY_FAILURES.items():
        for fdef in failures:
            for _action, subtype_keys in fdef.actions:
                assert set(subtype_keys) <= known, f"{category}/{fdef.code}: unknown subtype"


def test_fitment_maps_models_to_valid_parts(world: World) -> None:
    part_ids = {p.global_part_id for p in world.parts}
    model_ids = {m.model_id for m in world.models}
    for (model_id, _failure), part_list in world.fitment.items():
        assert model_id in model_ids
        assert set(part_list) <= part_ids
        assert part_list, f"fitment for {model_id} is empty"


def test_world_is_deterministic_per_seed(tiny_profile: ProfileConfig) -> None:
    tenant = tiny_profile.tenants[0]
    n_oem = int(tenant.n_skus / (1 + DUP_SHARE))
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    world_a = build_world(rng_a, n_oem, count_equipment_models(tenant.n_skus))
    world_b = build_world(rng_b, n_oem, count_equipment_models(tenant.n_skus))
    assert [p.oem_part_number for p in world_a.parts] == [p.oem_part_number for p in world_b.parts]
