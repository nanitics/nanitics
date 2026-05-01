"""Tests for PlanStore: save/load/update/delete/list_plans, namespace filtering, error cases."""

from typing import Any

import pytest

from nanitics.capabilities.planning.models import Plan, PlanStatus
from nanitics.capabilities.planning.store import InMemoryPlanStore, PlanStore


@pytest.fixture
def store() -> InMemoryPlanStore:
    return InMemoryPlanStore()


def make_plan(**kwargs: Any) -> Plan:
    defaults: dict[str, Any] = {"name": "Test plan"}
    defaults.update(kwargs)
    return Plan(**defaults)


# ──────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────


class TestProtocol:
    def test_in_memory_store_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryPlanStore(), PlanStore)


# ──────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────


class TestSaveLoad:
    @pytest.mark.anyio
    async def test_save_and_load(self, store: InMemoryPlanStore) -> None:
        plan = make_plan()
        plan_id = await store.save(plan)
        assert plan_id == plan.id
        loaded = await store.load(plan_id)
        assert loaded is not None
        assert loaded.name == "Test plan"

    @pytest.mark.anyio
    async def test_load_missing_returns_none(self, store: InMemoryPlanStore) -> None:
        result = await store.load("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_save_overwrites_existing(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1", name="Original")
        await store.save(plan)
        updated_plan = make_plan(id="p1", name="Updated")
        await store.save(updated_plan)
        loaded = await store.load("p1")
        assert loaded is not None
        assert loaded.name == "Updated"


# ──────────────────────────────────────────────────────────
# Update
# ──────────────────────────────────────────────────────────


class TestUpdate:
    @pytest.mark.anyio
    async def test_update_existing_plan(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1")
        await store.save(plan)
        updated = plan.model_copy(update={"status": PlanStatus.completed})
        await store.update(updated)
        loaded = await store.load("p1")
        assert loaded is not None
        assert loaded.status == PlanStatus.completed

    @pytest.mark.anyio
    async def test_update_missing_raises(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="nonexistent")
        with pytest.raises(ValueError, match="not found"):
            await store.update(plan)


# ──────────────────────────────────────────────────────────
# Delete
# ──────────────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.anyio
    async def test_delete_existing(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1")
        await store.save(plan)
        await store.delete("p1")
        assert await store.load("p1") is None

    @pytest.mark.anyio
    async def test_delete_missing_is_noop(self, store: InMemoryPlanStore) -> None:
        await store.delete("nonexistent")  # should not raise


# ──────────────────────────────────────────────────────────
# List Plans / Namespace Filtering
# ──────────────────────────────────────────────────────────


class TestListPlans:
    @pytest.mark.anyio
    async def test_list_all(self, store: InMemoryPlanStore) -> None:
        await store.save(make_plan(id="p1", name="Plan 1"))
        await store.save(make_plan(id="p2", name="Plan 2"))
        plans = await store.list_plans()
        assert len(plans) == 2

    @pytest.mark.anyio
    async def test_list_empty_store(self, store: InMemoryPlanStore) -> None:
        plans = await store.list_plans()
        assert plans == []

    @pytest.mark.anyio
    async def test_list_with_namespace_filter(self, store: InMemoryPlanStore) -> None:
        await store.save(make_plan(id="p1", namespace="ns1"))
        await store.save(make_plan(id="p2", namespace="ns2"))
        await store.save(make_plan(id="p3", namespace="ns1"))
        plans = await store.list_plans(namespace="ns1")
        assert len(plans) == 2
        assert all(p.namespace == "ns1" for p in plans)

    @pytest.mark.anyio
    async def test_list_with_namespace_none_returns_all(self, store: InMemoryPlanStore) -> None:
        await store.save(make_plan(id="p1", namespace="ns1"))
        await store.save(make_plan(id="p2", namespace=None))
        plans = await store.list_plans(namespace=None)
        assert len(plans) == 2

    @pytest.mark.anyio
    async def test_list_with_nonexistent_namespace(self, store: InMemoryPlanStore) -> None:
        await store.save(make_plan(id="p1", namespace="ns1"))
        plans = await store.list_plans(namespace="nonexistent")
        assert plans == []
