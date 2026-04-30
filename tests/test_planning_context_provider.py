"""Tests for PlanningContextProvider: detail levels, formatting, empty plan, plan not found."""

from typing import Any

import pytest

from nanitics.capabilities.memory.context_provider import ContextProvider
from nanitics.capabilities.planning.context_provider import PlanningContextProvider
from nanitics.capabilities.planning.models import (
    Goal,
    GoalStatus,
    Plan,
    PlanStep,
    StepStatus,
)
from nanitics.capabilities.planning.store import InMemoryPlanStore
from nanitics.infrastructure.llm.protocol import Message


@pytest.fixture
def store() -> InMemoryPlanStore:
    return InMemoryPlanStore()


def make_messages() -> list[Message]:
    return [Message(role="user", content="Do the task")]


def make_plan(**kwargs: Any) -> Plan:
    defaults: dict[str, Any] = {"name": "Test plan"}
    defaults.update(kwargs)
    return Plan(**defaults)


# ──────────────────────────────────────────────────────────
# Protocol conformance
# ──────────────────────────────────────────────────────────


class TestProtocol:
    def test_satisfies_context_provider_protocol(self, store: InMemoryPlanStore) -> None:
        provider = PlanningContextProvider(store, plan_id="p1")
        assert isinstance(provider, ContextProvider)


# ──────────────────────────────────────────────────────────
# Plan not found
# ──────────────────────────────────────────────────────────


class TestPlanNotFound:
    @pytest.mark.anyio
    async def test_returns_none_when_plan_missing(self, store: InMemoryPlanStore) -> None:
        provider = PlanningContextProvider(store, plan_id="nonexistent")
        result = await provider.provide(make_messages())
        assert result is None


# ──────────────────────────────────────────────────────────
# Priority and protected values
# ──────────────────────────────────────────────────────────


class TestContextMetadata:
    @pytest.mark.anyio
    async def test_priority_is_5(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1")
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1")
        result = await provider.provide(make_messages())
        assert result is not None
        assert result.priority == 5

    @pytest.mark.anyio
    async def test_not_protected(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1")
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1")
        result = await provider.provide(make_messages())
        assert result is not None
        assert result.protected is False


# ──────────────────────────────────────────────────────────
# Minimal detail level
# ──────────────────────────────────────────────────────────


class TestMinimalDetail:
    @pytest.mark.anyio
    async def test_minimal_shows_name_and_progress(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            name="Deploy service",
            steps=[
                PlanStep(description="Build", status=StepStatus.completed),
                PlanStep(description="Test", status=StepStatus.completed),
                PlanStep(description="Deploy", status=StepStatus.not_started),
            ],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="minimal")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "Deploy service" in result.content
        assert "2/3" in result.content

    @pytest.mark.anyio
    async def test_minimal_empty_plan(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1", name="Empty plan")
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="minimal")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "0/0" in result.content


# ──────────────────────────────────────────────────────────
# Normal detail level (default)
# ──────────────────────────────────────────────────────────


class TestNormalDetail:
    @pytest.mark.anyio
    async def test_normal_is_default(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            steps=[PlanStep(description="Step 1", status=StepStatus.completed)],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "## Completed" in result.content

    @pytest.mark.anyio
    async def test_normal_sections(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            steps=[
                PlanStep(description="Done step", status=StepStatus.completed, result="ok"),
                PlanStep(description="Active step", status=StepStatus.in_progress),
                PlanStep(description="Remaining step", status=StepStatus.not_started),
                PlanStep(description="Failed step", status=StepStatus.failed, result="error"),
            ],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="normal")
        result = await provider.provide(make_messages())

        assert result is not None
        content = result.content
        assert "## Completed" in content
        assert "[✓] Done step" in content
        assert "(result: ok)" in content
        assert "## Current" in content
        assert "[→] Active step" in content
        assert "## Remaining" in content
        assert "[ ] Remaining step" in content
        assert "## Failed" in content
        assert "[✗] Failed step" in content

    @pytest.mark.anyio
    async def test_normal_shows_dependencies(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            steps=[
                PlanStep(id="s1", description="First"),
                PlanStep(description="Second", dependencies=["s1"]),
            ],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="normal")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "depends on: s1" in result.content

    @pytest.mark.anyio
    async def test_normal_shows_goals(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            goals=[
                Goal(description="Main goal", status=GoalStatus.active),
                Goal(
                    description="Parent goal",
                    subgoals=[Goal(description="Sub goal", status=GoalStatus.achieved)],
                ),
            ],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="normal")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "## Goals" in result.content
        assert "[active] Main goal" in result.content
        assert "[achieved] Sub goal" in result.content

    @pytest.mark.anyio
    async def test_normal_empty_plan(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(id="p1", name="Empty plan")
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="normal")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "Empty plan" in result.content
        assert "0/0" in result.content


# ──────────────────────────────────────────────────────────
# Full detail level
# ──────────────────────────────────────────────────────────


class TestFullDetail:
    @pytest.mark.anyio
    async def test_full_shows_all_step_details(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            name="Full plan",
            description="A detailed plan",
            steps=[
                PlanStep(id="s1", description="Step one", status=StepStatus.completed, result="done"),
                PlanStep(id="s2", description="Step two", status=StepStatus.in_progress),
                PlanStep(id="s3", description="Step three", dependencies=["s1", "s2"]),
            ],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="full")
        result = await provider.provide(make_messages())

        assert result is not None
        content = result.content
        assert "Full plan" in content
        assert "A detailed plan" in content
        assert "## Steps" in content
        assert "[✓] Step one (id: s1)" in content
        assert "result: done" in content
        assert "[→] Step two (id: s2)" in content
        assert "[ ] Step three (id: s3)" in content
        assert "depends on: s1, s2" in content

    @pytest.mark.anyio
    async def test_full_shows_goals(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            goals=[Goal(description="Top goal", status=GoalStatus.active)],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="full")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "## Goals" in result.content
        assert "[active] Top goal" in result.content

    @pytest.mark.anyio
    async def test_full_shows_skipped_steps(self, store: InMemoryPlanStore) -> None:
        plan = make_plan(
            id="p1",
            steps=[PlanStep(description="Skipped step", status=StepStatus.skipped)],
        )
        await store.save(plan)
        provider = PlanningContextProvider(store, plan_id="p1", detail="full")
        result = await provider.provide(make_messages())
        assert result is not None
        assert "[~] Skipped step" in result.content
