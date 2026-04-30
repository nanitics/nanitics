"""Tests for resumable orchestration.

Covers suspension, checkpoint saving, and resume for all 7 orchestrators.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import pytest

from nanitics.composition.durability.models import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointVersionError,
    RunCheckpoint,
    SuspensionInfo,
)
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.adapters import FunctionStep
from nanitics.composition.orchestration.conditional import Conditional
from nanitics.composition.orchestration.dag import DAG, DAGNode
from nanitics.composition.orchestration.loop import Loop
from nanitics.composition.orchestration.mapreduce import MapReduce
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.composition.orchestration.pipeline import Pipeline, Stage
from nanitics.composition.orchestration.protocol import FailurePolicy
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.events import (
    CheckpointSavedEvent,
    ExecutionResumedEvent,
    ExecutionSuspendedEvent,
)
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


def make_suspending_step(name: str, agent_name: str | None = None) -> FunctionStep:
    """A step that raises SuspendExecution on first call, succeeds on second."""

    async def suspend(x):
        raise SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="test-suspension",
                request_id="test-request",
                request_type="approval",
                prompt="Approve?",
                agent_name=agent_name,
            ),
        )

    return FunctionStep(name=name, fn=suspend)


def make_resumable_step(name: str, call_count: dict[str, int] | None = None) -> FunctionStep:
    """A step that suspends on first call, returns a value on second."""
    if call_count is None:
        call_count = {}
    call_count.setdefault(name, 0)

    async def fn(x):
        call_count[name] += 1
        if call_count[name] == 1:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="test-suspension",
                    request_id="test-request",
                    request_type="approval",
                    prompt="Approve?",
                ),
            )
        return f"{x}-resumed"

    return FunctionStep(name=name, fn=fn)


def make_checkpoint(
    state: dict[str, Any],
    run_id: str = "test-run",
    checkpoint_type: Literal["orchestration", "agent"] = "orchestration",
) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        checkpoint_type=checkpoint_type,
        state=state,
        suspension_info=SuspensionInfo(
            suspension_id="test-suspension",
            request_id="test-request",
            request_type="approval",
            prompt="Approve?",
        ),
    )


# ── Sequential ─────────────────────────────────────────────


class TestSequentialSuspension:
    async def test_suspension_saves_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        seq = Sequential(
            name="seq",
            steps=[make_step("s1", add_one), make_suspending_step("s2"), make_step("s3", add_one)],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute(5)

        # Verify checkpoint saved
        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "sequential"
        assert cp.state["suspended_step_index"] == 1
        assert cp.state["completed_results"] == {"s1": {"output": 6, "metadata": {}}}
        assert cp.state["last_output"] == 6

    async def test_suspension_emits_events(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        seq = Sequential(
            name="seq",
            steps=[make_step("s1"), make_suspending_step("s2")],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")

        saved_events = [e for e in emitter.events if isinstance(e, CheckpointSavedEvent)]
        assert len(saved_events) == 1
        assert saved_events[0].checkpoint_type == "orchestration"

        suspended_events = [e for e in emitter.events if isinstance(e, ExecutionSuspendedEvent)]
        assert len(suspended_events) == 1
        assert suspended_events[0].step_name == "s2"
        assert suspended_events[0].suspension_type == "hitl"

    async def test_resume_skips_completed_steps(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        executed: list[str] = []

        async def track_s1(x):
            executed.append("s1")
            return x + 1

        async def track_s2(x):
            executed.append("s2")
            return x + 10

        async def track_s3(x):
            executed.append("s3")
            return x + 100

        seq = Sequential(
            name="seq",
            steps=[make_step("s1", track_s1), make_step("s2", track_s2), make_step("s3", track_s3)],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "sequential",
                "suspended_step_index": 1,
                "completed_results": {"s1": {"output": 6, "metadata": {}}},
                "last_output": 6,
                "original_input": 5,
            },
            run_id="run-1",
        )

        result = await seq.execute(5, resume_from=checkpoint)

        assert result.output == 116  # 6 + 10 + 100
        assert "s1" not in executed  # Skipped
        assert "s2" in executed
        assert "s3" in executed

    async def test_resume_emits_resumed_event(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        seq = Sequential(
            name="seq",
            steps=[make_step("s1"), make_step("s2")],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "sequential",
                "suspended_step_index": 1,
                "completed_results": {"s1": {"output": "done", "metadata": {}}},
                "last_output": "done",
                "original_input": "input",
            },
        )

        await seq.execute("input", resume_from=checkpoint)

        resumed_events = [e for e in emitter.events if isinstance(e, ExecutionResumedEvent)]
        assert len(resumed_events) == 1
        assert resumed_events[0].resumed_from_step == "s2"

    async def test_suspension_at_first_step(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        seq = Sequential(
            name="seq",
            steps=[make_suspending_step("s1"), make_step("s2")],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["suspended_step_index"] == 0
        assert cp.state["completed_results"] == {}

    async def test_suspension_at_last_step(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        seq = Sequential(
            name="seq",
            steps=[make_step("s1", add_one), make_step("s2", add_one), make_suspending_step("s3")],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute(0)

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["suspended_step_index"] == 2
        assert cp.state["completed_results"] == {
            "s1": {"output": 1, "metadata": {}},
            "s2": {"output": 2, "metadata": {}},
        }

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        seq = Sequential(
            name="seq",
            steps=[make_step("s1"), make_suspending_step("s2")],
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")


# ── Pipeline ───────────────────────────────────────────────


class TestPipelineSuspension:
    async def test_suspension_saves_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        pipeline = Pipeline(
            name="pipe",
            stages=[
                Stage(make_step("s1", add_one)),
                Stage(make_suspending_step("s2")),
                Stage(make_step("s3", add_one)),
            ],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await pipeline.execute(5)

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "pipeline"
        assert cp.state["suspended_stage_index"] == 1
        assert cp.state["last_output"] == 6

    async def test_resume_skips_completed_stages(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        executed: list[str] = []

        async def track_s1(x):
            executed.append("s1")
            return x + 1

        async def track_s2(x):
            executed.append("s2")
            return x + 10

        pipeline = Pipeline(
            name="pipe",
            stages=[
                Stage(make_step("s1", track_s1)),
                Stage(make_step("s2", track_s2)),
            ],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "pipeline",
                "suspended_stage_index": 1,
                "completed_results": {"s1": {"output": 6, "metadata": {}}},
                "last_output": 6,
                "original_input": 5,
            },
        )

        result = await pipeline.execute(5, resume_from=checkpoint)
        assert result.output == 16
        assert "s1" not in executed
        assert "s2" in executed

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        pipeline = Pipeline(
            name="pipe",
            stages=[Stage(make_suspending_step("s1"))],
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await pipeline.execute("input")


# ── DAG ────────────────────────────────────────────────────


class TestDAGSuspension:
    async def test_suspension_drains_and_saves_checkpoint(self) -> None:
        """Node B suspends, node A completes. Drain collects A's result."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def slow_a(x):
            await asyncio.sleep(0.01)
            return "a-done"

        dag = DAG(
            name="dag",
            nodes={
                "a": DAGNode(step=make_step("a", slow_a)),
                "b": DAGNode(step=make_suspending_step("b")),
                "c": DAGNode(step=make_step("c"), depends_on=["a", "b"]),
            },
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "dag"
        assert "a" in cp.state["completed_nodes"]
        assert cp.state["completed_nodes"]["a"] == "a-done"
        assert cp.state["suspended_node"] == "b"

    async def test_resume_skips_completed_nodes(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        executed: list[str] = []

        async def track_a(x):
            executed.append("a")
            return "a-done"

        async def track_b(x):
            executed.append("b")
            return "b-done"

        async def track_c(x):
            executed.append("c")
            return "c-done"

        dag = DAG(
            name="dag",
            nodes={
                "a": DAGNode(step=make_step("a", track_a)),
                "b": DAGNode(step=make_step("b", track_b)),
                "c": DAGNode(step=make_step("c", track_c), depends_on=["a", "b"]),
            },
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "dag",
                "completed_nodes": {"a": "a-done"},
                "suspended_node": "b",
                "original_input": "input",
            },
        )

        result = await dag.execute("input", resume_from=checkpoint)
        assert "a" not in executed
        assert "b" in executed
        assert "c" in executed
        # c depends on a and b, so it should get both results
        assert result.output == "c-done"

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        dag = DAG(
            name="dag",
            nodes={"a": DAGNode(step=make_suspending_step("a"))},
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("input")


# ── Parallel ───────────────────────────────────────────────


class TestParallelSuspension:
    async def test_suspension_drains_other_branches(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def normal_step(x):
            return f"{x}-done"

        parallel = Parallel(
            name="par",
            steps=[make_step("a", normal_step), make_suspending_step("b"), make_step("c", normal_step)],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await parallel.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "parallel"
        assert cp.state["completed_branches"]["a"] == "input-done"
        assert cp.state["completed_branches"]["c"] == "input-done"

    async def test_resume_only_re_executes_suspended_branch(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        executed: list[str] = []

        async def track_a(x):
            executed.append("a")
            return "a-result"

        async def track_b(x):
            executed.append("b")
            return "b-result"

        async def track_c(x):
            executed.append("c")
            return "c-result"

        parallel = Parallel(
            name="par",
            steps=[make_step("a", track_a), make_step("b", track_b), make_step("c", track_c)],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "parallel",
                "completed_branches": {"a": "a-result", "c": "c-result"},
                "suspended_branch": "b",
                "original_input": "input",
            },
        )

        result = await parallel.execute("input", resume_from=checkpoint)
        assert "a" not in executed  # Restored from checkpoint
        assert "b" in executed  # Re-executed
        assert "c" not in executed  # Restored from checkpoint
        assert result.output == ["a-result", "b-result", "c-result"]

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        parallel = Parallel(
            name="par",
            steps=[make_suspending_step("a")],
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await parallel.execute("input")

    async def test_multiple_concurrent_suspensions_only_first_captured(self) -> None:
        """When two parallel branches both suspend, only the first is raised."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        parallel = Parallel(
            name="par",
            steps=[
                make_suspending_step("a", agent_name="agent-a"),
                make_suspending_step("b", agent_name="agent-b"),
            ],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await parallel.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "parallel"
        # Only one branch is recorded as suspended
        assert cp.state["suspended_branch"] in ("a", "b")
        # The other branch is not in completed_branches (it also suspended, not completed)
        assert len(cp.state["completed_branches"]) == 0


# ── Loop ───────────────────────────────────────────────────


class TestLoopSuspension:
    async def test_suspension_mid_iteration(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        call_count = 0

        async def suspend_on_second(x):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise SuspendExecution(
                    suspension_info=SuspensionInfo(
                        suspension_id="test-suspension",
                        request_id="test-req",
                        request_type="approval",
                        prompt="Approve?",
                    ),
                )
            return x + 1

        loop = Loop(
            name="loop",
            step=make_step("step", suspend_on_second),
            condition=lambda r, i: i >= 5,
            max_iterations=5,
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await loop.execute(0)

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "loop"
        assert cp.state["iteration"] == 2

    async def test_resume_continues_from_iteration(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        iterations_executed: list[int] = []

        async def track(x):
            iterations_executed.append(x)
            return x + 1

        loop = Loop(
            name="loop",
            step=make_step("step", track),
            condition=lambda r, i: i >= 3,
            max_iterations=5,
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "loop",
                "iteration": 2,
                "last_result": {"output": 1, "metadata": {}},
                "original_input": 0,
            },
        )

        result = await loop.execute(0, resume_from=checkpoint)
        # Resume from iteration 2 with input 1 (from last_result.output)
        # Iteration 2: track(1) → 2, condition(2, 2) → False, continue
        # Iteration 3: track(2) → 3, condition(3, 3) → True, stop
        assert 1 in iterations_executed
        assert 2 in iterations_executed
        assert result.metadata["iterations"] == 3
        assert result.output == 3

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        loop = Loop(
            name="loop",
            step=make_suspending_step("step"),
            condition=lambda r, i: i >= 3,
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await loop.execute("input")


# ── Conditional ────────────────────────────────────────────


class TestConditionalSuspension:
    async def test_suspension_after_routing(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        conditional = Conditional(
            name="cond",
            router=lambda x: "branch_b",
            branches={
                "branch_a": make_step("a"),
                "branch_b": make_suspending_step("b"),
            },
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await conditional.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "conditional"
        assert cp.state["selected_branch"] == "branch_b"

    async def test_resume_skips_routing(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        router_called = False

        def router(x):
            nonlocal router_called
            router_called = True
            return "branch_a"

        async def branch_b_fn(x):
            return "b-result"

        conditional = Conditional(
            name="cond",
            router=router,
            branches={
                "branch_a": make_step("a"),
                "branch_b": make_step("b", branch_b_fn),
            },
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "conditional",
                "selected_branch": "branch_b",
                "original_input": "input",
            },
        )

        result = await conditional.execute("input", resume_from=checkpoint)
        assert not router_called  # Router not called on resume
        assert result.output == "b-result"
        assert result.metadata["selected_branch"] == "branch_b"

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        conditional = Conditional(
            name="cond",
            router=lambda x: "a",
            branches={"a": make_suspending_step("a")},
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await conditional.execute("input")


# ── MapReduce ──────────────────────────────────────────────


class TestMapReduceSuspension:
    async def test_suspension_drains_and_saves(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def suspend_on_two(x):
            if x == 2:
                raise SuspendExecution(
                    suspension_info=SuspensionInfo(
                        suspension_id="test-suspension",
                        request_id="test-req",
                        request_type="approval",
                        prompt="Approve?",
                    ),
                )
            return x * 10

        mr = MapReduce(
            name="mr",
            step=make_step("s", suspend_on_two),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await mr.execute([1, 2, 3])

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "mapreduce"
        assert cp.state["split_items"] == [1, 2, 3]
        # Items 0 and 2 should be completed
        completed = cp.state["completed_items"]
        assert completed.get("0") == 10 or completed.get("2") == 30

    async def test_resume_only_re_executes_remaining(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        executed_items: list[int] = []

        async def track(x):
            executed_items.append(x)
            return x * 10

        mr = MapReduce(
            name="mr",
            step=make_step("s", track),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: sorted([r.output for r in results]),
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "mapreduce",
                "completed_items": {"0": 10, "2": 30},
                "suspended_item_index": 1,
                "split_items": [1, 2, 3],
                "original_input": [1, 2, 3],
            },
        )

        result = await mr.execute([1, 2, 3], resume_from=checkpoint)
        # Only item index 1 should be re-executed
        assert 2 in executed_items  # items[1] == 2
        assert 1 not in executed_items
        assert 3 not in executed_items
        assert result.output == [10, 20, 30]

    async def test_resume_with_async_reducer(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def async_reduce(results: list[Any]) -> int:
            return sum(r.output for r in results)

        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=async_reduce,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "mapreduce",
                "completed_items": {"0": 10, "2": 30},
                "suspended_item_index": 1,
                "split_items": [1, 2, 3],
                "original_input": [1, 2, 3],
            },
        )

        result = await mr.execute([1, 2, 3], resume_from=checkpoint)
        assert result.output == 10 + 2 + 30  # items[0]=10, items[1]=2 (identity), items[2]=30

    async def test_resume_best_effort_tracks_failures(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def fail_on_two(x: int) -> int:
            if x == 2:
                raise ValueError("item 2 failed")
            return x * 10

        mr = MapReduce(
            name="mr",
            step=make_step("s", fail_on_two),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: sorted([r.output for r in results]),
            failure_policy=FailurePolicy.BEST_EFFORT,
            checkpoint_store=store,
            run_id="run-1",
        )

        # Resume: item 0 completed, item 1 suspended, item 2 needs re-execution
        # items[1]=2 will fail, items[2]=3 will succeed
        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "mapreduce",
                "completed_items": {"0": 10},
                "suspended_item_index": 1,
                "split_items": [1, 2, 3],
                "original_input": [1, 2, 3],
            },
        )

        result = await mr.execute([1, 2, 3], resume_from=checkpoint)
        assert result.metadata["failed_items"] == [1]
        assert result.output == [10, 30]

    async def test_resume_with_cancelled_token(self) -> None:
        from nanitics import CancellationToken

        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        token = CancellationToken()

        async def cancel_and_return(x: int) -> int:
            token.cancel()
            return x * 10

        mr = MapReduce(
            name="mr",
            step=make_step("s", cancel_and_return),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            failure_policy=FailurePolicy.BEST_EFFORT,
            cancellation_token=token,
            checkpoint_store=store,
            run_id="run-1",
        )

        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "mapreduce",
                "completed_items": {"0": 10},
                "suspended_item_index": 1,
                "split_items": [1, 2, 3],
                "original_input": [1, 2, 3],
            },
        )

        result = await mr.execute([1, 2, 3], resume_from=checkpoint)
        assert result.metadata["terminated"] == "cancelled"

    async def test_suspension_preserves_checkpoint_data(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def suspend_with_data(x: object) -> None:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="test-suspension",
                    request_id="test-req",
                    request_type="approval",
                    prompt="Approve?",
                ),
                checkpoint_data={"agent_state": "saved"},
            )

        mr = MapReduce(
            name="mr",
            step=make_step("s", suspend_with_data),
            emitter=emitter,
            splitter=lambda x: [x],
            reducer=lambda results: results,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await mr.execute("test")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {"agent_state": "saved"}

    async def test_without_checkpoint_store_propagates(self) -> None:
        emitter = make_emitter()

        async def suspend(x):
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="test-suspension",
                    request_id="test-req",
                    request_type="approval",
                    prompt="Approve?",
                ),
            )

        mr = MapReduce(
            name="mr",
            step=make_step("s", suspend),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: results,
        )

        with pytest.raises(SuspendExecution):
            await mr.execute([1])


# ── Cross-Cutting ──────────────────────────────────────────


class TestCheckpointVersionValidation:
    async def test_version_mismatch_raises(self) -> None:
        emitter = make_emitter()

        seq = Sequential(
            name="seq",
            steps=[make_step("s1")],
            emitter=emitter,
        )

        checkpoint = RunCheckpoint(
            run_id="run-1",
            checkpoint_type="orchestration",
            schema_version=999,
            state={},
            suspension_info=SuspensionInfo(
                suspension_id="test",
                request_id="test",
                request_type="approval",
                prompt="test",
            ),
        )

        with pytest.raises(CheckpointVersionError) as exc_info:
            await seq.execute("input", resume_from=checkpoint)
        assert exc_info.value.expected_version == CHECKPOINT_SCHEMA_VERSION
        assert exc_info.value.actual_version == 999


class TestNestedOrchestratorPropagation:
    async def test_inner_without_store_propagates_to_outer(self) -> None:
        """Inner Sequential without store lets SuspendExecution propagate.
        Outer Sequential with store catches it and checkpoints."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        inner = Sequential(
            name="inner",
            steps=[make_suspending_step("inner-step")],
            emitter=emitter,
            # No checkpoint_store — propagates SuspendExecution
        )

        outer = Sequential(
            name="outer",
            steps=[make_step("outer-s1"), inner, make_step("outer-s3")],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await outer.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["suspended_step_index"] == 1

    async def test_agent_checkpoint_data_preserved(self) -> None:
        """SuspendExecution with checkpoint_data is preserved in the checkpoint."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def suspend_with_data(x):
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="test-suspension",
                    request_id="test-req",
                    request_type="approval",
                    prompt="Approve?",
                ),
                checkpoint_data={"agent_type": "react", "messages": ["msg1"]},
            )

        seq = Sequential(
            name="seq",
            steps=[make_step("s1", suspend_with_data)],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {
            "agent_type": "react",
            "messages": ["msg1"],
        }


# ── Checkpoint State Serialization ──────────────────────────


class TestCheckpointStateSerialization:
    """Tests that Pydantic BaseModel instances in checkpoint state are
    normalized to plain dicts via _normalize_for_serialization."""

    async def test_pydantic_model_output_normalized_in_completed_results(self) -> None:
        """A step returning a Pydantic model has its output normalized in the checkpoint."""
        from pydantic import BaseModel as PydanticBaseModel

        class StepOutput(PydanticBaseModel):
            score: int
            label: str

        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def return_model(x: Any) -> StepOutput:
            return StepOutput(score=42, label="high")

        seq = Sequential(
            name="seq",
            steps=[
                FunctionStep(name="s1", fn=return_model),
                make_suspending_step("s2"),
            ],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        # completed_results should contain a plain dict, not a Pydantic model
        s1_result = cp.state["completed_results"]["s1"]
        assert s1_result["output"] == {"score": 42, "label": "high"}
        assert isinstance(s1_result["output"], dict)

    async def test_pydantic_model_as_last_output_normalized(self) -> None:
        """last_output containing a Pydantic model is normalized to a dict."""
        from pydantic import BaseModel as PydanticBaseModel

        class StepOutput(PydanticBaseModel):
            value: int

        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def return_model(x: Any) -> StepOutput:
            return StepOutput(value=99)

        seq = Sequential(
            name="seq",
            steps=[
                FunctionStep(name="s1", fn=return_model),
                make_suspending_step("s2"),
            ],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        assert cp.state["last_output"] == {"value": 99}
        assert isinstance(cp.state["last_output"], dict)

    async def test_nested_pydantic_models_normalize_recursively(self) -> None:
        """Nested Pydantic models are fully normalized to JSON-serializable dicts."""
        import json

        from pydantic import BaseModel as PydanticBaseModel

        class Finding(PydanticBaseModel):
            source: str
            confidence: float

        class Report(PydanticBaseModel):
            title: str
            findings: list[Finding]

        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def return_report(x: Any) -> Report:
            return Report(
                title="Analysis",
                findings=[
                    Finding(source="crm", confidence=0.95),
                    Finding(source="web", confidence=0.7),
                ],
            )

        seq = Sequential(
            name="seq",
            steps=[
                FunctionStep(name="s1", fn=return_report),
                make_suspending_step("s2"),
            ],
            emitter=emitter,
            checkpoint_store=store,
            run_id="run-1",
        )

        with pytest.raises(SuspendExecution):
            await seq.execute("input")

        cp = await store.load("run-1")
        assert cp is not None
        output = cp.state["completed_results"]["s1"]["output"]
        assert output == {
            "title": "Analysis",
            "findings": [
                {"source": "crm", "confidence": 0.95},
                {"source": "web", "confidence": 0.7},
            ],
        }
        # Verify it's fully JSON-serializable
        json.dumps(cp.state)

    async def test_resume_from_normalized_checkpoint(self) -> None:
        """Resuming from a checkpoint with dict values (post-normalization) works."""
        emitter = make_emitter()
        received_input: list[Any] = []

        async def capture_s2(x: Any) -> str:
            received_input.append(x)
            return "final"

        seq = Sequential(
            name="seq",
            steps=[
                make_step("s1"),
                FunctionStep(name="s2", fn=capture_s2),
            ],
            emitter=emitter,
            run_id="run-1",
        )

        # Simulate a checkpoint where model output was normalized to a dict
        checkpoint = make_checkpoint(
            state={
                "orchestrator_type": "sequential",
                "suspended_step_index": 1,
                "completed_results": {
                    "s1": {"output": {"score": 42, "label": "high"}, "metadata": {}},
                },
                "last_output": {"score": 42, "label": "high"},
                "original_input": "input",
            },
            run_id="run-1",
        )

        result = await seq.execute("input", resume_from=checkpoint)
        assert result.output == "final"
        # The resumed step receives the dict
        assert received_input == [{"score": 42, "label": "high"}]

    async def test_non_model_outputs_unchanged(self) -> None:
        """Non-model outputs (str, int, dict, list, None) pass through unmodified."""
        from nanitics.composition.orchestration.workflow import Workflow

        assert Workflow._normalize_for_serialization("hello") == "hello"
        assert Workflow._normalize_for_serialization(42) == 42
        assert Workflow._normalize_for_serialization(None) is None
        assert Workflow._normalize_for_serialization({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}
        assert Workflow._normalize_for_serialization([1, "two", {"three": 3}]) == [1, "two", {"three": 3}]
