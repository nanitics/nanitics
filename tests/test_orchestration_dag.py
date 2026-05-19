import asyncio

import pytest

from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.dag import DAG, DAGNode
from nanitics.composition.orchestration.protocol import FailurePolicy, Step, StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.events import (
    ExecutionSuspendedEvent,
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.safety import CancellationToken
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Construction Validation ────────────────────────────────


class TestDAGConstruction:
    def test_empty_nodes_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="at least one node"):
            DAG(name="empty", nodes={}, emitter=emitter)

    def test_dangling_dependency_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="does not exist"):
            DAG(
                name="dangling",
                nodes={
                    "A": DAGNode(step=make_step("A"), depends_on=["B"]),
                },
                emitter=emitter,
            )

    def test_cycle_detection_raises(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="cycle"):
            DAG(
                name="cycle",
                nodes={
                    "A": DAGNode(step=make_step("A"), depends_on=["B"]),
                    "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                },
                emitter=emitter,
            )

    def test_three_node_cycle_detected(self) -> None:
        emitter = make_emitter()
        with pytest.raises(ValueError, match="cycle"):
            DAG(
                name="cycle3",
                nodes={
                    "A": DAGNode(step=make_step("A"), depends_on=["C"]),
                    "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                    "C": DAGNode(step=make_step("C"), depends_on=["B"]),
                },
                emitter=emitter,
            )

    def test_valid_graph_accepted(self) -> None:
        emitter = make_emitter()
        dag = DAG(
            name="valid",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
            },
            emitter=emitter,
        )
        assert isinstance(dag, Step)

    def test_single_node_dag(self) -> None:
        emitter = make_emitter()
        dag = DAG(
            name="single",
            nodes={"A": DAGNode(step=make_step("A"))},
            emitter=emitter,
        )
        assert dag.name == "single"


# ── Execution ──────────────────────────────────────────────


class TestDAGExecution:
    async def test_linear_chain(self) -> None:
        """A→B→C matches Sequential behavior."""
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        dag = DAG(
            name="chain",
            nodes={
                "A": DAGNode(step=make_step("A", add_one)),
                "B": DAGNode(step=make_step("B", add_one), depends_on=["A"]),
                "C": DAGNode(step=make_step("C", add_one), depends_on=["B"]),
            },
            emitter=emitter,
        )

        result = await dag.execute(0)
        assert result.output == 3  # 0+1+1+1

    async def test_diamond(self) -> None:
        """A→B, A→C, B+C→D"""
        emitter = make_emitter()

        async def double(x):
            return x * 2

        async def triple(x):
            return x * 3

        async def combine(x):
            # Multi-dep: x is dict
            return x["B"] + x["C"]

        dag = DAG(
            name="diamond",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B", double), depends_on=["A"]),
                "C": DAGNode(step=make_step("C", triple), depends_on=["A"]),
                "D": DAGNode(step=make_step("D", combine), depends_on=["B", "C"]),
            },
            emitter=emitter,
        )

        result = await dag.execute(5)
        # A passes 5, B=10, C=15, D=10+15=25
        assert result.output == 25

    async def test_wide_fan_out(self) -> None:
        """A→B, A→C, A→D independently — multiple terminal nodes."""
        emitter = make_emitter()

        async def add_suffix(x):
            return f"{x}-B"

        async def add_suffix_c(x):
            return f"{x}-C"

        async def add_suffix_d(x):
            return f"{x}-D"

        dag = DAG(
            name="fan-out",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B", add_suffix), depends_on=["A"]),
                "C": DAGNode(step=make_step("C", add_suffix_c), depends_on=["A"]),
                "D": DAGNode(step=make_step("D", add_suffix_d), depends_on=["A"]),
            },
            emitter=emitter,
        )

        result = await dag.execute("root")
        # Multiple terminal nodes → dict
        assert result.output == {
            "B": "root-B",
            "C": "root-C",
            "D": "root-D",
        }

    async def test_single_source_node(self) -> None:
        emitter = make_emitter()

        async def double(x):
            return x * 2

        dag = DAG(
            name="single",
            nodes={"A": DAGNode(step=make_step("A", double))},
            emitter=emitter,
        )

        result = await dag.execute(7)
        assert result.output == 14


# ── Data Flow ──────────────────────────────────────────────


class TestDAGDataFlow:
    async def test_source_nodes_receive_dag_input(self) -> None:
        emitter = make_emitter()
        received: list = []

        async def capture(x):
            received.append(x)
            return x

        dag = DAG(
            name="source",
            nodes={"A": DAGNode(step=make_step("A", capture))},
            emitter=emitter,
        )

        await dag.execute("dag-input")
        assert received == ["dag-input"]

    async def test_single_dep_receives_output_directly(self) -> None:
        emitter = make_emitter()
        received: list = []

        async def produce(x):
            return {"key": "value"}

        async def capture(x):
            received.append(x)
            return x

        dag = DAG(
            name="single-dep",
            nodes={
                "A": DAGNode(step=make_step("A", produce)),
                "B": DAGNode(step=make_step("B", capture), depends_on=["A"]),
            },
            emitter=emitter,
        )

        await dag.execute("input")
        # B receives A's output directly (not wrapped in dict)
        assert received == [{"key": "value"}]

    async def test_multi_dep_receives_dict_of_outputs(self) -> None:
        emitter = make_emitter()
        received: list = []

        async def produce_x(inp):
            return "X"

        async def produce_y(inp):
            return "Y"

        async def capture(x):
            received.append(x)
            return x

        dag = DAG(
            name="multi-dep",
            nodes={
                "A": DAGNode(step=make_step("A", produce_x)),
                "B": DAGNode(step=make_step("B", produce_y)),
                "C": DAGNode(step=make_step("C", capture), depends_on=["A", "B"]),
            },
            emitter=emitter,
        )

        await dag.execute("input")
        assert received == [{"A": "X", "B": "Y"}]


# ── Output ─────────────────────────────────────────────────


class TestDAGOutput:
    async def test_single_terminal_node_output(self) -> None:
        emitter = make_emitter()

        async def double(x):
            return x * 2

        dag = DAG(
            name="single-terminal",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B", double), depends_on=["A"]),
            },
            emitter=emitter,
        )

        result = await dag.execute(5)
        assert result.output == 10  # Single terminal → direct output

    async def test_multiple_terminal_nodes_produce_dict(self) -> None:
        emitter = make_emitter()

        dag = DAG(
            name="multi-terminal",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                "C": DAGNode(step=make_step("C"), depends_on=["A"]),
            },
            emitter=emitter,
        )

        result = await dag.execute("x")
        assert result.output == {"B": "x", "C": "x"}


# ── Concurrency ────────────────────────────────────────────


class TestDAGConcurrency:
    async def test_independent_nodes_run_concurrently(self) -> None:
        emitter = make_emitter()
        concurrent_count = 0
        max_concurrent = 0

        async def tracked(x):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return x

        dag = DAG(
            name="concurrent",
            nodes={
                "A": DAGNode(step=make_step("A", tracked)),
                "B": DAGNode(step=make_step("B", tracked)),
                "C": DAGNode(step=make_step("C", tracked)),
            },
            emitter=emitter,
        )

        await dag.execute("x")
        assert max_concurrent == 3

    async def test_max_concurrency_limits_parallelism(self) -> None:
        emitter = make_emitter()
        concurrent_count = 0
        max_concurrent = 0

        async def tracked(x):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return x

        dag = DAG(
            name="limited",
            nodes={
                "A": DAGNode(step=make_step("A", tracked)),
                "B": DAGNode(step=make_step("B", tracked)),
                "C": DAGNode(step=make_step("C", tracked)),
                "D": DAGNode(step=make_step("D", tracked)),
            },
            emitter=emitter,
            max_concurrency=2,
        )

        await dag.execute("x")
        assert max_concurrent <= 2


# ── Failure Handling ───────────────────────────────────────


class TestDAGFailure:
    async def test_all_or_nothing_failure_cancels_dependents(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise ValueError("node failed")

        dag = DAG(
            name="aon",
            nodes={
                "A": DAGNode(step=make_step("A", fail)),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.ALL_OR_NOTHING,
        )

        with pytest.raises(ValueError, match="node failed"):
            await dag.execute("x")

    async def test_best_effort_independent_branches_complete(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise ValueError("fail")

        async def succeed(x):
            return "ok"

        dag = DAG(
            name="be",
            nodes={
                "A": DAGNode(step=make_step("A", fail)),
                "B": DAGNode(step=make_step("B", succeed)),
                "C": DAGNode(step=make_step("C"), depends_on=["A"]),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert "A" in result.metadata["failed_nodes"]
        assert result.metadata["failed_nodes"]["A"]["error_type"] == "ValueError"
        assert result.metadata["failed_nodes"]["A"]["error_message"] == "fail"
        assert "C" in result.metadata["skipped_nodes"]
        assert result.metadata["node_results"]["B"] == "ok"

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].failed_step == "A"
        assert error_events[0].error_type == "ValueError"
        assert error_events[0].error_message == "fail"
        assert error_events[0].workflow_name == "be"

    async def test_best_effort_transitive_dependents_skipped(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise ValueError("fail")

        dag = DAG(
            name="be-transitive",
            nodes={
                "A": DAGNode(step=make_step("A", fail)),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                "C": DAGNode(step=make_step("C"), depends_on=["B"]),
                "D": DAGNode(step=make_step("D")),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert "A" in result.metadata["failed_nodes"]
        assert result.metadata["failed_nodes"]["A"]["error_type"] == "ValueError"
        assert "B" in result.metadata["skipped_nodes"]
        assert "C" in result.metadata["skipped_nodes"]
        assert "D" not in result.metadata.get("failed_nodes", {})
        assert "D" not in result.metadata.get("skipped_nodes", [])


# ── Cancellation ───────────────────────────────────────────


class TestDAGCancellation:
    async def test_respects_cancellation_token(self) -> None:
        emitter = make_emitter()
        token = CancellationToken()
        executed: list[str] = []

        async def cancel_after(x):
            executed.append("A")
            token.cancel()
            return x

        async def should_not_run(x):
            executed.append("B")
            return x

        dag = DAG(
            name="cancel",
            nodes={
                "A": DAGNode(step=make_step("A", cancel_after)),
                "B": DAGNode(step=make_step("B", should_not_run), depends_on=["A"]),
            },
            emitter=emitter,
            cancellation_token=token,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert result.metadata["terminated"] == "cancelled"
        assert executed == ["A"]
        assert "B" not in result.metadata.get("node_results", {})


# ── Events ─────────────────────────────────────────────────


class TestDAGEvents:
    async def test_workflow_start_with_topology_metadata(self) -> None:
        emitter = make_emitter()

        dag = DAG(
            name="evented",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
            },
            emitter=emitter,
        )

        await dag.execute("x")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        assert len(start_events) == 1
        start = start_events[0]
        assert start.workflow_type == "dag"
        assert start.step_count == 2
        assert set(start.metadata["nodes"]) == {"A", "B"}
        assert ["A", "B"] in start.metadata["edges"]

    async def test_step_complete_per_node(self) -> None:
        emitter = make_emitter()

        dag = DAG(
            name="evented",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                "C": DAGNode(step=make_step("C"), depends_on=["A"]),
            },
            emitter=emitter,
        )

        await dag.execute("x")

        step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
        step_names = {e.step_name for e in step_events}
        assert step_names == {"A", "B", "C"}

    async def test_complete_event(self) -> None:
        emitter = make_emitter()

        dag = DAG(
            name="evented",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
            },
            emitter=emitter,
        )

        await dag.execute("x")

        complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].workflow_type == "dag"
        assert complete_events[0].total_steps_executed == 2

    async def test_error_event_on_failure(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise RuntimeError("boom")

        dag = DAG(
            name="err",
            nodes={"A": DAGNode(step=make_step("A", fail))},
            emitter=emitter,
        )

        with pytest.raises(RuntimeError):
            await dag.execute("x")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].workflow_type == "dag"


# ── Nesting ────────────────────────────────────────────────


class TestDAGNesting:
    async def test_dag_node_contains_sequential(self) -> None:
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        inner_seq = Sequential(
            name="inner-seq",
            steps=[make_step("s1", add_one), make_step("s2", add_one)],
            emitter=emitter,
        )

        dag = DAG(
            name="dag-with-seq",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=DAGNode(step=inner_seq).step, depends_on=["A"]),
            },
            emitter=emitter,
        )

        result = await dag.execute(0)
        # A passes 0, inner_seq: 0+1+1=2
        assert result.output == 2


# ── Partial Cycle Detection ───────────────────────────────


class TestDAGPartialCycle:
    def test_partial_cycle_detected(self) -> None:
        """A graph where some nodes are valid but a subset forms a cycle."""
        emitter = make_emitter()
        with pytest.raises(ValueError, match="cycle involving nodes"):
            DAG(
                name="partial-cycle",
                nodes={
                    "A": DAGNode(step=make_step("A")),
                    "B": DAGNode(step=make_step("B"), depends_on=["A", "C"]),
                    "C": DAGNode(step=make_step("C"), depends_on=["B"]),
                },
                emitter=emitter,
            )


# ── Suspension / Checkpoint ───────────────────────────────


def _make_suspension() -> SuspendExecution:
    return SuspendExecution(
        suspension_info=SuspensionInfo(
            suspension_id="test-suspension",
            request_id="test-request",
            request_type="approval",
            prompt="Approve?",
            agent_name="test-agent",
        ),
    )


def _make_checkpoint(state: dict) -> RunCheckpoint:
    return RunCheckpoint(
        run_id="test-run",
        checkpoint_type="orchestration",
        state=state,
        suspension_info=SuspensionInfo(
            suspension_id="test-suspension",
            request_id="test-request",
            request_type="approval",
            prompt="Approve?",
        ),
    )


class TestDAGSuspension:
    async def test_suspension_saves_checkpoint_and_emits_event(self) -> None:
        """Suspension in a DAG node saves checkpoint with completed nodes and emits event."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def succeed(x: object) -> str:
            return "done"

        async def suspend(x: object) -> None:
            raise _make_suspension()

        dag = DAG(
            name="dag-suspend",
            nodes={
                "A": DAGNode(step=make_step("A", succeed)),
                "B": DAGNode(step=make_step("B", suspend), depends_on=["A"]),
            },
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        assert cp.state["orchestrator_type"] == "dag"
        assert cp.state["completed_nodes"] == {"A": "done"}
        assert cp.state["suspended_node"] == "B"

        suspended_events = [e for e in emitter.events if isinstance(e, ExecutionSuspendedEvent)]
        assert len(suspended_events) == 1
        assert suspended_events[0].step_name == "B"
        assert suspended_events[0].agent_name == "test-agent"

    async def test_suspension_with_checkpoint_data(self) -> None:
        """SuspendExecution with checkpoint_data includes it in the checkpoint state."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()

        async def suspend(x: object) -> None:
            raise SuspendExecution(
                suspension_info=SuspensionInfo(
                    suspension_id="test-suspension",
                    request_id="test-request",
                    request_type="approval",
                    prompt="Approve?",
                ),
                checkpoint_data={"agent_state": "paused"},
            )

        dag = DAG(
            name="dag-cp-data",
            nodes={"A": DAGNode(step=make_step("A", suspend))},
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        assert cp.state["agent_checkpoint"] == {"agent_state": "paused"}

    async def test_suspension_without_checkpoint_store_still_raises(self) -> None:
        """Suspension propagates even without a checkpoint store."""
        emitter = make_emitter()

        async def suspend(x: object) -> None:
            raise _make_suspension()

        dag = DAG(
            name="dag-no-store",
            nodes={"A": DAGNode(step=make_step("A", suspend))},
            emitter=emitter,
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("input")

    async def test_suspension_drains_in_flight_tasks(self) -> None:
        """When one node suspends, other in-flight nodes are drained and their results kept."""
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        order: list[str] = []

        async def slow_succeed(x: object) -> str:
            order.append("B-start")
            await asyncio.sleep(0.05)
            order.append("B-done")
            return "B-result"

        async def fast_suspend(x: object) -> None:
            order.append("A-suspend")
            raise _make_suspension()

        dag = DAG(
            name="drain",
            nodes={
                "A": DAGNode(step=make_step("A", fast_suspend)),
                "B": DAGNode(step=make_step("B", slow_succeed)),
            },
            emitter=emitter,
            checkpoint_store=store,
            run_id="test-run",
        )

        with pytest.raises(SuspendExecution):
            await dag.execute("input")

        cp = await store.load("test-run")
        assert cp is not None
        # B should have completed and been drained
        assert cp.state["completed_nodes"].get("B") == "B-result"

    async def test_resume_from_checkpoint(self) -> None:
        """Resuming from a checkpoint skips completed nodes and re-runs from suspension point."""
        emitter = make_emitter()
        executed: list[str] = []

        async def track_a(x: object) -> str:
            executed.append("A")
            return "A-result"

        async def track_b(x: object) -> str:
            executed.append("B")
            return "B-result"

        async def track_c(x: object) -> str:
            executed.append("C")
            return "C-result"

        dag = DAG(
            name="resume",
            nodes={
                "A": DAGNode(step=make_step("A", track_a)),
                "B": DAGNode(step=make_step("B", track_b), depends_on=["A"]),
                "C": DAGNode(step=make_step("C", track_c), depends_on=["A"]),
            },
            emitter=emitter,
        )

        checkpoint = _make_checkpoint(
            state={
                "orchestrator_type": "dag",
                "completed_nodes": {"A": "A-result"},
                "suspended_node": "B",
                "original_input": "input",
            },
        )

        result = await dag.execute("input", resume_from=checkpoint)
        # A should be skipped (already completed), B and C should run
        assert "A" not in executed
        assert "B" in executed
        assert "C" in executed
        assert result.output is not None


# ── ALL_OR_NOTHING with concurrent in-flight ──────────────


class TestDAGAllOrNothingConcurrent:
    async def test_all_or_nothing_cancels_concurrent_in_flight(self) -> None:
        """When a node fails under ALL_OR_NOTHING with other nodes in-flight, those are cancelled."""
        emitter = make_emitter()

        async def slow_node(x: object) -> str:
            await asyncio.sleep(1)
            return "should-not-finish"

        async def fail_fast(x: object) -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")

        dag = DAG(
            name="aon-concurrent",
            nodes={
                "A": DAGNode(step=make_step("A", fail_fast)),
                "B": DAGNode(step=make_step("B", slow_node)),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.ALL_OR_NOTHING,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await dag.execute("x")


# ── Cancellation with concurrent in-flight ────────────────


class TestDAGCancellationConcurrent:
    async def test_cancellation_cancels_pending_in_flight_task(self) -> None:
        """When a node returns cancelled, genuinely pending in-flight tasks are cancelled."""
        emitter = make_emitter()
        token = CancellationToken()

        async def cancel_and_sleep(x: object) -> str:
            token.cancel()
            await asyncio.sleep(0.5)  # Yields control; will be cancelled
            return "unreachable"

        # Three source nodes launched concurrently. A runs first (dict order / FIFO):
        # A cancels token then yields at sleep → A is pending.
        # B checks token (cancelled) → returns cancelled → B completes.
        # C checks token (cancelled) → returns cancelled → C completes.
        # asyncio.wait returns B and C as done, A as pending.
        # Processing B: A is in in_flight with done()=False → t.cancel() called.
        dag = DAG(
            name="cancel-pending",
            nodes={
                "A": DAGNode(step=make_step("A", cancel_and_sleep)),
                "B": DAGNode(step=make_step("B")),
                "C": DAGNode(step=make_step("C")),
            },
            emitter=emitter,
            cancellation_token=token,
        )

        result = await dag.execute("x")
        assert result.metadata["terminated"] == "cancelled"

    async def test_cancellation_drains_and_cancels_in_flight(self) -> None:
        """When a node returns cancelled result with other tasks in-flight, those are cancelled and gathered."""
        emitter = make_emitter()

        async def return_cancelled(x: object) -> StepResult:
            return StepResult(output=None, metadata={"terminated": "cancelled"})

        async def slow_task(x: object) -> str:
            await asyncio.sleep(0.5)
            return "unreachable"

        # A returns cancelled immediately; B sleeps forever.
        # asyncio.wait returns A as done, B as pending.
        # Processing A: B is in in_flight, B.done()=False → B gets cancelled, then gathered.
        dag = DAG(
            name="cancel-drain",
            nodes={
                "A": DAGNode(step=make_step("A", return_cancelled)),
                "B": DAGNode(step=make_step("B", slow_task)),
            },
            emitter=emitter,
        )

        result = await dag.execute("x")
        assert result.output is None


# ── Best-effort skip dependents of in-flight failed ───────


class TestDAGBestEffortInFlight:
    async def test_best_effort_cancels_in_flight_dependent(self) -> None:
        """When a node fails under BEST_EFFORT, already-in-flight dependents are cancelled."""
        emitter = make_emitter()

        async def fail_after_delay(x: object) -> None:
            await asyncio.sleep(0.01)
            raise ValueError("fail-A")

        async def slow_independent(x: object) -> str:
            await asyncio.sleep(0.05)
            return "C-done"

        dag = DAG(
            name="be-inflight",
            nodes={
                "A": DAGNode(step=make_step("A", fail_after_delay)),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                "C": DAGNode(step=make_step("C", slow_independent)),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert "A" in result.metadata["failed_nodes"]
        assert "B" in result.metadata["skipped_nodes"]
        assert result.metadata["node_results"]["C"] == "C-done"

    async def test_best_effort_skips_already_skipped_dependent(self) -> None:
        """When a dependent is already in skipped_nodes, it's not re-added."""
        emitter = make_emitter()

        async def fail(x: object) -> None:
            raise ValueError("fail")

        dag = DAG(
            name="be-double-skip",
            nodes={
                "A": DAGNode(step=make_step("A", fail)),
                "B": DAGNode(step=make_step("B", fail)),
                # C depends on both A and B — both fail, so C gets skip-attempted twice
                "C": DAGNode(step=make_step("C"), depends_on=["A", "B"]),
                "D": DAGNode(step=make_step("D")),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert "C" in result.metadata["skipped_nodes"]
        # C should appear exactly once in skipped
        assert result.metadata["skipped_nodes"].count("C") == 1

    async def test_best_effort_skipped_dep_not_unblocked_on_success(self) -> None:
        """After a node succeeds, its dependents that are already skipped are not re-added to ready."""
        emitter = make_emitter()

        async def fail(x: object) -> None:
            raise ValueError("fail")

        async def slow_succeed(x: object) -> str:
            await asyncio.sleep(0.05)
            return str(x)

        # A (source) fails instantly, C (source) succeeds after delay.
        # A fails first → B skipped → D skipped (transitively).
        # Then C completes → unblocks D → but D already in skipped_nodes → continue.
        dag = DAG(
            name="be-skip-unblock",
            nodes={
                "A": DAGNode(step=make_step("A", fail)),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
                "C": DAGNode(step=make_step("C", slow_succeed)),
                "D": DAGNode(step=make_step("D"), depends_on=["B", "C"]),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert "A" in result.metadata["failed_nodes"]
        assert "B" in result.metadata["skipped_nodes"]
        assert "D" in result.metadata["skipped_nodes"]
        assert result.metadata["node_results"]["C"] == "x"


# ── No terminal nodes (all failed/skipped) ────────────────


class TestDAGNoTerminalNodes:
    async def test_all_nodes_failed_returns_none(self) -> None:
        """When all nodes fail or are skipped, output is None."""
        emitter = make_emitter()

        async def fail(x: object) -> None:
            raise ValueError("fail")

        dag = DAG(
            name="all-fail",
            nodes={
                "A": DAGNode(step=make_step("A", fail)),
                "B": DAGNode(step=make_step("B"), depends_on=["A"]),
            },
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await dag.execute("x")
        assert result.output is None


@pytest.mark.parametrize("value", [0, -1])
def test_dag_rejects_non_positive_max_concurrency(value: int) -> None:
    emitter = make_emitter()

    async def noop(inp: str) -> StepResult:
        return StepResult(output=inp)  # pragma: no cover

    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        DAG(
            name="test",
            nodes={"A": DAGNode(step=make_step("A", noop))},
            emitter=emitter,
            max_concurrency=value,
        )
