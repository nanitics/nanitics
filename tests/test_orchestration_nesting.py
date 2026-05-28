from typing import Any

from nanitics.composition.orchestration.conditional import Conditional
from nanitics.composition.orchestration.dag import DAG, DAGNode
from nanitics.composition.orchestration.loop import Loop
from nanitics.composition.orchestration.mapreduce import MapReduce
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.composition.orchestration.pipeline import Pipeline, Stage
from nanitics.composition.orchestration.protocol import StepResult
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.events import (
    SpanEndEvent,
    SpanStartEvent,
    WorkflowCompleteEvent,
    WorkflowStartEvent,
)
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Sequential containing Parallel ─────────────────────────


class TestSequentialContainingParallel:
    async def test_sequential_with_parallel_step(self) -> None:
        """A Sequential whose second step is a Parallel."""
        emitter = make_emitter()

        async def add_prefix(x):
            return f"prefixed-{x}"

        parallel = Parallel(
            name="fan-out",
            steps=[
                make_step("upper", fn=lambda x: _async(str(x).upper())),
                make_step("lower", fn=lambda x: _async(str(x).lower())),
            ],
            emitter=emitter,
        )

        seq = Sequential(
            name="seq-with-parallel",
            steps=[
                make_step("prefix", add_prefix),
                parallel,
            ],
            emitter=emitter,
        )

        result = await seq.execute("Hello")
        # prefix: "prefixed-Hello", parallel fans out to upper + lower
        assert result.output == ["PREFIXED-HELLO", "prefixed-hello"]

    async def test_sequential_parallel_preserves_metadata(self) -> None:
        emitter = make_emitter()

        parallel = Parallel(
            name="par",
            steps=[make_step("a"), make_step("b")],
            emitter=emitter,
        )

        seq = Sequential(
            name="seq",
            steps=[make_step("first"), parallel],
            emitter=emitter,
        )

        result = await seq.execute("data")
        assert "intermediate_results" in result.metadata
        par_step = result.metadata["intermediate_results"]["par"]
        assert isinstance(par_step, StepResult)
        assert "total_steps_executed" in par_step.metadata


# ── Sequential exposing Parallel failed_steps ───────────────


class TestSequentialExposesParallelFailedSteps:
    async def test_failed_steps_accessible_through_intermediate_results(self) -> None:
        """Sequential wrapping a BEST_EFFORT Parallel exposes failed_steps."""
        from nanitics.composition.orchestration.parallel import FailurePolicy

        emitter = make_emitter()

        async def succeed(x):
            return x

        async def fail(x):
            raise RuntimeError("boom")

        parallel = Parallel(
            name="parallel",
            steps=[
                make_step("good", succeed),
                make_step("bad", fail),
            ],
            emitter=emitter,
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        seq = Sequential(
            name="seq",
            steps=[make_step("pre", succeed), parallel],
            emitter=emitter,
        )

        result = await seq.execute("data")
        par_step = result.metadata["intermediate_results"]["parallel"]
        assert isinstance(par_step, StepResult)
        assert "bad" in par_step.metadata["failed_steps"]


# ── Conditional with Sequential branch ──────────────────────


class TestConditionalWithSequentialBranch:
    async def test_conditional_routes_to_sequential(self) -> None:
        """Conditional selects a Sequential branch."""
        emitter = make_emitter()

        async def double(x):
            return x * 2

        async def add_ten(x):
            return x + 10

        seq_branch = Sequential(
            name="double-then-add",
            steps=[
                make_step("double", double),
                make_step("add-ten", add_ten),
            ],
            emitter=emitter,
        )

        cond = Conditional(
            name="route",
            router=lambda x: "process" if isinstance(x, int) else "pass",
            branches={
                "process": seq_branch,
                "pass": make_step("identity"),
            },
            emitter=emitter,
        )

        result = await cond.execute(5)
        assert result.output == 20  # (5*2) + 10
        assert result.metadata["selected_branch"] == "process"

    async def test_conditional_non_matching_branch(self) -> None:
        emitter = make_emitter()

        seq_default = Sequential(
            name="default-seq",
            steps=[make_step("passthrough")],
            emitter=emitter,
        )

        cond = Conditional(
            name="route",
            router=lambda x: "unknown",
            branches={"known": make_step("known")},
            default=seq_default,
            emitter=emitter,
        )

        result = await cond.execute("data")
        assert result.output == "data"


# ── Loop body is Sequential ────────────────────────────────


class TestLoopWithSequentialBody:
    async def test_loop_with_sequential_body(self) -> None:
        """Loop body is a Sequential that increments and doubles."""
        emitter = make_emitter()

        async def increment(x):
            return x + 1

        async def double(x):
            return x * 2

        body = Sequential(
            name="inc-double",
            steps=[
                make_step("increment", increment),
                make_step("double", double),
            ],
            emitter=emitter,
        )

        loop = Loop(
            name="loop-seq",
            step=body,
            condition=lambda result, iteration: result.output >= 20,
            max_iterations=10,
            emitter=emitter,
        )

        # Iteration 1: (0+1)*2 = 2
        # Iteration 2: (2+1)*2 = 6
        # Iteration 3: (6+1)*2 = 14
        # Iteration 4: (14+1)*2 = 30 → condition met (>= 20)
        result = await loop.execute(0)
        assert result.output == 30
        assert result.metadata["iterations"] == 4


# ── Three-level nesting: Loop → Sequential → FunctionStep ──


class TestThreeLevelNesting:
    async def test_loop_sequential_function(self) -> None:
        """Three levels: Loop wrapping a Sequential of FunctionSteps."""
        emitter = make_emitter()

        async def append_a(x):
            return x + "a"

        async def append_b(x):
            return x + "b"

        body = Sequential(
            name="append-ab",
            steps=[
                make_step("append-a", append_a),
                make_step("append-b", append_b),
            ],
            emitter=emitter,
        )

        loop = Loop(
            name="build-string",
            step=body,
            condition=lambda result, iteration: len(result.output) >= 6,
            max_iterations=10,
            emitter=emitter,
        )

        # Iteration 1: "" + "a" + "b" = "ab"
        # Iteration 2: "ab" + "a" + "b" = "abab"
        # Iteration 3: "abab" + "a" + "b" = "ababab" → len=6, stop
        result = await loop.execute("")
        assert result.output == "ababab"
        assert result.metadata["iterations"] == 3


# ── Span hierarchy verification ─────────────────────────────


class TestSpanHierarchy:
    async def test_nested_span_hierarchy(self) -> None:
        """Verify span nesting is correct for Sequential containing Parallel."""
        emitter = make_emitter()

        parallel = Parallel(
            name="inner-parallel",
            steps=[make_step("p1"), make_step("p2")],
            emitter=emitter,
        )

        seq = Sequential(
            name="outer-seq",
            steps=[make_step("first"), parallel],
            emitter=emitter,
        )

        await seq.execute("x")

        # Collect span events
        span_starts = [e for e in emitter.events if isinstance(e, SpanStartEvent)]
        span_ends = [e for e in emitter.events if isinstance(e, SpanEndEvent)]

        # We expect spans for: outer-seq, first, inner-parallel, p1, p2
        span_names = [s.name for s in span_starts]
        assert "outer-seq" in span_names
        assert "first" in span_names
        assert "inner-parallel" in span_names
        assert "p1" in span_names
        assert "p2" in span_names

        # Verify outer-seq span starts first and ends last
        assert span_starts[0].name == "outer-seq"
        assert span_ends[-1].name == "outer-seq"

    async def test_three_level_span_hierarchy(self) -> None:
        """Verify span nesting for Loop → Sequential → FunctionStep."""
        emitter = make_emitter()

        body = Sequential(
            name="inner-seq",
            steps=[make_step("step-a")],
            emitter=emitter,
        )

        loop = Loop(
            name="outer-loop",
            step=body,
            condition=lambda result, iteration: iteration >= 2,
            max_iterations=5,
            emitter=emitter,
        )

        await loop.execute("data")

        span_starts = [e for e in emitter.events if isinstance(e, SpanStartEvent)]
        span_names = [s.name for s in span_starts]

        # outer-loop span should be first
        assert span_names[0] == "outer-loop"
        # iteration spans and inner-seq spans should be present
        assert "inner-seq-iteration-1" in span_names
        assert "inner-seq-iteration-2" in span_names

    async def test_workflow_events_emitted_at_each_nesting_level(self) -> None:
        """Each nested workflow emits its own start/complete events."""
        emitter = make_emitter()

        inner = Sequential(
            name="inner",
            steps=[make_step("s1")],
            emitter=emitter,
        )

        outer = Sequential(
            name="outer",
            steps=[inner],
            emitter=emitter,
        )

        await outer.execute("data")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]

        # Both outer and inner should emit start/complete
        start_names = [e.workflow_name for e in start_events]
        complete_names = [e.workflow_name for e in complete_events]

        assert "outer" in start_names
        assert "inner" in start_names
        assert "outer" in complete_names
        assert "inner" in complete_names


# ── Async helper ────────────────────────────────────────────


async def _async(value: Any) -> Any:
    return value


# ── Pipeline inside Sequential ──────────────────────────────


class TestPipelineInsideSequential:
    async def test_pipeline_as_sequential_step(self) -> None:
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        async def double(x):
            return x * 2

        pipeline = Pipeline(
            name="inner-pipeline",
            stages=[
                Stage(make_step("add", add_one)),
                Stage(make_step("double", double)),
            ],
            emitter=emitter,
        )

        seq = Sequential(
            name="outer-seq",
            steps=[make_step("identity"), pipeline],
            emitter=emitter,
        )

        result = await seq.execute(5)
        # identity: 5, pipeline: (5+1)*2 = 12
        assert result.output == 12


# ── MapReduce inside Sequential ─────────────────────────────


class TestMapReduceInsideSequential:
    async def test_mapreduce_as_sequential_step(self) -> None:
        emitter = make_emitter()

        async def double(x):
            return x * 2

        mr = MapReduce(
            name="inner-mr",
            step=make_step("double", double),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: sum(r.output for r in results),
        )

        seq = Sequential(
            name="outer-seq",
            steps=[make_step("identity"), mr],
            emitter=emitter,
        )

        result = await seq.execute([1, 2, 3])
        # identity passes through [1,2,3], MR doubles and sums: 2+4+6=12
        assert result.output == 12


# ── DAG containing Pipeline and MapReduce ────────────────────


class TestDAGWithPipelineAndMapReduce:
    async def test_dag_with_pipeline_and_mapreduce_nodes(self) -> None:
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        async def double(x):
            return x * 2

        pipeline = Pipeline(
            name="pipeline-node",
            stages=[
                Stage(make_step("add", add_one)),
                Stage(make_step("double", double)),
            ],
            emitter=emitter,
        )

        mr = MapReduce(
            name="mr-node",
            step=make_step("double", double),
            emitter=emitter,
            splitter=lambda x: [x, x, x],
            reducer=lambda results: sum(r.output for r in results),
        )

        async def combine(x):
            return x["pipeline"] + x["mapreduce"]

        dag = DAG(
            name="outer-dag",
            nodes={
                "source": DAGNode(step=make_step("source")),
                "pipeline": DAGNode(step=pipeline, depends_on=["source"]),
                "mapreduce": DAGNode(step=mr, depends_on=["source"]),
                "combine": DAGNode(
                    step=make_step("combine", combine),
                    depends_on=["pipeline", "mapreduce"],
                ),
            },
            emitter=emitter,
        )

        result = await dag.execute(5)
        # source: 5
        # pipeline: (5+1)*2 = 12
        # mapreduce: double(5)*3 = 30
        # combine: 12 + 30 = 42
        assert result.output == 42


# ── Sequential inside DAG node ───────────────────────────────


class TestSequentialInsideDAG:
    async def test_sequential_as_dag_node(self) -> None:
        emitter = make_emitter()

        async def add_one(x):
            return x + 1

        inner_seq = Sequential(
            name="inner-seq",
            steps=[make_step("s1", add_one), make_step("s2", add_one)],
            emitter=emitter,
        )

        dag = DAG(
            name="outer-dag",
            nodes={
                "A": DAGNode(step=make_step("A")),
                "B": DAGNode(step=inner_seq, depends_on=["A"]),
            },
            emitter=emitter,
        )

        result = await dag.execute(0)
        assert result.output == 2  # 0 + 1 + 1


# ── Event hierarchy for nested workflows ─────────────────────


class TestNestedWorkflowEventHierarchy:
    async def test_nested_dag_pipeline_events(self) -> None:
        """Verify event hierarchy across DAG containing Pipeline."""
        emitter = make_emitter()

        pipeline = Pipeline(
            name="inner-pipeline",
            stages=[Stage(make_step("s1"))],
            emitter=emitter,
        )

        dag = DAG(
            name="outer-dag",
            nodes={
                "A": DAGNode(step=pipeline),
            },
            emitter=emitter,
        )

        await dag.execute("x")

        start_events = [e for e in emitter.events if isinstance(e, WorkflowStartEvent)]
        complete_events = [e for e in emitter.events if isinstance(e, WorkflowCompleteEvent)]

        start_names = [e.workflow_name for e in start_events]
        complete_names = [e.workflow_name for e in complete_events]

        assert "outer-dag" in start_names
        assert "inner-pipeline" in start_names
        assert "outer-dag" in complete_names
        assert "inner-pipeline" in complete_names

        # DAG starts first
        assert start_names.index("outer-dag") < start_names.index("inner-pipeline")


# ── Recursive usage aggregation ───────────────────────────


class TestRecursiveUsageAggregation:
    async def test_workflow_step_folds_inner_aggregate_into_outer(self) -> None:
        from nanitics.composition.orchestration.adapters import AgentStep, WorkflowStep
        from nanitics.infrastructure import MockLLMClient
        from nanitics.infrastructure.observability.events import Usage
        from nanitics.strategies import ReasoningAgent
        from tests.testing_helpers import make_response, make_usage

        emitter = make_emitter()

        def agent_step(name: str, in_t: int, out_t: int):
            client = MockLLMClient([make_response("ok", usage=make_usage(in_t, out_t))])
            return AgentStep(ReasoningAgent(name=name, llm_client=client, emitter=emitter, system_prompt="t"))

        inner = Sequential(
            name="inner",
            steps=[agent_step("a", 1, 2), agent_step("b", 3, 4)],
            emitter=emitter,
        )
        outer = Sequential(
            name="outer",
            steps=[WorkflowStep(inner), agent_step("c", 5, 6)],
            emitter=emitter,
        )
        result = await outer.execute("task")
        # Inner aggregates to (4,6); outer folds inner aggregate + outer agent (5,6) = (9,12).
        assert result.usage == Usage(input_tokens=9, output_tokens=12)
