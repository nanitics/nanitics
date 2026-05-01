"""Tests for WorkflowStructureEvent emission from all workflow types."""

from nanitics import InMemoryEmitter
from nanitics.composition.orchestration.adapters import WorkflowStep
from nanitics.composition.orchestration.conditional import Conditional
from nanitics.composition.orchestration.dag import DAG, DAGNode
from nanitics.composition.orchestration.loop import Loop
from nanitics.composition.orchestration.mapreduce import MapReduce
from nanitics.composition.orchestration.parallel import Parallel
from nanitics.composition.orchestration.pipeline import Pipeline, Stage
from nanitics.composition.orchestration.sequential import Sequential
from nanitics.infrastructure.observability.events import (
    WorkflowStepCompleteEvent,
    WorkflowStructureEvent,
)
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


def get_structure_event(emitter: InMemoryEmitter, workflow_name: str | None = None) -> WorkflowStructureEvent:
    """Return the single ``WorkflowStructureEvent`` on ``emitter``.

    Nested workflows bind their inner emitter to the outer emitter, so inner
    workflows' structure events are forwarded into the outer emitter's
    ``events`` list. Tests with nested workflows must pass ``workflow_name``
    to disambiguate.
    """
    events = [e for e in emitter.events if isinstance(e, WorkflowStructureEvent)]
    if workflow_name is not None:
        events = [e for e in events if e.workflow_name == workflow_name]
    assert len(events) == 1, f"Expected 1 WorkflowStructureEvent (workflow_name={workflow_name!r}), got {len(events)}"
    return events[0]


# ── Sequential ─────────────────────────────────────────────


class TestSequentialStructure:
    async def test_emits_structure_event(self) -> None:
        emitter = make_emitter()
        seq = Sequential(
            name="seq",
            steps=[make_step("step_a"), make_step("step_b"), make_step("step_c")],
            emitter=emitter,
        )
        await seq.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "seq"
        assert event.workflow_type == "sequential"
        assert len(event.steps) == 3

        assert event.steps[0].name == "step_a"
        assert event.steps[0].step_type == "function"
        assert event.steps[0].index == 0

        assert event.steps[1].name == "step_b"
        assert event.steps[1].index == 1

        assert event.steps[2].name == "step_c"
        assert event.steps[2].index == 2


# ── Parallel ───────────────────────────────────────────────


class TestParallelStructure:
    async def test_emits_structure_event(self) -> None:
        emitter = make_emitter()
        par = Parallel(
            name="par",
            steps=[make_step("p1"), make_step("p2")],
            emitter=emitter,
        )
        await par.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "par"
        assert event.workflow_type == "parallel"
        assert len(event.steps) == 2

        for step_def in event.steps:
            assert step_def.parallel_group == "parallel"
            assert step_def.step_type == "function"


# ── DAG ────────────────────────────────────────────────────


class TestDAGStructure:
    async def test_emits_structure_event_with_dependencies(self) -> None:
        emitter = make_emitter()
        dag = DAG(
            name="dag",
            nodes={
                "fetch": DAGNode(step=make_step("fetch")),
                "parse": DAGNode(step=make_step("parse"), depends_on=["fetch"]),
                "validate": DAGNode(step=make_step("validate"), depends_on=["fetch"]),
                "combine": DAGNode(step=make_step("combine"), depends_on=["parse", "validate"]),
            },
            emitter=emitter,
        )
        await dag.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "dag"
        assert event.workflow_type == "dag"
        assert len(event.steps) == 4

        step_map = {s.name: s for s in event.steps}
        assert step_map["fetch"].depends_on == []
        assert step_map["parse"].depends_on == ["fetch"]
        assert step_map["validate"].depends_on == ["fetch"]
        assert sorted(step_map["combine"].depends_on) == ["parse", "validate"]


# ── Conditional ────────────────────────────────────────────


class TestConditionalStructure:
    async def test_emits_structure_event_with_branches(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="cond",
            router=lambda x: "a",
            branches={
                "a": make_step("branch_a"),
                "b": make_step("branch_b"),
            },
            emitter=emitter,
        )
        await cond.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "cond"
        assert event.workflow_type == "conditional"
        assert len(event.steps) == 2

        step_map = {s.metadata.get("branch"): s for s in event.steps}
        assert "a" in step_map
        assert "b" in step_map

    async def test_includes_default_branch(self) -> None:
        emitter = make_emitter()
        cond = Conditional(
            name="cond-default",
            router=lambda x: "a",
            branches={"a": make_step("branch_a")},
            default=make_step("fallback"),
            emitter=emitter,
        )
        await cond.execute("input")

        event = get_structure_event(emitter)
        assert len(event.steps) == 2
        default_steps = [s for s in event.steps if s.metadata.get("branch") == "default"]
        assert len(default_steps) == 1
        assert default_steps[0].name == "fallback"


# ── Loop ───────────────────────────────────────────────────


class TestLoopStructure:
    async def test_emits_structure_event(self) -> None:
        emitter = make_emitter()
        loop = Loop(
            name="loop",
            step=make_step("body"),
            condition=lambda result, iteration: True,
            max_iterations=5,
            emitter=emitter,
        )
        await loop.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "loop"
        assert event.workflow_type == "loop"
        assert len(event.steps) == 1
        assert event.steps[0].name == "body"
        assert event.steps[0].metadata["max_iterations"] == 5


# ── Pipeline ──────────────────────────────────────────────


class TestPipelineStructure:
    async def test_emits_structure_event(self) -> None:
        emitter = make_emitter()
        pipeline = Pipeline(
            name="pipe",
            stages=[
                Stage(make_step("stage_a")),
                Stage(make_step("stage_b")),
            ],
            emitter=emitter,
        )
        await pipeline.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "pipe"
        assert event.workflow_type == "pipeline"
        assert len(event.steps) == 2
        assert event.steps[0].name == "stage_a"
        assert event.steps[0].index == 0
        assert event.steps[1].name == "stage_b"
        assert event.steps[1].index == 1


# ── MapReduce ──────────────────────────────────────────────


class TestMapReduceStructure:
    async def test_emits_structure_event(self) -> None:
        emitter = make_emitter()
        mr = MapReduce(
            name="mr",
            step=make_step("mapper"),
            emitter=emitter,
            splitter=lambda x: [1, 2, 3],
            reducer=lambda results: sum(r.output for r in results),
        )
        await mr.execute("input")

        event = get_structure_event(emitter)
        assert event.workflow_name == "mr"
        assert event.workflow_type == "map_reduce"
        assert len(event.steps) == 2
        assert event.steps[0].name == "mapper-map"
        assert event.steps[0].step_type == "function"
        assert event.steps[1].name == "reduce"
        assert event.steps[1].step_type == "function"


# ── Step Duration Tracking ─────────────────────────────────


class TestStepDuration:
    async def test_sequential_step_has_duration(self) -> None:
        emitter = make_emitter()
        seq = Sequential(
            name="timed",
            steps=[make_step("s1")],
            emitter=emitter,
        )
        await seq.execute("input")

        step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
        assert len(step_events) == 1
        assert step_events[0].step_duration_ms is not None
        assert step_events[0].step_duration_ms >= 0

    async def test_step_output(self) -> None:
        async def produce(x):
            return "hello world"

        emitter = make_emitter()
        seq = Sequential(
            name="preview",
            steps=[make_step("s1", produce)],
            emitter=emitter,
        )
        await seq.execute("input")

        step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
        assert len(step_events) == 1
        assert step_events[0].step_output == "hello world"

    async def test_step_output_full_content(self) -> None:
        async def produce_long(x):
            return "x" * 500

        emitter = make_emitter()
        seq = Sequential(
            name="long-preview",
            steps=[make_step("s1", produce_long)],
            emitter=emitter,
        )
        await seq.execute("input")

        step_events = [e for e in emitter.events if isinstance(e, WorkflowStepCompleteEvent)]
        assert len(step_events) == 1
        assert step_events[0].step_output is not None
        assert len(step_events[0].step_output) == 500


# ── Step Type Classification ──────────────────────────────


class TestStepTypeClassification:
    async def test_function_step_classified(self) -> None:
        emitter = make_emitter()
        seq = Sequential(
            name="fn-type",
            steps=[make_step("fn_step")],
            emitter=emitter,
        )
        await seq.execute("input")

        event = get_structure_event(emitter)
        assert event.steps[0].step_type == "function"

    async def test_workflow_step_classified(self) -> None:
        inner_emitter = make_emitter()
        inner = Sequential(
            name="inner",
            steps=[make_step("inner_step")],
            emitter=inner_emitter,
        )
        emitter = make_emitter()
        outer = Sequential(
            name="outer",
            steps=[WorkflowStep(inner)],
            emitter=emitter,
        )
        await outer.execute("input")

        event = get_structure_event(emitter, workflow_name="outer")
        assert event.steps[0].step_type == "workflow"
        assert event.steps[0].metadata["workflow_name"] == "inner"
        assert event.steps[0].metadata["workflow_type"] == "sequential"


class TestWorkflowStepDirectExecute:
    async def test_passthrough_returns_wrapped_workflow_result(self) -> None:
        inner_emitter = make_emitter()
        inner = Sequential(
            name="inner",
            steps=[make_step("inner_step")],
            emitter=inner_emitter,
        )
        step = WorkflowStep(inner)

        result = await step.execute("hello")

        assert result.output == "hello"
