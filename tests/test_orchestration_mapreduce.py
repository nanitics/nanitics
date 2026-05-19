import asyncio

import pytest

from nanitics.composition.orchestration.mapreduce import MapReduce
from nanitics.composition.orchestration.protocol import FailurePolicy, Step
from nanitics.infrastructure.observability.events import (
    WorkflowCompleteEvent,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowStepCompleteEvent,
)
from nanitics.safety import CancellationToken
from tests.testing_helpers import make_emitter, make_step

# ── Helpers ────────────────────────────────────────────────


# ── Construction ───────────────────────────────────────────


class TestMapReduceConstruction:
    def test_satisfies_step_protocol(self) -> None:
        emitter = make_emitter()
        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
        )
        assert isinstance(mr, Step)

    def test_name_property(self) -> None:
        emitter = make_emitter()
        mr = MapReduce(
            name="my-mapreduce",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
        )
        assert mr.name == "my-mapreduce"


# ── Execution ──────────────────────────────────────────────


class TestMapReduceExecution:
    async def test_basic_split_map_reduce(self) -> None:
        emitter = make_emitter()

        async def double(x):
            return x * 2

        mr = MapReduce(
            name="mr",
            step=make_step("double", double),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: sum(r.output for r in results),
        )

        result = await mr.execute([1, 2, 3])
        assert result.output == 12  # 2 + 4 + 6

    async def test_single_item_collection(self) -> None:
        emitter = make_emitter()

        async def upper(x):
            return x.upper()

        mr = MapReduce(
            name="mr",
            step=make_step("upper", upper),
            emitter=emitter,
            splitter=lambda x: [x],
            reducer=lambda results: results[0].output,
        )

        result = await mr.execute("hello")
        assert result.output == "HELLO"

    async def test_empty_collection(self) -> None:
        emitter = make_emitter()

        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: [],
            reducer=lambda results: "empty",
        )

        result = await mr.execute("anything")
        assert result.output == "empty"
        assert result.metadata["total_items"] == 0
        assert result.metadata["total_steps_executed"] == 0

    async def test_metadata_tracks_items(self) -> None:
        emitter = make_emitter()

        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
        )

        result = await mr.execute([1, 2, 3])
        assert result.metadata["total_items"] == 3
        assert result.metadata["total_steps_executed"] == 3


# ── Concurrency ────────────────────────────────────────────


class TestMapReduceConcurrency:
    async def test_max_concurrency_limits_simultaneous_execution(self) -> None:
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

        mr = MapReduce(
            name="mr",
            step=make_step("tracked", tracked),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            max_concurrency=2,
        )

        await mr.execute([1, 2, 3, 4])
        assert max_concurrent <= 2

    async def test_unlimited_concurrency(self) -> None:
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

        mr = MapReduce(
            name="mr",
            step=make_step("tracked", tracked),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            max_concurrency=None,
        )

        await mr.execute([1, 2, 3, 4])
        # Without concurrency limit, all 4 should run concurrently
        assert max_concurrent == 4


# ── Failure Handling ───────────────────────────────────────


class TestMapReduceFailure:
    async def test_all_or_nothing_cancels_on_first_failure(self) -> None:
        emitter = make_emitter()

        async def fail_on_two(x):
            if x == 2:
                raise ValueError("item 2 failed")
            await asyncio.sleep(0.1)
            return x

        mr = MapReduce(
            name="mr",
            step=make_step("s", fail_on_two),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            failure_policy=FailurePolicy.ALL_OR_NOTHING,
        )

        with pytest.raises(ValueError, match="item 2 failed"):
            await mr.execute([1, 2, 3])

    async def test_best_effort_collects_partial_results(self) -> None:
        emitter = make_emitter()

        async def fail_on_two(x):
            if x == 2:
                raise ValueError("item 2 failed")
            return x * 10

        mr = MapReduce(
            name="mr",
            step=make_step("s", fail_on_two),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            failure_policy=FailurePolicy.BEST_EFFORT,
        )

        result = await mr.execute([1, 2, 3])
        assert sorted(result.output) == [10, 30]
        assert result.metadata["failed_items"] == [1]
        assert result.metadata["total_steps_executed"] == 2


# ── Cancellation ───────────────────────────────────────────


class TestMapReduceCancellation:
    async def test_respects_cancellation_token(self) -> None:
        emitter = make_emitter()
        token = CancellationToken()
        executed: list[int] = []

        async def track_and_cancel(x):
            executed.append(x)
            if x == 1:
                token.cancel()
                await asyncio.sleep(0.05)
            return x

        mr = MapReduce(
            name="mr",
            step=make_step("s", track_and_cancel),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
            max_concurrency=1,
            failure_policy=FailurePolicy.BEST_EFFORT,
            cancellation_token=token,
        )

        result = await mr.execute([1, 2, 3])
        # Item 1 completes, then cancellation should prevent later items
        assert 1 in executed
        assert result.metadata.get("terminated") == "cancelled"


# ── Events ─────────────────────────────────────────────────


class TestMapReduceEvents:
    async def test_emits_workflow_events(self) -> None:
        emitter = make_emitter()

        mr = MapReduce(
            name="evented",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=lambda results: [r.output for r in results],
        )

        await mr.execute([1, 2, 3])

        events = emitter.events
        start_events = [e for e in events if isinstance(e, WorkflowStartEvent)]
        step_events = [e for e in events if isinstance(e, WorkflowStepCompleteEvent)]
        complete_events = [e for e in events if isinstance(e, WorkflowCompleteEvent)]

        assert len(start_events) == 1
        assert start_events[0].workflow_type == "map_reduce"
        assert start_events[0].step_count == 0  # Unknown at start

        assert len(step_events) == 3
        for i, event in enumerate(sorted(step_events, key=lambda e: e.step_index)):
            assert event.step_name == "s"
            assert event.step_index == i

        assert len(complete_events) == 1
        assert complete_events[0].total_steps_executed == 3

    async def test_emits_error_event_on_failure(self) -> None:
        emitter = make_emitter()

        async def fail(x):
            raise RuntimeError("boom")

        mr = MapReduce(
            name="err",
            step=make_step("s", fail),
            emitter=emitter,
            splitter=lambda x: [x],
            reducer=lambda results: results,
        )

        with pytest.raises(RuntimeError):
            await mr.execute("x")

        error_events = [e for e in emitter.events if isinstance(e, WorkflowErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].workflow_type == "map_reduce"


# ── Async Splitter / Reducer ──────────────────────────────


class TestMapReduceAsync:
    async def test_async_splitter(self) -> None:
        emitter = make_emitter()

        async def async_split(x):
            return list(range(x))

        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=async_split,
            reducer=lambda results: [r.output for r in results],
        )

        result = await mr.execute(3)
        assert sorted(result.output) == [0, 1, 2]

    async def test_async_reducer(self) -> None:
        emitter = make_emitter()

        async def async_reduce(results):
            return sum(r.output for r in results)

        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=lambda x: x,
            reducer=async_reduce,
        )

        result = await mr.execute([10, 20, 30])
        assert result.output == 60

    async def test_async_splitter_empty_collection(self) -> None:
        emitter = make_emitter()

        async def async_split(x: object) -> list[object]:
            return []

        async def async_reduce(results: list[object]) -> str:
            return "empty"

        mr = MapReduce(
            name="mr",
            step=make_step("s"),
            emitter=emitter,
            splitter=async_split,
            reducer=async_reduce,
        )

        result = await mr.execute("anything")
        assert result.output == "empty"
        assert result.metadata["total_items"] == 0

    async def test_async_splitter_and_reducer(self) -> None:
        emitter = make_emitter()

        async def async_split(x):
            return x

        async def async_reduce(results):
            return [r.output for r in results]

        async def double(x):
            return x * 2

        mr = MapReduce(
            name="mr",
            step=make_step("double", double),
            emitter=emitter,
            splitter=async_split,
            reducer=async_reduce,
        )

        result = await mr.execute([1, 2, 3])
        assert sorted(result.output) == [2, 4, 6]
