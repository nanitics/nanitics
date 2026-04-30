"""Tests for TracedExecutor — run lifecycle management with event persistence."""

from __future__ import annotations

import asyncio

import pytest

from nanitics import (
    EventEmitter,
    InMemoryPersistentTraceStore,
    MockLLMClient,
    ReasoningAgent,
    TracedExecutor,
)
from nanitics.composition.durability.models import SuspensionInfo
from nanitics.composition.durability.suspension import SuspendExecution
from tests.testing_helpers import make_response


async def _successful_fn(emitter: EventEmitter, run_id: str) -> str:
    """Minimal callable that emits an event and returns."""
    del run_id  # unused in this factory
    agent = ReasoningAgent(
        name="test-agent",
        llm_client=MockLLMClient(responses=[make_response("done")]),
        emitter=emitter,
        system_prompt="test",
    )
    result = await agent.run("hello")
    return result.output


async def _failing_fn(emitter: EventEmitter, run_id: str) -> str:
    """Callable that emits events then raises."""
    del run_id  # unused in this factory
    agent = ReasoningAgent(
        name="test-agent",
        llm_client=MockLLMClient(responses=[]),
        emitter=emitter,
        system_prompt="test",
    )
    result = await agent.run("hello")
    return result.output


async def _suspending_fn(emitter: EventEmitter, run_id: str) -> str:
    """Callable that emits an event then suspends."""
    del run_id  # unused in this factory
    with emitter.span("before-suspend"):
        pass
    raise SuspendExecution(
        suspension_info=SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="human_input",
            prompt="Need input",
        ),
    )


@pytest.fixture
def store() -> InMemoryPersistentTraceStore:
    return InMemoryPersistentTraceStore()


@pytest.fixture
def executor(store: InMemoryPersistentTraceStore) -> TracedExecutor:
    return TracedExecutor(store)


async def test_successful_execution(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    run_id, result = await executor.execute(_successful_fn)

    assert result == "done"
    assert isinstance(run_id, str)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.status == "completed"
    assert run.result is not None

    events = await store.query_events(run_id)
    assert len(events) > 0


async def test_failed_execution_persists_events(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    with pytest.raises(ValueError, match="no more scripted responses"):
        await executor.execute(_failing_fn)

    runs = await store.list_runs(status="failed")
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "failed"
    assert run.error is not None

    # Events from before the failure should be persisted
    events = await store.query_events(run.id)
    assert len(events) > 0


async def test_suspended_execution(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    with pytest.raises(SuspendExecution):
        await executor.execute(_suspending_fn)

    runs = await store.list_runs(status="suspended")
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "suspended"

    events = await store.query_events(run.id)
    assert len(events) > 0


async def test_metadata_passed_through(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    metadata = {"agent": "extraction", "doc_id": "abc-123"}
    run_id, _ = await executor.execute(_successful_fn, metadata=metadata)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.metadata["agent"] == "extraction"
    assert run.metadata["doc_id"] == "abc-123"


async def test_queue_receives_events(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    _run_id, _ = await executor.execute(_successful_fn, queue=queue)

    assert not queue.empty()
    event = queue.get_nowait()
    assert event["event_type"] == "trace"
    assert "payload" in event


async def test_default_metadata_is_empty(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    run_id, _ = await executor.execute(_successful_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.metadata == {}


async def test_result_stored_as_serialized_string(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    run_id, _result = await executor.execute(_successful_fn)

    run = await store.get_run(run_id)
    assert run is not None
    assert run.result is not None
    # _successful_fn returns result.output which is a plain string
    assert run.result == "done"


async def test_plain_string_result(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    async def _string_fn(emitter: EventEmitter, run_id: str) -> str:
        del emitter, run_id
        return "plain-result"

    run_id, result = await executor.execute(_string_fn)

    assert result == "plain-result"
    run = await store.get_run(run_id)
    assert run is not None
    assert run.result == "plain-result"


async def test_pydantic_result_serialized_as_json(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    import json

    from pydantic import BaseModel

    class MyResult(BaseModel):
        value: int
        label: str

    async def _pydantic_fn(emitter: EventEmitter, run_id: str) -> MyResult:
        del emitter, run_id
        return MyResult(value=42, label="test")

    run_id, result = await executor.execute(_pydantic_fn)

    assert result.value == 42
    run = await store.get_run(run_id)
    assert run is not None
    parsed = json.loads(run.result)  # type: ignore[arg-type]
    assert parsed["value"] == 42
    assert parsed["label"] == "test"


async def test_dict_result_serialized_as_json(executor: TracedExecutor, store: InMemoryPersistentTraceStore) -> None:
    import json

    async def _dict_fn(emitter: EventEmitter, run_id: str) -> dict[str, int]:
        del emitter, run_id
        return {"a": 1, "b": 2}

    run_id, result = await executor.execute(_dict_fn)

    assert result == {"a": 1, "b": 2}
    run = await store.get_run(run_id)
    assert run is not None
    parsed = json.loads(run.result)  # type: ignore[arg-type]
    assert parsed == {"a": 1, "b": 2}


async def test_non_serializable_result_falls_back_to_json_default(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    async def _opaque_fn(emitter: EventEmitter, run_id: str) -> Opaque:
        del emitter, run_id
        return Opaque()

    run_id, _ = await executor.execute(_opaque_fn)

    run = await store.get_run(run_id)
    assert run is not None
    # json.dumps with default=str wraps the str() output in JSON
    assert run.result == '"opaque-value"'


async def test_factory_receives_same_run_id_as_returned(
    executor: TracedExecutor, store: InMemoryPersistentTraceStore
) -> None:
    """The ``run_id`` the factory sees equals the ``run_id`` returned.

    This invariant is the whole point of passing ``run_id`` into the
    factory — callers that key external durable state on the Observatory
    ``run_id`` before ``execute`` returns must be able to trust the
    in-factory value without inferring it from the emitter's
    ``trace_id``.
    """
    captured: dict[str, str] = {}

    async def _factory(emitter: EventEmitter, run_id: str) -> str:
        del emitter
        captured["run_id"] = run_id
        return "ok"

    returned_run_id, result = await executor.execute(_factory)

    assert result == "ok"
    assert captured["run_id"] == returned_run_id
    # Surface a second signal: the run record the executor wrote is
    # keyed on that same identifier.
    run = await store.get_run(returned_run_id)
    assert run is not None
    assert run.id == returned_run_id
