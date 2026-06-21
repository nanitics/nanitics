"""Continue a run that parked on a budget limiter (``suspend_on_budget``).

A ReAct agent built with ``suspend_on_budget=True`` and run under step-level
durability parks itself as a ``"budget_exhausted"`` suspension when it hits
``max_iterations`` / ``max_tool_calls``, instead of ending with an empty result.
The host continues it by rebuilding the agent on a larger budget and calling
:meth:`ResumeService.continue_run` (or :meth:`DurableRun.continue_exhausted`),
which re-enters the same ReAct loop from where it ran out of budget.

These tests cover the full surface:

- A parked run surfaces an :class:`ExhaustedRun` carrying the partial work, and
  persists a re-enterable budget checkpoint (cursor *at* the agent step).
- Continuing with a **larger** ceiling reaches a terminal answer.
- Continuing with the **same** ceiling re-parks (a host can offer Continue again).
- Continuing with a **smaller** ceiling than already consumed is rejected.
- The same machinery covers ``max_tool_calls``.
- Without durability (no checkpoint sink), ``suspend_on_budget`` is inert and
  exhaustion stays terminal.
- The continue guards reject non-budget checkpoints.
- ``DurableRun.continue_exhausted`` is the single-process equivalent.
"""

from __future__ import annotations

import pytest

from nanitics.composition.durability.resume import (
    DurableRun,
    ExhaustedRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
)
from nanitics.composition.durability.store import InMemoryCheckpointStore
from nanitics.hitl import InMemoryHitlRequestStore
from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.infrastructure.observability.events import ExecutionSuspendedEvent
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter, make_response

_RUN_ID = "continue-after-exhaustion"


@tool(name="add", description="Add two numbers")
async def add_tool(a: int, b: int) -> str:
    return str(a + b)


def _tool_turn(tc_id: str) -> LLMResponse:
    """An LLM turn that calls ``add`` — keeps the loop in tool-calling mode so
    every step consumes the iteration budget without ever finishing."""
    return make_response(
        content="adding",
        tool_calls=[ToolCall(id=tc_id, name="add", arguments={"a": 1, "b": 2})],
    )


def _build_agent(
    *,
    responses: list[LLMResponse],
    max_iterations: int = 10,
    max_tool_calls: int | None = None,
    suspend_on_budget: bool = True,
    emitter: object | None = None,
) -> ReActAgent:
    return ReActAgent(
        name="enricher",
        llm_client=MockLLMClient(responses),
        emitter=emitter or make_emitter(),
        system_prompt="You are a test agent",
        tools=[add_tool],
        run_id=_RUN_ID,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        suspend_on_budget=suspend_on_budget,
    )


def _durable(agent: ReActAgent, checkpoint_store: InMemoryCheckpointStore) -> DurableRun:
    return DurableRun(
        agent,
        hitl_store=InMemoryHitlRequestStore(),
        checkpoint_store=checkpoint_store,
        run_id=_RUN_ID,
        step_checkpoints=True,
    )


async def _park_on_iteration_limit(
    checkpoint_store: InMemoryCheckpointStore,
    *,
    emitter: object | None = None,
) -> ExhaustedRun:
    """First pass: a 2-iteration agent that only ever calls tools, so it parks
    on ``iteration_limit`` after two checkpointed tool batches."""
    agent = _build_agent(
        responses=[_tool_turn("tc1"), _tool_turn("tc2")],
        max_iterations=2,
        emitter=emitter,
    )
    result = await _durable(agent, checkpoint_store).start("enrich ACME Corp")
    assert isinstance(result, ExhaustedRun)
    return result


class TestParkOnExhaustion:
    async def test_parks_as_exhausted_run_with_partial_text(self) -> None:
        store = InMemoryCheckpointStore()
        parked = await _park_on_iteration_limit(store)

        assert parked.run_id == _RUN_ID
        assert parked.suspension_info.suspension_type == "budget_exhausted"
        assert parked.suspension_info.request_type == "iteration_limit"
        # The last assistant turn is surfaced as the partial work.
        assert parked.last_assistant_text == "adding"
        assert parked.checkpoint_id

    async def test_checkpoint_is_re_enterable_budget_park(self) -> None:
        """The persisted checkpoint points *at* the agent step (not past it) and
        is tagged a budget park — the precondition the continue path requires."""
        store = InMemoryCheckpointStore()
        await _park_on_iteration_limit(store)

        cursor = await store.load(_RUN_ID)
        assert cursor is not None
        assert cursor.checkpoint_reason == "budget_exhausted"
        assert cursor.suspension_info is not None
        assert cursor.suspension_info.suspension_type == "budget_exhausted"
        # suspended at the single agent step (index 0), so resume re-enters it.
        assert cursor.state["suspended_step_index"] == 0
        assert cursor.state.get("agent_checkpoint") is not None

    async def test_emits_budget_exhausted_suspension_event(self) -> None:
        store = InMemoryCheckpointStore()
        emitter = make_emitter()
        await _park_on_iteration_limit(store, emitter=emitter)

        suspended = [e for e in emitter.events if isinstance(e, ExecutionSuspendedEvent)]
        assert suspended
        assert all(e.suspension_type == "budget_exhausted" for e in suspended)


class TestContinueRun:
    async def test_larger_ceiling_continues_to_finish(self) -> None:
        """Rebuild with a raised ceiling → the run resumes from its checkpoint
        and reaches a terminal answer instead of re-parking."""
        store = InMemoryCheckpointStore()
        await _park_on_iteration_limit(store)

        resume_client = MockLLMClient([make_response(content="ACME Corp: a paper-clip maker.")])

        def factory(ctx: ResumeContext) -> DurableRun:
            agent = ReActAgent(
                name="enricher",
                llm_client=resume_client,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[add_tool],
                run_id=ctx.run_id,
                max_iterations=4,  # fresh budget rides on the rebuilt ceiling
                suspend_on_budget=True,
            )
            return DurableRun(
                agent,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
                step_checkpoints=True,
            )

        service = ResumeService(
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            factory=factory,
        )
        resumed = await service.continue_run(_RUN_ID)

        assert isinstance(resumed, ResumeResult)
        assert resumed.output == "ACME Corp: a paper-clip maker."
        # The loop was genuinely re-entered (the queued answer was consumed).
        assert len(resume_client.calls) == 1

    async def test_same_ceiling_re_parks(self) -> None:
        """Rebuild with the same ceiling → restore(2) into max=2 re-breaks on the
        first step and parks again, so a host can offer Continue repeatedly."""
        store = InMemoryCheckpointStore()
        await _park_on_iteration_limit(store)

        def factory(ctx: ResumeContext) -> DurableRun:
            agent = _build_agent(responses=[make_response(content="unreached")], max_iterations=2)
            return _durable(agent, ctx.checkpoint_store)

        service = ResumeService(
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            factory=factory,
        )
        resumed = await service.continue_run(_RUN_ID)

        assert isinstance(resumed, ExhaustedRun)
        assert resumed.suspension_info.suspension_type == "budget_exhausted"

    async def test_smaller_ceiling_than_consumed_is_rejected(self) -> None:
        """A continue budget below the already-consumed count is rejected by the
        limiter — documents the host's obligation to grant at least the count."""
        store = InMemoryCheckpointStore()
        await _park_on_iteration_limit(store)

        def factory(ctx: ResumeContext) -> DurableRun:
            agent = _build_agent(responses=[make_response(content="unreached")], max_iterations=1)
            return _durable(agent, ctx.checkpoint_store)

        service = ResumeService(
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            factory=factory,
        )
        with pytest.raises(ValueError, match="exceeds max_iterations"):
            await service.continue_run(_RUN_ID)

    async def test_rejects_non_durable_factory(self) -> None:
        store = InMemoryCheckpointStore()
        await _park_on_iteration_limit(store)

        service = ResumeService(
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            factory=lambda _ctx: "not a durable run",  # type: ignore[arg-type,return-value]
        )
        with pytest.raises(TypeError, match="must return a DurableRun"):
            await service.continue_run(_RUN_ID)

    async def test_rejects_missing_checkpoint(self) -> None:
        service = ResumeService(
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=InMemoryCheckpointStore(),
            factory=lambda _ctx: _durable(  # pragma: no cover - never reached
                _build_agent(responses=[]), InMemoryCheckpointStore()
            ),
        )
        with pytest.raises(ValueError, match="No checkpoint to continue"):
            await service.continue_run("missing")


class TestToolCallLimit:
    async def test_parks_and_continues_on_tool_call_limit(self) -> None:
        """The same park/continue cycle covers ``max_tool_calls``."""
        store = InMemoryCheckpointStore()
        # One tool call per turn, max_tool_calls=1 → the second batch trips it.
        agent = _build_agent(
            responses=[_tool_turn("tc1"), _tool_turn("tc2")],
            max_iterations=10,
            max_tool_calls=1,
        )
        parked = await _durable(agent, store).start("go")
        assert isinstance(parked, ExhaustedRun)
        assert parked.suspension_info.request_type == "tool_call_limit"

        resume_client = MockLLMClient([make_response(content="done")])

        def factory(ctx: ResumeContext) -> DurableRun:
            agent2 = ReActAgent(
                name="enricher",
                llm_client=resume_client,
                emitter=make_emitter(),
                system_prompt="You are a test agent",
                tools=[add_tool],
                run_id=ctx.run_id,
                max_iterations=10,
                max_tool_calls=4,
                suspend_on_budget=True,
            )
            return DurableRun(
                agent2,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
                run_id=ctx.run_id,
                step_checkpoints=True,
            )

        service = ResumeService(
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            factory=factory,
        )
        resumed = await service.continue_run(_RUN_ID)
        assert isinstance(resumed, ResumeResult)
        assert resumed.output == "done"


class TestInertWithoutDurability:
    async def test_suspend_on_budget_inert_without_sink(self) -> None:
        """Without step-level durability (no sink) ``suspend_on_budget`` is inert:
        exhaustion ends the run normally with the budget termination reason."""
        store = InMemoryCheckpointStore()
        agent = _build_agent(responses=[_tool_turn("tc1"), _tool_turn("tc2")], max_iterations=2)
        durable = DurableRun(
            agent,
            hitl_store=InMemoryHitlRequestStore(),
            checkpoint_store=store,
            run_id=_RUN_ID,
            step_checkpoints=False,  # no sink → no budget suspension
        )
        result = await durable.start("go")

        assert isinstance(result, ResumeResult)
        assert result.output is None
        intermediate = result.metadata["intermediate_results"]
        last = next(reversed(list(intermediate.values())))
        assert last.metadata["termination_reason"] == "iteration_limit"


class TestContinueGuards:
    async def test_continue_exhausted_rejects_crash_cursor(self) -> None:
        """``continue_exhausted`` rejects a plain step/crash cursor (no
        suspension) — that routes through ``resume_from_checkpoint`` instead."""
        store = InMemoryCheckpointStore()
        # A non-suspending agent that completes: leaves only a step cursor.
        agent = _build_agent(responses=[make_response(content="done")], suspend_on_budget=False)
        durable = _durable(agent, store)
        result = await durable.start("go")
        assert isinstance(result, ResumeResult)

        with pytest.raises(ValueError, match="not a budget-exhaustion park"):
            await durable.continue_exhausted()

    async def test_continue_exhausted_continues_in_process(self) -> None:
        """``DurableRun.continue_exhausted`` is the single-process equivalent of
        ``ResumeService.continue_run`` — rebuild the run, continue from the park."""
        store = InMemoryCheckpointStore()
        await _park_on_iteration_limit(store)

        agent = _build_agent(responses=[make_response(content="finished")], max_iterations=4)
        resumed = await _durable(agent, store).continue_exhausted()

        assert isinstance(resumed, ResumeResult)
        assert resumed.output == "finished"


class TestDegradedState:
    async def test_missing_checkpoint_after_budget_park_raises(self) -> None:
        """Defensive: a budget suspension with no persisted checkpoint surfaces a
        RuntimeError rather than a blank payload. Never fires in normal
        operation (the orchestrator persists on every suspension)."""
        from nanitics.composition.durability.models import SuspensionInfo
        from nanitics.composition.durability.suspension import SuspendExecution

        store = InMemoryCheckpointStore()  # stays empty
        durable = _durable(_build_agent(responses=[]), store)
        exc = SuspendExecution(
            suspension_info=SuspensionInfo(
                suspension_id="s",
                suspension_type="budget_exhausted",
                request_id="",
                request_type="iteration_limit",
                prompt="",
            )
        )
        with pytest.raises(RuntimeError, match="no checkpoint was persisted"):
            await durable._build_exhausted_run(exc)


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
