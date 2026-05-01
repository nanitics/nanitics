"""Tests for ``DurableRun`` and ``ResumeService``.

Unit coverage of the two cooperating classes that wrap the durable-HITL
suspend/resume cycle: :class:`DurableRun` (the suspend-side wrapper
that converts ``SuspendExecution`` into a :class:`SuspendedRun` value)
and :class:`ResumeService` (the resume-side dispatcher that persists
the response and drives a factory-built :class:`DurableRun` forward).

Tests exercise workflow-level suspension via :class:`ApprovalGate` +
:class:`DurableHumanInputProvider`, which drives the same
``Sequential._run`` suspend/resume machinery the higher-level agent
path uses — without depending on the ``AgentStep`` resume-state wiring.
"""

from __future__ import annotations

import pytest

from nanitics import (
    InMemoryCheckpointStore,
    InMemoryHitlRequestStore,
    MockLLMClient,
    ReActAgent,
    Sequential,
)
from nanitics.collaboration.approval_gate import ApprovalGate
from nanitics.collaboration.durable_provider import DurableHumanInputProvider
from nanitics.collaboration.protocol import (
    HumanDecision,
    HumanInputResponse,
)
from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.resume import (
    DurableRun,
    ResumeContext,
    ResumeResult,
    ResumeService,
    SuspendedRun,
)
from nanitics.composition.durability.suspension import SuspendExecution
from nanitics.composition.orchestration.adapters import FunctionStep
from tests.testing_helpers import make_emitter

_RUN_ID = "durable-run-test"


def _echo_step(name: str = "echo") -> FunctionStep:
    async def fn(value: object) -> object:
        return value

    return FunctionStep(name=name, fn=fn)


def _build_gate_workflow(
    *,
    hitl_store: InMemoryHitlRequestStore,
    checkpoint_store: InMemoryCheckpointStore,
    run_id: str = _RUN_ID,
    gate_name: str = "review",
    trailing_step_name: str | None = None,
) -> Sequential:
    """Build a Sequential with an ApprovalGate (and optional trailing step)."""
    provider = DurableHumanInputProvider(request_store=hitl_store)
    gate = ApprovalGate(provider=provider, name=gate_name, run_id=run_id)
    steps: list = [gate]
    if trailing_step_name is not None:
        steps.append(_echo_step(trailing_step_name))
    return Sequential(
        name="gate-workflow",
        steps=steps,
        emitter=make_emitter(),
        checkpoint_store=checkpoint_store,
        run_id=run_id,
    )


class TestDurableRunConstruction:
    def test_wraps_workflow_without_run_id_arg(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        assert durable.run_id == _RUN_ID

    def test_run_id_mismatch_raises(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=checkpoint_store, run_id="a")
        with pytest.raises(ValueError, match="conflicts with workflow"):
            DurableRun(
                workflow,
                hitl_store=hitl_store,
                checkpoint_store=checkpoint_store,
                run_id="b",
            )

    def test_workflow_without_checkpoint_store_raises(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = Sequential(
            name="no-cp",
            steps=[_echo_step()],
            emitter=make_emitter(),
            run_id=_RUN_ID,
        )
        with pytest.raises(ValueError, match="checkpoint_store configured"):
            DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)

    def test_workflow_checkpoint_store_mismatch_raises(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        cp1 = InMemoryCheckpointStore()
        cp2 = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=cp1)
        with pytest.raises(ValueError, match="must match the workflow"):
            DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=cp2)

    def test_non_agent_non_workflow_raises(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        with pytest.raises(TypeError, match="must be an Agent or Workflow"):
            DurableRun(
                object(),  # type: ignore[arg-type]
                hitl_store=hitl_store,
                checkpoint_store=checkpoint_store,
            )


class TestDurableRun:
    async def test_completed_run_returns_resume_result(self) -> None:
        """Workflow that doesn't suspend returns ResumeResult with matching output."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = Sequential(
            name="passthrough",
            steps=[_echo_step(name="pass")],
            emitter=make_emitter(),
            checkpoint_store=checkpoint_store,
            run_id=_RUN_ID,
        )
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)

        result = await durable.start("hello")

        assert isinstance(result, ResumeResult)
        assert result.run_id == _RUN_ID
        assert result.output == "hello"

    async def test_suspended_run_returns_suspended_run_handle(self) -> None:
        """ApprovalGate suspension returns a SuspendedRun whose fields line up."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)

        result = await durable.start("pending")

        assert isinstance(result, SuspendedRun)
        assert result.run_id == _RUN_ID
        assert result.suspension_info.request_id == result.pending_request.request_id
        # The checkpoint_id resolves to a persisted checkpoint for the run.
        loaded = await checkpoint_store.load(_RUN_ID)
        assert loaded is not None
        assert loaded.checkpoint_id == result.checkpoint_id

    async def test_suspended_run_captures_pending_request_from_store(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            gate_name="review",
        )
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)

        result = await durable.start("please review")

        assert isinstance(result, SuspendedRun)
        # The default ApprovalGate prompt is "Approve proceeding?"
        assert result.pending_request.prompt == "Approve proceeding?"
        assert result.pending_request.run_id == _RUN_ID


class TestResumeService:
    async def _build(
        self,
    ) -> tuple[
        InMemoryHitlRequestStore,
        InMemoryCheckpointStore,
        Sequential,
        DurableRun,
    ]:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        return hitl_store, checkpoint_store, workflow, durable

    async def test_resume_completes_run(self) -> None:
        hitl_store, checkpoint_store, _, durable = await self._build()
        suspended = await durable.start("payload")
        assert isinstance(suspended, SuspendedRun)

        def factory(ctx: ResumeContext) -> DurableRun:
            workflow = _build_gate_workflow(hitl_store=ctx.hitl_store, checkpoint_store=ctx.checkpoint_store)
            return DurableRun(
                workflow,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(
            request_id=suspended.pending_request.request_id,
            decision=HumanDecision.APPROVE,
        )
        result = await service.resume(suspended.run_id, response)

        assert isinstance(result, ResumeResult)
        assert result.run_id == _RUN_ID
        assert result.output == "payload"

    async def test_resume_returns_suspended_run_on_nested_suspension(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)

        def build() -> Sequential:
            return Sequential(
                name="two-gates",
                steps=[
                    ApprovalGate(provider=provider, name="first", run_id=_RUN_ID),
                    ApprovalGate(provider=provider, name="second", run_id=_RUN_ID),
                ],
                emitter=make_emitter(),
                checkpoint_store=checkpoint_store,
                run_id=_RUN_ID,
            )

        durable = DurableRun(build(), hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        first_suspended = await durable.start("payload")
        assert isinstance(first_suspended, SuspendedRun)

        def factory(ctx: ResumeContext) -> DurableRun:
            return DurableRun(
                build(),
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )

        second_suspended = await service.resume(
            first_suspended.run_id,
            HumanInputResponse(
                request_id=first_suspended.pending_request.request_id,
                decision=HumanDecision.APPROVE,
            ),
        )

        assert isinstance(second_suspended, SuspendedRun)
        assert second_suspended.pending_request.request_id != first_suspended.pending_request.request_id

        final = await service.resume(
            second_suspended.run_id,
            HumanInputResponse(
                request_id=second_suspended.pending_request.request_id,
                decision=HumanDecision.APPROVE,
            ),
        )
        assert isinstance(final, ResumeResult)
        assert final.output == "payload"

    async def test_missing_checkpoint_raises(self) -> None:
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()

        def factory(ctx: ResumeContext) -> DurableRun:  # pragma: no cover
            raise AssertionError("factory must not be invoked when checkpoint is missing")

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        response = HumanInputResponse(request_id="anything", decision=HumanDecision.APPROVE)
        with pytest.raises(ValueError, match="No checkpoint for run_id='missing'"):
            await service.resume("missing", response)

    async def test_response_request_id_mismatch_raises(self) -> None:
        hitl_store, checkpoint_store, _, durable = await self._build()
        suspended = await durable.start("payload")
        assert isinstance(suspended, SuspendedRun)

        def factory(ctx: ResumeContext) -> DurableRun:  # pragma: no cover
            raise AssertionError("factory must not be invoked on request_id mismatch")

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        bogus = HumanInputResponse(request_id="not-the-right-id", decision=HumanDecision.APPROVE)
        with pytest.raises(ValueError, match=r"expected .* got 'not-the-right-id'"):
            await service.resume(suspended.run_id, bogus)

    async def test_factory_receives_resume_context_with_stores_and_checkpoint(
        self,
    ) -> None:
        hitl_store, checkpoint_store, _, durable = await self._build()
        suspended = await durable.start("payload")
        assert isinstance(suspended, SuspendedRun)

        captured: list[ResumeContext] = []

        def factory(ctx: ResumeContext) -> DurableRun:
            captured.append(ctx)
            workflow = _build_gate_workflow(hitl_store=ctx.hitl_store, checkpoint_store=ctx.checkpoint_store)
            return DurableRun(
                workflow,
                hitl_store=ctx.hitl_store,
                checkpoint_store=ctx.checkpoint_store,
            )

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        await service.resume(
            suspended.run_id,
            HumanInputResponse(
                request_id=suspended.pending_request.request_id,
                decision=HumanDecision.APPROVE,
            ),
        )

        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.run_id == _RUN_ID
        assert ctx.hitl_store is hitl_store
        assert ctx.checkpoint_store is checkpoint_store
        assert ctx.checkpoint.checkpoint_id == suspended.checkpoint_id

    async def test_resume_does_not_invoke_factory_when_response_save_fails(
        self,
    ) -> None:
        """If ``save_response`` raises, the factory must not have been called."""
        hitl_store, checkpoint_store, _, durable = await self._build()
        suspended = await durable.start("payload")
        assert isinstance(suspended, SuspendedRun)

        class FailingStore:
            def __init__(self, inner: InMemoryHitlRequestStore) -> None:
                self._inner = inner

            async def save_request(self, *args: object, **kwargs: object) -> None:
                await self._inner.save_request(*args, **kwargs)  # pragma: no cover

            async def save_response(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("simulated store failure")

            async def get_response(self, *args: object, **kwargs: object) -> object:
                return await self._inner.get_response(*args, **kwargs)  # pragma: no cover

            async def get_pending_requests(self, *args: object, **kwargs: object) -> object:
                return await self._inner.get_pending_requests(*args, **kwargs)  # pragma: no cover

        failing = FailingStore(hitl_store)
        factory_calls = 0

        def factory(ctx: ResumeContext) -> DurableRun:  # pragma: no cover
            nonlocal factory_calls
            factory_calls += 1
            raise AssertionError("factory must not be invoked when save_response failed")

        service = ResumeService(
            hitl_store=failing,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )

        with pytest.raises(RuntimeError, match="simulated store failure"):
            await service.resume(
                suspended.run_id,
                HumanInputResponse(
                    request_id=suspended.pending_request.request_id,
                    decision=HumanDecision.APPROVE,
                ),
            )
        assert factory_calls == 0

    async def test_factory_must_return_durable_run(self) -> None:
        hitl_store, checkpoint_store, _, durable = await self._build()
        suspended = await durable.start("payload")
        assert isinstance(suspended, SuspendedRun)

        def factory(ctx: ResumeContext) -> DurableRun:
            return "not a durable run"  # type: ignore[return-value]

        service = ResumeService(
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            factory=factory,
        )
        with pytest.raises(TypeError, match="must return a DurableRun"):
            await service.resume(
                suspended.run_id,
                HumanInputResponse(
                    request_id=suspended.pending_request.request_id,
                    decision=HumanDecision.APPROVE,
                ),
            )


class TestDurableRunWrappedAgent:
    """Cover the Agent-wrapping construction branch.

    Verifies the construction-time wrap and the ``start()``
    suspend path against an agent whose tool triggers a
    ``DurableHumanInputProvider`` suspension on the very first LLM call.
    """

    async def test_agent_is_wrapped_and_suspension_returns_suspended_run(self) -> None:
        from nanitics import ToolCall, tool
        from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
        from tests.testing_helpers import make_response

        @tool(name="add", description="Add two numbers")
        async def add_tool(a: int, b: int) -> str:
            return str(a + b)  # pragma: no cover

        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        provider = DurableHumanInputProvider(request_store=hitl_store)
        wrapped = ApprovalWrappedTool(tool=add_tool, provider=provider)
        client = MockLLMClient(
            [
                make_response(
                    content="Adding",
                    tool_calls=[ToolCall(id="tc1", name="add", arguments={"a": 1, "b": 2})],
                )
            ]
        )
        agent = ReActAgent(
            name="test-agent",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="You are a test agent",
            tools=[wrapped],
            run_id=_RUN_ID,
        )

        durable = DurableRun(
            agent,
            hitl_store=hitl_store,
            checkpoint_store=checkpoint_store,
            run_id=_RUN_ID,
        )
        assert durable.run_id == _RUN_ID

        result = await durable.start("add 1 and 2")
        assert isinstance(result, SuspendedRun)
        assert result.run_id == _RUN_ID
        assert result.pending_request.request_id == result.suspension_info.request_id

    def test_agent_without_run_id_raises(self) -> None:
        from tests.testing_helpers import make_response

        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        client = MockLLMClient([make_response(content="hi")])
        agent = ReActAgent(
            name="a",
            llm_client=client,
            emitter=make_emitter(),
            system_prompt="p",
            tools=[],
        )
        with pytest.raises(ValueError, match="requires a run_id"):
            DurableRun(agent, hitl_store=hitl_store, checkpoint_store=checkpoint_store)


class TestDurableRunDefensiveFailures:
    """Exercise the two ``RuntimeError`` paths that guard upstream bugs.

    These paths should never fire in normal operation: the orchestrator
    persists a checkpoint on every suspension, and the provider saves
    the request before raising. We simulate the degraded states
    directly so the diagnostics are covered.
    """

    async def test_missing_checkpoint_after_suspension_raises(self) -> None:
        """If somehow no checkpoint was persisted, _build_suspended_run surfaces a RuntimeError."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()  # stays empty
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        # Fabricate a SuspendExecution manually and drive _build_suspended_run.
        info = SuspensionInfo(
            suspension_id="s",
            request_id="r",
            request_type="approval",
            prompt="?",
            agent_name=None,
        )
        exc = SuspendExecution(suspension_info=info)
        with pytest.raises(RuntimeError, match="no checkpoint was persisted"):
            await durable._build_suspended_run(exc)

    async def test_missing_pending_request_raises(self) -> None:
        """If the HITL store lacks the pending request, surface a RuntimeError."""
        hitl_store = InMemoryHitlRequestStore()
        checkpoint_store = InMemoryCheckpointStore()
        workflow = _build_gate_workflow(hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        durable = DurableRun(workflow, hitl_store=hitl_store, checkpoint_store=checkpoint_store)
        # Persist a checkpoint (so the first guard passes) but don't
        # persist any HITL request — the store's pending list is empty.
        info = SuspensionInfo(
            suspension_id="s",
            request_id="r-missing",
            request_type="approval",
            prompt="?",
            agent_name=None,
        )
        checkpoint = RunCheckpoint(
            run_id=_RUN_ID,
            checkpoint_type="orchestration",
            state={},
            suspension_info=info,
        )
        await checkpoint_store.save(checkpoint)

        exc = SuspendExecution(suspension_info=info)
        with pytest.raises(RuntimeError, match="no pending request"):
            await durable._build_suspended_run(exc)
