"""Tests for ApprovalWrappedTool: approve, reject, modify paths, event emission."""

from uuid import uuid4

import pytest

from nanitics import Tool
from nanitics.collaboration.approval_wrapped import ApprovalWrappedTool
from nanitics.collaboration.protocol import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
)
from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.infrastructure.llm.protocol import ToolCall
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)
from nanitics.strategies.tools import ToolRegistry
from nanitics.strategies.tools.function_tool import tool
from nanitics.strategies.tools.protocol import ToolResult
from tests.testing_helpers import make_emitter


def make_provider(
    decision: HumanDecision = HumanDecision.APPROVE,
    content: str | None = None,
    metadata: dict | None = None,
) -> CallbackHumanInputProvider:
    return CallbackHumanInputProvider(
        callback=lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=content,
            metadata=metadata or {},
        )
    )


@tool(name="dangerous_action", description="A dangerous action that needs approval")
async def dangerous_action(target: str, force: bool = False) -> str:
    return f"Executed on {target} (force={force})"


async def _dispatch_wrapped(
    wrapped: ApprovalWrappedTool,
    arguments: dict,
    *,
    run_id: str | None = "test-run",
    tool_call_id: str = "tc-1",
    agent_name: str | None = None,
    registry_emitter: InMemoryEmitter | None = None,
) -> ToolResult:
    """Build a one-tool registry for ``wrapped`` and dispatch a single call.

    Routes the execution through a ``ToolRegistry`` so that
    ``_current_tool_context`` carries ``run_id`` and ``tool_call_id`` —
    the ambient identity the wrapper reads to derive ``request_id``.
    """
    state: dict[str, object] = {}
    if run_id is not None:
        state["run_id"] = run_id
    if agent_name is not None:
        state["agent_name"] = agent_name
    registry = ToolRegistry(emitter=registry_emitter, tool_state=state)
    registry.register(wrapped)
    call = ToolCall(id=tool_call_id, name=wrapped.schema.name, arguments=arguments)
    return await registry.dispatch(call)


class TestApprovalWrappedToolSchema:
    def test_preserves_tool_name(self) -> None:
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=make_provider())
        assert wrapped.schema.name == "dangerous_action"

    def test_preserves_tool_description(self) -> None:
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=make_provider())
        assert wrapped.schema.description == "A dangerous action that needs approval"

    def test_preserves_tool_parameters(self) -> None:
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=make_provider())
        assert wrapped.schema.parameters == dangerous_action.schema.parameters

    def test_satisfies_tool_protocol(self) -> None:
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=make_provider())
        assert isinstance(wrapped, Tool)

    def test_requires_approval_is_true(self) -> None:
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=make_provider())
        assert wrapped.schema.requires_approval is True

    def test_unwrapped_tool_requires_approval_is_false(self) -> None:
        assert dangerous_action.schema.requires_approval is False


class TestApprovalWrappedToolExecution:
    async def test_approve_delegates_to_inner_tool(self) -> None:
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
        )
        result = await _dispatch_wrapped(wrapped, {"target": "server-1"})
        assert "Executed on server-1" in result.content

    async def test_reject_returns_message_without_executing(self) -> None:
        call_count = 0

        @tool(name="tracked", description="tracks calls")
        async def tracked_tool(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        wrapped = ApprovalWrappedTool(
            tool=tracked_tool,
            provider=make_provider(HumanDecision.REJECT, content="Too risky"),
        )
        result = await _dispatch_wrapped(wrapped, {"x": "test"})
        assert "rejected" in result.content.lower()
        assert "Too risky" in result.content
        assert call_count == 0
        assert result.executed is False

    async def test_override_passes_altered_params(self) -> None:
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(
                HumanDecision.OVERRIDE,
                content="Use force=True",
                metadata={"modified_params": {"force": True}},
            ),
        )
        result = await _dispatch_wrapped(wrapped, {"target": "server-1", "force": False})
        assert "force=True" in result.content

    async def test_escalate_returns_rejection(self) -> None:
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.ESCALATE, content="Need manager"),
        )
        result = await _dispatch_wrapped(wrapped, {"target": "server-1"})
        assert "rejected" in result.content.lower()


class TestApprovalWrappedToolEvents:
    async def test_emits_request_and_response_events(self) -> None:
        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await _dispatch_wrapped(wrapped, {"target": "server-1"})
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(req_events) == 1
        assert req_events[0].tool_name == "dangerous_action"
        assert req_events[0].request_type == "approval"
        assert len(resp_events) == 1
        assert resp_events[0].decision == "approve"
        assert resp_events[0].wait_duration_ms >= 0

    async def test_prompt_is_concise(self) -> None:
        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await _dispatch_wrapped(wrapped, {"target": "server-1"})
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert req_events[0].prompt == "Approve tool 'dangerous_action' execution?"

    async def test_context_contains_params_and_description(self) -> None:
        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await _dispatch_wrapped(wrapped, {"target": "server-1", "force": True})
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert req_events[0].context is not None
        assert "server-1" in req_events[0].context
        assert "A dangerous action that needs approval" in req_events[0].context

    async def test_rejection_emits_events(self) -> None:
        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.REJECT, content="No"),
            emitter=emitter,
        )
        await _dispatch_wrapped(wrapped, {"target": "server-1"})
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(resp_events) == 1
        assert resp_events[0].decision == "reject"
        assert resp_events[0].has_content is True

    async def test_no_emitter_works(self) -> None:
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
        )
        result = await _dispatch_wrapped(wrapped, {"target": "server-1"})
        assert "Executed" in result.content

    async def test_event_agent_name_from_tool_state(self) -> None:
        """``HumanInputRequestEvent.agent_name`` mirrors ``tool_state['agent_name']``.

        Pins Step 3 contract on wrapped-tool emission: the agent name is
        threaded from the ambient ToolContext.state onto the event, not only
        onto the request.
        """
        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await _dispatch_wrapped(wrapped, {"target": "server-1"}, agent_name="mailer")
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].agent_name == "mailer"

    async def test_event_metadata_contains_tool_name_and_parameters(self) -> None:
        """``HumanInputRequestEvent.metadata`` mirrors the request's metadata.

        Pins Fork 3: wrapped-tool approvals expose ``{tool_name, parameters}``
        on the event so machine consumers don't need to re-parse the
        human-facing ``context`` prose.
        """
        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )
        await _dispatch_wrapped(wrapped, {"target": "server-1", "force": True})
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].metadata == {
            "tool_name": "dangerous_action",
            "parameters": {"target": "server-1", "force": True},
        }


class TestApprovalWrappedToolAgentName:
    async def test_agent_name_sourced_from_tool_state(self) -> None:
        captured: list = []

        def capture(req):
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.APPROVE,
            )

        provider = CallbackHumanInputProvider(callback=capture)
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=provider)
        await _dispatch_wrapped(wrapped, {"target": "server-1"}, agent_name="my-agent")
        assert captured[0].agent_name == "my-agent"

    async def test_agent_name_defaults_to_none(self) -> None:
        captured: list = []

        def capture(req):
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.APPROVE,
            )

        provider = CallbackHumanInputProvider(callback=capture)
        wrapped = ApprovalWrappedTool(tool=dangerous_action, provider=provider)
        await _dispatch_wrapped(wrapped, {"target": "server-1"})
        assert captured[0].agent_name is None


class TestApprovalWrappedToolDeterministicIdentity:
    async def test_request_id_derived_from_run_id_and_tool_call_id(self) -> None:
        captured: list = []

        def capture(req):
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.APPROVE,
            )

        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=CallbackHumanInputProvider(callback=capture),
        )
        await _dispatch_wrapped(
            wrapped,
            {"target": "server-1"},
            run_id="r",
            tool_call_id="tc-42",
        )
        assert captured[0].request_id == "r:tc-42"
        assert captured[0].run_id == "r"

    async def test_missing_run_id_raises(self) -> None:
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
        )
        with pytest.raises(ToolExecutionError):
            await _dispatch_wrapped(wrapped, {"target": "server-1"}, run_id=None)

    async def test_missing_tool_context_raises(self) -> None:
        # Direct .execute(...) outside a registry — no ToolContext ambient.
        wrapped = ApprovalWrappedTool(
            tool=dangerous_action,
            provider=make_provider(HumanDecision.APPROVE),
        )
        with pytest.raises(ValueError, match="ApprovalWrappedTool requires a ToolContext"):
            await wrapped.execute(target="server-1")


class TestApprovalWrappedToolRegistryDispatch:
    """Registry-level emission gating for the approval wrapper.

    Pins the contract that ``ApprovalWrappedTool`` reject/escalate paths
    return ``executed=False`` and therefore suppress the registry's
    ``ToolInvokeEvent`` / ``ToolResultEvent`` pair, while the HITL
    request/response events remain visible on the wrapper's emitter.
    The approve path retains the complementary invariant — full
    invoke/result pair on the registry's emitter.
    """

    async def test_reject_suppresses_registry_events(self) -> None:
        """Reject path: registry emits zero invoke/result; HITL events present."""
        call_count = 0

        @tool(name="counter", description="counts calls")
        async def counter(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        # Distinct emitters: the registry sees tool events; the wrapper sees HITL events.
        registry_emitter = InMemoryEmitter(trace_id="reg-trace")
        wrapper_emitter = InMemoryEmitter(trace_id="wrap-trace")
        wrapped = ApprovalWrappedTool(
            tool=counter,
            provider=make_provider(HumanDecision.REJECT, content="No"),
            emitter=wrapper_emitter,
        )

        registry = ToolRegistry(emitter=registry_emitter, tool_state={"run_id": "r"})
        registry.register(wrapped)

        result = await registry.dispatch(ToolCall(id=str(uuid4()), name="counter", arguments={"x": "v"}))

        assert result.executed is False
        assert call_count == 0

        assert [e for e in registry_emitter.events if isinstance(e, ToolInvokeEvent)] == []
        assert [e for e in registry_emitter.events if isinstance(e, ToolResultEvent)] == []

        req_events = [e for e in wrapper_emitter.events if isinstance(e, HumanInputRequestEvent)]
        resp_events = [e for e in wrapper_emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(req_events) == 1
        assert len(resp_events) == 1
        assert resp_events[0].decision == "reject"

    async def test_approve_emits_registry_events(self) -> None:
        """Approve path: registry emits invoke+result; HITL events present."""
        call_count = 0

        @tool(name="counter", description="counts calls")
        async def counter(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=counter,
            provider=make_provider(HumanDecision.APPROVE),
            emitter=emitter,
        )

        registry = ToolRegistry(emitter=emitter, tool_state={"run_id": "r"})
        registry.register(wrapped)

        result = await registry.dispatch(ToolCall(id=str(uuid4()), name="counter", arguments={"x": "v"}))

        assert result.executed is True
        assert call_count == 1

        invoke_events = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke_events) == 1
        assert len(result_events) == 1
        assert result_events[0].success is True

        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(req_events) == 1
        assert len(resp_events) == 1
        assert resp_events[0].decision == "approve"

    async def test_escalate_suppresses_registry_events(self) -> None:
        """Escalate path: same suppression invariant as reject."""
        call_count = 0

        @tool(name="counter", description="counts calls")
        async def counter(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        emitter = make_emitter()
        wrapped = ApprovalWrappedTool(
            tool=counter,
            provider=make_provider(HumanDecision.ESCALATE, content="Need manager"),
            emitter=emitter,
        )

        registry = ToolRegistry(emitter=emitter, tool_state={"run_id": "r"})
        registry.register(wrapped)

        result = await registry.dispatch(ToolCall(id=str(uuid4()), name="counter", arguments={"x": "v"}))

        assert result.executed is False
        assert call_count == 0

        assert [e for e in emitter.events if isinstance(e, ToolInvokeEvent)] == []
        assert [e for e in emitter.events if isinstance(e, ToolResultEvent)] == []

        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(resp_events) == 1
        assert resp_events[0].decision == "escalate"
