"""Tests for HITL tools: factory, execution, event emission."""

import pytest

from nanitics.collaboration.protocol import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputRequest,
    HumanInputResponse,
)
from nanitics.collaboration.tools import (
    create_ask_human_tool,
    create_hitl_tools,
    create_request_approval_tool,
)
from nanitics.infrastructure.errors import ToolExecutionError
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    HumanInputRequestEvent,
    HumanInputResponseEvent,
)
from nanitics.strategies import (
    Tool,
    ToolRegistry,
)
from nanitics.strategies.tools.function_tool import FunctionTool
from nanitics.tracing import ToolCall
from tests.testing_helpers import make_emitter


def make_provider(
    decision: HumanDecision = HumanDecision.APPROVE,
    content: str | None = None,
) -> CallbackHumanInputProvider:
    return CallbackHumanInputProvider(
        callback=lambda req: HumanInputResponse(
            request_id=req.request_id,
            decision=decision,
            content=content,
        )
    )


def _dispatch_tool(
    tool: FunctionTool,
    *,
    arguments: dict[str, object],
    tool_call_id: str = "tc-1",
    run_id: str | None = "test-run",
    agent_name: str | None = None,
    emitter: InMemoryEmitter | None = None,
) -> tuple[ToolRegistry, ToolCall]:
    """Build a one-tool registry and return it with a matching ToolCall.

    Dispatches nothing — caller awaits ``registry.dispatch(call)``. Kept
    sync so tests can assemble inputs without an await boundary.
    """
    state: dict[str, object] = {}
    if run_id is not None:
        state["run_id"] = run_id
    if agent_name is not None:
        state["agent_name"] = agent_name
    registry = ToolRegistry(emitter=emitter, tool_state=state)
    registry.register(tool)
    call = ToolCall(id=tool_call_id, name=tool.schema.name, arguments=arguments)
    return registry, call


async def _run_tool(
    tool: FunctionTool,
    *,
    arguments: dict[str, object],
    tool_call_id: str = "tc-1",
    run_id: str | None = "test-run",
    agent_name: str | None = None,
    emitter: InMemoryEmitter | None = None,
) -> str:
    registry, call = _dispatch_tool(
        tool,
        arguments=arguments,
        tool_call_id=tool_call_id,
        run_id=run_id,
        agent_name=agent_name,
        emitter=emitter,
    )
    result = await registry.dispatch(call)
    return result.content


# ──────────────────────────────────────────────────────────
# Individual Tool Factories
# ──────────────────────────────────────────────────────────


class TestCreateAskHumanTool:
    def test_returns_single_function_tool(self) -> None:
        tool = create_ask_human_tool(make_provider())
        assert isinstance(tool, FunctionTool)

    def test_tool_name(self) -> None:
        tool = create_ask_human_tool(make_provider())
        assert tool.schema.name == "ask_human"

    def test_description_mentions_only_way(self) -> None:
        tool = create_ask_human_tool(make_provider())
        assert "communicate" in tool.schema.description
        assert "not visible" in tool.schema.description

    async def test_executes_correctly(self) -> None:
        tool = create_ask_human_tool(
            make_provider(HumanDecision.ANSWER, content="hello"),
        )
        content = await _run_tool(tool, arguments={"question": "Hi?"})
        assert "hello" in content


class TestCreateRequestApprovalTool:
    def test_returns_single_function_tool(self) -> None:
        tool = create_request_approval_tool(make_provider())
        assert isinstance(tool, FunctionTool)

    def test_tool_name(self) -> None:
        tool = create_request_approval_tool(make_provider())
        assert tool.schema.name == "request_approval"

    def test_description_mentions_only_way(self) -> None:
        tool = create_request_approval_tool(make_provider())
        assert "authorization" in tool.schema.description
        assert "not visible" in tool.schema.description

    async def test_executes_correctly(self) -> None:
        tool = create_request_approval_tool(
            make_provider(HumanDecision.APPROVE),
        )
        content = await _run_tool(
            tool,
            arguments={"action": "deploy", "reason": "release"},
        )
        assert "Approved" in content


# ──────────────────────────────────────────────────────────
# Tool Factory
# ──────────────────────────────────────────────────────────


class TestCreateHitlTools:
    def test_returns_two_tools(self) -> None:
        tools = create_hitl_tools(make_provider())
        assert len(tools) == 2

    def test_tools_are_function_tools(self) -> None:
        tools = create_hitl_tools(make_provider())
        for t in tools:
            assert isinstance(t, FunctionTool)
            assert isinstance(t, Tool)

    def test_tool_names(self) -> None:
        tools = create_hitl_tools(make_provider())
        names = {t.schema.name for t in tools}
        assert names == {"request_approval", "ask_human"}

    def test_tool_schemas_have_descriptions(self) -> None:
        tools = create_hitl_tools(make_provider())
        for t in tools:
            assert t.schema.description


# ──────────────────────────────────────────────────────────
# request_approval Tool
# ──────────────────────────────────────────────────────────


def _pick(tools: list[FunctionTool], name: str) -> FunctionTool:
    return next(t for t in tools if t.schema.name == name)


class TestRequestApprovalTool:
    async def test_approve(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.APPROVE))
        content = await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "delete files", "reason": "cleanup"},
        )
        assert "Approved" in content

    async def test_reject(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.REJECT, content="Too risky"))
        content = await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "delete files", "reason": "cleanup"},
        )
        assert "Rejected" in content
        assert "Too risky" in content

    async def test_override(self) -> None:
        tools = create_hitl_tools(
            make_provider(HumanDecision.OVERRIDE, content="Only delete temp files"),
        )
        content = await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "delete files", "reason": "cleanup"},
        )
        assert "overrides" in content
        assert "Only delete temp files" in content

    async def test_escalate(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.ESCALATE, content="Need manager"))
        content = await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "big purchase", "reason": "needed"},
        )
        assert "Escalated" in content

    async def test_revise_falls_through_to_generic_format(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.REVISE, content="Please revise step 2"))
        content = await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "deploy", "reason": "release"},
        )
        assert "Decision:" in content
        assert "Please revise step 2" in content

    async def test_details_does_not_change_request_type(self) -> None:
        """``details`` routes into the prompt but never changes ``request_type``.

        Both branches (with and without ``details``) emit ``"approval"`` —
        pins the removal of the old ``PLAN_REVIEW if details else APPROVAL``
        branch.
        """
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        provider = CallbackHumanInputProvider(callback=capture)
        tools = create_hitl_tools(provider)
        await _run_tool(
            _pick(tools, "request_approval"),
            arguments={
                "action": "deploy",
                "reason": "release",
                "details": "Step 1: build\nStep 2: deploy",
            },
            tool_call_id="tc-with-details",
        )
        await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "deploy", "reason": "release"},
            tool_call_id="tc-without-details",
        )
        assert len(captured) == 2
        assert captured[0].request_type.value == "approval"
        assert captured[1].request_type.value == "approval"

    async def test_request_id_derived_from_run_id_and_tool_call_id(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "deploy", "reason": "release"},
            run_id="r",
            tool_call_id="tc-42",
        )
        assert captured[0].request_id == "r:tc-42"
        assert captured[0].run_id == "r"

    async def test_agent_name_sourced_from_tool_state(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "deploy", "reason": "release"},
            run_id="r",
            agent_name="a",
        )
        assert captured[0].agent_name == "a"

    async def test_agent_name_defaults_to_none_when_absent_from_state(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(request_id=req.request_id, decision=HumanDecision.APPROVE)

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        await _run_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "deploy", "reason": "release"},
        )
        assert captured[0].agent_name is None

    async def test_missing_run_id_raises(self) -> None:
        tools = create_hitl_tools(make_provider())
        registry, call = _dispatch_tool(
            _pick(tools, "request_approval"),
            arguments={"action": "deploy", "reason": "release"},
            run_id=None,
        )
        with pytest.raises(ToolExecutionError):
            await registry.dispatch(call)

    async def test_missing_tool_call_id_raises(self) -> None:
        # Direct .execute(...) bypasses the registry so no ToolContext is set;
        # the tool must still refuse to emit a request with undefined identity.
        tools = create_hitl_tools(make_provider())
        tool = _pick(tools, "request_approval")
        with pytest.raises(ValueError, match="HITL tools require a ToolContext"):
            await tool.execute(action="deploy", reason="release")


# ──────────────────────────────────────────────────────────
# ask_human Tool
# ──────────────────────────────────────────────────────────


class TestAskHumanTool:
    async def test_answer(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.ANSWER, content="42"))
        content = await _run_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "What is the answer?"},
        )
        assert "42" in content

    async def test_answer_with_options(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ANSWER,
                content="red",
            )

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        content = await _run_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "What color?", "options": ["red", "blue", "green"]},
        )
        assert "red" in content
        assert captured[0].options == ["red", "blue", "green"]

    async def test_answer_with_context(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ANSWER,
                content="yes",
            )

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        await _run_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "Continue?", "context_info": "We are 50% done"},
        )
        assert captured[0].context == "We are 50% done"

    async def test_no_answer_content(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.ANSWER, content=None))
        content = await _run_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "Anything?"},
        )
        assert "no answer" in content.lower()

    async def test_request_id_derived_from_run_id_and_tool_call_id(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ANSWER,
                content="ok",
            )

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        await _run_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "Continue?"},
            run_id="r",
            tool_call_id="tc-77",
        )
        assert captured[0].request_id == "r:tc-77"
        assert captured[0].run_id == "r"

    async def test_agent_name_sourced_from_tool_state(self) -> None:
        captured: list[HumanInputRequest] = []

        def capture(req: HumanInputRequest) -> HumanInputResponse:
            captured.append(req)
            return HumanInputResponse(
                request_id=req.request_id,
                decision=HumanDecision.ANSWER,
                content="yes",
            )

        tools = create_hitl_tools(CallbackHumanInputProvider(callback=capture))
        await _run_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "Continue?"},
            run_id="r",
            agent_name="ask-agent",
        )
        assert captured[0].agent_name == "ask-agent"

    async def test_missing_run_id_raises(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.ANSWER, content="x"))
        registry, call = _dispatch_tool(
            _pick(tools, "ask_human"),
            arguments={"question": "What?"},
            run_id=None,
        )
        with pytest.raises(ToolExecutionError):
            await registry.dispatch(call)

    async def test_missing_tool_call_id_raises(self) -> None:
        tools = create_hitl_tools(make_provider(HumanDecision.ANSWER, content="x"))
        tool = _pick(tools, "ask_human")
        with pytest.raises(ValueError, match="HITL tools require a ToolContext"):
            await tool.execute(question="What?")


# ──────────────────────────────────────────────────────────
# Event Emission
# ──────────────────────────────────────────────────────────


class TestHitlToolEvents:
    def _make_registry(self, tools: list[FunctionTool], emitter: InMemoryEmitter) -> ToolRegistry:
        registry = ToolRegistry(emitter=emitter, tool_state={"run_id": "r"})
        for t in tools:
            registry.register(t)
        return registry

    async def test_request_approval_emits_request_and_response_events(self) -> None:
        emitter = make_emitter()
        tools = create_hitl_tools(make_provider(HumanDecision.APPROVE))
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="request_approval",
                arguments={"action": "delete", "reason": "cleanup"},
            )
        )
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(req_events) == 1
        assert req_events[0].request_type == "approval"
        assert req_events[0].trace_id == "test-trace"
        assert len(resp_events) == 1
        assert resp_events[0].decision == "approve"
        assert resp_events[0].has_content is False
        assert resp_events[0].wait_duration_ms >= 0

    async def test_ask_human_emits_request_and_response_events(self) -> None:
        emitter = make_emitter()
        tools = create_hitl_tools(make_provider(HumanDecision.ANSWER, content="yes"))
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="ask_human",
                arguments={"question": "Continue?"},
            )
        )
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(req_events) == 1
        assert req_events[0].request_type == "question"
        assert len(resp_events) == 1
        assert resp_events[0].decision == "answer"
        assert resp_events[0].has_content is True

    async def test_rejection_emits_events(self) -> None:
        emitter = make_emitter()
        tools = create_hitl_tools(make_provider(HumanDecision.REJECT, content="No way"))
        registry = self._make_registry(tools, emitter)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="request_approval",
                arguments={"action": "destroy", "reason": "testing"},
            )
        )
        resp_events = [e for e in emitter.events if isinstance(e, HumanInputResponseEvent)]
        assert len(resp_events) == 1
        assert resp_events[0].decision == "reject"
        assert resp_events[0].has_content is True

    async def test_request_approval_event_carries_agent_name_and_metadata(self) -> None:
        """``agent_name`` and ``metadata`` flow from ambient tool state onto the event.

        Pins Step 3 contract on ``create_request_approval_tool``: the event
        mirrors the request's ``agent_name`` (from ``tool_state``) and
        ``metadata`` (``{"action", "reason"}``) so machine consumers can
        filter HITL events without re-parsing prose.
        """
        emitter = make_emitter()
        tools = create_hitl_tools(make_provider(HumanDecision.APPROVE))
        registry = ToolRegistry(emitter=emitter, tool_state={"run_id": "r", "agent_name": "approver"})
        for t in tools:
            registry.register(t)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="request_approval",
                arguments={"action": "delete", "reason": "cleanup"},
            )
        )
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].agent_name == "approver"
        assert req_events[0].metadata == {"action": "delete", "reason": "cleanup"}

    async def test_ask_human_event_carries_agent_name_and_metadata(self) -> None:
        """``agent_name`` and ``metadata`` flow from ambient tool state onto the event.

        Pins Step 3 contract on ``create_ask_human_tool``: the event mirrors
        the request's ``agent_name`` (from ``tool_state``) and ``metadata``
        (``{"question"}``).
        """
        emitter = make_emitter()
        tools = create_hitl_tools(make_provider(HumanDecision.ANSWER, content="yes"))
        registry = ToolRegistry(emitter=emitter, tool_state={"run_id": "r", "agent_name": "asker"})
        for t in tools:
            registry.register(t)
        await registry.dispatch(
            ToolCall(
                id="1",
                name="ask_human",
                arguments={"question": "Continue?"},
            )
        )
        req_events = [e for e in emitter.events if isinstance(e, HumanInputRequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].agent_name == "asker"
        assert req_events[0].metadata == {"question": "Continue?"}
