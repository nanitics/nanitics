"""Tests for ToolRegistry's integration with ToolResultPolicy."""

from typing import Any

import pytest

from nanitics.capabilities.context.token_counter import EstimateTokenCounter
from nanitics.capabilities.context.tool_result import (
    ErrorOnLargeToolResult,
    ToolResultContext,
    TruncateToolResult,
)
from nanitics.infrastructure.errors import ToolError, ToolResultTooLargeError
from nanitics.infrastructure.llm.protocol import ToolCall, ToolSchema
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    ToolInvokeEvent,
    ToolResultEvent,
    ToolResultPolicyAppliedEvent,
)
from nanitics.strategies.tools import ToolRegistry
from nanitics.strategies.tools.protocol import ToolResult


class _BigTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="big", description="returns lots of data", parameters={})

    async def execute(self, **params: Any) -> ToolResult:
        return ToolResult(content="x" * 1000)


class _SmallTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="small", description="small", parameters={})

    async def execute(self, **params: Any) -> ToolResult:
        return ToolResult(content="hi")


class _WrapperTool:
    """Returns ``executed=False`` to model a wrapper short-circuit."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="wrapped", description="wrapper", parameters={})

    async def execute(self, **params: Any) -> ToolResult:
        return ToolResult(content="x" * 1000, executed=False)


class _RaisingTool:
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(name="raises", description="raises", parameters={})

    async def execute(self, **params: Any) -> ToolResult:
        raise ToolError("intentional")


class TestConstructor:
    def test_policy_without_counter_raises(self) -> None:
        with pytest.raises(ValueError, match="token_counter"):
            ToolRegistry(tool_result_policy=TruncateToolResult(max_tokens=10))

    def test_policy_with_counter_ok(self) -> None:
        ToolRegistry(
            tool_result_policy=TruncateToolResult(max_tokens=10),
            token_counter=EstimateTokenCounter(),
        )

    def test_no_policy_no_counter_ok(self) -> None:
        ToolRegistry()


class TestPolicyHook:
    async def test_policy_applied_on_success(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(
            emitter=emitter,
            tool_result_policy=TruncateToolResult(max_tokens=5),
            token_counter=EstimateTokenCounter(),
        )
        registry.register(_BigTool())
        result = await registry.dispatch(ToolCall(id="c1", name="big", arguments={}))
        assert result.metadata["truncated"] is True

        # ToolResultEvent records the post-policy content
        result_events = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is True
        assert result_events[0].result == result.content

    async def test_policy_not_applied_on_wrapper_suppressed(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(
            emitter=emitter,
            tool_result_policy=TruncateToolResult(max_tokens=5),
            token_counter=EstimateTokenCounter(),
        )
        registry.register(_WrapperTool())
        result = await registry.dispatch(ToolCall(id="c1", name="wrapped", arguments={}))
        # No truncation marker — policy did NOT run on a wrapper-suppressed result
        assert "truncated" not in result.metadata
        # And no invoke/result events were emitted
        assert [e for e in emitter.events if isinstance(e, ToolInvokeEvent)] == []
        assert [e for e in emitter.events if isinstance(e, ToolResultEvent)] == []

    async def test_policy_not_applied_on_tool_error(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(
            emitter=emitter,
            tool_result_policy=TruncateToolResult(max_tokens=5),
            token_counter=EstimateTokenCounter(),
        )
        registry.register(_RaisingTool())
        with pytest.raises(ToolError):
            await registry.dispatch(ToolCall(id="c1", name="raises", arguments={}))
        # Policy did not produce any event
        assert [e for e in emitter.events if isinstance(e, ToolResultPolicyAppliedEvent)] == []

    async def test_policy_raise_routes_through_tool_error_branch(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(
            emitter=emitter,
            tool_result_policy=ErrorOnLargeToolResult(max_tokens=5),
            token_counter=EstimateTokenCounter(),
        )
        registry.register(_BigTool())
        with pytest.raises(ToolResultTooLargeError):
            await registry.dispatch(ToolCall(id="c1", name="big", arguments={}))

        invoke = [e for e in emitter.events if isinstance(e, ToolInvokeEvent)]
        result_evt = [e for e in emitter.events if isinstance(e, ToolResultEvent)]
        assert len(invoke) == 1
        assert len(result_evt) == 1
        assert result_evt[0].success is False
        # Invoke before result
        assert emitter.events.index(invoke[0]) < emitter.events.index(result_evt[0])

    async def test_no_policy_behaves_as_before(self) -> None:
        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(emitter=emitter)
        registry.register(_SmallTool())
        result = await registry.dispatch(ToolCall(id="c1", name="small", arguments={}))
        assert result.content == "hi"
        assert result.metadata == {}


class TestContextPropagation:
    async def test_context_carries_tool_call_and_emitter(self) -> None:
        """The policy receives the ToolCall and the emitter through ToolResultContext."""
        seen: list[ToolResultContext] = []

        class _Spy:
            async def apply(self, result: ToolResult, context: ToolResultContext) -> ToolResult:
                seen.append(context)
                return result

            def reset(self) -> None:
                pass

        emitter = InMemoryEmitter(trace_id="t1")
        registry = ToolRegistry(
            emitter=emitter,
            tool_result_policy=_Spy(),
            token_counter=EstimateTokenCounter(),
        )
        registry.register(_SmallTool())
        await registry.dispatch(ToolCall(id="c1", name="small", arguments={"a": 1}))
        assert len(seen) == 1
        assert seen[0].tool_call.id == "c1"
        assert seen[0].tool_call.name == "small"
        assert seen[0].tool_call.arguments == {"a": 1}
        assert seen[0].emitter is emitter
