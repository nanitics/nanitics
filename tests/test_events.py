import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from nanitics.infrastructure.errors import LLMRateLimitError, ToolExecutionError
from nanitics.infrastructure.observability.events import (
    AgentCompleteEvent,
    AgentErrorEvent,
    AgentStartEvent,
    AgentStepEvent,
    HumanInputRequestEvent,
    HumanInputResponseEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    SpanEndEvent,
    SpanStartEvent,
    ToolInvokeEvent,
    ToolResultEvent,
    TraceEvent,
    Usage,
)

TRACE_ID = "trace-001"
SPAN_ID = "span-001"


def _base_fields(**overrides: Any) -> dict[str, Any]:
    defaults = {"trace_id": TRACE_ID, "span_id": SPAN_ID}
    defaults.update(overrides)
    return defaults


class TestUsage:
    def test_create(self):
        u = Usage(input_tokens=100, output_tokens=50)
        assert u.input_tokens == 100
        assert u.output_tokens == 50
        assert u.total_tokens == 150

    def test_total_tokens_is_computed(self):
        u = Usage(input_tokens=100, output_tokens=50)
        assert u.total_tokens == u.input_tokens + u.output_tokens
        # total_tokens is derived, not stored: inputs that try to set it
        # are ignored, not honored.
        v = Usage.model_validate({"input_tokens": 10, "output_tokens": 5, "total_tokens": 999})
        assert v.total_tokens == 15
        # It still round-trips through model_dump so downstream consumers see it.
        assert u.model_dump()["total_tokens"] == 150

    def test_frozen(self):
        u = Usage(input_tokens=1, output_tokens=1)
        with pytest.raises(ValidationError):
            u.input_tokens = 5


class TestBaseEventDefaults:
    def test_auto_event_id_and_timestamp(self):
        evt = AgentStartEvent(
            **_base_fields(),
            agent_name="test",
            task_input="do stuff",
            tools_available=["t1"],
        )
        assert evt.event_id  # non-empty UUID string
        assert isinstance(evt.timestamp, datetime)
        assert evt.timestamp.tzinfo is not None

    def test_overridable_defaults(self):
        evt = AgentStartEvent(
            **_base_fields(),
            event_id="custom-id",
            agent_name="test",
            task_input="do stuff",
            tools_available=[],
        )
        assert evt.event_id == "custom-id"


class TestAgentEvents:
    def test_agent_start(self):
        evt = AgentStartEvent(
            **_base_fields(),
            agent_name="researcher",
            task_input="find papers",
            model_name="claude-3",
            tools_available=["search", "read"],
        )
        assert evt.event_type == "agent.start"
        assert evt.agent_name == "researcher"
        assert evt.tools_available == ["search", "read"]

    def test_agent_step(self):
        evt = AgentStepEvent(
            **_base_fields(),
            agent_name="researcher",
            step_number=1,
            thought="I should search",
            action="search",
            observation="found 3 results",
        )
        assert evt.event_type == "agent.step"
        assert evt.agent_name == "researcher"
        assert evt.step_number == 1

    def test_agent_step_artifact_defaults_none(self):
        evt = AgentStepEvent(
            **_base_fields(),
            agent_name="researcher",
            step_number=1,
        )
        assert evt.artifact is None

    def test_agent_step_artifact_round_trip(self):
        evt = AgentStepEvent(
            **_base_fields(),
            agent_name="researcher",
            step_number=1,
            artifact={"k": 1, "nested": {"x": "y"}},
        )
        dumped = evt.model_dump()
        assert dumped["artifact"] == {"k": 1, "nested": {"x": "y"}}
        restored = AgentStepEvent.model_validate(dumped)
        assert restored.artifact == {"k": 1, "nested": {"x": "y"}}

    def test_agent_step_discriminator_resolves_with_artifact(self):
        # The TraceEvent discriminator must still select AgentStepEvent
        # when the artifact field is populated.
        adapter = TypeAdapter(TraceEvent)
        payload = AgentStepEvent(
            **_base_fields(),
            agent_name="researcher",
            step_number=1,
            artifact={"plan": ["a", "b"]},
        ).model_dump()
        restored = adapter.validate_python(payload)
        assert isinstance(restored, AgentStepEvent)
        assert restored.artifact == {"plan": ["a", "b"]}

    def test_agent_complete(self):
        evt = AgentCompleteEvent(
            **_base_fields(),
            agent_name="researcher",
            output="Done",
            total_steps=3,
            termination_reason="completed",
        )
        assert evt.event_type == "agent.complete"
        assert evt.output == "Done"
        assert evt.total_steps == 3

    def test_agent_complete_no_output(self):
        evt = AgentCompleteEvent(
            **_base_fields(),
            agent_name="researcher",
            total_steps=25,
            termination_reason="iteration_limit",
        )
        assert evt.output is None

    def test_agent_error(self):
        evt = AgentErrorEvent(
            **_base_fields(),
            agent_name="researcher",
            error_type="LLMRateLimitError",
            error_message="Rate limited",
            error_metadata={"retry_after": 30.0},
            step_number=5,
        )
        assert evt.event_type == "agent.error"
        assert evt.error_metadata["retry_after"] == 30.0


class TestLLMEvents:
    def test_llm_request(self):
        evt = LLMRequestEvent(
            **_base_fields(),
            model_name="claude-3",
            system_prompt="You are helpful",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "search"}],
        )
        assert evt.event_type == "llm.request"
        assert len(evt.messages) == 1

    def test_llm_response(self):
        evt = LLMResponseEvent(
            **_base_fields(),
            model_name="claude-3",
            content="Hello!",
            usage=Usage(input_tokens=10, output_tokens=5),
            duration_ms=250.0,
        )
        assert evt.event_type == "llm.response"
        assert evt.usage.total_tokens == 15
        assert evt.duration_ms == 250.0


class TestToolEvents:
    def test_tool_invoke(self):
        evt = ToolInvokeEvent(
            **_base_fields(),
            tool_call_id="tc-1",
            tool_name="search",
            parameters={"query": "test"},
        )
        assert evt.event_type == "tool.invoke"
        assert evt.tool_call_id == "tc-1"
        assert evt.parameters["query"] == "test"

    def test_tool_result_success(self):
        evt = ToolResultEvent(
            **_base_fields(),
            tool_call_id="tc-1",
            tool_name="search",
            result="found 3 items",
            success=True,
            duration_ms=100.0,
        )
        assert evt.success is True
        assert evt.tool_call_id == "tc-1"
        assert evt.error is None

    def test_tool_result_failure(self):
        evt = ToolResultEvent(
            **_base_fields(),
            tool_call_id="tc-1",
            tool_name="search",
            error="connection timeout",
            success=False,
            duration_ms=5000.0,
        )
        assert evt.success is False
        assert evt.result is None


class TestSpanEvents:
    def test_span_start(self):
        evt = SpanStartEvent(**_base_fields(), name="llm_call")
        assert evt.event_type == "span.start"
        assert evt.name == "llm_call"

    def test_span_end(self):
        evt = SpanEndEvent(**_base_fields(), name="llm_call", duration_ms=500.0)
        assert evt.event_type == "span.end"
        assert evt.duration_ms == 500.0


class TestSerialization:
    def test_model_dump(self):
        evt = AgentStartEvent(
            **_base_fields(),
            agent_name="test",
            task_input="do stuff",
            tools_available=["t1"],
        )
        d = evt.model_dump()
        assert d["event_type"] == "agent.start"
        assert d["agent_name"] == "test"
        assert "event_id" in d
        assert "timestamp" in d

    def test_model_dump_json(self):
        evt = AgentStartEvent(
            **_base_fields(),
            agent_name="test",
            task_input="do stuff",
            tools_available=[],
        )
        json_str = evt.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "agent.start"


class TestImmutability:
    def test_frozen(self):
        evt = AgentStartEvent(
            **_base_fields(),
            agent_name="test",
            task_input="do stuff",
            tools_available=[],
        )
        with pytest.raises(ValidationError):
            evt.agent_name = "changed"


class TestDiscriminatedUnion:
    adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)

    @pytest.mark.parametrize(
        ("event_cls", "extra_fields", "expected_type"),
        [
            (
                AgentStartEvent,
                {
                    "agent_name": "a",
                    "task_input": "t",
                    "tools_available": [],
                },
                "agent.start",
            ),
            (
                AgentStepEvent,
                {"agent_name": "a", "step_number": 1},
                "agent.step",
            ),
            (
                AgentCompleteEvent,
                {
                    "agent_name": "a",
                    "total_steps": 1,
                    "termination_reason": "completed",
                },
                "agent.complete",
            ),
            (
                AgentErrorEvent,
                {
                    "agent_name": "a",
                    "error_type": "E",
                    "error_message": "m",
                    "error_metadata": {},
                },
                "agent.error",
            ),
            (
                LLMRequestEvent,
                {"model_name": "m", "messages": []},
                "llm.request",
            ),
            (
                LLMResponseEvent,
                {
                    "model_name": "m",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "duration_ms": 1.0,
                },
                "llm.response",
            ),
            (
                ToolInvokeEvent,
                {"tool_call_id": "tc-1", "tool_name": "t", "parameters": {}},
                "tool.invoke",
            ),
            (
                ToolResultEvent,
                {"tool_call_id": "tc-1", "tool_name": "t", "success": True, "duration_ms": 1.0},
                "tool.result",
            ),
            (
                SpanStartEvent,
                {"name": "s"},
                "span.start",
            ),
            (
                SpanEndEvent,
                {"name": "s", "duration_ms": 1.0},
                "span.end",
            ),
            (
                HumanInputRequestEvent,
                {
                    "request_id": "r1",
                    "request_type": "approval",
                    "prompt": "Approve?",
                },
                "hitl.request",
            ),
            (
                HumanInputResponseEvent,
                {
                    "request_id": "r1",
                    "decision": "approve",
                    "has_content": False,
                    "wait_duration_ms": 500,
                },
                "hitl.response",
            ),
        ],
    )
    def test_roundtrip(self, event_cls, extra_fields, expected_type):
        evt = event_cls(**_base_fields(), **extra_fields)
        json_data = evt.model_dump_json()
        restored = self.adapter.validate_json(json_data)
        assert type(restored) is event_cls
        assert restored.event_type == expected_type
        assert restored.event_id == evt.event_id


class TestErrorAsEventMetadata:
    def test_nanitics_error_to_dict_in_event(self):
        err = LLMRateLimitError("Rate limited", retry_after=30.0)
        evt = AgentErrorEvent(
            **_base_fields(),
            agent_name="test",
            error_type=type(err).__name__,
            error_message=err.message,
            error_metadata=err.to_dict(),
        )
        assert evt.error_metadata["retry_after"] == 30.0
        assert evt.error_metadata["message"] == "Rate limited"

    def test_tool_execution_error_to_dict_in_event(self):
        original = ValueError("bad value")
        try:
            raise ToolExecutionError("tool broke", tool_name="calc") from original
        except ToolExecutionError as err:
            evt = AgentErrorEvent(
                **_base_fields(),
                agent_name="test",
                error_type=type(err).__name__,
                error_message=err.message,
                error_metadata=err.to_dict(),
            )
            assert evt.error_metadata["original_error_type"] == "ValueError"


class TestHumanInputEvents:
    def test_request_event(self):
        evt = HumanInputRequestEvent(
            **_base_fields(),
            request_id="req-1",
            request_type="approval",
            prompt="Approve deleting files?",
            agent_name="cleanup-agent",
            tool_name="delete_files",
        )
        assert evt.event_type == "hitl.request"
        assert evt.request_id == "req-1"
        assert evt.request_type == "approval"
        assert evt.prompt == "Approve deleting files?"
        assert evt.agent_name == "cleanup-agent"
        assert evt.tool_name == "delete_files"

    def test_request_event_optional_fields(self):
        evt = HumanInputRequestEvent(
            **_base_fields(),
            request_id="req-2",
            request_type="question",
            prompt="What color?",
        )
        assert evt.agent_name is None
        assert evt.tool_name is None

    def test_response_event(self):
        evt = HumanInputResponseEvent(
            **_base_fields(),
            request_id="req-1",
            decision="approve",
            has_content=False,
            wait_duration_ms=1500,
        )
        assert evt.event_type == "hitl.response"
        assert evt.request_id == "req-1"
        assert evt.decision == "approve"
        assert evt.has_content is False
        assert evt.wait_duration_ms == 1500

    def test_response_event_with_content(self):
        evt = HumanInputResponseEvent(
            **_base_fields(),
            request_id="req-1",
            decision="modify",
            has_content=True,
            wait_duration_ms=3000,
        )
        assert evt.has_content is True
