from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from nanitics.infrastructure.observability.events import (
    CodeExecutionEvent,
    CodeExecutionResultEvent,
    TraceEvent,
)
from nanitics.safety.sandbox.protocol import ExecutionResult, Sandbox, SandboxConfig

TRACE_ID = "trace-001"
SPAN_ID = "span-001"


def _base_fields(**overrides: Any) -> dict[str, Any]:
    defaults = {"trace_id": TRACE_ID, "span_id": SPAN_ID}
    defaults.update(overrides)
    return defaults


class TestSandboxConfig:
    def test_defaults(self) -> None:
        config = SandboxConfig()
        assert config.image == "python:3.13-slim"
        assert config.timeout == 30.0
        assert config.memory_limit_mb == 256
        assert config.cpu_count == 1.0
        assert config.working_directory == "/sandbox"
        assert config.environment == {}

    def test_custom_values(self) -> None:
        config = SandboxConfig(
            image="python:3.12",
            timeout=60.0,
            memory_limit_mb=512,
            cpu_count=2.0,
            working_directory="/work",
            environment={"KEY": "value"},
        )
        assert config.image == "python:3.12"
        assert config.timeout == 60.0
        assert config.memory_limit_mb == 512
        assert config.cpu_count == 2.0
        assert config.working_directory == "/work"
        assert config.environment == {"KEY": "value"}

    def test_frozen(self) -> None:
        config = SandboxConfig()
        with pytest.raises(ValidationError):
            config.timeout = 99.0


class TestExecutionResult:
    def test_success_result(self) -> None:
        result = ExecutionResult(
            stdout="hello\n",
            stderr="",
            return_value="42",
            success=True,
            duration_ms=15.5,
        )
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.return_value == "42"
        assert result.success is True
        assert result.error is None
        assert result.duration_ms == 15.5

    def test_error_result(self) -> None:
        result = ExecutionResult(
            stdout="",
            stderr="Traceback...",
            success=False,
            error="NameError: name 'x' is not defined",
            duration_ms=2.0,
        )
        assert result.success is False
        assert result.error == "NameError: name 'x' is not defined"
        assert result.return_value is None

    def test_frozen(self) -> None:
        result = ExecutionResult(stdout="", stderr="", success=True, duration_ms=0.0)
        with pytest.raises(ValidationError):
            result.stdout = "changed"


class TestSandboxProtocol:
    def test_structural_subtyping(self) -> None:
        class FakeSandbox:
            async def start(self) -> None: ...
            async def execute(self, code: str) -> ExecutionResult:  # type: ignore[empty-body]
                ...
            async def reset(self) -> None: ...
            async def cleanup(self) -> None: ...
            async def __aenter__(self) -> "FakeSandbox": ...  # type: ignore[empty-body]
            async def __aexit__(self, *args: object) -> None: ...

        assert isinstance(FakeSandbox(), Sandbox)

    def test_non_conforming_rejected(self) -> None:
        class Incomplete:
            async def start(self) -> None: ...

        assert not isinstance(Incomplete(), Sandbox)


class TestCodeExecutionEvent:
    def test_create(self) -> None:
        event = CodeExecutionEvent(
            **_base_fields(),
            agent_name="coder",
            code="print('hi')",
            step_number=1,
        )
        assert event.event_type == "code.execution"
        assert event.agent_name == "coder"
        assert event.code == "print('hi')"
        assert event.step_number == 1

    def test_trace_event_deserialization(self) -> None:
        adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
        data = {
            **_base_fields(),
            "event_type": "code.execution",
            "agent_name": "coder",
            "code": "x = 1",
            "step_number": 1,
        }
        event = adapter.validate_python(data)
        assert isinstance(event, CodeExecutionEvent)


class TestCodeExecutionResultEvent:
    def test_create(self) -> None:
        event = CodeExecutionResultEvent(
            **_base_fields(),
            agent_name="coder",
            stdout="42\n",
            stderr="",
            return_value="42",
            success=True,
            duration_ms=10.0,
            step_number=1,
        )
        assert event.event_type == "code.execution.result"
        assert event.success is True
        assert event.return_value == "42"

    def test_error_event(self) -> None:
        event = CodeExecutionResultEvent(
            **_base_fields(),
            agent_name="coder",
            stdout="",
            stderr="Traceback...",
            success=False,
            error="ValueError",
            duration_ms=5.0,
            step_number=2,
        )
        assert event.success is False
        assert event.error == "ValueError"

    def test_trace_event_deserialization(self) -> None:
        adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)
        data = {
            **_base_fields(),
            "event_type": "code.execution.result",
            "agent_name": "coder",
            "stdout": "",
            "stderr": "",
            "success": True,
            "duration_ms": 1.0,
            "step_number": 1,
        }
        event = adapter.validate_python(data)
        assert isinstance(event, CodeExecutionResultEvent)
