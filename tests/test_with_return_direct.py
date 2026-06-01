"""Tests for ``with_return_direct`` on ``FunctionTool`` and ``AgentTool``.

Deriving a tool-terminating variant of a tool defined once: the copy flips
``return_direct`` and preserves everything else (wrapped function, parameter
schema, ``ToolContext`` injection, and the other SDK-side ``ToolSchema`` flags).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from nanitics.composition.multi_agent.agent_tool import AgentTool
from nanitics.composition.multi_agent.context_transfer import RawOutputTransfer
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import Usage
from nanitics.strategies.agents.base import AgentResult
from nanitics.strategies.tools import FunctionTool, ToolContext, tool
from nanitics.strategies.tools.context import _current_tool_context

# --------------------------------------------------------------------------- #
# FunctionTool
# --------------------------------------------------------------------------- #


@tool(name="echo", description="Echo the text back")
async def _echo(text: str) -> str:
    return f"echo: {text}"


@pytest.mark.asyncio
async def test_function_tool_flips_and_preserves_original() -> None:
    assert _echo.schema.return_direct is False

    variant = _echo.with_return_direct()

    assert variant is not _echo
    assert variant.schema.return_direct is True
    assert _echo.schema.return_direct is False  # original untouched
    assert variant.schema.name == "echo"
    assert variant.parameters_model is _echo.parameters_model
    assert (await variant.execute(text="hi")).content == "echo: hi"


@pytest.mark.asyncio
async def test_function_tool_value_false() -> None:
    @tool(name="terminal", description="A terminating tool", return_direct=True)
    async def terminal() -> str:
        return "done"

    variant = terminal.with_return_direct(False)

    assert variant.schema.return_direct is False
    assert (await variant.execute()).content == "done"


@pytest.mark.asyncio
async def test_function_tool_raw_schema_path() -> None:
    async def impl(city: str) -> str:
        return f"weather in {city}"

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    base = FunctionTool(
        fn=impl,
        name="weather",
        description="Get weather",
        parameters_schema=schema,
    )

    variant = base.with_return_direct()

    assert variant.parameters_model is None
    assert variant.schema.return_direct is True
    assert variant.schema.parameters == schema
    assert (await variant.execute(city="Oslo")).content == "weather in Oslo"


def test_function_tool_preserves_other_schema_flags() -> None:
    # requires_approval / timeout_seconds are SDK-side flags the constructor
    # does not currently accept; the copy must still carry them forward.
    tweaked = _echo.with_return_direct(False)
    tweaked._schema = tweaked._schema.model_copy(update={"requires_approval": True, "timeout_seconds": 5.0})

    variant = tweaked.with_return_direct()

    assert variant.schema.return_direct is True
    assert variant.schema.requires_approval is True
    assert variant.schema.timeout_seconds == 5.0


@pytest.mark.asyncio
async def test_function_tool_preserves_context_injection() -> None:
    captured: list[ToolContext | None] = []

    @tool(name="needs_ctx", description="Captures the injected context")
    async def needs_ctx(text: str, context: ToolContext) -> str:
        captured.append(context)
        return text

    variant = needs_ctx.with_return_direct()

    ctx = ToolContext(run_id="run-1")
    token = _current_tool_context.set(ctx)
    try:
        result = await variant.execute(text="payload")
    finally:
        _current_tool_context.reset(token)

    assert result.content == "payload"
    assert captured == [ctx]


# --------------------------------------------------------------------------- #
# AgentTool
# --------------------------------------------------------------------------- #


def _make_result() -> AgentResult:
    return AgentResult(
        output="delegate output",
        total_steps=1,
        termination_reason="complete",
        messages=[],
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _make_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.name = "delegate"
    agent.run = AsyncMock(return_value=_make_result())
    handle = Mock()

    async def _forward(*a, **kw):
        return await agent.run(*a, **kw)

    handle.run = _forward
    agent.bind = Mock(return_value=handle)
    agent.set_cancellation_token = Mock()
    return agent


def _make_agent_tool() -> AgentTool:
    return AgentTool(
        agent=_make_agent(),
        emitter=InMemoryEmitter(trace_id="t"),
        description="Delegate work",
        name="worker",
        transfer_strategy=RawOutputTransfer(),
        caller_name="boss",
        thread_key="thread-1",
    )


def test_agent_tool_flips_and_preserves_original() -> None:
    base = _make_agent_tool()
    assert base.schema.return_direct is False

    variant = base.with_return_direct()

    assert variant is not base
    assert variant.schema.return_direct is True
    assert base.schema.return_direct is False  # original untouched
    assert variant.schema.name == "worker"
    assert variant.schema.description == "Delegate work"
    assert variant._transfer_strategy is base._transfer_strategy
    assert variant._thread_key == "thread-1"
    assert variant._caller_name == "boss"


def test_agent_tool_value_false() -> None:
    base = AgentTool(
        agent=_make_agent(),
        emitter=InMemoryEmitter(trace_id="t"),
        description="Delegate work",
        return_direct=True,
    )

    variant = base.with_return_direct(False)

    assert variant.schema.return_direct is False


@pytest.mark.asyncio
async def test_agent_tool_variant_still_delegates() -> None:
    variant = _make_agent_tool().with_return_direct()

    result = await variant.execute(task="do the thing")

    assert result.content == "delegate output"
