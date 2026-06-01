"""Tests for ``replace`` on ``FunctionTool`` and ``AgentTool``.

``replace`` returns a copy of the tool with the given schema metadata
overridden (name, description, and the SDK-side flags), preserving the wrapped
function, parameter schema, and ``ToolContext`` injection. The deprecated
``with_return_direct`` alias delegates to it.
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
# FunctionTool.replace
# --------------------------------------------------------------------------- #


@tool(name="echo", description="Echo the text back")
async def _echo(text: str) -> str:
    return f"echo: {text}"


async def test_replace_return_direct_flips_and_preserves_original() -> None:
    assert _echo.schema.return_direct is False

    variant = _echo.replace(return_direct=True)

    assert variant is not _echo
    assert variant.schema.return_direct is True
    assert _echo.schema.return_direct is False  # original untouched
    assert variant.schema.name == "echo"
    assert variant.parameters_model is _echo.parameters_model
    assert (await variant.execute(text="hi")).content == "echo: hi"


def test_replace_no_args_copies_unchanged() -> None:
    variant = _echo.replace()

    assert variant is not _echo
    assert variant.schema.return_direct is False
    assert variant.schema.name == "echo"
    assert variant.schema.description == "Echo the text back"


async def test_replace_name_and_description() -> None:
    variant = _echo.replace(name="shout", description="Shout it")

    assert variant.schema.name == "shout"
    assert variant.schema.description == "Shout it"
    assert _echo.schema.name == "echo"  # original untouched
    assert (await variant.execute(text="hi")).content == "echo: hi"


def test_replace_approval_and_timeout_preserve_return_direct() -> None:
    base = _echo.replace(return_direct=True)

    variant = base.replace(requires_approval=True, timeout_seconds=5.0)

    assert variant.schema.requires_approval is True
    assert variant.schema.timeout_seconds == 5.0
    assert variant.schema.return_direct is True  # carried forward


def test_replace_timeout_none_is_distinct_from_unset() -> None:
    # A tool with timeout set; replace(timeout_seconds=None) must clear it,
    # not be treated as "unchanged" the way an unset argument would.
    base = _echo.replace(timeout_seconds=5.0)
    assert base.schema.timeout_seconds == 5.0

    variant = base.replace(timeout_seconds=None)

    assert variant.schema.timeout_seconds is None


async def test_replace_raw_schema_path() -> None:
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

    variant = base.replace(return_direct=True)

    assert variant.parameters_model is None
    assert variant.schema.return_direct is True
    assert variant.schema.parameters == schema
    assert (await variant.execute(city="Oslo")).content == "weather in Oslo"


async def test_replace_preserves_context_injection() -> None:
    captured: list[ToolContext | None] = []

    @tool(name="needs_ctx", description="Captures the injected context")
    async def needs_ctx(text: str, context: ToolContext) -> str:
        captured.append(context)
        return text

    variant = needs_ctx.replace(return_direct=True)

    ctx = ToolContext(run_id="run-1")
    token = _current_tool_context.set(ctx)
    try:
        result = await variant.execute(text="payload")
    finally:
        _current_tool_context.reset(token)

    assert result.content == "payload"
    assert captured == [ctx]


def test_function_tool_with_return_direct_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="replace"):
        variant = _echo.with_return_direct()
    assert variant.schema.return_direct is True

    with pytest.warns(DeprecationWarning, match="replace"):
        off = _echo.replace(return_direct=True).with_return_direct(False)
    assert off.schema.return_direct is False


# --------------------------------------------------------------------------- #
# AgentTool.replace
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


def test_agent_tool_replace_return_direct() -> None:
    base = _make_agent_tool()
    assert base.schema.return_direct is False

    variant = base.replace(return_direct=True)

    assert variant is not base
    assert variant.schema.return_direct is True
    assert base.schema.return_direct is False  # original untouched
    assert variant.schema.name == "worker"
    assert variant._transfer_strategy is base._transfer_strategy
    assert variant._thread_key == "thread-1"
    assert variant._caller_name == "boss"


def test_agent_tool_replace_name_and_description() -> None:
    variant = _make_agent_tool().replace(name="helper", description="Help out")

    assert variant.schema.name == "helper"
    assert variant.schema.description == "Help out"


def test_agent_tool_replace_no_args_copies_unchanged() -> None:
    base = _make_agent_tool()

    variant = base.replace()

    assert variant is not base
    assert variant.schema.name == "worker"
    assert variant.schema.description == "Delegate work"
    assert variant.schema.return_direct is False


async def test_agent_tool_replace_still_delegates() -> None:
    variant = _make_agent_tool().replace(return_direct=True)

    result = await variant.execute(task="do the thing")

    assert result.content == "delegate output"


def test_agent_tool_with_return_direct_deprecated() -> None:
    base = _make_agent_tool()

    with pytest.warns(DeprecationWarning, match="replace"):
        variant = base.with_return_direct()
    assert variant.schema.return_direct is True

    with pytest.warns(DeprecationWarning, match="replace"):
        off = base.with_return_direct(False)
    assert off.schema.return_direct is False
