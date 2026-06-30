"""Streaming-safety and cancellation behavior around the LLM-call retry path.

Two SDK guarantees beyond the existing transient-error retry:

* On a re-streamed attempt (transient retry *or* schema correction), an
  ``LLMStreamResetEvent`` is emitted before the new stream so a consumer can
  discard the partial tokens of the abandoned attempt.
* A cancellation that fires while an LLM call is in flight — including during
  the retry backoff that lives inside the call — stops the run immediately
  with ``termination_reason="cancelled"`` rather than waiting it out.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from nanitics.infrastructure.errors import LLMOverloadedError
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import (
    LLMResponse,
    Message,
    SystemPromptSection,
    ToolSchema,
)
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    LLMStreamResetEvent,
    SafetyCancellationEvent,
    Usage,
)
from nanitics.safety.cancellation import CancellationToken
from nanitics.strategies.agents.react import ReActAgent


def _usage() -> Usage:
    return Usage(input_tokens=1, output_tokens=1)


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], usage=_usage(), model="m", stop_reason="end_turn")


def _raise_overloaded(_messages: list[Message]) -> LLMResponse:
    raise LLMOverloadedError("overloaded", status_code=529)


def _reset_events(emitter: InMemoryEmitter) -> list[LLMStreamResetEvent]:
    return [e for e in emitter.events if isinstance(e, LLMStreamResetEvent)]


class TestStreamReset:
    """A re-streamed attempt announces a reset so partial tokens are discarded."""

    async def test_transient_retry_emits_stream_reset_when_streaming(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace")
        client = MockLLMClient([_raise_overloaded, _text("done")])
        agent = ReActAgent(
            name="r",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[],
            streaming=True,
        )

        result = await agent.run("go")

        assert result.output == "done"
        resets = _reset_events(emitter)
        assert len(resets) == 1
        assert resets[0].agent_name == "r"
        # The reset shares the LLM span so a consumer can target the right stream.
        token_or_response = [e for e in emitter.events if e.event_type in ("llm.response", "llm.stream.reset")]
        assert resets[0].span_id == token_or_response[0].span_id

    async def test_no_stream_reset_when_not_streaming(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace")
        client = MockLLMClient([_raise_overloaded, _text("done")])
        agent = ReActAgent(
            name="r",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[],
            streaming=False,
        )

        result = await agent.run("go")

        assert result.output == "done"
        assert _reset_events(emitter) == []

    async def test_no_stream_reset_on_a_clean_first_attempt(self) -> None:
        emitter = InMemoryEmitter(trace_id="trace")
        client = MockLLMClient([_text("done")])
        agent = ReActAgent(
            name="r",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[],
            streaming=True,
        )

        result = await agent.run("go")

        assert result.output == "done"
        assert _reset_events(emitter) == []


class _BlockingLLMClient:
    """LLM client whose ``generate`` blocks until cancelled.

    Signals ``started`` once entered, then awaits forever, so a test can
    deterministically cancel an in-flight LLM call with no real sleeps.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()

    @property
    def model(self) -> str | None:
        return None

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        output_schema: type | None = None,
        on_token: Callable[[str], None] | None = None,
        system_prompt_sections: list[SystemPromptSection] | None = None,
    ) -> LLMResponse:
        self.started.set()
        await asyncio.Event().wait()  # blocks forever
        return _text("unreachable")  # pragma: no cover


class TestLLMCallCancellation:
    """An in-flight LLM call (generate or its retry backoff) is cancellable."""

    async def test_cancel_during_llm_call_returns_cancelled(self) -> None:
        token = CancellationToken()
        emitter = InMemoryEmitter(trace_id="trace")
        client = _BlockingLLMClient()
        agent = ReActAgent(
            name="r",
            llm_client=client,
            emitter=emitter,
            system_prompt="be brief",
            tools=[],
            cancellation_token=token,
        )

        result = None  # type: ignore[assignment]

        async def _run() -> None:
            nonlocal result
            result = await agent.run("go")

        task = asyncio.create_task(_run())
        await client.started.wait()
        token.cancel()
        await asyncio.wait_for(task, timeout=2.0)

        assert result is not None
        assert result.termination_reason == "cancelled"
        assert result.output is None
        assert sum(1 for e in emitter.events if isinstance(e, SafetyCancellationEvent)) == 1
