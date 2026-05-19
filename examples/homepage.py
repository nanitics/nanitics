"""Two-layer composition — ``researcher`` → ``reviewer``.

``researcher`` is a ``ReActAgent`` with a ``search`` tool. ``reviewer`` is a
``ReActAgent`` that wraps the researcher in an ``AgentTool`` and composes the
final answer from the delegated result. The composition uses only the core
surface — no specialized strategies, no evaluators — to mirror the Start here
path on the website (Tools, Memory, Multi-Agent Foundations).

The visible portion of this file (delimited by the ``HOMEPAGE VISIBLE`` start/end
comment markers below) is fetched at build time by the website and rendered
verbatim on the homepage proof section. Imports, ``MockLLMClient`` setup, the
static mock corpus, the closing assertions, and the ``asyncio.run(main())`` wrapper
sit outside the markers so the snippet stays focused on composition.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio
import random
from typing import Any

from examples.helpers import make_emitter, make_response
from nanitics.composition import AgentTool
from nanitics.infrastructure import (
    DelegationEvent,
    LLMResponse,
    LLMResponseEvent,
    MockLLMClient,
)
from nanitics.strategies import (
    AgentResult,
    ReActAgent,
    tool,
)
from nanitics.tracing import (
    InMemoryEmitter,
    ToolCall,
)

# Delay ranges (seconds) injected into the mock LLM and tool calls so the captured
# trace shows realistic per-span durations rather than every span clocking 0ms.
# Pure mocks would otherwise make the duration column on the homepage screenshot
# read as constant zeros.
_LLM_DELAY_RANGE = (0.4, 0.7)
_TOOL_DELAY_RANGE = (0.06, 0.15)


async def _jittered_sleep(low: float, high: float) -> None:
    await asyncio.sleep(random.uniform(low, high))


# Static mock corpus served by the search tool. Stable IDs R-1, R-2, R-3 let the
# researcher cite hits as [R-1], [R-2], etc. The corpus is returned in full for
# any query so the example is deterministic without query-routing logic.
_SEARCH_CORPUS: list[tuple[str, str]] = [
    ("R-1", "Q3 baseline: exponential backoff with up to 5 attempts."),
    ("R-2", "Q4 change: backoff is now jittered; max attempts lowered from 5 to 3."),
    ("R-3", "Q4 change: retried POSTs require an idempotency-key header."),
]


@tool("search", "Search engineering notes for retry-policy records.")
async def search(query: str) -> str:
    await _jittered_sleep(*_TOOL_DELAY_RANGE)
    return "\n".join(f"[{rid}] {snippet}" for rid, snippet in _SEARCH_CORPUS)


class _DelayedMockLLMClient:
    """Wraps a ``MockLLMClient`` and inserts a jittered sleep before each
    response, so per-span durations in the captured trace look realistic
    rather than reading as a column of zeros.
    """

    def __init__(self, inner: MockLLMClient) -> None:
        self._inner = inner

    @property
    def model(self) -> str | None:
        return self._inner.model

    async def generate(self, **kwargs: Any) -> LLMResponse:
        await _jittered_sleep(*_LLM_DELAY_RANGE)
        return await self._inner.generate(**kwargs)


# Scripted mocks: 2 researcher responses (search → cite [R-1] [R-2]), 2 reviewer
# responses (delegate → final composed answer). Four LLM responses total on the
# shared timeline.
_RESEARCHER_RESPONSES = [
    make_response(
        "I'll search the engineering notes for retry-policy records.",
        tool_calls=[
            ToolCall(
                id="r-tc-1",
                name="search",
                arguments={"query": "retry policy changes last quarter"},
            )
        ],
        stop_reason="tool_use",
    ),
    make_response(
        "Two changes landed last quarter: backoff is now jittered, and the maximum number of retry "
        "attempts dropped from 5 to 3 [R-1] [R-2]."
    ),
]
_REVIEWER_RESPONSES = [
    make_response(
        "I'll delegate this to the researcher.",
        tool_calls=[
            ToolCall(
                id="rev-tc-1",
                name="researcher",
                arguments={"task": "What changed in our retry policy last quarter?"},
            )
        ],
        stop_reason="tool_use",
    ),
    make_response(
        "Two changes landed last quarter: jittered backoff, and the maximum retry attempts dropped from "
        "5 to 3 [R-1] [R-2]."
    ),
]


async def main(emitter: InMemoryEmitter | None = None) -> tuple[AgentResult, InMemoryEmitter]:
    if emitter is None:
        emitter = make_emitter("homepage")
    researcher_llm = _DelayedMockLLMClient(MockLLMClient(responses=_RESEARCHER_RESPONSES))
    reviewer_llm = _DelayedMockLLMClient(MockLLMClient(responses=_REVIEWER_RESPONSES))

    # --- HOMEPAGE VISIBLE START ---
    researcher = ReActAgent(
        name="researcher",
        llm_client=researcher_llm,
        emitter=emitter,
        system_prompt="Research the question using search() and cite results as [R-N].",
        tools=[search],
    )
    reviewer = ReActAgent(
        name="reviewer",
        llm_client=reviewer_llm,
        emitter=emitter,
        system_prompt="Delegate research to the specialist, then compose the final answer.",
        tools=[
            AgentTool(
                agent=researcher,
                emitter=emitter,
                caller_name="reviewer",
                description="Delegate research to the specialist researcher.",
            )
        ],
    )
    result = await reviewer.run("What changed in our retry policy last quarter?")
    # --- HOMEPAGE VISIBLE END ---

    # --- Trace-shape invariants. Full coverage in tests/test_homepage_trace_shape.py. ---
    assert result.termination_reason == "complete"
    assert isinstance(result.output, str) and result.output, "reviewer final answer is non-empty"

    delegation_events = [e for e in emitter.events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1
    assert delegation_events[0].caller_agent == "reviewer"
    assert delegation_events[0].delegate_agent == "researcher"
    assert delegation_events[0].task == "What changed in our retry policy last quarter?"

    llm_responses = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
    assert len(llm_responses) == 4, (
        f"Expected 4 LLMResponseEvents (2 researcher + 2 reviewer), got {len(llm_responses)}"
    )

    print("--- Homepage example: composing ReActAgent + AgentTool ---")
    print(f"  Reviewer final answer: {result.output}")
    print(f"  Delegation recorded: {delegation_events[0].caller_agent} → {delegation_events[0].delegate_agent}")
    print(f"  LLM responses on shared timeline: {len(llm_responses)}")
    print("✓ Trace shape: reviewer delegates → researcher searches → researcher drafts → reviewer composes")

    return result, emitter


if __name__ == "__main__":
    asyncio.run(main())
