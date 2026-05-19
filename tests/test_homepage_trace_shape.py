"""Pin the trace-shape contract for ``examples/homepage.py``.

The website displays the example's snippet and an Observatory screenshot of
its run side by side; this test prevents silent drift between the two
surfaces. Three independent assertions:

- ``test_trace_event_sequence`` — every event the screenshot is expected to
  show, asserted by type, count, and content invariant.
- ``test_runs_with_no_api_keys_in_environment`` — the example structurally
  does not depend on any provider key reachable in the environment.
- ``test_opens_no_network_sockets`` — the example performs no outbound
  network I/O at any layer.
"""

import os
import socket

import pytest

from examples import homepage
from nanitics.infrastructure import (
    AgentCompleteEvent,
    AgentStartEvent,
    DelegationEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    ToolInvokeEvent,
    ToolResultEvent,
)

_KNOWN_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "VOYAGE_API_KEY",
)


async def test_trace_event_sequence() -> None:
    """Run the homepage example and assert every trace-shape invariant.

    The trace exercises the two-layer agent-as-tool composition: ``reviewer``
    delegates to ``researcher``, which calls the ``search`` tool once and
    drafts a cited answer, then ``reviewer`` composes the final response.
    """
    result, emitter = await homepage.main()
    events = emitter.events

    # --- Termination ---
    assert result.termination_reason == "complete"
    assert isinstance(result.output, str) and len(result.output) > 0

    # --- LLM request/response counts: 2 researcher + 2 reviewer ---
    llm_requests = [e for e in events if isinstance(e, LLMRequestEvent)]
    llm_responses = [e for e in events if isinstance(e, LLMResponseEvent)]
    assert len(llm_responses) == 4
    assert len(llm_requests) == len(llm_responses) == 4

    # --- Agent start/complete pairs ---
    starts_by_name: dict[str, list[AgentStartEvent]] = {}
    completes_by_name: dict[str, list[AgentCompleteEvent]] = {}
    for event in events:
        if isinstance(event, AgentStartEvent):
            starts_by_name.setdefault(event.agent_name, []).append(event)
        elif isinstance(event, AgentCompleteEvent):
            completes_by_name.setdefault(event.agent_name, []).append(event)

    assert len(starts_by_name.get("reviewer", [])) == 1
    assert len(completes_by_name.get("reviewer", [])) == 1
    assert len(starts_by_name.get("researcher", [])) == 1
    assert len(completes_by_name.get("researcher", [])) == 1

    # --- Delegation (reviewer → researcher) ---
    delegation_events = [e for e in events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1
    assert delegation_events[0].caller_agent == "reviewer"
    assert delegation_events[0].delegate_agent == "researcher"
    assert delegation_events[0].task == "What changed in our retry policy last quarter?"

    # --- Tool invocations: 1 search call inside researcher, 1 researcher call inside reviewer ---
    search_invokes = [e for e in events if isinstance(e, ToolInvokeEvent) and e.tool_name == "search"]
    search_results = [e for e in events if isinstance(e, ToolResultEvent) and e.tool_name == "search"]
    assert len(search_invokes) == 1
    assert len(search_results) == 1

    researcher_invokes = [e for e in events if isinstance(e, ToolInvokeEvent) and e.tool_name == "researcher"]
    researcher_results = [e for e in events if isinstance(e, ToolResultEvent) and e.tool_name == "researcher"]
    assert len(researcher_invokes) == 1
    assert len(researcher_results) == 1

    # --- Citations on the researcher's end-of-turn draft ---
    researcher_start_idx = events.index(starts_by_name["researcher"][0])
    researcher_complete_idx = events.index(completes_by_name["researcher"][0])
    end_of_turn_drafts = [
        e
        for i, e in enumerate(events)
        if researcher_start_idx < i < researcher_complete_idx and isinstance(e, LLMResponseEvent) and not e.tool_calls
    ]
    assert end_of_turn_drafts, "expected at least one end-of-turn researcher LLMResponseEvent"
    assert isinstance(end_of_turn_drafts[-1].content, str)
    assert end_of_turn_drafts[-1].content.count("[") == 2  # cites [R-1] and [R-2]


async def test_runs_with_no_api_keys_in_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the example does not depend on any provider API key.

    Removes every named provider key plus a wildcard ``*_API_KEY`` sweep,
    then runs ``main()`` and asserts the run completes.
    """
    for key in _KNOWN_PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in [k for k in os.environ if k.endswith("_API_KEY")]:
        monkeypatch.delenv(key, raising=False)

    result, _ = await homepage.main()
    assert result.termination_reason == "complete"


async def test_opens_no_network_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the example performs no outbound network I/O.

    Patches ``socket.socket`` with a sentinel that raises on construction.
    Any code path that opens a socket fails the test with a clear error.
    """

    def _blocked_socket(*args: object, **kwargs: object) -> socket.socket:
        raise RuntimeError("homepage example must not open network sockets — run is meant to be fully mocked")

    monkeypatch.setattr(socket, "socket", _blocked_socket)

    result, _ = await homepage.main()
    assert result.termination_reason == "complete"
