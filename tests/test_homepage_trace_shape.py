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
    EvaluationEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    ReflectionGeneratedEvent,
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

    The full set of invariants lives in ``technical-spec.md`` §1 of Phase
    2.2. Each assertion below maps to one row of that table.
    """
    result, emitter = await homepage.main()
    events = emitter.events

    # --- §1.1 Termination ---
    assert result.termination_reason == "complete"  # T1
    assert isinstance(result.output, str) and len(result.output) > 0  # T2

    # --- §1.2 LLM request/response counts ---
    llm_requests = [e for e in events if isinstance(e, LLMRequestEvent)]
    llm_responses = [e for e in events if isinstance(e, LLMResponseEvent)]
    assert len(llm_responses) == 7  # L1 (4 researcher + 1 reflection + 2 coordinator)
    assert len(llm_requests) == len(llm_responses) == 7  # L2

    # --- §1.3 Agent start/complete pairs (per named agent) ---
    starts_by_name: dict[str, list[AgentStartEvent]] = {}
    completes_by_name: dict[str, list[AgentCompleteEvent]] = {}
    for event in events:
        if isinstance(event, AgentStartEvent):
            starts_by_name.setdefault(event.agent_name, []).append(event)
        elif isinstance(event, AgentCompleteEvent):
            completes_by_name.setdefault(event.agent_name, []).append(event)

    assert len(starts_by_name.get("coordinator", [])) == 1  # A1
    assert len(completes_by_name.get("coordinator", [])) == 1  # A1
    assert len(starts_by_name.get("grounded", [])) == 1  # A2
    assert len(completes_by_name.get("grounded", [])) == 1  # A2
    assert len(starts_by_name.get("researcher", [])) == 2  # A3 (attempt 1 + attempt 2)
    assert len(completes_by_name.get("researcher", [])) == 2  # A3

    # --- §1.4 Evaluation verdict sequence ---
    eval_events = [e for e in events if isinstance(e, EvaluationEvent)]
    assert len(eval_events) == 2  # E1
    assert (eval_events[0].verdict, eval_events[1].verdict) == ("revise", "accept")  # E2

    # --- §1.5 Reflection ---
    reflection_events = [e for e in events if isinstance(e, ReflectionGeneratedEvent)]
    assert len(reflection_events) == 1  # R1
    reflection_index = events.index(reflection_events[0])
    first_eval_index = events.index(eval_events[0])
    second_eval_index = events.index(eval_events[1])
    assert first_eval_index < reflection_index < second_eval_index  # R2

    # --- §1.6 Delegation (coordinator → grounded) ---
    delegation_events = [e for e in events if isinstance(e, DelegationEvent)]
    assert len(delegation_events) == 1  # D1
    assert delegation_events[0].caller_agent == "coordinator"  # D2
    assert delegation_events[0].delegate_agent == "grounded"  # D3
    assert delegation_events[0].task == "What changed in our retry policy last quarter?"  # D4

    # --- §1.7 Tool invocations ---
    search_invokes = [e for e in events if isinstance(e, ToolInvokeEvent) and e.tool_name == "search"]
    search_results = [e for e in events if isinstance(e, ToolResultEvent) and e.tool_name == "search"]
    assert len(search_invokes) == 2  # I1
    assert len(search_results) == 2  # I2

    grounded_invokes = [e for e in events if isinstance(e, ToolInvokeEvent) and e.tool_name == "grounded"]
    grounded_results = [e for e in events if isinstance(e, ToolResultEvent) and e.tool_name == "grounded"]
    assert len(grounded_invokes) == 1  # I3
    assert len(grounded_results) == 1  # I4

    # --- §1.8 Citations on the inner-researcher drafts ---
    # Identify the two researcher attempts by their AgentStart/AgentComplete index windows.
    researcher_starts = [events.index(e) for e in starts_by_name["researcher"]]
    researcher_completes = [events.index(e) for e in completes_by_name["researcher"]]
    researcher_windows = sorted(zip(researcher_starts, researcher_completes, strict=True))

    def _end_of_turn_draft_in_window(start_idx: int, end_idx: int) -> str:
        """Return the content of the final no-tool-calls LLM response in a researcher window."""
        candidates = [
            e
            for i, e in enumerate(events)
            if start_idx < i < end_idx and isinstance(e, LLMResponseEvent) and not e.tool_calls
        ]
        # The last no-tool-calls response in the window is the end-of-turn draft.
        assert candidates, "expected at least one end-of-turn researcher LLMResponseEvent"
        content = candidates[-1].content
        assert isinstance(content, str)
        return content

    attempt_one_text = _end_of_turn_draft_in_window(*researcher_windows[0])
    attempt_two_text = _end_of_turn_draft_in_window(*researcher_windows[1])
    assert attempt_one_text.count("[") == 1  # C1: cites [R-1] once
    assert attempt_two_text.count("[") == 2  # C2: cites [R-1] and [R-2]


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
