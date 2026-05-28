"""Information continuity vs behavioral continuity — same task, two substrates.

Two scenarios produce a haiku and revise it. They use different SDK
substrates and surface the prior draft to the model in structurally
different ways.

Scenario A — ``InMemoryWorkingMemory``. One multi-step ``agent.run`` in
which step 1 journals the draft into a ``working_memory`` section. Step
2's prepared message list contains the journaled draft inside a
``<nanitics:context provider="working_memory">…</nanitics:context>``
envelope. The wrapper structurally signals "injected context" — the
model does not treat it as its own prior turn.

Scenario B — ``thread_key`` + ``InMemoryThreadStore``. Two separate
``agent.run`` calls keyed to the same thread. Run 2's prepared message
list contains run 1's response as an unwrapped ``assistant``-role
message. The model treats it as its own prior work and revises it
directly.

WorkingMemory is information continuity (what the agent should know).
Threads are behavioral continuity (what the agent produced). The two
substrates sit on different axes and compose.

Related guide: docs/guides/memory.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.composition import InMemoryThreadStore
from nanitics.infrastructure import LLMRequestEvent, MockLLMClient
from nanitics.memory import (
    InMemoryWorkingMemory,
    WorkingMemoryProvider,
)
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import ToolCall


@tool("note_revision_request", "Acknowledge that a revision was requested.")
async def note_revision_request() -> str:
    """Tool present only to keep the ReAct loop going from step 1 to step 2."""
    return "Revision requested."


async def _run_scenario_a_working_memory() -> list[dict]:
    """Scenario A: one run, two steps. Step 1 journals; step 2 revises.

    Returns the messages the LLM saw on step 2 (the revision step).
    """
    emitter = make_emitter("substrate-a")
    memory = InMemoryWorkingMemory()
    provider = WorkingMemoryProvider(memory)

    client = MockLLMClient(
        [
            # Step 1: draft the haiku and journal it; emit a tool call so
            # the ReAct loop continues into step 2.
            make_response(
                content=(
                    "Drafting the haiku.\n"
                    "<working_memory>\n"
                    "## Latest draft\n"
                    "Quiet morning rain / soft on the empty rooftops / waking the city.\n"
                    "</working_memory>"
                ),
                tool_calls=[ToolCall(id="tc-1", name="note_revision_request", arguments={})],
                stop_reason="tool_use",
            ),
            # Step 2: revise — the journaled draft is now present in the
            # LLM's message list as a <nanitics:context> envelope.
            make_response(
                content=("Quiet morning rain / soft on the empty rooftops / neighbours wake slowly."),
            ),
        ]
    )

    agent = ReActAgent(
        name="poet-a",
        llm_client=client,
        emitter=emitter,
        system_prompt=(
            "You are a haiku poet. Journal your latest draft into "
            "<working_memory>## Latest draft\\n…</working_memory> so you can "
            "revise it on the next step."
        ),
        tools=[note_revision_request],
        context_providers=[provider],
        working_memory=memory,
    )

    await agent.run("Draft a haiku about a quiet morning, then revise it for warmer imagery.")

    # Two LLMRequestEvents — one per step. The second is what step 2 saw.
    req_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
    assert len(req_events) == 2, f"Expected two LLM requests in one run; got {len(req_events)}"
    return req_events[1].messages


async def _run_scenario_b_threads() -> list[dict]:
    """Scenario B: two runs, same thread key. Returns run-2's messages."""
    emitter = make_emitter("substrate-b")
    store = InMemoryThreadStore()

    client = MockLLMClient(
        [
            # Run 1: draft the haiku.
            make_response(content="Quiet morning rain / soft on the empty rooftops / waking the city."),
            # Run 2: revise — run 1's response is replayed unwrapped as an
            # assistant-role message, so the agent treats it as its own work.
            make_response(content="Quiet morning rain / soft on the empty rooftops / neighbours wake slowly."),
        ]
    )

    agent = ReActAgent(
        name="poet-b",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a haiku poet.",
        tools=[],
        thread_store=store,
    )

    await agent.run("Draft a haiku about a quiet morning.", thread_key="poem-1")
    await agent.run("Revise the previous draft for warmer imagery.", thread_key="poem-1")

    # One LLMRequestEvent per run. Index 1 is run 2's request.
    req_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
    assert len(req_events) == 2, f"Expected two LLM requests across two runs; got {len(req_events)}"
    return req_events[1].messages


async def main() -> None:
    # --- Section 1: Run both scenarios ---
    print("--- Section 1: Run both scenarios ---")

    scenario_a_messages = await _run_scenario_a_working_memory()
    scenario_b_messages = await _run_scenario_b_threads()

    print(f"  Scenario A (WorkingMemory, step 2)  message count: {len(scenario_a_messages)}")
    print(f"  Scenario B (thread_key, run 2)      message count: {len(scenario_b_messages)}")

    print("✓ Section 1 passed")

    # --- Section 2: Scenario A — <nanitics:context> wrapper ---
    print("\n--- Section 2: Scenario A — <nanitics:context> wrapper ---")

    # WorkingMemoryProvider emits the journaled draft inside a
    # <nanitics:context provider="working_memory">…</nanitics:context>
    # envelope on a user-role message.
    a_contents = [m.get("content") or "" for m in scenario_a_messages]
    a_wrapped = [c for c in a_contents if '<nanitics:context provider="working_memory"' in c]
    assert a_wrapped, (
        'Scenario A\'s step-2 messages must contain at least one <nanitics:context provider="working_memory"> envelope.'
    )

    # The journaled draft appears inside the wrapper.
    assert any("Quiet morning rain" in c for c in a_wrapped), (
        "The journaled draft must appear inside the working_memory wrapper."
    )

    # Scenario A does have prior assistant messages in step 2 (the ReAct
    # loop carries each step's assistant turn forward), but the *journaled
    # draft itself* is surfaced via the wrapper — not as a standalone
    # assistant-role message that the model treats as its own prior turn
    # in the substrate-distinction sense.
    print("  Scenario A:")
    print('    Found <nanitics:context provider="working_memory"> wrapper ✓')
    print("    Journaled draft is inside the wrapper ✓")
    print(f"    Wrapped chunk preview: {a_wrapped[0][:80]}...")

    print("✓ Section 2 passed")

    # --- Section 3: Scenario B — unwrapped assistant replay ---
    print("\n--- Section 3: Scenario B — unwrapped assistant replay ---")

    # In Scenario B run 1's response is replayed as an assistant-role
    # message with no wrapper around it.
    b_assistant_turns = [m for m in scenario_b_messages if m.get("role") == "assistant"]
    assert b_assistant_turns, (
        "Scenario B's run-2 messages must contain at least one "
        "unwrapped assistant-role turn (the replayed prior draft)."
    )
    for m in b_assistant_turns:
        content = m.get("content") or ""
        assert "<nanitics:context" not in content, (
            "Replayed thread messages must bypass the <nanitics:context> wrapper; "
            f"found wrapper bytes in assistant content: {content!r}"
        )

    b_contents = [m.get("content") or "" for m in scenario_b_messages]
    b_wrapped = [c for c in b_contents if "<nanitics:context" in c]
    assert not b_wrapped, "Scenario B must contain no <nanitics:context> envelopes."

    # Concrete: the prior draft is exactly the assistant-role content.
    assistant_contents = [m["content"] for m in b_assistant_turns]
    assert any("Quiet morning rain" in c for c in assistant_contents), (
        "The prior haiku draft must appear verbatim as an assistant turn."
    )

    print("  Scenario B:")
    print("    Found unwrapped assistant-role turn ✓")
    print("    No <nanitics:context> envelope present ✓")
    print(f"    Assistant turn preview: {b_assistant_turns[0]['content'][:80]}...")

    print("✓ Section 3 passed")

    # --- Section 4: Substrate-distinction summary ---
    print("\n--- Section 4: Substrate-distinction summary ---")
    print("  WorkingMemory: information continuity (what the agent should know).")
    print('    → Surfaces as <nanitics:context provider="working_memory">…</nanitics:context>.')
    print("    → Model reads it as injected context, not its own prior turn.")
    print("  thread_key:    behavioral continuity (what the agent produced).")
    print("    → Surfaces as an unwrapped assistant-role message.")
    print("    → Model treats it as its own prior work and revises in place.")
    print("  The two substrates compose; they sit on different axes.")
    print("✓ Section 4 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
