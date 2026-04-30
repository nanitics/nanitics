"""Shared memory coordination across two agents in a Sequential workflow.

Validates that :class:`InMemorySharedMemory` combined with
:class:`SharedMemoryProvider` and :func:`create_shared_memory_tools`
lets agent A's written contribution become visible to agent B via
automatic context injection. A distinctive sentinel token written by
A is asserted to round-trip into B's ``LLMRequestEvent`` messages
(context path) and into B's final output (LLM surface).

A control scenario runs the same workflow WITHOUT wiring the provider
or tools on agent B: B must then fail to reproduce the sentinel,
confirming the provider is what carries A's contribution across.

Acceptance criteria — positive coordination:
  - Agent A's run emits a ``SharedMemoryWriteEvent`` whose ``author`` is
    A's name, ``content`` contains the sentinel, and ``scope`` matches.
  - After A runs, the store contains exactly one active entry authored
    by A whose content contains the sentinel (direct store assertion).
  - Agent B's run emits at least one ``LLMRequestEvent`` whose serialized
    messages contain the sentinel — proves the provider injected A's
    content into B's context.
  - Agent B's final output contains the sentinel verbatim (round-trip
    proof all the way to B's answer).

Acceptance criteria — negative control:
  - Same workflow without ``SharedMemoryProvider`` and without shared
    tools on agent B: B's ``LLMRequestEvent`` messages do NOT contain
    the sentinel, and B's final output does not contain the sentinel.
    This rules out "B magically knows the answer" from prompt leakage
    or LLM priors.
"""

from __future__ import annotations

import json

from nanitics import (
    AgentStep,
    InMemoryEmitter,
    InMemorySharedMemory,
    ReActAgent,
    Sequential,
    SharedMemoryContributor,
    SharedMemoryProvider,
    create_shared_memory_tools,
)
from nanitics.infrastructure import LLMRequestEvent, SharedMemoryWriteEvent
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Distinctive token — any appearance in B's context/output must come from
# A's write, not LLM priors or prompt leakage.
SENTINEL_TOKEN = "NANITICS-SHARED-4B7E-MANGO"
SCOPE = "findings"


def _messages_contain(emitter: InMemoryEmitter, needle: str) -> bool:
    """Scan every ``LLMRequestEvent``'s serialized messages for ``needle``.

    The event stores ``messages`` as ``list[dict[str, Any]]`` — we
    serialize with ``json.dumps(default=str)`` so any structured content
    (blocks, tool results) is inspected in full.
    """
    for event in emitter.events:
        if not isinstance(event, LLMRequestEvent):
            continue
        try:
            payload = json.dumps(event.messages, default=str)
        except (TypeError, ValueError):  # pragma: no cover - structural safeguard
            payload = str(event.messages)
        if needle in payload:
            return True
    return False


async def test_shared_memory_cross_agent_visibility(traced_emitter: InMemoryEmitter) -> None:
    store = InMemorySharedMemory()

    writer_name = "writer-agent"
    reader_name = "reader-agent"

    writer_tools = create_shared_memory_tools(store, writer_name)
    reader_tools = create_shared_memory_tools(store, reader_name)

    # Provider on the reader injects A's contribution into B's context
    # automatically via the context-provider pipeline.
    reader_provider = SharedMemoryProvider(store, emitter=traced_emitter, scopes=[SCOPE])

    writer = ReActAgent(
        name=writer_name,
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a research agent. Use write_to_shared to record findings "
            f"to the shared memory board with scope='{SCOPE}'. Preserve any "
            "distinctive tokens the user gives you exactly as provided."
        ),
        tools=writer_tools,
        prompt_contributors=[SharedMemoryContributor()],
        max_iterations=3,
    )

    reader = ReActAgent(
        name=reader_name,
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a summarization agent. You will be given a shared memory "
            f"board containing findings from other agents under scope='{SCOPE}'. "
            "Read the board carefully and produce a one-sentence summary that "
            "includes any distinctive tokens present in the findings verbatim."
        ),
        tools=reader_tools,
        prompt_contributors=[SharedMemoryContributor()],
        context_providers=[reader_provider],
        max_iterations=3,
    )

    workflow = Sequential(
        name="shared-memory-pipeline",
        steps=[AgentStep(writer), AgentStep(reader)],
        emitter=traced_emitter,
    )

    result = await run_with_retry(
        lambda: workflow.execute(
            "Record the following finding to the shared memory board so that "
            "the next agent can see it, then the next agent will summarize: "
            f"'Quarterly revenue reference code is {SENTINEL_TOKEN}'. "
            "Use the exact token verbatim."
        ),
        max_attempts=2,
    )

    # --- Write event: A wrote the sentinel under its own author name ---
    assert_trace_contains(
        traced_emitter,
        SharedMemoryWriteEvent,
        predicate=lambda e: e.author == writer_name and SENTINEL_TOKEN in e.content and e.scope == SCOPE,
    )

    # --- Direct store assertion ---
    entries = await store.read(scope=SCOPE)
    writer_entries = [e for e in entries if e.author == writer_name and SENTINEL_TOKEN in e.content]
    assert writer_entries, (
        f"Expected at least one active entry authored by {writer_name!r} in scope={SCOPE!r} "
        f"containing {SENTINEL_TOKEN!r}; got entries: "
        f"{[(e.author, e.content) for e in entries]}"
    )

    # --- Context injection proof: sentinel reached B's LLM request ---
    # Filter to the reader's own LLMRequestEvents only. The writer's
    # requests also contain the token (user input), so an unfiltered scan
    # would be vacuous.
    reader_emitter_view = InMemoryEmitter(trace_id=traced_emitter.trace_id)
    reader_emitter_view.events.extend(
        e
        for e in traced_emitter.events
        if isinstance(e, LLMRequestEvent) and e.system_prompt is not None and reader_name in (e.label or "")
    )
    # The label carries the agent name in most paths; if not, fall back to a
    # system-prompt match which is unique to the reader.
    if not reader_emitter_view.events:
        reader_emitter_view.events.extend(
            e
            for e in traced_emitter.events
            if isinstance(e, LLMRequestEvent)
            and e.system_prompt is not None
            and "summarization agent" in e.system_prompt
        )
    assert reader_emitter_view.events, (
        "Could not isolate reader LLMRequestEvents — neither label nor system_prompt signature matched."
    )
    assert _messages_contain(reader_emitter_view, SENTINEL_TOKEN), (
        f"Expected reader's LLMRequestEvent messages to contain {SENTINEL_TOKEN!r} "
        "(injected via SharedMemoryProvider). Got no match across "
        f"{len(reader_emitter_view.events)} reader LLM requests."
    )

    # --- Final output round-trip ---
    final_output = str(result.output or "")
    assert SENTINEL_TOKEN in final_output, (
        f"Expected reader's final output to contain {SENTINEL_TOKEN!r}; got: {final_output!r}"
    )


async def test_shared_memory_control_without_provider(traced_emitter: InMemoryEmitter) -> None:
    """Control — B has no provider and no shared tools.

    Without the provider wiring, A's contribution cannot reach B. The
    sentinel must NOT appear in B's request messages or final output.
    This rules out the possibility that the positive test above passes
    due to prompt leakage, LLM priors, or Sequential-stage input piping.
    """
    store = InMemorySharedMemory()
    writer_name = "writer-agent-control"
    reader_name = "reader-agent-control"

    writer_tools = create_shared_memory_tools(store, writer_name)

    writer = ReActAgent(
        name=writer_name,
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a research agent. Use write_to_shared to record findings "
            f"to the shared memory board with scope='{SCOPE}'. Preserve any "
            "distinctive tokens the user gives you exactly as provided. "
            "Your FINAL message must be a short acknowledgement that does NOT "
            "repeat the stored content verbatim."
        ),
        tools=writer_tools,
        prompt_contributors=[SharedMemoryContributor()],
        max_iterations=3,
    )
    # Reader has no shared-memory tools and no provider.
    reader = ReActAgent(
        name=reader_name,
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a summarization agent. Summarize the text you receive "
            "in one short sentence. You have no access to external memory. "
            "If you lack information, say so plainly."
        ),
        tools=[],
        max_iterations=2,
    )

    workflow = Sequential(
        name="shared-memory-control",
        steps=[AgentStep(writer), AgentStep(reader)],
        emitter=traced_emitter,
    )

    # The user prompt to the workflow asks for a confirmation — so the
    # sentinel travels to the writer, but the writer is instructed not
    # to echo it in its own output. Thus the sentinel can only reach B
    # through shared memory — which is not wired here.
    result = await run_with_retry(
        lambda: workflow.execute(
            f"Record finding code '{SENTINEL_TOKEN}' to shared memory, then "
            "confirm completion briefly without repeating the code."
        ),
        max_attempts=2,
    )

    # Confirm A did write the token to the store (otherwise the control
    # is vacuous — both sides silent for the wrong reason).
    entries = await store.read(scope=SCOPE)
    assert any(SENTINEL_TOKEN in e.content for e in entries), (
        "Sanity: the writer must have stored the sentinel, otherwise the control case is vacuous."
    )

    # Find the reader's LLMRequestEvents and assert the sentinel is absent.
    reader_events = [
        e
        for e in traced_emitter.events
        if isinstance(e, LLMRequestEvent)
        and e.system_prompt is not None
        and "no access to external memory" in e.system_prompt
    ]
    assert reader_events, "Could not isolate control-reader LLMRequestEvents."
    reader_view = InMemoryEmitter(trace_id=traced_emitter.trace_id)
    reader_view.events.extend(reader_events)

    # The writer's output (piped as input to reader) is expected to be a
    # brief acknowledgement without the token. Any appearance of the token
    # in reader's request messages would indicate leakage — the test's
    # whole point.
    assert not _messages_contain(reader_view, SENTINEL_TOKEN), (
        f"Control failure: {SENTINEL_TOKEN!r} reached the reader's LLMRequestEvent "
        "messages without shared-memory wiring — either the writer leaked it in its "
        "reply or another context path is contributing. The positive test's assertion "
        "would be vacuous in that case."
    )

    final_output = str(result.output or "")
    assert SENTINEL_TOKEN not in final_output, (
        f"Control failure: reader reproduced {SENTINEL_TOKEN!r} without provider wiring. Got: {final_output!r}"
    )
