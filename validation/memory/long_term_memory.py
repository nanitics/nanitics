"""Long-term memory round-trip across two agent runs.

Validates that :class:`InMemoryLongTermStore` wired into a
:class:`ReActAgent` via :func:`create_long_term_memory_tools` persists
data across separate ``agent.run()`` calls. A distinctive sentinel token
is written in run 1 and recalled in run 2 — the token's presence in the
store between runs and in the final output of run 2 is the defining
property (a broken store or missing namespace plumbing would fail the
round-trip). A second scenario covers the negative case: retrieval for a
key that was never written.

Acceptance criteria — positive round-trip:
  - Run 1 emits a ``ToolInvokeEvent`` for the ``store_memory`` tool.
  - The sentinel token appears verbatim under the agreed key in the
    store immediately after run 1 (direct assertion on store state).
  - Run 2 emits a ``ToolInvokeEvent`` for the ``recall_memory`` tool.
  - Run 2 emits a ``LongTermRetrieveEvent`` with ``found=True`` whose
    ``value`` equals the sentinel (pins the recall payload round-trip).
  - Run 2's final output contains the sentinel token verbatim —
    proves the recalled value reached the LLM and was surfaced.

Acceptance criteria — negative case (missing key):
  - No ``store_memory`` tool ever ran for the target key, so the store
    is empty for that key.
  - ``recall_memory`` is invoked and emits a ``LongTermRetrieveEvent``
    with ``found=False``.
  - The agent's final output communicates absence of the requested
    memory (LLM-as-judge check).
"""

from __future__ import annotations

from nanitics import (
    InMemoryEmitter,
    InMemoryLongTermStore,
    ReActAgent,
    create_long_term_memory_tools,
)
from nanitics.infrastructure import LongTermRetrieveEvent, ToolInvokeEvent
from validation.helpers import (
    assert_result_satisfies,
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)

# Distinctive sentinel — guarantees any match in output is the recalled
# value, not an LLM confabulation.
SENTINEL_KEY = "project_launch_code"
SENTINEL_VALUE = "NANITICS-LTM-9F2C-ZEBRA"


async def test_long_term_memory_round_trip(traced_emitter: InMemoryEmitter) -> None:
    store = InMemoryLongTermStore()
    tools = create_long_term_memory_tools(store)

    agent = ReActAgent(
        name="long-term-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a memory-augmented assistant with long-term memory tools "
            "(store_memory, recall_memory, delete_memory, list_memory_keys). "
            "When the user asks you to remember something, use store_memory with "
            "a descriptive key and the exact value they provided. When they ask "
            "you to look something up, use recall_memory and report the exact "
            "value you retrieve."
        ),
        tools=tools,
        max_iterations=5,
    )

    # --- Run 1: write the sentinel into long-term memory ---
    await run_with_retry(
        lambda: agent.run(
            f"Please remember this for later. Store it under the key "
            f"'{SENTINEL_KEY}' with the exact value '{SENTINEL_VALUE}'. "
            "Do not paraphrase the value."
        ),
        max_attempts=2,
    )

    # Tool-invocation proof for write.
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "store_memory",
    )

    # Direct assertion on the store — the sentinel must be present verbatim
    # under the agreed key. This pins the write path regardless of what the
    # agent says in run 1's textual output.
    stored_value = await store.retrieve(SENTINEL_KEY)
    assert stored_value == SENTINEL_VALUE, (
        f"Expected store[{SENTINEL_KEY!r}] == {SENTINEL_VALUE!r} after run 1; got: {stored_value!r}"
    )

    # --- Run 2: recall the sentinel on the same agent + same store ---
    result2 = await run_with_retry(
        lambda: agent.run(
            f"Look up the value I asked you to remember under the key "
            f"'{SENTINEL_KEY}'. Report the exact stored value verbatim in "
            "your final answer."
        ),
        max_attempts=2,
    )

    # Tool-invocation proof for read.
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "recall_memory",
    )

    # Payload round-trip: the retrieve event itself must show found=True
    # and carry the sentinel as its value. Without this the test could
    # pass if the agent echoed the sentinel from its own prompt context.
    assert_trace_contains(
        traced_emitter,
        LongTermRetrieveEvent,
        predicate=lambda e: e.key == SENTINEL_KEY and e.found is True and e.value == SENTINEL_VALUE,
    )

    # Final output must contain the sentinel verbatim — proves the recalled
    # value actually reached the LLM's final message.
    final_output = str(result2.output or "")
    assert SENTINEL_VALUE in final_output, (
        f"Expected run 2's output to contain the sentinel {SENTINEL_VALUE!r} verbatim; got: {final_output!r}"
    )


async def test_long_term_memory_missing_key(traced_emitter: InMemoryEmitter) -> None:
    """Negative case — retrieval for a never-written key.

    The agent must attempt a recall, observe absence via
    :class:`LongTermRetrieveEvent` with ``found=False``, and communicate
    that absence in its final output. A silent default-value masking bug
    in ``retrieve`` (e.g. returning ``""`` instead of ``None``) would
    cause this test to fail.
    """
    store = InMemoryLongTermStore()
    tools = create_long_term_memory_tools(store)

    agent = ReActAgent(
        name="long-term-agent-missing",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt=(
            "You are a memory-augmented assistant. Use recall_memory to look up "
            "stored values. If the tool reports no value is stored under the "
            "requested key, tell the user clearly that nothing is stored there."
        ),
        tools=tools,
        max_iterations=4,
    )

    missing_key = "nonexistent_project_code"
    # Sanity: nothing stored yet — guards against this test silently
    # passing if seeding leaked from elsewhere.
    assert await store.retrieve(missing_key) is None, "Store must be empty for this test to be meaningful."

    result = await run_with_retry(
        lambda: agent.run(f"Please look up the value stored under the key '{missing_key}' and tell me what you find."),
        max_attempts=2,
    )

    # recall_memory must have been invoked and surfaced a not-found result.
    assert_trace_contains(
        traced_emitter,
        ToolInvokeEvent,
        predicate=lambda e: e.tool_name == "recall_memory",
    )
    assert_trace_contains(
        traced_emitter,
        LongTermRetrieveEvent,
        predicate=lambda e: e.key == missing_key and e.found is False and e.value is None,
    )

    await assert_result_satisfies(
        str(result.output or ""),
        (
            "The output clearly states that no value is stored under the requested "
            f"key '{missing_key}' (or equivalent phrasing indicating absence). It must "
            "NOT fabricate a value."
        ),
    )
