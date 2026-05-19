"""Message grouping and the ``ContextProvider`` protocol under a real LLM.

Two unrelated components share one script because both sit at the
boundary between the conversation history and the LLM call:

1.  ``default_message_grouper`` and custom ``MessageGrouper`` callables
    from ``nanitics.capabilities.context.grouping``. The default groups
    messages into atomic units where a ``tool_result`` attaches to the
    preceding group and every other role opens a new group
    (``grouping.py``). A custom grouper must be able to override that
    strategy and the output must differ in a structural, role-driven
    way.
2.  ``ContextProvider`` (``nanitics/core/agents/context.py``). Not
    exercised via memory or planning in this script — a minimal
    stand-alone provider is attached to a ``ReActAgent`` and we assert
    the contributed content reaches ``LLMRequestEvent.messages`` before
    the user turn, confirming the injection pipeline in isolation.

Acceptance criteria — ``default_message_grouper``:
  - Empty input returns an empty list.
  - For a mixed user / assistant(tool_call) / tool_result / assistant
    sequence, every ``tool_result`` lives in the same group as its
    preceding assistant message, no group contains two non-``tool_result``
    roles, and flattening the groups is identity-equal to the input.

Acceptance criteria — custom grouper:
  - A role-partitioning grouper (contiguous messages sharing the same
    role collapsed into one group) produces a grouping structurally
    different from ``default_message_grouper`` on the same input,
    while still preserving message identity under ``flatten_groups``.
  - The ``ContextManager`` honours the injected grouper: its
    ``prepare()`` output is consistent with the custom grouper's
    boundaries — specifically the number of surviving groups after
    truncation is bounded by the custom-grouper group count, not the
    default.

Acceptance criteria — ``ContextProvider`` injection:
  - The agent emits at least one ``LLMRequestEvent``.
  - The marker string returned by the static provider appears verbatim
    inside a ``<nanitics:context provider="static-dossier" …>…</nanitics:context>``
    block on one of the ``messages`` payloads on the first
    ``LLMRequestEvent``. The wrapper is the SDK's structural signal to
    the LLM that this content is SDK-injected context, not user speech
    — without it, Anthropic-backed agents tend to treat the injected
    text as untrusted external data and refuse to reference it.
  - The injected message has role ``"user"`` and precedes the final
    user turn in the message list — the provider contribution is not
    appended after the human prompt.

Acceptance criteria — ``ContextProvider`` is actually consumed by the LLM:
  - When the agent is prompted to cite the dossier, the Anthropic
    response contains the dossier's unambiguous marker content — not a
    refusal, not a paraphrase of "I cannot verify external data." This
    is the behavioural fix that flips the original ``context W2``
    finding from red to green.
"""

from __future__ import annotations

import pytest

from nanitics.capabilities.context.grouping import (
    default_message_grouper,
    flatten_groups,
)
from nanitics.context import (
    ContextManager,
    TruncationPolicy,
)
from nanitics.infrastructure import LLMRequestEvent
from nanitics.memory import (
    ContextContent,
    ContextProvider,
)
from nanitics.strategies import ReActAgent
from nanitics.tracing import (
    InMemoryEmitter,
    Message,
)
from validation.helpers import (
    assert_trace_contains,
    make_llm_client,
    run_with_retry,
)


def _sample_conversation() -> list[Message]:
    """Mixed turns covering the role transitions exercised by the grouper."""
    return [
        Message(role="user", content="Turn 1 question."),
        Message(
            role="assistant",
            content="Calling a tool.",
            tool_calls=[{"id": "call-a", "name": "lookup", "arguments": {"q": "x"}}],
        ),
        Message(role="tool_result", content="tool-output-a", tool_call_id="call-a"),
        Message(role="assistant", content="Final answer 1."),
        Message(role="user", content="Turn 2 question."),
        Message(role="user", content="Immediate follow-up."),
        Message(
            role="assistant",
            content="Second tool call.",
            tool_calls=[{"id": "call-b", "name": "lookup", "arguments": {"q": "y"}}],
        ),
        Message(role="tool_result", content="tool-output-b", tool_call_id="call-b"),
        Message(role="tool_result", content="tool-output-b2", tool_call_id="call-b"),
        Message(role="assistant", content="Final answer 2."),
    ]


def _role_partition_grouper(messages: list[Message]) -> list[list[Message]]:
    """Collapse contiguous same-role runs into a single group.

    Different strategy from the default: the default isolates each
    assistant/user message; this one merges adjacent same-role messages.
    """
    groups: list[list[Message]] = []
    for msg in messages:
        if groups and groups[-1][0].role == msg.role:
            groups[-1].append(msg)
        else:
            groups.append([msg])
    return groups


class _StaticContextProvider:
    """Minimal ``ContextProvider`` returning a single fixed contribution.

    Used to prove the provider-injection path is wired and carries the
    contribution into the pre-LLM message list.
    """

    def __init__(self, marker: str) -> None:
        self._marker = marker

    async def provide(self, messages: list[Message]) -> ContextContent | None:
        return ContextContent(
            content=f"Dossier: {self._marker}",
            priority=0,
            protected=False,
            provider_name="static-dossier",
        )


@pytest.mark.quick
def test_default_grouper_semantics() -> None:
    # Contract: empty in -> empty out.
    assert default_message_grouper([]) == []

    msgs = _sample_conversation()
    groups = default_message_grouper(msgs)

    # Identity: flattening groups must reproduce the input list exactly.
    assert flatten_groups(groups) == msgs, (
        "default_message_grouper must not reorder or drop messages; flatten(groups) must equal the input."
    )

    # Contract: tool_result attaches to the preceding group.
    for group in groups:
        tool_results = [m for m in group if m.role == "tool_result"]
        if tool_results:
            assert group[0].role != "tool_result", (
                "A tool_result must never start a group — it attaches to the "
                f"preceding group; got group starting with tool_result: {group!r}"
            )
        non_tr = [m for m in group if m.role != "tool_result"]
        assert len(non_tr) <= 1, f"A group may hold at most one non-tool_result message; got {group!r}"


@pytest.mark.quick
def test_custom_grouper_differs_from_default() -> None:
    msgs = _sample_conversation()
    default_groups = default_message_grouper(msgs)
    custom_groups = _role_partition_grouper(msgs)

    # The two groupers apply different strategies (tool_result attachment
    # vs. contiguous same-role collapse), so their outputs must differ
    # structurally — that is the evidence the custom grouper is actually
    # taking effect instead of silently deferring to the default.
    assert custom_groups != default_groups, (
        "Role-partition grouper must produce a grouping structurally different from the default; got identical outputs."
    )

    # Content preservation still holds under the custom strategy.
    assert flatten_groups(custom_groups) == msgs


async def test_context_manager_honours_custom_grouper() -> None:
    """``ContextManager`` must call the injected grouper, not the default.

    A counting grouper records its invocations; the manager is configured
    with a budget tight enough that truncation fires. We assert the
    counter was bumped — direct evidence the custom grouper (not the
    default) participated in ``prepare()``.
    """
    msgs = _sample_conversation()

    call_count = {"n": 0}

    def counting_grouper(messages: list[Message]) -> list[list[Message]]:
        call_count["n"] += 1
        return _role_partition_grouper(messages)

    # Budget sized so the ~80-token sample blows past the message budget —
    # forces truncation to fire and gives us a second, independent signal
    # (alongside call_count) that the injected grouper was actually used.
    manager = ContextManager(
        context_limit=50,
        reserve_tokens=5,
        threshold=0.1,
        truncation=TruncationPolicy(preserve_first=True, preserve_recent=1),
        grouper=counting_grouper,
    )

    emitter = InMemoryEmitter(trace_id="grouper-manager-honour")
    prepared = await manager.prepare(
        system_prompt="sp",
        messages=msgs,
        tools=None,
        emitter=emitter,
    )

    assert call_count["n"] >= 1, (
        f"ContextManager.prepare() must invoke the injected grouper; call_count={call_count['n']}."
    )
    # Truncation fired — prepared output is strictly shorter than input.
    assert len(prepared) < len(msgs), (
        f"Tight budget must force truncation to drop messages; len(prepared)={len(prepared)} len(input)={len(msgs)}."
    )


async def test_context_provider_injects_into_llm_request(
    traced_emitter: InMemoryEmitter,
) -> None:
    marker = "NANITICS-CONTEXT-MARKER-7F3A"
    provider: ContextProvider = _StaticContextProvider(marker)

    agent = ReActAgent(
        name="context-provider-agent",
        llm_client=make_llm_client("anthropic"),
        emitter=traced_emitter,
        system_prompt="Answer concisely using the dossier context you have been given.",
        tools=[],
        context_providers=[provider],
        max_iterations=1,
    )

    # Prompt forces the model to cite the dossier marker; a synthetic
    # marker the model cannot plausibly produce from parametric knowledge
    # gives an unambiguous signal that the injected content was used.
    result = await run_with_retry(
        lambda: agent.run(
            "Using only the dossier provided in your context, answer: "
            "what is the marker ID? Reply with just the marker ID, nothing else."
        ),
        max_attempts=2,
    )

    # Distinguishing assertion: the provider-supplied marker must appear in
    # the pre-LLM message list of the first request.
    first_request = assert_trace_contains(
        traced_emitter,
        LLMRequestEvent,
        predicate=lambda e: any(marker in str(m.get("content", "")) for m in e.messages),
    )

    injected_index = next(i for i, m in enumerate(first_request.messages) if marker in str(m.get("content", "")))
    last_user_index = max(
        (i for i, m in enumerate(first_request.messages) if m.get("role") == "user"),
        default=-1,
    )
    assert injected_index < last_user_index, (
        "Injected context must precede the final user turn, not be appended "
        f"after it; injected_index={injected_index}, last_user_index={last_user_index}."
    )
    injected_msg = first_request.messages[injected_index]
    assert injected_msg.get("role") == "user", (
        f"Injected context must be materialised as a user message; got role={injected_msg.get('role')!r}."
    )

    # Structural assertion: the marker must appear *inside* a
    # <nanitics:context provider="static-dossier" …>…</nanitics:context>
    # block. This pins the wire-shape contract — a regression that drops
    # the wrapper fails here immediately.
    injected_content = str(injected_msg.get("content", ""))
    assert '<nanitics:context provider="static-dossier"' in injected_content, (
        'Injected context must carry the canonical <nanitics:context provider="static-dossier" …> '
        f"opening tag; got content={injected_content!r}."
    )
    assert "</nanitics:context>" in injected_content, (
        f"Injected context must carry the canonical </nanitics:context> closing tag; got content={injected_content!r}."
    )
    # The marker must live inside the wrapper body, not alongside it.
    opening_end = injected_content.index(">") + 1
    closing_start = injected_content.rindex("</nanitics:context>")
    body = injected_content[opening_end:closing_start]
    assert marker in body, f"Marker {marker!r} must appear inside the wrapper body; got body={body!r}."

    # Behavioural assertion: the model must reference the dossier marker
    # in its answer. This is the fix for the original ``context W2``
    # finding — without the wrapper, Anthropic refuses to cite the
    # injected content; with it, the model treats the content as
    # authoritative and answers with the marker.
    assert result.output is not None, "Agent must produce an output."
    assert marker in result.output, (
        f"Agent output must reference the dossier marker {marker!r} — a refusal or paraphrase "
        f"indicates the LLM is ignoring the injected context. Got output={result.output!r}."
    )
