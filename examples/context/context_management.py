"""Context management: token counting, message grouping, truncation, summarization, and ContextManager.

Demonstrates the context management pipeline that keeps long-running agents within their token
budget. Builds up from primitives (token counting, message grouping) through ContextManager
orchestration (truncation, summarization, combined strategies with protected messages).

Related guide: docs/guides/context-management.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.context import (
    ContextManager,
    ContextUsage,
    EstimateTokenCounter,
    SummarizationPolicy,
    TruncationPolicy,
    count_message_tokens,
    default_message_grouper,
)
from nanitics.infrastructure import (
    ContextSummarizationEvent,
    ContextTruncationEvent,
    MockLLMClient,
)
from nanitics.tracing import (
    Message,
    ToolCall,
)


async def main() -> None:
    # --- Section 1: Token Counting ---
    print("--- Section 1: Token Counting ---")

    counter = EstimateTokenCounter(chars_per_token=4.0)

    # count_text: character-based estimation
    text = "Hello, world!"  # 13 chars → max(1, int(13 / 4.0)) = 3
    text_tokens = counter.count_text(text)
    assert text_tokens == 3, f"Expected 3, got {text_tokens}"
    print(f"  '{text}' ({len(text)} chars) → {text_tokens} tokens")

    # count_message_tokens: 4-token overhead + content tokens
    msg = Message(role="user", content="Hello")  # 5 chars → max(1, int(5/4.0)) = 1
    msg_tokens = count_message_tokens(msg, counter)
    assert msg_tokens == 5, f"Expected 5 (4 overhead + 1 content), got {msg_tokens}"
    print(f"  Message('Hello') → {msg_tokens} tokens (4 overhead + 1 content)")

    # Tool calls: name + JSON'd arguments add tokens
    tc_msg = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="tc-1", name="search", arguments={"query": "test"})],
    )
    tc_tokens = count_message_tokens(tc_msg, counter)
    # 4 overhead + "search" (6 chars → 1) + '{"query": "test"}' (17 chars → 4) = 9
    assert tc_tokens == 9, f"Expected 9, got {tc_tokens}"
    print(f"  Message with tool_call → {tc_tokens} tokens (4 overhead + name + args)")
    print("✓ EstimateTokenCounter provides deterministic, character-based token estimation")

    # --- Section 2: Message Grouping ---
    print("\n--- Section 2: Message Grouping ---")

    messages = [
        Message(role="user", content="What's the weather?"),
        Message(
            role="assistant",
            content="Let me check.",
            tool_calls=[ToolCall(id="tc-1", name="get_weather", arguments={"city": "Paris"})],
        ),
        Message(role="tool_result", content="Sunny, 22°C", tool_call_id="tc-1"),
        Message(role="assistant", content="It's sunny and 22°C in Paris."),
    ]

    groups = default_message_grouper(messages)

    # 3 groups: [user], [assistant+tc, tool_result], [assistant]
    assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"
    assert len(groups[0]) == 1 and groups[0][0].role == "user"
    assert len(groups[1]) == 2 and groups[1][0].role == "assistant" and groups[1][1].role == "tool_result"
    assert len(groups[2]) == 1 and groups[2][0].role == "assistant"

    for i, group in enumerate(groups):
        roles = [m.role for m in group]
        print(f"  Group {i}: {roles}")
    print("✓ tool_result attaches to preceding assistant group; other messages start new groups")

    # --- Section 3: ContextUsage ---
    print("\n--- Section 3: ContextUsage ---")

    cm = ContextManager(
        context_limit=500,
        reserve_tokens=100,
        threshold=0.9,
        truncation=TruncationPolicy(),
    )

    system_prompt = "You are a helpful assistant."  # 28 chars → 7 tokens
    usage_messages = [
        Message(role="user", content="Hello there"),  # 4 + max(1,int(11/4))=2 → 6
        Message(role="assistant", content="Hi! How can I help?"),  # 4 + max(1,int(19/4))=4 → 8
    ]

    usage = cm.current_usage(system_prompt, usage_messages)

    assert isinstance(usage, ContextUsage)
    assert usage.system_tokens == 7  # int(28/4) = 7
    assert usage.message_tokens == 14  # 6 + 8
    assert usage.total_tokens == 21  # 7 + 0 (no tools) + 14
    assert usage.context_limit == 500
    # available = 500 - 100 - 21 = 379
    assert usage.available_tokens == 379
    # utilization = 21 / (500 - 100) = 0.0525
    assert abs(usage.utilization - 0.0525) < 0.001

    print(f"  system_tokens: {usage.system_tokens}")
    print(f"  message_tokens: {usage.message_tokens}")
    print(f"  total_tokens: {usage.total_tokens}")
    print(f"  available_tokens: {usage.available_tokens}")
    print(f"  utilization: {usage.utilization:.1%}")
    print("✓ current_usage() reports token breakdown without triggering management")

    # --- Section 4: Truncation ---
    print("\n--- Section 4: Truncation ---")

    # Token math:
    #   context_limit=120, reserve_tokens=20, threshold=0.9
    #   available = 120 - 20 = 100
    #   threshold_tokens = 100 * 0.9 = 90
    #   system_prompt = "Be helpful" → 10 chars → int(10/4) = 2 tokens
    #   Each message: 34 chars → int(34/4)=8 + 4 overhead = 12 tokens
    #   11 messages × 12 = 132 + 2 system = 134 > 90 → triggers truncation
    #   message_budget = 100 - 2 = 98
    #   Protected: first(12) + recent 2(24) = 36 → remaining 62 for expendable
    #   Keeps groups 4–8 from expendable (60 tokens ≤ 62), drops groups 1–3
    #   Result: 8 messages (groups 0, 4, 5, 6, 7, 8, 9, 10)

    emitter = make_emitter("truncation-demo")
    trunc_cm = ContextManager(
        context_limit=120,
        reserve_tokens=20,
        threshold=0.9,
        truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
    )

    sys_prompt = "Be helpful"  # 10 chars → 2 tokens
    trunc_messages = [Message(role="user", content=f"Message {i:02d} with padding text here.") for i in range(11)]
    sample_content = "Message 00 with padding text here."
    sample_tokens = count_message_tokens(Message(role="user", content=sample_content), counter)
    total_msg_tokens = sample_tokens * 11
    sys_tokens = counter.count_text(sys_prompt)
    total_before = total_msg_tokens + sys_tokens
    print(f"  Token math: {len(sample_content)} chars/msg → {sample_tokens} tokens/msg")
    print(f"  {11} messages × {sample_tokens} = {total_msg_tokens} + {sys_tokens} system = {total_before} total")
    print(f"  Threshold: {100 * 0.9:.0f} tokens (available={100}, threshold=0.9)")

    result = await trunc_cm.prepare(sys_prompt, trunc_messages, None, emitter)

    assert len(result) < len(trunc_messages), "Truncation should reduce message count"
    # First message preserved (preserve_first=True)
    assert result[0].content == trunc_messages[0].content, "First message should be preserved"
    # Last 2 messages preserved (preserve_recent=2)
    assert result[-1].content == trunc_messages[-1].content, "Last message should be preserved"
    assert result[-2].content == trunc_messages[-2].content, "Second-to-last should be preserved"

    # Verify event emitted
    trunc_events = [e for e in emitter.events if isinstance(e, ContextTruncationEvent)]
    assert len(trunc_events) == 1
    te = trunc_events[0]
    assert te.messages_before == 11
    assert te.messages_after == len(result)
    assert te.messages_before > te.messages_after
    assert len(te.removed_messages) > 0

    print(f"  Before: {te.messages_before} messages ({te.tokens_before} tokens)")
    print(f"  After: {te.messages_after} messages ({te.tokens_after} tokens)")
    print(f"  Removed: {len(te.removed_messages)} messages")
    print("✓ Truncation drops oldest middle groups; first and recent groups preserved")

    # --- Section 5: Summarization ---
    print("\n--- Section 5: Summarization ---")

    # Same token math as Section 4 to trigger management
    summary_text = "User asked several numbered messages with padding text."
    mock_client = MockLLMClient(responses=[make_response(summary_text)])
    emitter = make_emitter("summarization-demo")

    summ_cm = ContextManager(
        context_limit=120,
        reserve_tokens=20,
        threshold=0.9,
        summarization=SummarizationPolicy(llm_client=mock_client),
    )

    summ_messages = [Message(role="user", content=f"Message {i:02d} with padding text here.") for i in range(11)]

    result = await summ_cm.prepare(sys_prompt, summ_messages, None, emitter)

    # Find the summary message
    summary_msgs = [m for m in result if m.content and m.content.startswith("[Summary of prior conversation]")]
    assert len(summary_msgs) == 1, f"Expected 1 summary message, got {len(summary_msgs)}"
    assert summary_text in summary_msgs[0].content

    # First message preserved
    assert result[0].content == summ_messages[0].content, "First message should be preserved"
    # Recent messages preserved
    assert result[-1].content == summ_messages[-1].content, "Last message should be preserved"

    # Verify event emitted
    summ_events = [e for e in emitter.events if isinstance(e, ContextSummarizationEvent)]
    assert len(summ_events) == 1
    se = summ_events[0]
    assert se.summary_text == summary_text

    print(f"  Summary: '{se.summary_text}'")
    print(f"  Messages summarized: {se.messages_summarized}")
    print(f"  Original tokens: {se.original_tokens} → Summary tokens: {se.summary_tokens}")
    print("✓ Summarization compresses middle messages via LLM; [Summary] message injected")

    # --- Section 6: Combined Strategies + Protected Messages ---
    print("\n--- Section 6: Combined Strategies + Protected Messages ---")

    mock_client = MockLLMClient(responses=[make_response("Combined summary of conversation.")])
    emitter = make_emitter("combined-demo")

    combined_cm = ContextManager(
        context_limit=120,
        reserve_tokens=20,
        threshold=0.9,
        truncation=TruncationPolicy(preserve_first=True, preserve_recent=2),
        summarization=SummarizationPolicy(llm_client=mock_client),
    )

    # Create messages with one protected message in the expendable zone (index 3)
    combined_messages = []
    for i in range(11):
        if i == 3:
            combined_messages.append(
                Message(
                    role="user",
                    content="PROTECTED: important context here!!",
                    metadata={"protected": True},
                )
            )
        else:
            combined_messages.append(Message(role="user", content=f"Message {i:02d} with padding text here."))

    result = await combined_cm.prepare(sys_prompt, combined_messages, None, emitter)

    # Protected message must survive
    protected_in_result = [m for m in result if m.metadata and m.metadata.get("protected")]
    assert len(protected_in_result) == 1, "Protected message should survive truncation"
    assert "PROTECTED" in protected_in_result[0].content

    # First and recent messages preserved
    assert result[0].content == combined_messages[0].content
    assert result[-1].content == combined_messages[-1].content

    print(f"  Output messages: {len(result)} (from {len(combined_messages)} input)")
    print(f"  Protected message preserved: '{protected_in_result[0].content}'")
    print("✓ Combined truncation + summarization; protected messages survive in expendable zone")


if __name__ == "__main__":
    asyncio.run(main())
