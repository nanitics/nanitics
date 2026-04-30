"""Error handling: classification, retry policies, self-correction, and graceful degradation.

Demonstrates how the SDK classifies errors into recovery categories, how RetryPolicy
configures exponential backoff for transient failures, how ErrorHandler generates
correction prompts that the LLM uses to self-correct, and what happens when the
correction budget is exhausted (graceful degradation).

Related guide: docs/guides/error-handling.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    ErrorCategory,
    ErrorHandler,
    LLMContextLengthError,
    LLMRateLimitError,
    MockLLMClient,
    ReActAgent,
    RetryPolicy,
    ToolCall,
    ToolParameterError,
    classify_error,
    tool,
)
from nanitics.infrastructure import (
    ErrorCorrectionEvent,
    ErrorDegradationEvent,
)


async def main() -> None:
    # --- Section 1: Error Classification ---
    print("--- Section 1: Error Classification ---")

    # Every error maps to one of three categories: RETRYABLE, CORRECTABLE, or FATAL.
    # The category determines the recovery strategy.

    # Rate limits are transient — retry with backoff
    rate_limit = LLMRateLimitError("Rate limit exceeded", retry_after=5.0)
    assert classify_error(rate_limit) == ErrorCategory.RETRYABLE

    # Bad tool parameters are the agent's mistake — send a correction prompt
    bad_params = ToolParameterError(
        "Invalid date format",
        tool_name="search",
        parameter_name="date",
        reason="Expected YYYY-MM-DD format",
    )
    assert classify_error(bad_params) == ErrorCategory.CORRECTABLE

    # Context overflow is unrecoverable — propagate or degrade
    context_overflow = LLMContextLengthError(
        "Input exceeds context window",
        token_count=250_000,
        token_limit=200_000,
    )
    assert classify_error(context_overflow) == ErrorCategory.FATAL

    print(f"  LLMRateLimitError    → {classify_error(rate_limit).value}")
    print(f"  ToolParameterError   → {classify_error(bad_params).value}")
    print(f"  LLMContextLengthError → {classify_error(context_overflow).value}")
    print("✓ classify_error maps each error type to a recovery category")

    # --- Section 2: Self-Correction — Tool Error Recovery ---
    print("\n--- Section 2: Self-Correction — Tool Error Recovery ---")

    # A tool that fails on the first call but succeeds on the second.
    # ValueError raised by the tool is wrapped as ToolExecutionError by the registry.
    call_count = 0

    @tool("search", "Search for articles on a topic")
    async def flaky_search(query: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Connection timeout")
        return f"Found 3 articles about '{query}'"

    client = MockLLMClient(
        responses=[
            # Step 1: LLM calls the search tool → tool fails → correction prompt injected
            make_response(
                "Let me search for that.",
                tool_calls=[ToolCall(id="tc-1", name="search", arguments={"query": "climate change"})],
                stop_reason="tool_use",
            ),
            # Step 2: LLM sees correction prompt, retries the tool → tool succeeds
            make_response(
                "Let me try the search again.",
                tool_calls=[ToolCall(id="tc-2", name="search", arguments={"query": "climate change"})],
                stop_reason="tool_use",
            ),
            # Step 3: LLM produces final answer using the search results
            make_response("Based on my search, I found 3 articles about climate change."),
        ]
    )
    emitter = make_emitter("error-s2")

    agent = ReActAgent(
        name="self-correcting-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful research assistant.",
        tools=[flaky_search],
        error_handler=ErrorHandler.default(),
    )

    result = await agent.run("Find articles about climate change")

    # Verify the agent recovered from the tool failure
    assert result.output == "Based on my search, I found 3 articles about climate change."
    assert result.total_steps == 3
    assert result.termination_reason == "complete"

    # The correction prompt replaced the failed tool result in the conversation
    messages = result.messages
    # messages[0] = user, [1] = assistant (tool_call), [2] = tool_result (correction),
    # [3] = assistant (retry tool_call), [4] = tool_result (success), [5] = assistant (final)
    correction_msg = messages[2]
    assert correction_msg.role == "tool_result"
    assert "failed during execution" in correction_msg.content
    assert "Attempt 1/3" in correction_msg.content

    success_msg = messages[4]
    assert success_msg.role == "tool_result"
    assert "Found 3 articles" in success_msg.content

    # An ErrorCorrectionEvent was emitted
    correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
    assert len(correction_events) == 1
    assert correction_events[0].error_type == "ToolExecutionError"
    assert correction_events[0].attempt == 1
    assert "failed during execution" in correction_events[0].correction_prompt

    print("  Conversation flow:")
    for msg in messages:
        if msg.tool_calls:
            print(f"    {msg.role}: {msg.content} → [{msg.tool_calls[0].name}]")
        elif msg.tool_call_id:
            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"    {msg.role}: {preview}")
        else:
            print(f"    {msg.role}: {msg.content}")
    print(
        f"  ErrorCorrectionEvent: error_type={correction_events[0].error_type}, attempt={correction_events[0].attempt}"
    )
    print("✓ Tool failure → correction prompt → LLM retries → success")

    # --- Section 3: Graceful Degradation — Budget Exhaustion ---
    print("\n--- Section 3: Graceful Degradation — Budget Exhaustion ---")

    # A tool that always fails — simulates a persistently offline service.
    @tool("database", "Query the user database")
    async def broken_database(query: str) -> str:
        raise ValueError("Database is offline")

    client = MockLLMClient(
        responses=[
            # Step 1: LLM calls the database tool → fails → correction prompt (budget: 1/1 used)
            make_response(
                "Let me query the database.",
                tool_calls=[ToolCall(id="tc-d1", name="database", arguments={"query": "select users"})],
                stop_reason="tool_use",
            ),
            # Step 2: LLM retries → fails again → budget exhausted → degradation message
            make_response(
                "Let me try the database again.",
                tool_calls=[ToolCall(id="tc-d2", name="database", arguments={"query": "select users"})],
                stop_reason="tool_use",
            ),
            # Step 3: LLM acknowledges the limitation and provides a partial answer
            make_response(
                "I was unable to query the database as it is currently offline. "
                "I cannot retrieve user data at this time."
            ),
        ]
    )
    emitter = make_emitter("error-s3")

    agent = ReActAgent(
        name="degrading-agent",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a helpful data assistant.",
        tools=[broken_database],
        error_handler=ErrorHandler(max_total_corrections=1),
    )

    result = await agent.run("How many users signed up last month?")

    # The agent completed with a partial answer
    assert "unable to query the database" in result.output
    assert result.total_steps == 3
    assert result.termination_reason == "complete"

    # The degradation message is visible in the conversation
    messages = result.messages
    # messages[0] = user, [1] = assistant (tool_call), [2] = tool_result (correction),
    # [3] = assistant (retry tool_call), [4] = tool_result (degradation), [5] = assistant (partial)
    degradation_msg = messages[4]
    assert degradation_msg.role == "tool_result"
    assert "failed repeatedly" in degradation_msg.content
    assert "clearly state what you could not accomplish" in degradation_msg.content

    # Both ErrorCorrectionEvent and ErrorDegradationEvent were emitted
    correction_events = [e for e in emitter.events if isinstance(e, ErrorCorrectionEvent)]
    degradation_events = [e for e in emitter.events if isinstance(e, ErrorDegradationEvent)]
    assert len(correction_events) == 1, f"Expected 1 correction event, got: {len(correction_events)}"
    assert len(degradation_events) == 1, f"Expected 1 degradation event, got: {len(degradation_events)}"
    assert "failed repeatedly" in degradation_events[0].degradation_message

    print("  Conversation flow:")
    for msg in messages:
        if msg.tool_calls:
            print(f"    {msg.role}: {msg.content} → [{msg.tool_calls[0].name}]")
        elif msg.tool_call_id:
            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"    {msg.role}: {preview}")
        else:
            print(f"    {msg.role}: {msg.content}")
    print(f"  ErrorCorrectionEvent: attempt={correction_events[0].attempt}")
    print(f"  ErrorDegradationEvent: {degradation_events[0].degradation_message[:60]}...")
    print("✓ Correction budget exhausted → degradation message → LLM provides partial answer")

    # --- Section 4: RetryPolicy and ErrorHandler Configuration ---
    print("\n--- Section 4: RetryPolicy and ErrorHandler Configuration ---")

    # RetryPolicy controls exponential backoff for RETRYABLE errors (rate limits,
    # server errors). It's passed to ErrorHandler and used internally — you don't
    # call retry_with_backoff directly.

    # Default policy: 5 attempts, 2s base delay, 2x exponential growth, max 60s, with jitter
    default_policy = RetryPolicy()
    assert default_policy.max_attempts == 5
    assert default_policy.base_delay == 2.0
    assert default_policy.max_delay == 60.0
    assert default_policy.exponential_base == 2.0
    assert default_policy.jitter is True
    print(
        f"  Default: {default_policy.max_attempts} attempts, {default_policy.base_delay}s base, "
        f"{default_policy.exponential_base}x growth, max {default_policy.max_delay}s, jitter={default_policy.jitter}"
    )

    # Custom policy: more aggressive retries for a resilient service
    aggressive_policy = RetryPolicy(
        max_attempts=5,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        jitter=True,
    )
    assert aggressive_policy.max_attempts == 5
    assert aggressive_policy.base_delay == 0.5
    print(
        f"  Aggressive: {aggressive_policy.max_attempts} attempts, "
        f"{aggressive_policy.base_delay}s base, {aggressive_policy.exponential_base}x growth"
    )

    # RetryPolicy is frozen (immutable) and validates its parameters
    try:
        RetryPolicy(max_attempts=0)  # Must be >= 1
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  Validation: max_attempts=0 rejected ✓")

    # ErrorHandler.default() uses standard retry + correction settings
    _ = ErrorHandler.default()
    print("  ErrorHandler.default(): standard retry + correction")

    # ErrorHandler.fail_fast() disables all recovery — errors propagate immediately
    _ = ErrorHandler.fail_fast()
    print("  ErrorHandler.fail_fast(): no retries, no corrections")

    # Custom ErrorHandler: pass a RetryPolicy + tune correction budgets
    _ = ErrorHandler(
        retry_policy=aggressive_policy,
        max_corrections=2,  # max corrections per individual tool error
        max_total_corrections=4,  # max corrections across all tools in a run
    )
    print("  Custom: aggressive retries + 2 per-tool / 4 total corrections")

    print("✓ RetryPolicy configures backoff; ErrorHandler ties retry + correction + degradation together")


if __name__ == "__main__":
    asyncio.run(main())
