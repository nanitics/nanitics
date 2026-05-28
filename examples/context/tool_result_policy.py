"""ToolResultPolicy: bounding individual tool results before they enter the message list.

Demonstrates the three default implementations:

- ``ErrorOnLargeToolResult`` — the recommended default. Raises
  ``ToolResultTooLargeError`` (a ``ToolError`` subclass) which the agent's
  error handler surfaces back to the LLM as a correction prompt.
- ``TruncateToolResult`` — opt-in head/tail truncation. Sets
  ``metadata["truncated"] = True`` and ``metadata["original_tokens"]``.
- ``SummarizeToolResult`` — opt-in LLM summarization. Falls back to
  truncate semantics if the LLM call fails or the summary is still over
  budget.

The policy hooks at ``ToolRegistry.dispatch`` and is symmetric to
``ContextManager`` for messages: each layer enforces its own invariant.

Related guide: docs/guides/context-management.md
"""

import asyncio

from nanitics.context import (
    ErrorOnLargeToolResult,
    SummarizeToolResult,
    TruncateToolResult,
)
from nanitics.errors import ToolResultTooLargeError
from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import InMemoryEmitter, ToolCall, Usage


@tool(name="dump", description="Returns a deliberately large payload")
async def dump_tool() -> str:
    return "BIG" + ("payload " * 1000)


def _response(content: str | None = None, tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=5),
        model="test-model",
        stop_reason="end_turn",
    )


async def main() -> None:
    # --- Section 1: ErrorOnLargeToolResult (recommended default) ---
    print("--- Section 1: ErrorOnLargeToolResult ---")
    tc = ToolCall(id="tc1", name="dump", arguments={})
    client = MockLLMClient(
        [
            _response(content="dumping", tool_calls=[tc]),
            # After the policy raises, the error-handler reformulates the
            # tool failure into a correction prompt. The agent answers "ok".
            _response(content="ok"),
        ]
    )
    agent = ReActAgent(
        name="error-agent",
        llm_client=client,
        emitter=InMemoryEmitter(trace_id="ex-1"),
        system_prompt="be terse",
        tools=[dump_tool],
        tool_result_policy=ErrorOnLargeToolResult(max_tokens=20),
    )
    result = await agent.run("dump it")
    assert result.output == "ok"
    print("  Over-budget tool result raised ToolResultTooLargeError,")
    print("  the error handler surfaced it to the LLM, and the agent recovered.")

    # The policy is also directly usable for testing.
    from nanitics.capabilities.context.tool_result import ToolResultContext
    from nanitics.context import EstimateTokenCounter
    from nanitics.strategies.tools.protocol import ToolResult

    counter = EstimateTokenCounter()
    direct_ctx = ToolResultContext(
        tool_call=ToolCall(id="x", name="dump", arguments={}),
        token_counter=counter,
    )
    try:
        await ErrorOnLargeToolResult(max_tokens=5).apply(ToolResult(content="x" * 1000), direct_ctx)
    except ToolResultTooLargeError as e:
        print(f"  Direct apply raised: tool={e.tool_name} tokens={e.result_tokens} budget={e.max_tokens}")

    # --- Section 2: TruncateToolResult (opt-in data loss) ---
    print("\n--- Section 2: TruncateToolResult ---")
    tc = ToolCall(id="tc2", name="dump", arguments={})
    client = MockLLMClient(
        [
            _response(content="dumping", tool_calls=[tc]),
            _response(content="done"),
        ]
    )
    agent = ReActAgent(
        name="trunc-agent",
        llm_client=client,
        emitter=InMemoryEmitter(trace_id="ex-2"),
        system_prompt="be terse",
        tools=[dump_tool],
        tool_result_policy=TruncateToolResult(max_tokens=10, head_tokens=2),
    )
    result = await agent.run("dump it")
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    truncated = tool_results[0]
    assert isinstance(truncated.content, str)
    assert "[…truncated…]" in truncated.content
    assert truncated.metadata.get("truncated") is True
    print("  Tool result contains marker: '[…truncated…]' present = True")
    print(f"  metadata.truncated = {truncated.metadata['truncated']}")
    print(f"  metadata.original_tokens = {truncated.metadata['original_tokens']}")

    # --- Section 3: SummarizeToolResult (opt-in LLM compression) ---
    print("\n--- Section 3: SummarizeToolResult ---")
    # The summary LLM is independent of the agent's LLM.
    summary_llm = MockLLMClient(
        [
            LLMResponse(
                content="ten lines of payload data",
                tool_calls=[],
                usage=Usage(input_tokens=50, output_tokens=10),
                model="m",
                stop_reason="end_turn",
            )
        ]
    )
    tc = ToolCall(id="tc3", name="dump", arguments={})
    client = MockLLMClient(
        [
            _response(content="dumping", tool_calls=[tc]),
            _response(content="done"),
        ]
    )
    agent = ReActAgent(
        name="sum-agent",
        llm_client=client,
        emitter=InMemoryEmitter(trace_id="ex-3"),
        system_prompt="be terse",
        tools=[dump_tool],
        tool_result_policy=SummarizeToolResult(max_tokens=50, llm_client=summary_llm),
    )
    result = await agent.run("dump it")
    tool_results = [m for m in result.messages if m.role == "tool_result"]
    summarized = tool_results[0]
    assert summarized.content == "ten lines of payload data"
    assert summarized.metadata.get("summarized") is True
    print(f"  Tool result replaced with summary: '{summarized.content}'")
    print(f"  metadata.summarized = {summarized.metadata['summarized']}")
    print(f"  metadata.original_tokens = {summarized.metadata['original_tokens']}")

    print("\n✓ All three impls bound tool output before it enters the message list.")
    print("  Reach for ErrorOnLargeToolResult first; use Truncate/Summarize only")
    print("  when surfacing the failure to the LLM is not actionable.")


if __name__ == "__main__":
    asyncio.run(main())
