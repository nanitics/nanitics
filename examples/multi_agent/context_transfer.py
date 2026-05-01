"""Context transfer: controlling what flows between agents.

Demonstrates the four built-in ContextTransferStrategy implementations:
RawOutputTransfer (final output only), TrajectoryTransfer (full message history),
SummaryTransfer (LLM-compressed summary), and CustomTransfer (user-defined extraction).
Each strategy takes an AgentResult and produces a string — shown here side-by-side
on the same result so you can compare the output.

Related guide: docs/guides/multi-agent-foundations.md
"""

import asyncio

from examples.helpers import make_response
from nanitics import (
    AgentResult,
    CustomTransfer,
    Message,
    MockLLMClient,
    RawOutputTransfer,
    SummaryTransfer,
    ToolCall,
    TrajectoryTransfer,
    Usage,
)


async def main() -> None:
    # --- Section 1: Build an AgentResult ---
    # All strategies operate on an AgentResult. We construct one manually
    # to keep the focus on the transfer strategies, not agent execution.

    messages = [
        Message(role="user", content="What is the capital of France?"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    name="lookup_country",
                    arguments={"country": "France"},
                )
            ],
        ),
        Message(
            role="tool_result",
            content="France — capital: Paris, population: 67 million",
            tool_call_id="tc_1",
        ),
        Message(
            role="assistant",
            content="The capital of France is Paris.",
        ),
    ]

    result = AgentResult(
        output="The capital of France is Paris.",
        total_steps=2,
        termination_reason="complete",
        messages=messages,
        usage=Usage(input_tokens=50, output_tokens=20),
    )

    assert result.output == "The capital of France is Paris."
    assert len(result.messages) == 4
    print("--- Section 1: AgentResult constructed ---")
    print(f"  Output: {result.output}")
    print(f"  Messages: {len(result.messages)}")
    print(f"  Steps: {result.total_steps}")
    print("✓ AgentResult ready for transfer strategy comparison")

    # --- Section 2: RawOutputTransfer ---
    # Extracts only result.output. Cheapest — no processing, no LLM calls.
    # Loses all reasoning trajectory; only the final answer transfers.

    raw = RawOutputTransfer()
    raw_text = await raw.extract(result)

    assert raw_text == "The capital of France is Paris."
    assert raw_text == result.output
    print("\n--- Section 2: RawOutputTransfer ---")
    print(f"  Extracted: {raw_text}")
    print("✓ RawOutputTransfer returns only the final output")

    # --- Section 3: TrajectoryTransfer ---
    # Formats the full message history including tool calls.
    # Most faithful — the receiving agent sees everything that happened.
    # Can be large for long-running agents.

    trajectory = TrajectoryTransfer()
    trajectory_text = await trajectory.extract(result)

    # The trajectory includes all messages and tool calls
    assert "USER: What is the capital of France?" in trajectory_text
    assert "TOOL_RESULT: France" in trajectory_text
    assert "lookup_country" in trajectory_text
    assert "ASSISTANT: The capital of France is Paris." in trajectory_text
    # Trajectory is always longer than raw output
    assert len(trajectory_text) > len(raw_text)
    print("\n--- Section 3: TrajectoryTransfer ---")
    print(f"  Extracted ({len(trajectory_text)} chars):")
    for line in trajectory_text.split("\n"):
        print(f"    {line}")
    print("✓ TrajectoryTransfer includes full message history with tool calls")

    # --- Section 4: SummaryTransfer ---
    # Uses an LLM to compress the trajectory into a concise summary.
    # Costs one additional LLM call per transfer. Good when the trajectory
    # is too large but the receiving agent needs more than just the output.

    summary_content = "The agent looked up France's capital using the lookup_country tool and determined it is Paris."
    summarizer_client = MockLLMClient(responses=[make_response(summary_content)])
    summary = SummaryTransfer(llm_client=summarizer_client)
    summary_text = await summary.extract(result)

    assert summary_text == summary_content
    # The summarizer received the formatted trajectory as input
    assert len(summarizer_client.calls) == 1
    summarizer_input = summarizer_client.calls[0]["messages"][0].content
    assert "lookup_country" in summarizer_input
    assert "Paris" in summarizer_input
    print("\n--- Section 4: SummaryTransfer ---")
    print(f"  Extracted: {summary_text}")
    print(f"  Summarizer received {len(summarizer_input)} chars of trajectory")
    print("✓ SummaryTransfer compresses trajectory via LLM call")

    # --- Section 5: CustomTransfer ---
    # User-defined extraction function for full control. Useful when you
    # need structured extraction — e.g., pulling specific fields, parsing
    # JSON from output, or combining output with metadata.

    def extract_report(r: AgentResult) -> str:
        tool_names = []
        for msg in r.messages:
            if msg.tool_calls:
                tool_names.extend(tc.name for tc in msg.tool_calls)
        return (
            f"Answer: {r.output}\nSteps taken: {r.total_steps}\n"
            f"Tools used: {', '.join(tool_names)}\nTokens: {r.usage.total_tokens}"
        )

    custom = CustomTransfer(fn=extract_report)
    custom_text = await custom.extract(result)

    assert "Answer: The capital of France is Paris." in custom_text
    assert "Steps taken: 2" in custom_text
    assert "Tools used: lookup_country" in custom_text
    assert "Tokens: 70" in custom_text
    print("\n--- Section 5: CustomTransfer ---")
    print("  Extracted:")
    for line in custom_text.split("\n"):
        print(f"    {line}")
    print("✓ CustomTransfer applies user-defined extraction logic")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
