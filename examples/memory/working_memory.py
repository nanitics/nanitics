"""Working memory: an in-run structured scratchpad that persists across agent steps.

Covers InMemoryWorkingMemory (write, read, update, clear, reset), WorkingMemoryContributor
for system prompt integration, WorkingMemoryProvider for automatic context injection, and
end-to-end agent integration where working memory evolves across a multi-step run.

Related guide: docs/guides/memory.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics import (
    ContextContent,
    InMemoryWorkingMemory,
    MockLLMClient,
    ReActAgent,
    SystemPromptContributor,
    ToolCall,
    WorkingMemoryContributor,
    WorkingMemoryProvider,
    tool,
)

# --- Tool used in Section 4 ---

FACTS = {
    "python": "Python is a general-purpose programming language created by Guido van Rossum.",
    "rust": "Rust is a systems programming language focused on safety and performance.",
}


@tool("lookup_fact", "Look up a fact about a topic")
async def lookup_fact(topic: str) -> str:
    return FACTS.get(topic.lower(), f"No fact found for '{topic}'")


async def main() -> None:
    # --- Section 1: InMemoryWorkingMemory — Write and Read ---
    print("--- Section 1: InMemoryWorkingMemory — Write and Read ---")

    memory = InMemoryWorkingMemory()

    # Empty memory returns None
    assert memory.read() is None, "Empty memory should return None"
    print("  read() on empty memory: None ✓")

    # Write content with ## Section headers
    memory.write("## Findings\n- Python is a programming language\n## Status\nIn progress")

    content = memory.read()
    assert content is not None
    assert content.startswith("[Working Memory]")
    assert "## Findings" in content
    assert "Python is a programming language" in content
    assert "## Status" in content
    assert "In progress" in content
    print("  read() after write:")
    for line in content.split("\n"):
        print(f"    {line}")

    # Content before the first ## header is silently discarded
    memory.write("some preamble text\n## Real Section\nactual content")
    content = memory.read()
    assert content is not None
    assert "some preamble" not in content, "Preamble before first ## should be discarded"
    assert "actual content" in content
    print("  Content before first ## header: discarded ✓")

    print("✓ Section 1 passed")

    # --- Section 2: Update, Clear, and Reset ---
    print("\n--- Section 2: Update, Clear, and Reset ---")

    # Start with two sections
    memory.write("## Findings\n- Python is a programming language\n## Status\nIn progress")

    # update() adds a section while preserving existing ones
    memory.update({"Next Steps": "Research Rust"})
    content = memory.read()
    assert content is not None
    assert "Findings" in content
    assert "Status" in content
    assert "Next Steps" in content
    assert "Research Rust" in content
    print("  update() adds section, preserves existing ✓")

    # update() can overwrite an existing section
    memory.update({"Status": "Complete"})
    content = memory.read()
    assert content is not None
    assert "Complete" in content
    assert "In progress" not in content, "Old status should be replaced"
    assert "Findings" in content, "Other sections should be untouched"
    print("  update() overwrites existing section ✓")

    # clear() removes all content
    memory.clear()
    assert memory.read() is None
    print("  clear(): memory is empty ✓")

    # reset() has the same effect as clear()
    # Note: ReActAgent calls reset() at the start of every run,
    # so pre-populated data does not survive into the agent loop.
    memory.write("## Temp\nsome data")
    assert memory.read() is not None
    memory.reset()
    assert memory.read() is None
    print("  reset(): same as clear() — agent calls this at run start ✓")

    print("✓ Section 2 passed")

    # --- Section 3: WorkingMemoryContributor and WorkingMemoryProvider ---
    print("\n--- Section 3: WorkingMemoryContributor and WorkingMemoryProvider ---")

    # WorkingMemoryContributor teaches the agent the <working_memory> format
    contributor = WorkingMemoryContributor()
    assert isinstance(contributor, SystemPromptContributor)
    print("  WorkingMemoryContributor is SystemPromptContributor ✓")

    section = contributor.system_prompt_section()
    assert section is not None
    key, instructions = section
    assert key == "working_memory"
    assert "<working_memory>" in instructions
    print(f"  Section key: {key!r}")
    print(f"  Instructions: {instructions[:80]}...")

    # WorkingMemoryProvider injects memory into LLM context
    memory = InMemoryWorkingMemory()
    provider = WorkingMemoryProvider(memory)

    # Empty memory → provider returns None (nothing to inject)
    result = await provider.provide([])
    assert result is None, "Provider should return None for empty memory"
    print("  Provider with empty memory: None ✓")

    # Populate memory, then provide returns ContextContent
    memory.write("## Research\nKey findings here")
    result = await provider.provide([])
    assert result is not None
    assert isinstance(result, ContextContent)
    assert result.protected is True, "Working memory should be protected from truncation"
    assert result.priority == 0, "Working memory should have highest priority"
    assert result.provider_name == "working_memory"
    assert "Key findings here" in result.content
    print(f"  Provider with content: protected={result.protected}, priority={result.priority} ✓")
    print(f"  provider_name: {result.provider_name!r} ✓")

    print("✓ Section 3 passed")

    # --- Section 4: Agent Integration — Working Memory Across Steps ---
    print("\n--- Section 4: Agent Integration — Working Memory Across Steps ---")

    # Script a 3-step research task where working memory evolves at each step.
    # Each <working_memory> block fully replaces the previous content.
    client = MockLLMClient(
        [
            # Step 1: Look up Python, record initial findings
            make_response(
                content=(
                    "Let me look up Python first.\n"
                    "<working_memory>\n"
                    "## Findings\n"
                    "- Python: general-purpose programming language\n"
                    "## Status\n"
                    "Researched 1/2 topics\n"
                    "</working_memory>"
                ),
                tool_calls=[ToolCall(id="tc-1", name="lookup_fact", arguments={"topic": "python"})],
                stop_reason="tool_use",
            ),
            # Step 2: Look up Rust, update findings (full replacement — includes Python too)
            make_response(
                content=(
                    "Now let me look up Rust.\n"
                    "<working_memory>\n"
                    "## Findings\n"
                    "- Python: general-purpose programming language\n"
                    "- Rust: systems programming language\n"
                    "## Status\n"
                    "Researched 2/2 topics\n"
                    "</working_memory>"
                ),
                tool_calls=[ToolCall(id="tc-2", name="lookup_fact", arguments={"topic": "rust"})],
                stop_reason="tool_use",
            ),
            # Step 3: Final answer (no tool calls), mark status complete
            make_response(
                content=(
                    "Both Python and Rust are popular programming languages. "
                    "Python excels at general-purpose tasks while Rust focuses on systems programming.\n"
                    "<working_memory>\n"
                    "## Findings\n"
                    "- Python: general-purpose programming language\n"
                    "- Rust: systems programming language\n"
                    "## Status\n"
                    "Complete\n"
                    "</working_memory>"
                ),
            ),
        ]
    )

    memory = InMemoryWorkingMemory()
    provider = WorkingMemoryProvider(memory)
    emitter = make_emitter()

    agent = ReActAgent(
        name="researcher",
        llm_client=client,
        emitter=emitter,
        system_prompt="You are a research assistant. Look up facts about requested topics.",
        tools=[lookup_fact],
        context_providers=[provider],
        working_memory=memory,  # Enables parsing + auto-adds WorkingMemoryContributor
    )

    result = await agent.run("Research Python and Rust")

    # The <working_memory> block is stripped from the output
    assert result.output is not None
    assert "Both Python and Rust" in result.output
    assert "<working_memory>" not in result.output, "Working memory block should be stripped"
    print(f"  Agent output: {result.output[:80]}...")
    print("  <working_memory> stripped from output ✓")

    # Agent completed in 3 steps
    assert result.total_steps == 3
    print(f"  Total steps: {result.total_steps} ✓")

    # Final memory state reflects the last <working_memory> block
    final_memory = memory.read()
    assert final_memory is not None
    assert "Python" in final_memory
    assert "Rust" in final_memory
    assert "Complete" in final_memory
    print("  Final memory state:")
    for line in final_memory.split("\n"):
        print(f"    {line}")

    print("✓ Section 4 passed")

    print("\nAll sections passed ✓")


if __name__ == "__main__":
    asyncio.run(main())
