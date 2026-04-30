from nanitics import (
    MockLLMClient,
    ReActAgent,
    ToolCall,
    tool,
)
from nanitics.capabilities.memory.context_provider import (
    ContextContent,
    ContextProvider,
)
from nanitics.capabilities.memory.working_memory import (
    InMemoryWorkingMemory,
    WorkingMemory,
    WorkingMemoryContributor,
    WorkingMemoryProvider,
)
from nanitics.core.agents.parsing import (
    parse_working_memory_update,
    strip_working_memory_block,
)
from nanitics.infrastructure.observability.events import (
    WorkingMemoryReadEvent,
    WorkingMemoryUpdateEvent,
)
from tests.testing_helpers import make_emitter, make_response

# ──────────────────────────────────────────────────────────
# InMemoryWorkingMemory Tests
# ──────────────────────────────────────────────────────────


class TestInMemoryWorkingMemory:
    def test_read_empty_returns_none(self) -> None:
        wm = InMemoryWorkingMemory()
        assert wm.read() is None

    def test_write_then_read(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Key Findings\n- item 1\n- item 2")
        content = wm.read()
        assert content is not None
        assert "[Working Memory]" in content
        assert "## Key Findings" in content
        assert "- item 1" in content
        assert "- item 2" in content

    def test_write_multiple_sections(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Section A\ncontent a\n\n## Section B\ncontent b")
        content = wm.read()
        assert content is not None
        assert "## Section A" in content
        assert "content a" in content
        assert "## Section B" in content
        assert "content b" in content

    def test_update_merges_sections(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Existing\nold content")
        wm.update({"New Section": "new content"})
        content = wm.read()
        assert content is not None
        assert "## Existing" in content
        assert "old content" in content
        assert "## New Section" in content
        assert "new content" in content

    def test_update_overwrites_existing_section(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Status\nold")
        wm.update({"Status": "new"})
        content = wm.read()
        assert content is not None
        assert "new" in content
        assert "old" not in content

    def test_clear_empties(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Data\nsome data")
        wm.clear()
        assert wm.read() is None

    def test_reset_empties(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Data\nsome data")
        wm.reset()
        assert wm.read() is None

    def test_is_working_memory_protocol(self) -> None:
        wm = InMemoryWorkingMemory()
        assert isinstance(wm, WorkingMemory)


# ──────────────────────────────────────────────────────────
# WorkingMemoryProvider Tests
# ──────────────────────────────────────────────────────────


class TestWorkingMemoryProvider:
    async def test_returns_content_when_memory_has_data(self) -> None:
        wm = InMemoryWorkingMemory()
        wm.write("## Status\nactive")
        provider = WorkingMemoryProvider(wm)

        result = await provider.provide([])
        assert result is not None
        assert isinstance(result, ContextContent)
        assert "Status" in result.content
        assert result.protected is True
        assert result.priority == 0

    async def test_returns_none_when_empty(self) -> None:
        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        result = await provider.provide([])
        assert result is None

    def test_is_context_provider(self) -> None:
        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)
        assert isinstance(provider, ContextProvider)


# ──────────────────────────────────────────────────────────
# WorkingMemoryContributor Tests
# ──────────────────────────────────────────────────────────


class TestWorkingMemoryContributor:
    def test_returns_section_tuple(self) -> None:
        contributor = WorkingMemoryContributor()
        result = contributor.system_prompt_section()
        assert result is not None
        name, content = result
        assert name == "working_memory"
        assert len(content) > 0

    def test_section_mentions_key_concepts(self) -> None:
        contributor = WorkingMemoryContributor()
        _, content = contributor.system_prompt_section()
        assert "working memory" in content.lower()
        assert "<working_memory>" in content
        assert "[Working Memory]" in content
        assert "replaces" in content.lower()
        assert "curated" in content.lower()


# ──────────────────────────────────────────────────────────
# parse_working_memory_update Tests
# ──────────────────────────────────────────────────────────


class TestParseWorkingMemoryUpdate:
    def test_extracts_content(self) -> None:
        text = "Some response.\n<working_memory>\n## Key\nvalue\n</working_memory>\nMore text."
        result = parse_working_memory_update(text)
        assert result is not None
        assert "## Key" in result
        assert "value" in result

    def test_no_tags_returns_none(self) -> None:
        text = "Just a normal response with no working memory tags."
        result = parse_working_memory_update(text)
        assert result is None

    def test_empty_tags_returns_none(self) -> None:
        text = "Response.\n<working_memory>\n   \n</working_memory>"
        result = parse_working_memory_update(text)
        assert result is None

    def test_handles_other_xml_like_tags_inside(self) -> None:
        text = "<working_memory>\n## Data\n<important>value</important>\n</working_memory>"
        result = parse_working_memory_update(text)
        assert result is not None
        assert "<important>value</important>" in result

    def test_only_first_block_extracted(self) -> None:
        text = "<working_memory>first</working_memory> text <working_memory>second</working_memory>"
        result = parse_working_memory_update(text)
        assert result == "first"


# ──────────────────────────────────────────────────────────
# strip_working_memory_block Tests
# ──────────────────────────────────────────────────────────


class TestStripWorkingMemoryBlock:
    def test_strips_block(self) -> None:
        text = "Before.\n<working_memory>\n## Key\nvalue\n</working_memory>\nAfter."
        result = strip_working_memory_block(text)
        assert "<working_memory>" not in result
        assert "</working_memory>" not in result
        assert "Before." in result
        assert "After." in result

    def test_no_block_returns_original(self) -> None:
        text = "Just text."
        result = strip_working_memory_block(text)
        assert result == "Just text."

    def test_only_block_returns_empty(self) -> None:
        text = "<working_memory>## Key\nvalue</working_memory>"
        result = strip_working_memory_block(text)
        assert result == ""


# ──────────────────────────────────────────────────────────
# Integration: ReActAgent with Working Memory
# ──────────────────────────────────────────────────────────


class TestReActAgentWorkingMemory:
    async def test_scratchpad_updated_from_llm_response(self) -> None:
        """LLM response with <working_memory> block updates the scratchpad."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hi"})
        responses = [
            make_response(
                content="Thinking...\n<working_memory>\n## Status\nProcessing step 1\n</working_memory>",
                tool_calls=[tool_call],
            ),
            make_response(content="Done."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        result = await agent.run("test input")

        # Working memory should have the update
        wm_content = wm.read()
        assert wm_content is not None
        assert "Processing step 1" in wm_content

        # The final output should not contain working memory tags
        assert result.output == "Done."

    async def test_working_memory_block_stripped_from_trajectory(self) -> None:
        """The <working_memory> block is stripped from assistant messages in trajectory."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hi"})
        responses = [
            make_response(
                content="Call echo.\n<working_memory>\n## Step\nStep 1\n</working_memory>",
                tool_calls=[tool_call],
            ),
            make_response(content="Final answer."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        result = await agent.run("test")

        # Check trajectory messages don't contain working_memory tags
        assistant_msgs = [m for m in result.messages if m.role == "assistant"]
        for msg in assistant_msgs:
            assert "<working_memory>" not in (msg.content or "")
            assert "</working_memory>" not in (msg.content or "")

        # First assistant message should have had the block stripped
        assert assistant_msgs[0].content == "Call echo."

    async def test_working_memory_injected_on_next_turn(self) -> None:
        """Working memory content is injected into the next LLM call."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hi"})
        responses = [
            make_response(
                content="Step 1.\n<working_memory>\n## Progress\nDone step 1\n</working_memory>",
                tool_calls=[tool_call],
            ),
            make_response(content="Final."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        await agent.run("test")

        # Second LLM call should have had working memory injected
        second_call_messages = client.calls[1]["messages"]
        wm_msgs = [m for m in second_call_messages if "[Working Memory]" in (m.content or "")]
        assert len(wm_msgs) == 1
        assert "Done step 1" in wm_msgs[0].content

    async def test_working_memory_reset_at_start(self) -> None:
        """Working memory is reset at the start of each run."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        wm.write("## Old\nold data")  # Pre-existing data

        provider = WorkingMemoryProvider(wm)
        responses = [make_response(content="Done.")]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        await agent.run("test")

        # Working memory should have been reset, so nothing injected
        sent_messages = client.calls[0]["messages"]
        wm_msgs = [m for m in sent_messages if "[Working Memory]" in (m.content or "")]
        assert len(wm_msgs) == 0

    async def test_no_working_memory_no_parsing(self) -> None:
        """Without working_memory parameter, <working_memory> tags are left intact."""
        responses = [make_response(content="Response with <working_memory>## Key\nval</working_memory> tags.")]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[],
        )

        result = await agent.run("test")

        # Tags should remain in output since no working_memory was configured
        assert "<working_memory>" in (result.output or "")

    async def test_working_memory_only_response_produces_placeholder(self) -> None:
        """LLM response with ONLY a working memory block gets a placeholder message."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        responses = [
            make_response(content="<working_memory>\n## Status\nSummarizing results\n</working_memory>"),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        result = await agent.run("test input")

        # Agent completes with the placeholder as output (WM-only = no tool calls = loop ends)
        assert result.output == "[Working memory updated]"

        # Working memory should have been updated
        wm_content = wm.read()
        assert wm_content is not None
        assert "Summarizing results" in wm_content

        # The assistant message in the trajectory should be the placeholder, not empty
        assistant_msgs = [m for m in result.messages if m.role == "assistant"]
        assert assistant_msgs[0].content == "[Working memory updated]"

    async def test_working_memory_only_response_with_output_schema(self) -> None:
        """WM-only response followed by structured output completes successfully."""
        from pydantic import BaseModel

        class Result(BaseModel):
            summary: str

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hi"})
        responses = [
            # Step 1: tool call with working memory
            make_response(
                content="Processing.\n<working_memory>\n## Step\nStep 1 done\n</working_memory>",
                tool_calls=[tool_call],
            ),
            # Step 2: only working memory (triggers break with placeholder)
            make_response(content="<working_memory>\n## Step\nAll steps done\n</working_memory>"),
            # Step 3: structured output call
            make_response(content='{"summary": "All done"}'),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
            output_schema=Result,
        )

        result = await agent.run("test")

        # Agent should complete with structured output
        assert result.parsed is not None
        assert isinstance(result.parsed, Result)
        assert result.parsed.summary == "All done"
        assert result.termination_reason == "complete"

        # No empty assistant messages in trajectory
        assistant_msgs = [m for m in result.messages if m.role == "assistant"]
        for msg in assistant_msgs:
            assert msg.content  # No empty content


# ──────────────────────────────────────────────────────────
# Event Emission Tests
# ──────────────────────────────────────────────────────────


class TestWorkingMemoryEvents:
    async def test_read_event_emitted_on_provider_injection(self) -> None:
        """WorkingMemoryReadEvent emitted when provider injects content."""
        wm = InMemoryWorkingMemory()
        wm.write("## Status\nactive")
        emitter = make_emitter()
        provider = WorkingMemoryProvider(wm, emitter=emitter)

        result = await provider.provide([])

        assert result is not None
        read_events = [e for e in emitter.events if isinstance(e, WorkingMemoryReadEvent)]
        assert len(read_events) == 1
        assert read_events[0].content is not None
        assert "active" in read_events[0].content
        assert read_events[0].token_count > 0
        assert read_events[0].trace_id == "test-trace"

    async def test_no_read_event_when_empty(self) -> None:
        """No WorkingMemoryReadEvent when memory is empty."""
        wm = InMemoryWorkingMemory()
        emitter = make_emitter()
        provider = WorkingMemoryProvider(wm, emitter=emitter)

        result = await provider.provide([])

        assert result is None
        read_events = [e for e in emitter.events if isinstance(e, WorkingMemoryReadEvent)]
        assert len(read_events) == 0

    async def test_no_read_event_without_emitter(self) -> None:
        """No error when provider has no emitter."""
        wm = InMemoryWorkingMemory()
        wm.write("## Data\nsome data")
        provider = WorkingMemoryProvider(wm)

        result = await provider.provide([])
        assert result is not None

    async def test_update_event_emitted_on_scratchpad_update(self) -> None:
        """WorkingMemoryUpdateEvent emitted when LLM response updates scratchpad."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        tool_call = ToolCall(id="tc1", name="echo", arguments={"text": "hi"})
        responses = [
            make_response(
                content="Thinking.\n<working_memory>\n## Progress\nStep 1 done\n</working_memory>",
                tool_calls=[tool_call],
            ),
            make_response(content="Done."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        await agent.run("test")

        update_events = [e for e in emitter.events if isinstance(e, WorkingMemoryUpdateEvent)]
        assert len(update_events) == 1
        assert update_events[0].previous_content is None  # First update, was empty
        assert "Step 1 done" in update_events[0].new_content
        assert update_events[0].source == "llm_output"
        assert update_events[0].trace_id == "test-trace"

    async def test_update_event_carries_previous_content(self) -> None:
        """WorkingMemoryUpdateEvent carries previous content on subsequent updates."""

        @tool(name="echo", description="Echo input")
        async def echo_tool(text: str) -> str:
            return text

        wm = InMemoryWorkingMemory()
        provider = WorkingMemoryProvider(wm)

        tc1 = ToolCall(id="tc1", name="echo", arguments={"text": "a"})
        tc2 = ToolCall(id="tc2", name="echo", arguments={"text": "b"})
        responses = [
            make_response(
                content="Step 1\n<working_memory>\n## Progress\nStep 1\n</working_memory>",
                tool_calls=[tc1],
            ),
            make_response(
                content="Step 2\n<working_memory>\n## Progress\nStep 2\n</working_memory>",
                tool_calls=[tc2],
            ),
            make_response(content="Done."),
        ]
        client = MockLLMClient(responses)
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="prompt",
            tools=[echo_tool],
            context_providers=[provider],
            working_memory=wm,
        )

        await agent.run("test")

        update_events = [e for e in emitter.events if isinstance(e, WorkingMemoryUpdateEvent)]
        assert len(update_events) == 2
        # First update: no previous content
        assert update_events[0].previous_content is None
        assert "Step 1" in update_events[0].new_content
        # Second update: previous content from first update
        assert update_events[1].previous_content is not None
        assert "Step 1" in update_events[1].previous_content
        assert "Step 2" in update_events[1].new_content
