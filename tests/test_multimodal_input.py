"""Tests for multimodal agent input (AgentInput type)."""

from __future__ import annotations

from nanitics import (
    ImageContentBlock,
    MockLLMClient,
    ReActAgent,
    ReasoningAgent,
    TextContentBlock,
    tool,
)
from nanitics.infrastructure import AgentStartEvent, LLMRequestEvent
from nanitics.strategies.agents.base import _input_to_text
from tests.testing_helpers import make_emitter, make_response

# ──────────────────────────────────────────────────────────
# _input_to_text helper
# ──────────────────────────────────────────────────────────


class TestInputToText:
    def test_string_passthrough(self) -> None:
        assert _input_to_text("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert _input_to_text("") == ""

    def test_mixed_content_blocks(self) -> None:
        blocks: list[TextContentBlock | ImageContentBlock] = [
            TextContentBlock(text="Describe this image:"),
            ImageContentBlock(media_type="image/png", data="base64data"),
            TextContentBlock(text="in detail"),
        ]
        assert _input_to_text(blocks) == "Describe this image: in detail"

    def test_image_only_blocks(self) -> None:
        blocks: list[TextContentBlock | ImageContentBlock] = [
            ImageContentBlock(media_type="image/png", data="base64data"),
        ]
        assert _input_to_text(blocks) == ""

    def test_empty_list(self) -> None:
        assert _input_to_text([]) == ""


# ──────────────────────────────────────────────────────────
# ReasoningAgent with multimodal input
# ──────────────────────────────────────────────────────────


class TestReasoningAgentMultimodal:
    async def test_string_input_backward_compatible(self) -> None:
        client = MockLLMClient([make_response("text output")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="Test prompt",
        )
        result = await agent.run("simple text task")

        assert result.output == "text output"
        assert result.total_steps == 1

    async def test_multimodal_input_reaches_llm(self) -> None:
        client = MockLLMClient([make_response("I see an image")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="Describe images",
        )

        blocks: list = [
            TextContentBlock(text="What is in this image?"),
            ImageContentBlock(media_type="image/png", data="base64imagedata"),
        ]
        result = await agent.run(blocks)

        assert result.output == "I see an image"
        # Verify the message sent to LLM contains the multimodal content
        assert len(result.messages) >= 1
        first_msg = result.messages[0]
        assert first_msg.role == "user"
        assert isinstance(first_msg.content, list)
        assert len(first_msg.content) == 2
        assert isinstance(first_msg.content[0], TextContentBlock)
        assert isinstance(first_msg.content[1], ImageContentBlock)

    async def test_start_event_has_text_only_task_input(self) -> None:
        client = MockLLMClient([make_response("output")])
        emitter = make_emitter()
        agent = ReasoningAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="Test",
        )

        blocks: list = [
            TextContentBlock(text="Analyze this"),
            ImageContentBlock(media_type="image/png", data="base64data"),
            TextContentBlock(text="carefully"),
        ]
        await agent.run(blocks)

        start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].task_input == "Analyze this carefully"


# ──────────────────────────────────────────────────────────
# ReActAgent with multimodal input
# ──────────────────────────────────────────────────────────


@tool(name="describe", description="Describe something")
async def describe_tool(subject: str) -> str:
    return f"Description of {subject}"


class TestReActAgentMultimodal:
    async def test_multimodal_input_initial_message(self) -> None:
        client = MockLLMClient([make_response("The image shows a cat")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="Describe images",
            tools=[describe_tool],
            max_iterations=5,
        )

        blocks: list = [
            TextContentBlock(text="What is in this image?"),
            ImageContentBlock(media_type="image/jpeg", data="base64catimage"),
        ]
        result = await agent.run(blocks)

        assert result.output == "The image shows a cat"
        # Verify initial message is multimodal
        first_msg = result.messages[0]
        assert first_msg.role == "user"
        assert isinstance(first_msg.content, list)
        assert len(first_msg.content) == 2

    async def test_llm_request_contains_multimodal_content(self) -> None:
        client = MockLLMClient([make_response("Analysis complete")])
        emitter = make_emitter()
        agent = ReActAgent(
            name="test",
            llm_client=client,
            emitter=emitter,
            system_prompt="Analyze",
            tools=[describe_tool],
            max_iterations=5,
        )

        blocks: list = [
            TextContentBlock(text="Analyze this document"),
            ImageContentBlock(media_type="image/png", data="base64doc"),
        ]
        await agent.run(blocks)

        llm_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        assert len(llm_events) >= 1
        # First LLM request should contain the multimodal user message
        first_request = llm_events[0]
        user_msg = first_request.messages[0]
        assert user_msg["role"] == "user"
        assert isinstance(user_msg["content"], list)
