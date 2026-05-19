"""Multimodal Input: passing text + images to agents.

Demonstrates how to use Agent.run() with multimodal input. Agents accept
either a plain string or a list of ContentBlock objects containing text
and image blocks. This enables vision tasks like image analysis, document
extraction, and diagram understanding.

Related guide: docs/guides/agent-types.md
"""

import asyncio
import base64

from examples.helpers import make_emitter, make_response
from nanitics.infrastructure import (
    AgentStartEvent,
    LLMRequestEvent,
    MockLLMClient,
)
from nanitics.strategies import ReasoningAgent
from nanitics.tracing import (
    ImageContentBlock,
    TextContentBlock,
)


async def main() -> None:
    # --- Section 1: Plain String Input (existing behavior) ---
    print("--- Section 1: Plain String Input ---")

    client = MockLLMClient(
        responses=[
            make_response("The sentiment is positive — the reviewer loves the product."),
        ]
    )
    emitter = make_emitter("multimodal-s1")

    agent = ReasoningAgent(
        name="classifier",
        llm_client=client,
        emitter=emitter,
        system_prompt="Classify the sentiment of the given text.",
    )

    # Plain string — works exactly as before
    result = await agent.run("I love this product!")

    assert result.output is not None
    assert "positive" in result.output.lower()
    print(f"Output: {result.output}")

    # --- Section 2: Multimodal Input (text + image) ---
    print("\n--- Section 2: Multimodal Input ---")

    client = MockLLMClient(
        responses=[
            make_response("The image shows an invoice from Acme Corp for €1,250.00 dated 2025-03-01."),
        ]
    )
    emitter = make_emitter("multimodal-s2")

    agent = ReasoningAgent(
        name="extractor",
        llm_client=client,
        emitter=emitter,
        system_prompt="Extract invoice data from the provided content.",
    )

    # Create a small placeholder image (1x1 white PNG) for demonstration
    # In production, this would be real image bytes from a file or API
    placeholder_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")

    # Pass multimodal input: text instruction + image
    multimodal_input = [
        TextContentBlock(text="Extract all invoice data from this scanned document."),
        ImageContentBlock(media_type="image/png", data=placeholder_png),
    ]
    result = await agent.run(multimodal_input)

    assert result.output is not None
    print(f"Output: {result.output}")

    # Verify the LLM received multimodal content
    llm_requests = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
    assert len(llm_requests) == 1
    user_msg = llm_requests[0].messages[0]
    assert isinstance(user_msg["content"], list)
    assert len(user_msg["content"]) == 2  # text block + image block
    print(f"LLM received {len(user_msg['content'])} content blocks (text + image)")

    # Verify AgentStartEvent has text-only task_input (no base64 image data)
    start_events = [e for e in emitter.events if isinstance(e, AgentStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].task_input == "Extract all invoice data from this scanned document."
    assert "PNG" not in start_events[0].task_input  # No image data in events
    print(f"Start event task_input: {start_events[0].task_input}")

    # --- Section 3: Multiple Images with Context ---
    print("\n--- Section 3: Multiple Images with Context ---")

    client = MockLLMClient(
        responses=[
            make_response("Page 1 shows the invoice header. Page 2 contains line items totaling €3,400."),
        ]
    )
    emitter = make_emitter("multimodal-s3")

    agent = ReasoningAgent(
        name="multi-page-extractor",
        llm_client=client,
        emitter=emitter,
        system_prompt="Analyze multi-page documents.",
    )

    # Multiple content blocks: instruction, two images, supplementary text
    multi_page_input = [
        TextContentBlock(text="Analyze this 2-page invoice:"),
        ImageContentBlock(media_type="image/png", data=placeholder_png),
        ImageContentBlock(media_type="image/png", data=placeholder_png),
        TextContentBlock(text="OCR extracted partial text: 'Acme Corp, Invoice #12345'"),
    ]
    result = await agent.run(multi_page_input)

    assert result.output is not None
    print(f"Output: {result.output}")

    # Verify all content blocks were passed through
    llm_requests = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
    user_msg = llm_requests[0].messages[0]
    assert isinstance(user_msg["content"], list)
    assert len(user_msg["content"]) == 4  # 2 text blocks + 2 image blocks
    print(f"LLM received {len(user_msg['content'])} content blocks")

    print("\n✓ All sections passed")


if __name__ == "__main__":
    asyncio.run(main())
