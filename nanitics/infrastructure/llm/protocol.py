from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from nanitics.infrastructure.observability.events import Usage


class SystemPromptSection(BaseModel):
    """A segment of the system prompt with caching metadata.

    Cache-aware LLM clients use sections to apply cache control
    selectively — stable sections get cached, volatile sections don't.
    """

    model_config = ConfigDict(frozen=True)

    content: str
    cacheable: bool = True


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call, used to correlate with
            the corresponding ``tool_result`` message.
        name: Name of the tool to invoke.
        arguments: Parsed arguments to pass to the tool.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class ToolSchema(BaseModel):
    """Schema describing a tool's interface for the LLM.

    Attributes:
        name: Tool name the LLM uses to invoke it.
        description: Human-readable description — the LLM reads this to
            decide when and how to use the tool.
        parameters: JSON Schema describing the tool's input parameters.
        requires_approval: If ``True``, indicates this tool needs human
            approval before execution.
        return_direct: If ``True``, a :class:`~nanitics.strategies.agents.react.ReActAgent`
            ends the run on the first call to this tool within a tool batch
            and uses that call's :class:`~nanitics.strategies.tools.protocol.ToolResult`
            content as the run output, skipping the closing LLM turn (and,
            when ``output_schema`` is set, the structured-synthesis call).
            SDK-side only, like ``requires_approval`` and ``timeout_seconds``:
            never serialized to any LLM provider. Defaults to ``False`` so
            existing tools are unaffected.
        human_channel: If ``True``, marks this tool as a two-way human-input
            channel (a question the agent can ask a person, e.g.
            :func:`~nanitics.collaboration.tools.create_ask_human_tool`). A
            :class:`~nanitics.strategies.agents.react.ReActAgent` reads this
            flag to make its environment guidance capability-aware (prefer
            asking over assuming) and to phrase the explicit-finish nudge.
            SDK-side only, never serialized to any LLM provider. Defaults to
            ``False`` so existing tools are unaffected.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False
    timeout_seconds: float | None = None
    return_direct: bool = False
    human_channel: bool = False


class TextContentBlock(BaseModel):
    """A text content block within a multi-part message."""

    model_config = ConfigDict(frozen=True)

    type: Literal["text"] = "text"
    text: str


class ImageContentBlock(BaseModel):
    """An image content block within a multi-part message.

    Attributes:
        media_type: MIME type (e.g., ``"image/png"``, ``"image/jpeg"``).
        data: Base64-encoded image data or a URL.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["image"] = "image"
    media_type: str
    data: str


ContentBlock = TextContentBlock | ImageContentBlock


class Message(BaseModel):
    """A single message in the agent conversation.

    Attributes:
        role: Who produced this message — ``"user"``, ``"assistant"``,
            or ``"tool_result"``.
        content: Text content, a list of content blocks for multi-part
            messages (text + images), or ``None`` for assistant messages
            that only contain tool calls.
        tool_calls: Tool calls requested by the assistant.
        tool_call_id: Links a ``tool_result`` back to its ``ToolCall``.
        name: Optional sender name for multi-agent scenarios.
        metadata: Arbitrary metadata (e.g., ``{"protected": True}`` for
            context-managed messages).
    """

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "tool_result"]
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] | None = None


class LLMResponse(BaseModel):
    """Response from an LLM generation call.

    Attributes:
        content: Text response from the model, or ``None`` if the
            response contains only tool calls.
        tool_calls: Tool calls the model wants to execute. Empty list
            if none were requested.
        usage: Token usage for this call.
        model: Model identifier string.
        stop_reason: Why generation stopped — ``"end_turn"``,
            ``"tool_use"``, or ``"max_tokens"``.
        parsed: Parsed structured output when ``output_schema`` was
            provided. ``None`` otherwise.
        reasoning_text: Free-text reasoning from the model, if any.
            Populated by cache-aware / thinking-aware providers with:

            - Anthropic: the concatenation of ``thinking`` content
              blocks and any ``text`` content blocks that precede a
              ``tool_use`` block (prose that narrates the action).
            - OpenAI / Mistral / LiteLLM: ``message.content`` when
              ``tool_calls`` is non-empty (prose that precedes the
              tool call).

            Never populated for:

            - Plain final-answer responses with no tool use and no
              thinking — the full response is in ``content``; there
              is no reasoning slice.
            - Empty-string content on the tool-use path — empty is
              not reasoning; the field is ``None``.

            Consumers must treat ``reasoning_text`` as a *view* onto
            ``content``, not a move from it. ``content`` remains the
            full response body for replay, caching, and existing
            Observatory rendering.
    """

    model_config = ConfigDict(frozen=True)

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage
    model: str
    stop_reason: str
    parsed: BaseModel | None = Field(default=None, exclude=True)
    reasoning_text: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for language model clients.

    Any object implementing ``generate()`` can serve as an LLM client.
    The SDK provides ``AnthropicLLMClient`` for production use and
    ``MockLLMClient`` for testing.
    """

    @property
    def model(self) -> str | None:
        """Model identifier, or ``None`` when not known statically (e.g. routing)."""
        ...

    async def generate(
        self,
        *,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        output_schema: type[BaseModel] | None = None,
        on_token: Callable[[str], None] | None = None,
        system_prompt_sections: list[SystemPromptSection] | None = None,
    ) -> LLMResponse: ...
