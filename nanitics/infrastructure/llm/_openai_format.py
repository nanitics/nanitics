"""Internal OpenAI-format helpers shared by the OpenAI and LiteLLM clients.

This module is internal (underscore prefix) and not re-exported from any
``__init__.py``. It exists because LiteLLM's ``acompletion()`` returns
OpenAI-shaped ``ChatCompletion`` objects and streams OpenAI-shaped
``ChatCompletionChunk`` objects. Both clients therefore need the same
conversion helpers — rather than duplicate them, both import from here.
"""

from __future__ import annotations

import json
from typing import Any

from nanitics.infrastructure.llm._openai_profiles import ModelProfile
from nanitics.infrastructure.llm.protocol import (
    ContentBlock,
    LLMResponse,
    Message,
    TextContentBlock,
    ToolCall,
    ToolSchema,
)
from nanitics.infrastructure.observability.events import Usage

STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"


def _content_block_to_openai(block: ContentBlock) -> dict[str, Any]:
    """Convert an SDK ContentBlock to OpenAI's chat-completions format."""
    if isinstance(block, TextContentBlock):
        return {"type": "text", "text": block.text}
    # ImageContentBlock
    if block.data.startswith(("http://", "https://")):
        url = block.data
    else:
        url = f"data:{block.media_type};base64,{block.data}"
    return {"type": "image_url", "image_url": {"url": url}}


def _to_openai_messages(system_prompt: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Convert SDK messages to OpenAI chat-completions format with a leading system message."""
    result: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, list):
                content: Any = [_content_block_to_openai(b) for b in msg.content]
            else:
                content = msg.content or ""
            result.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                if msg.content:
                    entry["content"] = msg.content
            else:
                # Defensive guard: OpenAI rejects empty assistant content
                entry["content"] = msg.content or " "
            result.append(entry)

        elif msg.role == "tool_result":
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )

    return result


def _build_openai_kwargs(
    *,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    tools: list[ToolSchema] | None,
    tool_choice: Any,
    profile: ModelProfile,
) -> dict[str, Any]:
    """Assemble the provider-native kwargs dict for ``chat.completions.create``.

    The ``profile.token_param`` selects whether the completion-cap kwarg is
    ``"max_tokens"`` (chat family) or ``"max_completion_tokens"``
    (reasoning family). Centralizing this here removes the inline dict
    construction in the OpenAI client and gives the per-model variance a
    single data-driven owner.

    ``tools=None`` omits both the ``tools`` and ``tool_choice`` keys
    entirely, matching OpenAI's no-tools calling convention. When tools
    are provided they are converted to OpenAI's function-tool format via
    :func:`_to_openai_tools` and ``tool_choice`` is stamped as-is.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        profile.token_param: max_tokens,
        "messages": messages,
    }
    if tools is not None:
        kwargs["tools"] = _to_openai_tools(tools)
        kwargs["tool_choice"] = tool_choice
    return kwargs


def _to_openai_tools(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Convert SDK tool schemas to OpenAI's function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _map_stop_reason(finish_reason: str | None) -> str:
    """Map OpenAI ``finish_reason`` to the SDK's stop-reason vocabulary."""
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _parse_tool_calls(raw_tool_calls: list[Any]) -> list[ToolCall]:
    """Parse OpenAI tool-call objects (or dicts in stream-accumulator form) into SDK ``ToolCall`` objects."""
    result: list[ToolCall] = []
    for tc in raw_tool_calls:
        if isinstance(tc, dict):
            tc_id = tc["id"]
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "{}")
        else:
            tc_id = tc.id
            name = tc.function.name
            arguments = tc.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments else {}
        result.append(ToolCall(id=tc_id, name=name, arguments=arguments))
    return result


def _from_openai_response(response: Any, model: str) -> LLMResponse:
    """Convert an OpenAI ``ChatCompletion`` response to an SDK ``LLMResponse``."""
    choice = response.choices[0]
    message = choice.message

    content = message.content
    raw_tool_calls = message.tool_calls or []
    tool_calls = _parse_tool_calls(list(raw_tool_calls))

    input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
    output_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0

    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )

    # Extract reasoning_text: on the tool-use path, ``message.content``
    # carries the prose-before-tool-call. On the final-answer path,
    # ``message.content`` is the final answer and belongs only in
    # ``content``. Empty-string content with ``tool_calls`` present maps
    # to ``None`` (empty is not reasoning).
    reasoning_text: str | None = None
    if tool_calls and isinstance(content, str) and content:
        reasoning_text = content

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        model=model,
        stop_reason=_map_stop_reason(choice.finish_reason),
        reasoning_text=reasoning_text,
    )
