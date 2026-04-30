"""Release-gate no-leakage invariant test.

Exercises every LLM client the SDK ships against a mock backend, collects
every event the instrumented wrapper emits, and asserts the JSON-serialized
payloads contain none of the credential-shaped strings the SDK's own
emission code would use if it were leaking.

This test guards the *SDK surface*. Adopter content (prompts, tool I/O,
custom events) flows through the same events — but scrubbing that content
is the adopter's job via :class:`~nanitics.RedactionHook`. The fields this
test inspects are the ones the SDK itself authors: event types, usage,
duration, response content, model name.

See ``docs/guides/observability.md#trace-surface-hygiene`` for the
framing.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest

from nanitics.infrastructure.llm.anthropic import AnthropicLLMClient
from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.litellm import LiteLLMClient
from nanitics.infrastructure.llm.mistral import MistralLLMClient
from nanitics.infrastructure.llm.openai import OpenAILLMClient
from nanitics.infrastructure.llm.protocol import Message, ToolSchema
from nanitics.infrastructure.observability.emitter import InMemoryEmitter

# --- Sentinels ---
#
# The SDK's own emission code would write credentials using one of these
# shapes if it leaked them. Each provider test constructs its client with
# the unique sentinel API key below; the assertion sweep then verifies
# none of the shapes appear anywhere in the serialized event payloads.

SENTINEL_API_KEY = "sk-test-SENTINEL-nanitics-leak-guard"

# Sentinel lists carry canonical casing for readability; comparisons
# lowercase both sides so any casing permutation (``Authorization``,
# ``AUTHORIZATION``, ``authorization``, ``x-API-Key``) is detected
# without enumeration.
CREDENTIAL_SHAPES_SUBSTRING: list[str] = [
    SENTINEL_API_KEY,
    "Authorization",
    "Bearer ",
    "x-api-key",
]
_CREDENTIAL_SUBSTRING_NEEDLES_LC: list[str] = [s.lower() for s in CREDENTIAL_SHAPES_SUBSTRING]
CREDENTIAL_SHAPES_DICT_KEY: list[str] = [
    "api_key",
]
_CREDENTIAL_DICT_KEY_NEEDLES_LC: list[str] = [s.lower() for s in CREDENTIAL_SHAPES_DICT_KEY]

# Minimal call inputs shared across the provider matrix. Exercise prompts,
# tool schemas, and tool-call responses so the full emission surface is
# covered.

SYSTEM_PROMPT = "You are a helpful assistant."
USER_MESSAGES: list[Message] = [Message(role="user", content="Search for pufferfish")]
TOOL_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        name="search",
        description="Search the web.",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )
]


# --- Assertion helpers ---


def _collect_strings(obj: object, path: str = "$") -> list[tuple[str, str]]:
    """Return every string value in *obj* paired with its JSON path.

    Recurses through dicts and lists. Non-string leaves are skipped. The
    path is useful when a failure reports the offending field.
    """
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_collect_strings(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out.extend(_collect_strings(item, f"{path}[{i}]"))
    elif isinstance(obj, str):
        out.append((path, obj))
    return out


def _collect_dict_keys(obj: object, path: str = "$") -> list[tuple[str, str]]:
    """Return every dict key in *obj* paired with its JSON path."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((path, k))
            out.extend(_collect_dict_keys(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out.extend(_collect_dict_keys(item, f"{path}[{i}]"))
    return out


def _assert_no_leakage(events: list[Any], provider: str) -> None:
    """Assert no emitted event payload contains any credential-shaped string.

    Scans every string value and every dict key in each event's
    ``model_dump(mode="json")`` payload. Matching is case-insensitive —
    any casing permutation (``Authorization`` / ``authorization`` /
    ``AUTHORIZATION``, ``x-api-key`` / ``X-API-Key`` / ``X-Api-Key``)
    fails the invariant. On failure, the assertion message names the
    provider, the event type, and the JSON path of the offending field.
    """
    assert events, f"{provider}: expected at least one emitted event"

    for event in events:
        payload = event.model_dump(mode="json")
        # Round-trip through JSON text to catch any odd nested shapes too.
        serialized = json.dumps(payload, default=str)
        serialized_lc = serialized.lower()

        # Shape 1 — substring appearance in any string value or the JSON
        # serialization. The latter catches credentials encoded inside
        # string-shaped blobs (e.g., a header dict serialized as a string).
        for needle_lc, needle_display in zip(
            _CREDENTIAL_SUBSTRING_NEEDLES_LC,
            CREDENTIAL_SHAPES_SUBSTRING,
            strict=True,
        ):
            if needle_lc in serialized_lc:
                # Find the specific path for a precise error message.
                offending = [(p, v) for p, v in _collect_strings(payload) if needle_lc in v.lower()]
                where = offending[0] if offending else ("<serialized-root>", serialized[:200])
                pytest.fail(
                    f"{provider}: credential-shaped substring {needle_display!r} found in "
                    f"{event.event_type} at {where[0]} = {where[1]!r}"
                )

        # Shape 2 — ``api_key`` appearing as a dict key anywhere in the
        # payload. This catches a leak that would surface as a structured
        # field rather than free text.
        for needle_lc, needle_display in zip(
            _CREDENTIAL_DICT_KEY_NEEDLES_LC,
            CREDENTIAL_SHAPES_DICT_KEY,
            strict=True,
        ):
            offending_keys = [p for p, k in _collect_dict_keys(payload) if k.lower() == needle_lc]
            if offending_keys:
                pytest.fail(
                    f"{provider}: credential-shaped dict key {needle_display!r} found in "
                    f"{event.event_type} at {offending_keys[0]}"
                )


# --- Mock response builders ---
#
# Each builder returns a canned response matching the provider's SDK
# response shape. The shape is intentionally minimal — enough to trigger
# ``LLMRequestEvent`` + ``LLMResponseEvent`` emission, including a
# tool-call response to exercise ``tool_calls`` serialization.


def _anthropic_tool_use_message() -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Searching now."

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.id = "toolu_01"
    tool_use_block.name = "search"
    tool_use_block.input = {"q": "pufferfish"}

    response = MagicMock(spec=anthropic.types.Message)
    response.content = [text_block, tool_use_block]
    response.stop_reason = "tool_use"
    response.usage = MagicMock()
    response.usage.input_tokens = 12
    response.usage.output_tokens = 7
    return response


def _anthropic_stream_ctx(message: MagicMock) -> AsyncMock:
    stream_obj = AsyncMock()
    stream_obj.get_final_message = AsyncMock(return_value=message)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=stream_obj)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _openai_tool_call_response() -> MagicMock:
    tc = MagicMock()
    tc.id = "call_01"
    tc.function = MagicMock()
    tc.function.name = "search"
    tc.function.arguments = json.dumps({"q": "pufferfish"})

    message = MagicMock()
    message.content = "Searching now."
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    usage = MagicMock()
    usage.prompt_tokens = 12
    usage.completion_tokens = 7

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _mistral_tool_call_response_json() -> dict[str, Any]:
    return {
        "id": "chat-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Searching now.",
                    "tool_calls": [
                        {
                            "id": "call_01",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"q": "pufferfish"}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19,
        },
        "model": "mistral-small-latest",
    }


def _mistral_httpx_response(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=data,
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )


# --- Per-provider exercise routines ---


async def _exercise_anthropic() -> list[Any]:
    client = AnthropicLLMClient(model="claude-test", api_key=SENTINEL_API_KEY)
    emitter = InMemoryEmitter(trace_id="trace-anthropic")
    instrumented = InstrumentedLLMClient(client, emitter=emitter)

    response_message = _anthropic_tool_use_message()
    with patch.object(
        client._client.messages,
        "stream",
        return_value=_anthropic_stream_ctx(response_message),
    ):
        await instrumented.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=USER_MESSAGES,
            tools=TOOL_SCHEMAS,
        )
    return emitter.events


async def _exercise_openai() -> list[Any]:
    client = OpenAILLMClient(model="gpt-4o-mini", api_key=SENTINEL_API_KEY)
    emitter = InMemoryEmitter(trace_id="trace-openai")
    instrumented = InstrumentedLLMClient(client, emitter=emitter)

    mock_response = _openai_tool_call_response()
    with patch.object(
        client._client.chat.completions,
        "create",
        new=AsyncMock(return_value=mock_response),
    ):
        await instrumented.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=USER_MESSAGES,
            tools=TOOL_SCHEMAS,
        )
    return emitter.events


async def _exercise_mistral() -> list[Any]:
    client = MistralLLMClient(model="mistral-small-latest", api_key=SENTINEL_API_KEY)
    emitter = InMemoryEmitter(trace_id="trace-mistral")
    instrumented = InstrumentedLLMClient(client, emitter=emitter)

    mock_response = _mistral_httpx_response(_mistral_tool_call_response_json())
    with patch.object(client._client, "post", return_value=mock_response):
        await instrumented.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=USER_MESSAGES,
            tools=TOOL_SCHEMAS,
        )
    return emitter.events


async def _exercise_litellm() -> list[Any]:
    client = LiteLLMClient(model="openai/gpt-4o-mini", api_key=SENTINEL_API_KEY)
    emitter = InMemoryEmitter(trace_id="trace-litellm")
    instrumented = InstrumentedLLMClient(client, emitter=emitter)

    mock_response = _openai_tool_call_response()
    with patch(
        "nanitics.infrastructure.llm.litellm.litellm.acompletion",
        new=AsyncMock(return_value=mock_response),
    ):
        await instrumented.generate(
            system_prompt=SYSTEM_PROMPT,
            messages=USER_MESSAGES,
            tools=TOOL_SCHEMAS,
        )
    return emitter.events


# --- The invariant test ---


class TestNoLeakageInvariant:
    """Release-gate guarantee: the SDK's own emission code never leaks credentials.

    Exercises every shipped LLM client against a mock backend, collects
    every emitted event, and asserts the serialized payloads contain none
    of the credential-shaped strings a leaking emission site would emit.
    """

    async def test_anthropic_client_does_not_leak_credentials(self) -> None:
        events = await _exercise_anthropic()
        _assert_no_leakage(events, provider="AnthropicLLMClient")

    async def test_openai_client_does_not_leak_credentials(self) -> None:
        events = await _exercise_openai()
        _assert_no_leakage(events, provider="OpenAILLMClient")

    async def test_mistral_client_does_not_leak_credentials(self) -> None:
        events = await _exercise_mistral()
        _assert_no_leakage(events, provider="MistralLLMClient")

    async def test_litellm_client_does_not_leak_credentials(self) -> None:
        events = await _exercise_litellm()
        _assert_no_leakage(events, provider="LiteLLMClient")


# --- Self-check: the invariant must be sensitive to a real leak ---
#
# A test that always passes is useless. The two tests below prove the
# assertion sweep catches both credential shapes by feeding it a
# hand-crafted synthetic event that contains the sentinel. If the
# assertion ever stops firing on these shapes, the release gate is no
# longer guarding anything.


class TestInvariantSensitivity:
    """The assertion sweep must fail on a synthetic leak."""

    async def test_substring_leak_is_detected(self) -> None:
        events = await _exercise_anthropic()
        leaking = events[0].model_copy(update={"system_prompt": f"secret {SENTINEL_API_KEY}"})
        with pytest.raises(pytest.fail.Exception, match="credential-shaped substring"):
            _assert_no_leakage([leaking], provider="synthetic")

    async def test_dict_key_leak_is_detected(self) -> None:
        events = await _exercise_anthropic()
        # ``LLMRequestEvent.messages`` is a list of dicts; inject an
        # ``api_key`` key inside one message dict.
        original = events[0]
        poisoned_messages = [*original.messages, {"role": "user", "content": "", "api_key": "leak"}]
        leaking = original.model_copy(update={"messages": poisoned_messages})
        with pytest.raises(pytest.fail.Exception, match="credential-shaped dict key"):
            _assert_no_leakage([leaking], provider="synthetic")

    async def test_lowercase_substring_leak_is_detected(self) -> None:
        """A lowercase ``authorization`` must trip the sweep — case-insensitive."""
        events = await _exercise_anthropic()
        leaking = events[0].model_copy(update={"system_prompt": "prelude authorization header follows"})
        with pytest.raises(pytest.fail.Exception, match="credential-shaped substring"):
            _assert_no_leakage([leaking], provider="synthetic")

    async def test_mixed_case_dict_key_leak_is_detected(self) -> None:
        """A ``API_KEY`` key must trip the sweep — case-insensitive."""
        events = await _exercise_anthropic()
        original = events[0]
        poisoned_messages = [*original.messages, {"role": "user", "content": "", "API_KEY": "leak"}]
        leaking = original.model_copy(update={"messages": poisoned_messages})
        with pytest.raises(pytest.fail.Exception, match="credential-shaped dict key"):
            _assert_no_leakage([leaking], provider="synthetic")
