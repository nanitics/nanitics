import pytest
from pydantic import BaseModel, ValidationError

from nanitics.errors import LLMSchemaViolationError
from nanitics.infrastructure import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
    ToolSchema,
)
from nanitics.tracing import (
    Message,
    ToolCall,
    Usage,
)

# --- Data Model Tests ---


class TestToolCall:
    def test_construction(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "hello"})
        assert tc.id == "tc-1"
        assert tc.name == "search"
        assert tc.arguments == {"query": "hello"}

    def test_frozen(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"query": "hello"})
        with pytest.raises(ValidationError):
            tc.name = "other"


class TestToolSchema:
    def test_construction(self) -> None:
        ts = ToolSchema(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert ts.name == "search"
        assert ts.description == "Search the web"


class TestMessage:
    def test_user_message(self) -> None:
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_assistant_message_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msg = Message(role="assistant", tool_calls=[tc])
        assert msg.tool_calls == [tc]

    def test_tool_result_message(self) -> None:
        msg = Message(role="tool_result", content="result data", tool_call_id="tc-1")
        assert msg.role == "tool_result"
        assert msg.tool_call_id == "tc-1"


class TestLLMResponse:
    def _usage(self) -> Usage:
        return Usage(input_tokens=10, output_tokens=20)

    def test_construction(self) -> None:
        resp = LLMResponse(
            content="Hello",
            usage=self._usage(),
            model="test-model",
            stop_reason="end_turn",
        )
        assert resp.content == "Hello"
        assert resp.tool_calls == []
        assert resp.model == "test-model"
        assert resp.stop_reason == "end_turn"
        assert resp.parsed is None

    def test_frozen(self) -> None:
        resp = LLMResponse(
            content="Hello",
            usage=self._usage(),
            model="test-model",
            stop_reason="end_turn",
        )
        with pytest.raises(ValidationError):
            resp.content = "other"

    def test_parsed_excluded_from_model_dump(self) -> None:
        resp = LLMResponse(
            content="Hello",
            usage=self._usage(),
            model="test-model",
            stop_reason="end_turn",
            parsed={"some": "value"},
        )
        dumped = resp.model_dump()
        assert "parsed" not in dumped

    def test_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        resp = LLMResponse(
            tool_calls=[tc],
            usage=self._usage(),
            model="test-model",
            stop_reason="tool_use",
        )
        assert len(resp.tool_calls) == 1
        assert resp.content is None

    def test_reasoning_text_defaults_none(self) -> None:
        resp = LLMResponse(
            content="Hello",
            usage=self._usage(),
            model="test-model",
            stop_reason="end_turn",
        )
        assert resp.reasoning_text is None

    def test_reasoning_text_round_trip(self) -> None:
        resp = LLMResponse(
            content="final answer",
            reasoning_text="thinking about it",
            usage=self._usage(),
            model="test-model",
            stop_reason="end_turn",
        )
        dumped = resp.model_dump()
        assert dumped["reasoning_text"] == "thinking about it"
        restored = LLMResponse.model_validate(dumped)
        assert restored.reasoning_text == "thinking about it"
        assert restored.content == "final answer"


# --- MockLLMClient Tests ---


class TestMockLLMClient:
    def _make_response(self, content: str = "Hello") -> LLMResponse:
        return LLMResponse(
            content=content,
            usage=Usage(input_tokens=10, output_tokens=20),
            model="mock-model",
            stop_reason="end_turn",
        )

    async def test_returns_responses_in_sequence(self) -> None:
        r1 = self._make_response("first")
        r2 = self._make_response("second")
        client = MockLLMClient(responses=[r1, r2])

        result1 = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        result2 = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        assert result1.content == "first"
        assert result2.content == "second"

    async def test_raises_on_exhaustion(self) -> None:
        client = MockLLMClient(responses=[self._make_response()])
        await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        with pytest.raises(ValueError, match="no more scripted responses"):
            await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])

    async def test_tracks_call_history(self) -> None:
        client = MockLLMClient(responses=[self._make_response()])
        tools = [ToolSchema(name="t", description="d", parameters={})]
        messages = [Message(role="user", content="hi")]

        await client.generate(system_prompt="sys", messages=messages, tools=tools)

        assert len(client.calls) == 1
        assert client.calls[0]["system_prompt"] == "sys"
        assert client.calls[0]["messages"] == messages
        assert client.calls[0]["tools"] == tools
        assert client.calls[0]["output_schema"] is None

    async def test_output_schema_parsing_success(self) -> None:
        class MyOutput(BaseModel):
            answer: str

        response = LLMResponse(
            content='{"answer": "42"}',
            usage=Usage(input_tokens=10, output_tokens=20),
            model="mock-model",
            stop_reason="end_turn",
        )
        client = MockLLMClient(responses=[response])

        result = await client.generate(
            system_prompt="test",
            messages=[Message(role="user", content="hi")],
            output_schema=MyOutput,
        )
        assert isinstance(result.parsed, MyOutput)
        assert result.parsed.answer == "42"

    async def test_output_schema_parsing_failure(self) -> None:
        class MyOutput(BaseModel):
            answer: str

        response = LLMResponse(
            content="not valid json",
            usage=Usage(input_tokens=10, output_tokens=20),
            model="mock-model",
            stop_reason="end_turn",
        )
        client = MockLLMClient(responses=[response])

        with pytest.raises(LLMSchemaViolationError, match="does not match MyOutput schema"):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                output_schema=MyOutput,
            )

    async def test_rejects_output_schema_with_tools(self) -> None:
        class MyOutput(BaseModel):
            answer: str

        client = MockLLMClient(responses=[self._make_response()])
        tools = [ToolSchema(name="t", description="d", parameters={})]

        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.generate(
                system_prompt="test",
                messages=[Message(role="user", content="hi")],
                tools=tools,
                output_schema=MyOutput,
            )

    async def test_callable_response(self) -> None:
        """Callable responses receive messages and return dynamic LLMResponse."""

        def dynamic_response(messages: list[Message]) -> LLMResponse:
            # Build response based on conversation content
            user_content = messages[-1].content or ""
            return LLMResponse(
                content=f"Echo: {user_content}",
                usage=Usage(input_tokens=10, output_tokens=20),
                model="mock-model",
                stop_reason="end_turn",
            )

        static = self._make_response("static")
        client = MockLLMClient(responses=[static, dynamic_response])

        # First call returns static response
        r1 = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        assert r1.content == "static"

        # Second call invokes the callable with messages
        r2 = await client.generate(system_prompt="test", messages=[Message(role="user", content="hello world")])
        assert r2.content == "Echo: hello world"

    async def test_protocol_conformance(self) -> None:
        client = MockLLMClient(responses=[])
        assert isinstance(client, LLMClient)

    async def test_reasoning_texts_overrides_static_response(self) -> None:
        """``reasoning_texts`` rebuilds static responses with the scripted value."""
        r1 = self._make_response("first")
        r2 = self._make_response("second")
        client = MockLLMClient(responses=[r1, r2], reasoning_texts=["thinking a", None])

        result1 = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        result2 = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        assert result1.reasoning_text == "thinking a"
        assert result1.content == "first"
        assert result2.reasoning_text is None
        assert result2.content == "second"

    async def test_reasoning_texts_length_mismatch_raises(self) -> None:
        """Mismatched-length ``reasoning_texts`` raises ``ValueError`` at construction."""
        r1 = self._make_response("first")
        with pytest.raises(ValueError, match="reasoning_texts must have the same length"):
            MockLLMClient(responses=[r1], reasoning_texts=["a", "b"])

    async def test_reasoning_texts_does_not_override_callable(self) -> None:
        """Callables own their own reasoning_text; the mock does not override them."""

        def dynamic_response(messages: list[Message]) -> LLMResponse:
            return LLMResponse(
                content="dynamic",
                reasoning_text="callable-reasoning",
                usage=Usage(input_tokens=1, output_tokens=1),
                model="mock-model",
                stop_reason="end_turn",
            )

        client = MockLLMClient(
            responses=[dynamic_response],
            reasoning_texts=["should-be-ignored"],
        )
        result = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        assert result.reasoning_text == "callable-reasoning"

    async def test_reasoning_texts_callable_returning_no_reasoning(self) -> None:
        """Callable returning a response with ``reasoning_text=None`` is not overridden
        even when ``reasoning_texts[i]`` is also ``None``."""

        def dynamic_response(messages: list[Message]) -> LLMResponse:
            return LLMResponse(
                content="dynamic",
                usage=Usage(input_tokens=1, output_tokens=1),
                model="mock-model",
                stop_reason="end_turn",
            )

        client = MockLLMClient(responses=[dynamic_response], reasoning_texts=[None])
        result = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        assert result.reasoning_text is None

    async def test_reasoning_texts_none_entry_leaves_response_untouched(self) -> None:
        """A ``None`` entry in ``reasoning_texts`` preserves the response's own
        ``reasoning_text`` field as-is."""
        r1 = LLMResponse(
            content="preset",
            reasoning_text="pre-scripted-on-response",
            usage=Usage(input_tokens=1, output_tokens=1),
            model="mock-model",
            stop_reason="end_turn",
        )
        client = MockLLMClient(responses=[r1], reasoning_texts=[None])
        result = await client.generate(system_prompt="test", messages=[Message(role="user", content="hi")])
        assert result.reasoning_text == "pre-scripted-on-response"
