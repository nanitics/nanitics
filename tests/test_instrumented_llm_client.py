import pytest

from nanitics.infrastructure.llm.instrumented import InstrumentedLLMClient
from nanitics.infrastructure.llm.mock import MockLLMClient
from nanitics.infrastructure.llm.protocol import (
    LLMClient,
    LLMResponse,
    Message,
    ToolSchema,
)
from nanitics.infrastructure.observability.emitter import InMemoryEmitter
from nanitics.infrastructure.observability.events import (
    LLMRequestEvent,
    LLMResponseEvent,
    Usage,
)


def _make_response(content: str = "Hello") -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5),
        model="test-model",
        stop_reason="end_turn",
    )


class TestInstrumentedLLMClient:
    @pytest.fixture
    def emitter(self) -> InMemoryEmitter:
        return InMemoryEmitter(trace_id="trace-1")

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        return MockLLMClient(responses=[_make_response()])

    @pytest.fixture
    def instrumented(self, mock_client: MockLLMClient, emitter: InMemoryEmitter) -> InstrumentedLLMClient:
        return InstrumentedLLMClient(mock_client, emitter=emitter, label="test_label")

    async def test_emits_request_and_response_events(
        self, instrumented: InstrumentedLLMClient, emitter: InMemoryEmitter
    ) -> None:
        messages = [Message(role="user", content="hi")]
        await instrumented.generate(system_prompt="sys", messages=messages)

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        response_events = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]

        assert len(request_events) == 1
        assert len(response_events) == 1

        req = request_events[0]
        assert req.trace_id == "trace-1"
        assert req.span_id == emitter.span_id
        assert req.system_prompt == "sys"
        assert req.model_name == ""
        assert len(req.messages) == 1

        resp = response_events[0]
        assert resp.trace_id == "trace-1"
        assert resp.model_name == "test-model"
        assert resp.content == "Hello"
        assert resp.usage.total_tokens == 15
        assert resp.duration_ms > 0

    async def test_label_appears_in_events(self, instrumented: InstrumentedLLMClient, emitter: InMemoryEmitter) -> None:
        messages = [Message(role="user", content="hi")]
        await instrumented.generate(system_prompt="sys", messages=messages)

        req = next(e for e in emitter.events if isinstance(e, LLMRequestEvent))
        resp = next(e for e in emitter.events if isinstance(e, LLMResponseEvent))

        assert req.label == "test_label"
        assert resp.label == "test_label"

    async def test_label_defaults_to_none(self, mock_client: MockLLMClient, emitter: InMemoryEmitter) -> None:
        instrumented = InstrumentedLLMClient(mock_client, emitter=emitter)
        messages = [Message(role="user", content="hi")]
        await instrumented.generate(system_prompt="sys", messages=messages)

        req = next(e for e in emitter.events if isinstance(e, LLMRequestEvent))
        resp = next(e for e in emitter.events if isinstance(e, LLMResponseEvent))

        assert req.label is None
        assert resp.label is None

    async def test_delegates_all_parameters(
        self, instrumented: InstrumentedLLMClient, mock_client: MockLLMClient
    ) -> None:
        messages = [Message(role="user", content="hi")]
        tools = [ToolSchema(name="search", description="Search", parameters={"type": "object"})]
        await instrumented.generate(system_prompt="sys", messages=messages, tools=tools)

        assert len(mock_client.calls) == 1
        call = mock_client.calls[0]
        assert call["system_prompt"] == "sys"
        assert call["messages"] == messages
        assert call["tools"] == tools

    async def test_exceptions_propagate(self, emitter: InMemoryEmitter) -> None:
        mock_client = MockLLMClient(responses=[])  # will raise on generate
        instrumented = InstrumentedLLMClient(mock_client, emitter=emitter)

        messages = [Message(role="user", content="hi")]
        with pytest.raises(ValueError, match="no more scripted responses"):
            await instrumented.generate(system_prompt="sys", messages=messages)

        # Request event is emitted before the call, but no response event
        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        response_events = [e for e in emitter.events if isinstance(e, LLMResponseEvent)]
        assert len(request_events) == 1
        assert len(response_events) == 0

    async def test_on_token_callback_passed_through(self, emitter: InMemoryEmitter) -> None:
        mock_client = MockLLMClient(responses=[_make_response("word1 word2")])
        instrumented = InstrumentedLLMClient(mock_client, emitter=emitter)

        tokens_received: list[str] = []
        messages = [Message(role="user", content="hi")]
        await instrumented.generate(
            system_prompt="sys",
            messages=messages,
            on_token=lambda t: tokens_received.append(t),
        )

        assert len(tokens_received) > 0
        assert "".join(tokens_received).strip() == "word1 word2"

    async def test_span_context_read_at_call_time(self, mock_client: MockLLMClient, emitter: InMemoryEmitter) -> None:
        # Construct without a span, then enter one before calling
        mock_client = MockLLMClient(responses=[_make_response(), _make_response()])
        instrumented = InstrumentedLLMClient(mock_client, emitter=emitter)

        messages = [Message(role="user", content="hi")]

        # Call outside any span
        await instrumented.generate(system_prompt="sys", messages=messages)
        req1 = next(e for e in emitter.events if isinstance(e, LLMRequestEvent))
        root_span = req1.span_id

        # Call inside a named span
        with emitter.span("inner"):
            await instrumented.generate(system_prompt="sys", messages=messages)

        request_events = [e for e in emitter.events if isinstance(e, LLMRequestEvent)]
        assert len(request_events) == 2

        req2 = request_events[1]
        assert req2.span_id != root_span
        assert req2.parent_span_id == root_span

    def test_satisfies_llm_client_protocol(self, instrumented: InstrumentedLLMClient) -> None:
        assert isinstance(instrumented, LLMClient)

    def test_model_delegates_to_inner_client(self, mock_client: MockLLMClient, emitter: InMemoryEmitter) -> None:
        instrumented = InstrumentedLLMClient(mock_client, emitter=emitter)
        assert instrumented.model is None  # MockLLMClient.model returns None

    def test_mock_client_model_is_none(self) -> None:
        client = MockLLMClient(responses=[])
        assert client.model is None

    async def test_model_name_from_inner_client(self, emitter: InMemoryEmitter) -> None:
        """LLMRequestEvent.model_name should reflect the inner client's model."""

        class _NamedClient(MockLLMClient):
            @property
            def model(self) -> str | None:
                return "claude-haiku-4-5-20251001"

        client = _NamedClient(responses=[_make_response()])
        instrumented = InstrumentedLLMClient(client, emitter=emitter)
        await instrumented.generate(system_prompt="sys", messages=[Message(role="user", content="hi")])

        req = next(e for e in emitter.events if isinstance(e, LLMRequestEvent))
        assert req.model_name == "claude-haiku-4-5-20251001"

    async def test_reasoning_text_survives_pass_through(self, emitter: InMemoryEmitter) -> None:
        """``LLMResponse.reasoning_text`` must survive the instrumented wrapper."""
        inner = MockLLMClient(
            responses=[_make_response(content="content-body")],
            reasoning_texts=["reasoning-from-model"],
        )
        instrumented = InstrumentedLLMClient(inner, emitter=emitter)
        result = await instrumented.generate(
            system_prompt="sys",
            messages=[Message(role="user", content="hi")],
        )
        assert result.reasoning_text == "reasoning-from-model"
        assert result.content == "content-body"
