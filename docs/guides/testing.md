# Testing

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Tests for agentic systems answer a narrow question: **is the code correct?** They use mocks for the non-deterministic parts (LLMs, embeddings, sandboxes, HTTP) so the same inputs always produce the same outputs. A separate concern — "do the prompts produce good outputs?" — belongs in the validation suite, which runs against real LLMs and is governed by evaluators rather than assertions. Keep the two separate: `tests/` is for the code you wrote; `validation/` is for the prompts and agent designs you're iterating on. Nanitics ships a validation suite under [`validation/`](../../validation/) that demonstrates the pattern on real services. See [Evaluation](evaluation.md) for runtime evaluators, which are the runtime-guarding counterpart and a common source of naming confusion.

## Testing with MockLLMClient

`MockLLMClient` is the primitive for every agent test. It returns scripted `LLMResponse` objects in order and records every `generate()` call on its `calls` attribute for assertions.

### Scripted responses

Use a static list when the conversation shape is fixed — the agent will call the LLM N times and you know what each response should look like.

<!-- verify: skip — illustrative, top-level await requires pytest async context -->
```python
from nanitics.infrastructure import MockLLMClient
from nanitics.strategies import ReActAgent
from nanitics.tracing import InMemoryEmitter
from examples.helpers import make_response

llm = MockLLMClient(responses=[
    make_response("The answer is 42."),
])
agent = ReActAgent(name="test", llm_client=llm, emitter=InMemoryEmitter("t"))
result = await agent.run("What is the answer?")
assert result.output == "The answer is 42."
```

### Callable responses (dependent outputs)

When a later response must reference a value produced by an earlier tool call (a run ID, a database row, the prior message), pass a callable `(messages) -> LLMResponse` in the responses list. The callable is invoked with the full message list at the moment the LLM would be called, so the response can close over the tool result.

<!-- verify: skip — illustrative, uses `...` placeholder for tool_calls -->
```python
def second_response(messages):
    last_tool_result = messages[-1].content
    return make_response(f"Result was: {last_tool_result}")

llm = MockLLMClient(responses=[
    make_response("", tool_calls=[...]),  # first turn: call a tool
    second_response,                       # second turn: reference the result
])
```

### Asserting on calls

`llm.calls` captures `system_prompt`, `messages`, `tools`, and `output_schema` per invocation. Assert on what the agent *asked* the LLM to do — tool availability, schema structure, message ordering.

<!-- verify: skip — assertion snippet, depends on the preceding agent run -->
```python
assert len(llm.calls) == 2
assert "get_weather" in [t.name for t in llm.calls[0]["tools"]]
```

## Asserting on traces and events

Test output (`AgentResult.output`, `AgentResult.total_steps`) for *what* happened. Test events for *how* it happened — which tool ran, whether a retry fired, whether a gate approved. Read events from `emitter.events` and filter by the `event_type` discriminator string. Event class names (`AgentStepEvent`, `LLMResponseEvent`, `ToolResultEvent`) describe the payload shape; `event_type` selects them.

<!-- verify: skip — assertion snippet, depends on a captured emitter from a run -->
```python
tool_results = [e for e in emitter.events if e.event_type == "tool.result"]
assert len(tool_results) == 1
assert tool_results[0].success is True
```

See [Observability](observability.md) for the full event catalogue and classification into levels.

## Mocking tools

Mock at the seam. `MockSandbox` replaces `DockerSandbox` for code-execution tools with scripted `ExecutionResult` returns. `MockEmbeddingClient` returns deterministic vectors for semantic-memory tests. For HTTP-based tools, mount an `respx` route fixture and inject it at the HTTP-client layer. Isolate the tool under test only when its logic is non-trivial; otherwise, test through the agent loop so the assertions cover registration, invocation, and result handling together.

## Testing multi-agent systems

Give each agent its own `MockLLMClient` with the responses that agent should produce. The shared `InMemoryEmitter` records every delegation, handoff, and broadcast event — assert on `event_type == "agent.delegate"`, `"agent.handoff"`, etc. For coordinators (orchestrator, supervisor, bus), assert on the coordinator's result object *and* on the event sequence. See [Multi-Agent Foundations](multi-agent-foundations.md) and [Multi-Agent Coordination](multi-agent-coordination.md) for the patterns being exercised.

## Coverage expectations

The full quality gate enforces 100% line coverage (`just check --cov-fail-under=100`). `# pragma: no cover` is reserved for genuinely untestable lines — see `CONTRIBUTING.md` for the policy, not this guide.

## See also

- [`examples/helpers.py`](../../examples/helpers.py) — `make_response`, `make_emitter`, `make_usage` helpers used across examples and tests
- [`tests/conftest.py`](../../tests/conftest.py) — fixture patterns for shared test setup
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) — coverage policy and quality gates
- [Evaluation](evaluation.md) — runtime evaluators (distinct from test-time assertions)
