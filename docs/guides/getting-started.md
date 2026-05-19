# Getting Started

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Build your first AI agent in a few lines of code. This guide walks you through creating a ReAct agent, giving it a tool, running it, and inspecting the result.

## Installation

```bash
pip install nanitics
```

Anthropic and OpenAI clients ship by default — no extras needed for Claude or GPT models. Optional extras cover the other providers and tool integrations:

```bash
pip install nanitics[mistral]     # Mistral models
pip install nanitics[litellm]     # 100+ providers (Bedrock, Vertex, Gemini, Ollama, ...) via LiteLLM
pip install nanitics[mcp]         # Model Context Protocol tools (see tools.md)
pip install nanitics[tools]       # Everything needed by the four built-in reference tools (web search, HTTP, file read, code execution)
```

Multiple optional extras can be installed together — e.g., `pip install nanitics[litellm,mcp]`.

Running a local model? See [Using a local LLM](./local-llms.md) — any OpenAI-compatible server (Ollama, vLLM, LM Studio) works with `OpenAILLMClient`.

## Your First Agent

```python
import asyncio
import os
from nanitics.infrastructure import AnthropicLLMClient
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import InMemoryEmitter


# 1. Define a tool
@tool("get_weather", "Get the current weather for a city")
async def get_weather(city: str) -> str:
    return f"Sunny, 22°C in {city}"


# 2. Create an LLM client
llm = AnthropicLLMClient(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

# 3. Create an emitter for observability
emitter = InMemoryEmitter("trace-001")

# 4. Build and run the agent
agent = ReActAgent(
    name="weather-agent",
    llm_client=llm,
    emitter=emitter,
    system_prompt="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = asyncio.run(agent.run("What's the weather in Amsterdam?"))
```

> Swap `AnthropicLLMClient` for `OpenAILLMClient(model="gpt-4o-mini")` (reads `OPENAI_API_KEY`) to use OpenAI instead — the rest of the code is unchanged. See [`examples/providers/openai_client.py`](../../examples/providers/openai_client.py).

## What Just Happened

The agent runs a **ReAct loop** — it reasons about what to do, takes an action (calls a tool), observes the result, and repeats until it has a final answer.

In this example:
1. The agent received the task "What's the weather in Amsterdam?"
2. It called the `get_weather` tool with `city="Amsterdam"`
3. It observed the result: "Sunny, 22°C in Amsterdam"
4. It produced a final text response

The loop runs until the LLM returns a response without tool calls (the final answer), or until the iteration limit is reached.

## AgentResult

`agent.run()` returns an `AgentResult` with everything you need to inspect the run:

```python
print(result.output)              # "The weather in Amsterdam is sunny at 22°C."
print(result.total_steps)         # 2
print(result.termination_reason)  # "complete"
print(result.usage.total_tokens)  # 165

# Full conversation history
for msg in result.messages:
    print(f"{msg.role}: {msg.content}")
```

See the `AgentResult` docstring for all available fields.

## Observing Execution

The `EventEmitter` records everything that happens during a run. Use it for debugging, logging, or building application features.

```python
# Print all events
for event in emitter.events:
    print(f"{type(event).__name__}: {event}")

# Listen to events in real-time
def on_event(event):
    print(f"[{type(event).__name__}] {event}")

emitter.add_listener(on_event)
```

Key events emitted during a run:
- `AgentStartEvent` — agent begins execution
- `LLMRequestEvent` / `LLMResponseEvent` — each LLM call
- `ToolInvokeEvent` / `ToolResultEvent` — each tool call
- `AgentStepEvent` — each reasoning step (thought, action, observation)
- `AgentCompleteEvent` — agent finishes

See [Observability](observability.md) for the full event catalog.

## Testing with MockLLMClient

For tests, use `MockLLMClient` to script deterministic responses without calling the API:

```python
from nanitics.infrastructure import LLMResponse, MockLLMClient
from nanitics.tracing import ToolCall, Usage

llm = MockLLMClient(responses=[
    LLMResponse(
        content="Let me check the weather.",
        tool_calls=[ToolCall(id="1", name="get_weather", arguments={"city": "Amsterdam"})],
        usage=Usage(input_tokens=50, output_tokens=20),
        model="mock",
        stop_reason="tool_use",
    ),
    LLMResponse(
        content="The weather in Amsterdam is sunny at 22°C.",
        tool_calls=[],
        usage=Usage(input_tokens=80, output_tokens=15),
        model="mock",
        stop_reason="end_turn",
    ),
])
```

The rest of the code stays the same — all agent types work with any `LLMClient`.

## Running against a real LLM

To run your first end-to-end agent against a live provider, install Nanitics, set `ANTHROPIC_API_KEY`, and run the flagship quickstart example:

```bash
pip install nanitics
export ANTHROPIC_API_KEY=sk-ant-...
python examples/providers/real_llm_quickstart.py
```

(If you're working from a clone of the repo rather than an installed wheel, substitute `uv run python` for `python` — `uv` resolves the project's dev dependencies automatically.)

[`examples/providers/real_llm_quickstart.py`](../../examples/providers/real_llm_quickstart.py) wires an `AnthropicLLMClient` into a `ReActAgent` with one tool, prints the full event trace, and summarises the token cost. It is the only example in the gallery that calls a real provider — every other example stays hermetic by using `MockLLMClient`. Without the env var the example prints a skip message and exits cleanly, so it is safe to run in CI.

## Next Steps

- **[Core Concepts](core-concepts.md)** — the agent loop, message types, LLM client protocol, and system prompts
- **[Tools](tools.md)** — defining tools, managing tool state, configuring registries
- **[Built-in Tools](built-in-tools.md)** — the four shipped tools (web search, HTTP, file read, code execution) and when to pick them
- **[Agent Types](agent-types.md)** — choosing the right agent type for your use case
- **[Testing](testing.md)** — `MockLLMClient`, trace assertions, multi-agent test patterns
- **[Streaming](streaming.md)** — shipping agent events out of the process as Server-Sent Events
- **[Examples directory](../../examples/)** — 63 runnable examples covering every SDK component. Run any with `python examples/<name>.py` from an installed environment, or `uv run python examples/<name>.py` from a repo clone.

> **Runnable example:** [`examples/agents/react_agent.py`](../../examples/agents/react_agent.py) — agent creation, tool use, result inspection, and event tracing with `MockLLMClient`.
