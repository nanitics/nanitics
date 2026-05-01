# Nanitics

Python SDK for building single-agent and multi-agent AI systems. It provides 7
agent strategies, 11 coordination patterns, 4 memory types, durable
human-in-the-loop, a built-in evaluation framework, and full observability —
all as composable building blocks that work together without framework
lock-in.

## Start here

- **[Getting Started](guides/getting-started.md)** — build your first agent in
  a few minutes.
- **Source docstrings** — for API details (signatures, fields,
  constraints), read the docstrings in the source tree under
  [`nanitics/`](../nanitics/) or in your editor. `nanitics.__all__` is
  the authoritative public surface.
- **[Examples](https://github.com/nanitics/nanitics/tree/main/examples)**
  — runnable scripts covering every SDK component. All use
  `MockLLMClient` for deterministic, API-key-free execution.

## Quick start

Install Nanitics:

```bash
pip install nanitics
```

Create a ReAct agent with a tool:

```python
import asyncio
from nanitics import ReActAgent, MockLLMClient, InMemoryEmitter, tool


@tool("greet", "Greet someone by name")
async def greet(name: str) -> str:
    return f"Hello, {name}!"


async def main():
    agent = ReActAgent(
        name="my-agent",
        llm_client=MockLLMClient(),
        emitter=InMemoryEmitter(),
        system_prompt="You are a helpful assistant.",
        tools=[greet],
    )
    result = await agent.run("Say hello to the world")
    print(result.output)


asyncio.run(main())
```

For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

## LLM providers

Nanitics supports multiple LLM providers:

| Provider  | Install                           | Client               |
| --------- | --------------------------------- | -------------------- |
| Anthropic | `pip install nanitics`            | `AnthropicLLMClient` |
| OpenAI    | `pip install nanitics`            | `OpenAILLMClient`    |
| Mistral   | `pip install nanitics[mistral]`   | `MistralLLMClient`   |
| LiteLLM   | `pip install nanitics[litellm]`   | `LiteLLMClient`      |

Anthropic and OpenAI clients ship by default — no extras needed.

For testing and development, use `MockLLMClient` — no API keys required.
