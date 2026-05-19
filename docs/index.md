# Nanitics

Python SDK for building single-agent and multi-agent AI systems.

Composable primitives — agent strategies, memory, planning, orchestration,
multi-agent coordination, evaluation, human-in-the-loop, observability —
that work together without framework lock-in. Every agent decision is
traceable through the built-in Observatory.

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
from nanitics.infrastructure import MockLLMClient
from nanitics.strategies import ReActAgent, tool
from nanitics.tracing import InMemoryEmitter


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

## Public API surface

The SDK exposes three namespaces:

- **`nanitics`** — recommended core. Primitives and load-bearing compositions for building most agentic systems: the three agent types (`ReActAgent`, `ReasoningAgent`, `CodeActAgent`), all five memory types, core workflows (`Sequential`, `Parallel`, `DAG`), multi-agent foundations (`AgentTool`, `Broadcast`, context transfer), `Blackboard`, `Supervisor`, `JudgeRouter`, HITL and durable suspension, planning, evaluation, context management, error handling, safety, observability, standard LLM and embedding clients, built-in tools.
- **`nanitics.patterns`** — named compositions over the core primitives. `create_orchestrator`, the `HandoffPayload`/`HandoffStep`/`create_handoff_chain` stack, and other sugar that could be rebuilt from primitives in a few lines but is named for discoverability.
- **`nanitics.specialized`** — specialized primitives that are structurally distinct but niche. Reach for them deliberately: `ReWOOAgent`, `ReflexionAgent`, `TreeOfThoughtAgent`, `LATSAgent`, the `Loop`/`Conditional`/`MapReduce`/`Pipeline` workflows, the `Bidding`/`Debate`/`Consensus` coordination patterns, `MessageBus`, `PeerNetwork`, `MistralLLMClient`, hierarchical-decomposition planning.

## LLM providers

Nanitics supports multiple LLM providers:

| Provider  | Install                           | Client                                       |
| --------- | --------------------------------- | -------------------------------------------- |
| Anthropic | `pip install nanitics`            | `AnthropicLLMClient`                         |
| OpenAI    | `pip install nanitics`            | `OpenAILLMClient`                            |
| LiteLLM   | `pip install nanitics[litellm]`   | `LiteLLMClient`                              |
| Mistral   | `pip install nanitics[mistral]`   | `nanitics.specialized.MistralLLMClient`      |

Anthropic and OpenAI clients ship by default — no extras needed. For Mistral, the native `MistralLLMClient` lives in `nanitics.specialized`; for most adopters `LiteLLMClient` covers Mistral too.

For testing and development, use `MockLLMClient` — no API keys required.
