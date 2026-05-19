# Nanitics

Python SDK for building single-agent and multi-agent AI systems.

Composable primitives — agent strategies, memory, planning, orchestration,
multi-agent coordination, evaluation, human-in-the-loop, observability —
that work together without framework lock-in. Every agent decision is
traceable through the built-in Observatory.

## Start here

The fastest path from zero to a running agent:

1. **[Getting Started](guides/getting-started.md)** — your first ReAct agent
2. **[Core Concepts](guides/core-concepts.md)** — the agent loop, messages, LLM clients
3. **[Tools](guides/tools.md)** — give the agent something to do
4. **[Memory](guides/memory.md)** — Working memory; when to reach for the others
5. **[Human-in-the-Loop](guides/human-in-the-loop.md)** — approval and revision gates
6. **[Multi-Agent Foundations](guides/multi-agent-foundations.md)** — compose two agents

That's the essential surface. For everything else — orchestration, evaluation, planning, observability, production guides, advanced reasoning strategies — see the [full guides index](guides/README.md).

For API details (signatures, fields, constraints), read the docstrings in your editor, in the source tree under [`nanitics/`](../nanitics/), or at [docs.nanitics.dev](https://docs.nanitics.dev/). The union of every public subpackage's `__all__` is the authoritative public surface. Runnable [examples](https://github.com/nanitics/nanitics/tree/main/examples) cover every SDK component using `MockLLMClient` — no API key required.

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

For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). The union of every public subpackage's `__all__` is the authoritative public surface.

## Public API surface

The public surface is hierarchical. Every name lives in a topic-named subpackage under `nanitics`; the top-level `nanitics` package itself exports only `__version__`. There are no flat re-exports — import from the subpackage directly.

- **`nanitics.strategies`** — the foundational `Agent` / `Tool` / `SystemPromptBuilder` primitives and the agent strategies built on top: `ReActAgent`, `ReasoningAgent`, `CodeActAgent` (plus the specialized strategies re-exported from `nanitics.specialized`).
- **`nanitics.memory`** — working, shared, semantic, episodic, and long-term memory stores.
- **`nanitics.composition`** — multi-agent foundations and workflows: `Sequential`, `Parallel`, `DAG`, `AgentTool`, `Broadcast`, `Blackboard`, `Supervisor`, `JudgeRouter`, durable runs and checkpointing.
- **`nanitics.tracing`** — events, emitters, trace stores, and level filtering for the Observatory.
- **`nanitics.errors`** — error classes and the error-handling capability surface.
- **`nanitics.hitl`** — human-in-the-loop primitives: approval, revision, human-input providers.
- **`nanitics.evaluation`** — output evaluators, verdicts, and contexts.
- **`nanitics.planning`** — goal- and plan-based planning primitives.
- **`nanitics.context`** — context management: token counting, summarization, truncation.
- **`nanitics.safety`** — cancellation tokens, iteration limits, sandboxes.
- **`nanitics.tools`** — curated reference `Tool` implementations.
- **`nanitics.infrastructure`** — LLM and embedding clients: `AnthropicLLMClient`, `OpenAILLMClient`, `LiteLLMClient`, `MockLLMClient`, `VoyageEmbeddingClient`, `MockEmbeddingClient`.
- **`nanitics.patterns`** — named compositions over the core primitives: `create_orchestrator`, the `HandoffPayload`/`HandoffStep`/`create_handoff_chain` stack, and other sugar that could be rebuilt from primitives in a few lines but is named for discoverability.
- **`nanitics.specialized`** — specialized primitives that are structurally distinct but niche. Reach for them deliberately: `ReWOOAgent`, `ReflexionAgent`, `TreeOfThoughtAgent`, `LATSAgent`, the `Loop`/`Conditional`/`MapReduce`/`Pipeline` workflows, the `Bidding`/`Debate`/`Consensus` coordination patterns, `MessageBus`, `PeerNetwork`, `MistralLLMClient`, hierarchical-decomposition planning.

The `patterns` and `specialized` namespaces signal adoption guidance, not maturity — every symbol there is part of the v1.0 surface. The union of every public subpackage's `__all__` is the authoritative public surface (see [deprecation-policy.md](deprecation-policy.md)).

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
