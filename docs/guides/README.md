# Nanitics SDK Guides

Decision guidance and composition patterns for the SDK. For API details (signatures, fields, constraints), see the docstrings in the source code. For usage patterns, see the [examples](../../examples/).

The guides are organized in four tiers. Read **Start here** in order; everything below is à la carte.

## Start here

The minimum you need to build something useful: an agent, a tool, a way to remember, a way to involve a human, and a way to compose two agents. About 30 minutes end to end.

| # | Guide | What it helps you decide |
|---|-------|------------------------|
| 1 | [Getting Started](getting-started.md) | First agent — walkthrough from zero to running |
| 2 | [Core Concepts](core-concepts.md) | Agent loop, messages, LLM clients, extension points |
| 3 | [Tools](tools.md) | When to use tools vs context providers, creation methods |
| 4 | [Memory](memory.md) | Working memory first; when to reach for the others |
| 5 | [Human-in-the-Loop](human-in-the-loop.md) | Approval gates, revision gates, durable HITL |
| 6 | [Multi-Agent Foundations](multi-agent-foundations.md) | Agent-as-tool, handoff, broadcast — the primitives you need first |

## Build on it

The rest of the core surface. No required order — pick what your task needs.

| Guide | What it helps you decide |
|-------|------------------------|
| [Agent Types](agent-types.md) | When to use `ReasoningAgent` or `CodeActAgent` over `ReActAgent` |
| [Built-in Tools](built-in-tools.md) | The four shipped tools — when to pick shipped vs MCP vs custom |
| [Error Handling](error-handling.md) | Recovery strategy: retry, correct, or degrade |
| [Context Management](context-management.md) | Truncation vs summarization trade-offs |
| [Orchestration](orchestration.md) | Sequential, Parallel, DAG — which workflow shape composes steps |
| [Multi-Agent Coordination](multi-agent-coordination.md) | Orchestrator, supervisor, blackboard, judge router |
| [Evaluation](evaluation.md) | When and how to gate agent output quality |
| [Planning](planning.md) | Planning strategy selection, goal-based planning |
| [Testing](testing.md) | Mock-based testing of agents, tools, and multi-agent systems |
| [Observability](observability.md) | Event levels, storage architecture, Observatory |
| [Safety](safety.md) | Iteration limits, cancellation, sandboxing |

## Ship it

Everything between "works on my laptop" and "running in production."

| Guide | What it helps you decide |
|-------|------------------------|
| [Building Applications](building-applications.md) | API server, SSE streaming, persistence, HITL endpoints |
| [Streaming](streaming.md) | Shipping events out of the process as Server-Sent Events |
| [Observatory Integration](observatory-integration.md) | Mount the Observatory UI; custom views and panels |
| [Security](security.md) | Trust boundary, prompt-injection posture, redaction, secrets |
| [Production](production.md) | Pre-launch operational decision index |
| [Deployment](deployment.md) | Full-stack compose, take-to-own-infra, resource and shutdown patterns |
| [Local LLMs](local-llms.md) | Running against Ollama, vLLM, and other OpenAI-compatible servers |
| [Architecture Guide](architecture-guide.md) | System design decision sequence |
| [Diagnosing Agent Issues](diagnosing-agent-issues.md) | Debugging agents that misbehave |

## Advanced patterns

Specialized primitives in `nanitics.specialized`. Reach for these deliberately — the core surface covers most needs. The [Advanced Patterns index](advanced-patterns.md) lists them with use-case framing and links to the guides that cover them.
