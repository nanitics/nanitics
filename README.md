<p align="center">
  <img src="https://raw.githubusercontent.com/nanitics/nanitics/main/assets/nanitics-logo.png" alt="Nanitics" width="160" />
</p>

# Nanitics

The Python SDK for production agents.

[![CI](https://github.com/nanitics/nanitics/actions/workflows/ci.yml/badge.svg)](https://github.com/nanitics/nanitics/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nanitics)](https://pypi.org/project/nanitics/)
[![Python](https://img.shields.io/pypi/pyversions/nanitics)](https://pypi.org/project/nanitics/)
[![License](https://img.shields.io/github/license/nanitics/nanitics)](LICENSE)

Build agentic systems your team can debug, extend, and own. Built and used by [Propodeum](https://propodeum.com) for production client engagements. Compose any architecture from reusable primitives. Trace every agent decision.

## Why Nanitics?

- **Your team owns the result.** Apache-2.0, typed, documented, with an explicit public API (`nanitics.__all__`). The codebase a client inherits at the end of an engagement is one their engineers can read, extend, and operate.
- **Trace-first observability.** Every agent loop, tool call, and coordination event emits a structured event. The built-in Observatory trace viewer turns that into a live debugging surface from day one. No instrumentation to add.
- **Real-services validation.** Public components are validated against real LLM providers before release, not just mocks. Mocks drive fast tests; real services prove correctness.

## Features

**Agent strategies** — Three committed strategies: [ReAct](docs/guides/agent-types.md), Reasoning, CodeAct. Advanced patterns (Reflexion, ReWOO, LATS, Tree of Thought) are available from `nanitics.experimental.strategies` under a separate stability contract.

**Memory** — [Working, episodic, long-term, and semantic memory](docs/guides/memory.md) for persistent agent state.

**Orchestration** — Compose agents into [pipelines, DAGs, loops, conditionals, and map-reduce workflows](docs/guides/orchestration.md).

**Multi-agent coordination** — Committed patterns: [handoff, supervisor, agent-as-tool](docs/guides/multi-agent-coordination.md). Advanced patterns (blackboard, debate, consensus, bidding, broadcast, message bus, peer network, judge router, dynamic orchestrator) are available from `nanitics.experimental.coordination`.

**Evaluation** — [Programmatic and LLM-based evaluators](docs/guides/evaluation.md) for assessing agent output quality.

**Human-in-the-Loop** — [Approval gates, revision gates, and durable HITL](docs/guides/human-in-the-loop.md) with checkpoint suspension for long-running workflows.

**Tools** — [Function tools, conditional tools, and tool composition](docs/guides/tools.md) with automatic schema generation.

**Observability** — [Event-based tracing](docs/guides/observability.md) with the Observatory trace viewer for inspecting agent execution.

**Planning** — [Upfront and adaptive planning](docs/guides/planning.md) with goal tracking and plan adherence evaluation.

**Safety** — [Iteration limits, cancellation tokens, and sandboxed code execution](docs/guides/safety.md).

## Quick Start

> For a full end-to-end run against a real LLM, see [the deployment guide](docs/guides/deployment.md).

Install Nanitics:

```bash
pip install nanitics
```

Create a ReAct agent with a tool, driven by a scripted `MockLLMClient` so the snippet runs without an API key:

```python
import asyncio
from nanitics import (
    InMemoryEmitter,
    LLMResponse,
    MockLLMClient,
    ReActAgent,
    ToolCall,
    Usage,
    tool,
)

@tool("greet", "Greet someone by name")
async def greet(name: str) -> str:
    return f"Hello, {name}!"

async def main():
    llm = MockLLMClient(responses=[
        LLMResponse(
            content="I'll greet them.",
            tool_calls=[ToolCall(id="1", name="greet", arguments={"name": "world"})],
            usage=Usage(input_tokens=50, output_tokens=20),
            model="mock",
            stop_reason="tool_use",
        ),
        LLMResponse(
            content="Hello, world!",
            tool_calls=[],
            usage=Usage(input_tokens=80, output_tokens=15),
            model="mock",
            stop_reason="end_turn",
        ),
    ])
    agent = ReActAgent(
        name="my-agent",
        llm_client=llm,
        emitter=InMemoryEmitter("trace-001"),
        system_prompt="You are a helpful assistant.",
        tools=[greet],
    )
    result = await agent.run("Say hello to the world")
    print(result.output)

asyncio.run(main())
```

To run the same agent against a real provider, set `ANTHROPIC_API_KEY` and swap `MockLLMClient(...)` for `AnthropicLLMClient(model="claude-haiku-4-5")`. Everything else above is unchanged.

See the [Getting Started guide](docs/guides/getting-started.md) for a full walkthrough. For the full API, read the docstrings in the source tree under [`nanitics/`](nanitics/).

## LLM Providers

Nanitics supports multiple LLM providers:

| Provider | Install | Client |
|----------|---------|--------|
| Anthropic | `pip install nanitics` | `AnthropicLLMClient` |
| OpenAI | `pip install nanitics` | `OpenAILLMClient` |
| Mistral | `pip install nanitics[mistral]` | `MistralLLMClient` |
| LiteLLM | `pip install nanitics[litellm]` | `LiteLLMClient` |

For testing and development, use `MockLLMClient` — no API keys required.

## Examples

The [examples directory](examples/) contains runnable examples covering every SDK component. All examples use `MockLLMClient` for deterministic, API-key-free execution.

See the [examples README](examples/README.md) for a complete index.

## Documentation

Primary entry points:

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/guides/getting-started.md) | Build your first agent |
| [Core Concepts](docs/guides/core-concepts.md) | The agent loop, tools, prompts, LLM clients |
| [Agent Types](docs/guides/agent-types.md) | Agent strategies and when to use each |
| [Multi-Agent Coordination](docs/guides/multi-agent-coordination.md) | Coordination patterns for multi-agent systems |
| [Deployment](docs/guides/deployment.md) | Full-stack compose, take-to-own-infra, resource and shutdown patterns |
| [API Reference](https://docs.nanitics.dev/) | Generated from source docstrings — signatures, fields, constraints |

For the complete catalogue — Memory, Orchestration, Evaluation, HITL, Tools, Planning, Context Management, Error Handling, Safety, Security, Observability, Building Applications, Architecture, SDK Internals, Diagnosing Agent Issues, Testing, Streaming, Production, Built-in Tools, Local LLMs — see the [full guides index](docs/guides/README.md).

## Project

- **Status**: Pre-1.0. The public API is `nanitics.__all__`; see the [deprecation policy](docs/deprecation-policy.md) for the change contract.
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, [DEVELOPMENT.md](DEVELOPMENT.md) for setup.
- **Governance**: [GOVERNANCE.md](GOVERNANCE.md) covers decisions and release cadence; [MAINTAINERS.md](MAINTAINERS.md) names the current maintainers.
- **Trademark**: [TRADEMARK.md](TRADEMARK.md) for who owns the Nanitics name and what uses are permitted.
- **Questions and ideas**: [GitHub Discussions](https://github.com/nanitics/nanitics/discussions); [Getting Help](docs/getting-help.md) for the channel split.
- **Security**: report vulnerabilities via [SECURITY.md](SECURITY.md); SDK security posture in the [security guide](docs/guides/security.md).

## About

Nanitics is built and maintained by [Propodeum](https://propodeum.com) for production client engagements with technical teams taking agentic AI from prototype to production. If you want help integrating it, see propodeum.com.

## License

Apache License 2.0. See [LICENSE](LICENSE) for the full text.
