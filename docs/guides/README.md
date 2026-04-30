# Nanitics SDK Guides

Decision guidance and composition patterns for the SDK. For API details (signatures, fields, constraints), see the docstrings in the source code. For usage patterns, see the [examples](../../examples/).

## Reading Order

| # | Guide | What it helps you decide |
|---|-------|------------------------|
| 1 | [Getting Started](getting-started.md) | First agent — walkthrough from zero to running |
| 2 | [Core Concepts](core-concepts.md) | Agent loop, messages, LLM clients, extension points |
| 3 | [Tools](tools.md) | When to use tools vs context providers, creation methods |
| 4 | [Built-in Tools](built-in-tools.md) | Reference tool catalog — when to pick shipped vs MCP vs custom |
| 5 | [Agent Types](agent-types.md) | Which agent type fits your task |
| 6 | [Error Handling](error-handling.md) | Recovery strategy: retry, correct, or degrade |
| 7 | [Context Management](context-management.md) | Truncation vs summarization trade-offs |
| 8 | [Memory](memory.md) | Which memory type for which need |
| 9 | [Evaluation](evaluation.md) | When and how to gate agent output quality |
| 10 | [Testing](testing.md) | Mock-based testing of agents, tools, and multi-agent systems |
| 11 | [Planning](planning.md) | Planning strategy selection, goal-based planning |
| 12 | [Orchestration](orchestration.md) | Which workflow pattern to compose steps |
| 13 | [Multi-Agent Foundations](multi-agent-foundations.md) | Agent-as-tool, handoff, broadcast, message bus, peer network |
| 14 | [Multi-Agent Coordination](multi-agent-coordination.md) | Orchestrator, supervisor, blackboard, bidding, debate, consensus |
| 15 | [Human-in-the-Loop](human-in-the-loop.md) | Approval wrapping vs gates vs HITL provider |
| 16 | [Safety](safety.md) | Iteration limits, cancellation, sandboxing |
| 17 | [Security](security.md) | Trust boundary, prompt-injection posture, redaction, DockerSandbox limits, API-key handling |
| 18 | [Observability](observability.md) | Event levels, storage architecture, observatory |
| 19 | [Observatory Integration](observatory-integration.md) | Mount the Observatory UI in your app; custom view/panel registration; for-production seams |
| 20 | [Streaming](streaming.md) | Shipping events out of the process as Server-Sent Events |
| 21 | [Building Applications](building-applications.md) | API server, SSE streaming, persistence, HITL endpoints |
| 22 | [Production](production.md) | Pre-launch operational decision index |
| 23 | [Deployment](deployment.md) | Full-stack compose, take-to-own-infra, resource and shutdown patterns |
| 24 | [Local LLMs](local-llms.md) | Running against Ollama, vLLM, and other OpenAI-compatible servers |
| 25 | [Architecture Guide](architecture-guide.md) | System design decision sequence |
| 26 | [Diagnosing Agent Issues](diagnosing-agent-issues.md) | Debugging agents that misbehave |
