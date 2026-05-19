# SDK Internals

Building a system **with** Nanitics? Read [docs/guides/architecture-guide.md](../guides/architecture-guide.md) first — it is the decision sequence for designing your agents. This doc covers the SDK's own internal structure, for readers who want to read the source.

Nanitics is a Python library for building AI agent systems: define agents, tools, prompts, and flows in Python code, with no server, database, or web framework on the library side. The sections below explain why the code is organised the way it is — the principles, the layer model, the package structure, the component protocols, the error model, and the dependency choices behind the surface.

## Principles

**Composition over inheritance.** Capabilities are assembled, not inherited. An agent *has* a memory system, *has* a planner, *has* tools — it doesn't inherit from MemoryAgent or PlanningAgent. This prevents combinatorial class hierarchies and allows any capability to be combined with any other.

**Dependency inversion at every boundary.** Components depend on protocols, not implementations. Agent code depends on the LLM *protocol*, not Anthropic. Orchestration depends on a *step* protocol, not specific agent types. Trace storage depends on an event *protocol*, not PostgreSQL. Implementations are injected, never imported directly by consumers.

**Protocols for boundaries, ABCs for shared behavior.** External integration points (LLM client, memory stores, trace backends, tool interface) use `typing.Protocol` — structural typing with no forced inheritance. ABCs exist only where subclasses genuinely reuse shared implementation logic (e.g., the agent loop's observe→think→act cycle). An ABC with only abstract methods should be a Protocol instead.

**Event-driven observability from day one.** Every runtime component emits structured events through an injected `EventEmitter`. Trace hierarchy (spans with parent-child relationships) is built into the agent loop and propagated through orchestration and multi-agent boundaries. Observability is not bolted on — it is a design constraint from the first line of code.

**Async-native.** All core protocols and implementations use `async`/`await`. Agent systems are I/O-heavy (LLM calls, tool execution, database writes). Starting async avoids a painful rewrite later and enables concurrent tool execution, parallel orchestration, and non-blocking trace emission.

**Fail-fast at boundaries.** Validate inputs at system edges (user input, LLM responses, tool parameters, external API responses). Within the system, trust typed interfaces and let exceptions propagate. No swallowed errors, no silent fallbacks in internal code.

**Pydantic v2 for all data boundaries.** Schemas, validation, serialization, and configuration use Pydantic models throughout. This gives us LLM structured output schemas, tool parameter definitions, event data models, configuration objects, and API request/response types from a single system.

## Layer Model

The SDK is organized in six layers. Dependencies flow downward only — each layer may import from layers below it, never above.

```
┌──────────────────────────────────────────────────────────┐
│  Nanitics SDK (framework)                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Collaboration                                     │  │
│  │  human-in-the-loop                                 │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  Composition                                       │  │
│  │  orchestration, multi-agent                        │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  Agent Capabilities                                │  │
│  │  memory, planning, context, errors, eval           │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  Agent Core                                        │  │
│  │  agents, tools, prompts                            │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  Safety                                            │  │
│  │  iteration limits, cancellation                     │  │
│  ├────────────────────────────────────────────────────┤  │
│  │  Infrastructure                                    │  │
│  │  llm, embeddings, observability                     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Infrastructure** is consumed by every other SDK layer. LLM abstraction provides the model interface. Embeddings provide vector representation for semantic memory. Observability provides event emission, trace storage, level classification, buffered collection, persistent per-event storage (with a Postgres implementation), and a reusable FastAPI trace router. The `observatory/` module builds on top of observability to provide a comprehensive backend API for trace visualization frontends — it exposes a router factory + service layer pattern with run management, span tree queries, agent-scoped views, workflow DAG, and SSE streaming.

**Safety** defines the constraint primitives that bound agent behavior: iteration limits, cooperative cancellation, and sandboxed code execution. It sits above infrastructure (it uses observability for tracking) and below core (agents and tools consume safety constraints). Safety does not import from upper layers — it defines the constraints, and upper layers apply them. The sandbox subsystem (`Sandbox` protocol, `DockerSandbox`, `MockSandbox`) provides isolated execution environments for running untrusted code with resource limits, read-only filesystems, and PID restrictions.

**Agent Core** defines the central abstractions: agent types and their reasoning loops, tool definitions and dispatch, prompt assembly. Core also defines the extension contracts (protocols) that capabilities implement: `OutputEvaluator`, `ContextProvider`, `ContextManagement`, `ErrorHandling`, `WorkingMemory`. These protocols live in Core because the agent loop depends on them — capabilities provide implementations. Core consumes safety primitives (e.g., the agent loop checks iteration limits and cancellation signals). Context-provider materialization (the `<nanitics:context>` wrapper the LLM sees around every provider contribution) is applied by the agent base — it is provider-adapter-agnostic and lives above `nanitics/infrastructure/llm/*`, so every `LLMClient` implementation serializes the wrapped message as an ordinary `role="user"` turn.

**Agent Capabilities** enhance individual agents: memory retention, planning strategies, context window management, error handling, output evaluation.

**Composition** combines agents and steps into workflows: orchestration patterns (sequential, parallel, pipeline, conditional, loop, map-reduce, DAG), multi-agent topologies (orchestrator, handoff, supervisor, etc.), and durability (checkpointing and suspension for resumable execution).

**Collaboration** integrates human participation into agent workflows through three mechanisms: **HITL tools** (`request_approval`, `ask_human`) that agents invoke voluntarily, **ApprovalWrappedTool** that enforces mandatory human approval on any tool, and **ApprovalGate** that pauses orchestrated workflows for human review. All share a single `HumanInputProvider` protocol — implementations determine the interaction medium (callback for tests/CLI, API endpoint for web).

Observability is the one true cross-cutting concern — every layer emits events and trace hierarchies flow through the full stack. Safety is *applied* at multiple layers but its dependency direction is strictly downward, like any other layer.

## Package Structure

The repository is a standalone SDK package.

```
nanitics/
├── nanitics/                        # Importable Python package
│   ├── __init__.py                  # Public API surface
│   ├── infrastructure/
│   │   ├── errors.py                # Exception hierarchy (NaniticsError and subclasses)
│   │   ├── llm/                     # LLM client protocol + implementations
│   │   │   ├── protocol.py          # LLMClient protocol
│   │   │   ├── anthropic.py         # Anthropic implementation
│   │   │   ├── openai.py            # OpenAI implementation
│   │   │   ├── mistral.py           # Mistral implementation
│   │   │   ├── litellm.py           # LiteLLM multi-provider adapter
│   │   │   ├── mock.py              # Deterministic mock for testing
│   │   │   └── routing.py           # RoutingLLMClient, routing strategies
│   │   ├── embeddings/              # Embedding client protocol + implementations
│   │   │   ├── protocol.py          # EmbeddingClient protocol
│   │   │   ├── voyage.py            # Voyage AI implementation
│   │   │   └── mock.py              # Deterministic mock for testing
│   │   └── observability/
│   │       ├── events.py            # Event types (Pydantic models)
│   │       ├── emitter.py           # EventEmitter protocol + implementations
│   │       ├── storage.py           # Trace storage protocol + implementations
│   │       ├── levels.py            # Level classification (info/debug/verbose)
│   │       ├── collector.py         # TraceCollector (buffer, flush, SSE queue)
│   │       └── postgres_store.py    # PostgresTraceStore (PersistentTraceStore)
│   ├── observatory/                 # Observatory backend API
│   │   ├── __init__.py              # Public API: create_observatory_router, ObservatoryService, models
│   │   ├── models.py                # Response models and request schemas (Pydantic)
│   │   ├── service.py               # Business logic composing PersistentTraceStore primitives
│   │   ├── router.py                # FastAPI route definitions (thin delegates to service)
│   │   └── streaming.py             # SSE streaming endpoint logic
│   ├── safety/                      # Iteration limits, cancellation, sandboxing
│   ├── core/
│   │   ├── agents/                  # Agent base, agent types, extension contracts
│   │   │   ├── base.py              # Agent base class and loop
│   │   │   ├── react.py             # ReAct agent
│   │   │   ├── reasoning.py         # Reasoning agent
│   │   │   ├── context.py           # ContextProvider, ContextContent, ContextManagement protocols
│   │   │   ├── errors.py            # ErrorHandling protocol
│   │   │   ├── evaluation.py        # OutputEvaluator protocol, evaluation models
│   │   │   ├── working_memory.py    # WorkingMemory protocol, WorkingMemoryContributor
│   │   │   └── parsing.py           # Working memory parsing utilities
│   │   ├── tools/                   # Tool protocol, registry, decorator
│   │   └── prompts/                 # System prompt builder, tool formatting
│   ├── capabilities/
│   │   ├── memory/                  # Memory protocols + implementations
│   │   ├── planning/                # Planning strategies
│   │   ├── context/                 # Context window management, summarization
│   │   ├── errors/                  # Error handling strategies (classification, retry, self-correction)
│   │   └── evaluation/              # Programmatic, LLM-based, composite
│   ├── composition/
│   │   ├── orchestration/           # Sequential, parallel, pipeline, conditional, loop, map-reduce, DAG, plan bridge
│   │   ├── multi_agent/             # Orchestrator, handoff, supervisor, etc.
│   │   └── durability/              # Checkpoints, suspension, resumable execution
│   └── collaboration/               # Human-in-the-loop: HITL tools, approval wrappers, gates
├── observatory/                     # React component library for trace visualization
│   ├── src/                         # TypeScript source (tree views, event renderers, SSE streaming)
│   │   ├── index.ts                 # Barrel export
│   │   ├── types.ts                 # SpanTreeNode, TraceEvent, Run types
│   │   ├── client.ts                # ObservatoryClient (REST), StreamingClient (SSE)
│   │   ├── registry.ts              # EventRendererRegistry (extensible event rendering)
│   │   ├── context.tsx              # ObservatoryProvider React context
│   │   └── components/              # TraceTree, TreeNode, EventDetailPanel, RunCard, pages
│   ├── dev/                         # Embedded Vite dev shell for standalone development
│   ├── tests/                       # Vitest component and client tests
│   └── package.json
├── tests/                           # SDK unit and integration tests
├── examples/                        # Runnable examples
├── docs/                            # Documentation
├── pyproject.toml                   # SDK package definition
├── justfile
└── .env.example
```

### Import Rules

These rules enforce the layer model and prevent circular dependencies.

**Within the SDK:**

- `nanitics.infrastructure` imports nothing from `nanitics` (except stdlib and third-party).
- `nanitics.safety` imports from `nanitics.infrastructure` only.
- `nanitics.strategies` imports from `nanitics.infrastructure` and `nanitics.safety`. Strategies define extension contracts (protocols) that upper layers implement.
- `nanitics.capabilities` imports from `nanitics.infrastructure`, `nanitics.safety`, and `nanitics.strategies`. Capabilities implement the strategies' extension contracts (`ContextManagement`, `ErrorHandling`, `OutputEvaluator`, etc.).
- `nanitics.composition` imports from layers below it (infrastructure, safety, strategies, capabilities).
- `nanitics.collaboration` imports from layers below it (infrastructure, safety, strategies, capabilities, composition).

### Public API

Adopters interact with the SDK through a layered import surface:

- **`import nanitics`** — The primary import for daily use. Re-exports agents, tools, prompts, LLM clients, memory, planning, composition, collaboration, and other core abstractions (~300 symbols). Application code should start here.
- **`import nanitics.infrastructure`** — Event classes (`*Event`, `BaseEvent`), trace levels (`TraceLevel`, `classify_level`, `is_level_included`, `LEVEL_ORDER`), and Postgres utilities (`get_schema_sql`). Used when subscribing to or inspecting trace events.
- **`import nanitics.observatory`** — Observatory backend API (`create_observatory_router`, `ObservatoryService`, models). Used when embedding the trace visualization backend in an application.
- **`import nanitics.composition`** — `CHECKPOINT_SCHEMA_VERSION` and other composition internals not needed in typical application code.

The internal layer structure is for readers of the source. The external surface — the names re-exported from `nanitics` — is what an adopter imports.

## Component Protocols

The key protocols that define component boundaries. Each is a `typing.Protocol` — implementations match structurally, no inheritance required.

### LLMClient

The boundary between agent logic and model providers. Defines a single async generation method that accepts a system prompt, message history, optional tool schemas, and an optional Pydantic model for structured output. Returns a unified response type. Everything below this — authentication, retries, provider-specific formatting — is implementation detail.

### Tool

The boundary between agent reasoning and external actions. A tool exposes a schema (name, description, parameters — used by the LLM to decide invocation) and an async execute method that returns a result. Tool parameters are validated against the schema before execution.

### EventEmitter

The boundary between runtime behavior and observability. Components emit structured events through this protocol. It also provides a span method that creates a context manager for hierarchical tracing — entering a span sets the parent for all events emitted within it.

### EmbeddingClient

The boundary between semantic capabilities and embedding providers. Defines a single async method for converting text into vector representations. Implementations handle provider-specific API calls (Voyage AI) or deterministic vectors (mock for testing).

### Memory Store Protocols

Memory uses separate protocols per memory type rather than a unified base — the access patterns are too different to abstract into one interface:

- **LongTermStore** — key-value CRUD for persistent facts (store, retrieve by key, list_keys, delete).
- **SemanticStore** — vector similarity search (store with embedding, query by similarity, delete).
- **EpisodeStore** — temporal episode storage (add episode, query recent/relevant, capacity management).

Each protocol has an in-memory implementation in the SDK and can have persistent implementations in applications.

### CheckpointStore

The boundary between durability and persistence. Defines async methods for saving and loading run checkpoints, enabling suspension and resumption of agent and workflow execution.

### HumanInputProvider

The boundary between collaboration and human interaction. Defines how the system requests input from humans — implementations determine the medium (callback for tests/CLI, HTTP endpoint for web).

### TraceStore

The boundary between observability and persistence. Defines async methods for saving a trace, retrieving a single trace by ID, and querying traces with filters to return summaries.

### Implementation Strategy

Protocol definitions for all component boundaries live in the SDK. The SDK also ships in-memory implementations as defaults — usable for testing, demos, and simple single-process use without any external dependencies. Persistent implementations (PostgreSQL-backed trace storage, vector-database-backed memory) can be provided as SDK optional extras (e.g., `nanitics[postgres]`) or by applications. This keeps the SDK dependency-free while making it usable out of the box.

## Error Model

A typed exception hierarchy that mirrors the three error sources identified in the theory.

```
NaniticsError
├── LLMError
│   ├── LLMRateLimitError       # Provider rate limit hit
│   ├── LLMContextLengthError   # Input exceeds model context window
│   ├── LLMProviderError        # Provider unavailable or internal error
│   └── LLMSchemaViolationError # Response doesn't match requested schema
├── ToolError
│   ├── ToolNotFoundError       # Tool not registered
│   ├── ToolParameterError      # Invalid tool parameters
│   └── ToolExecutionError      # Tool function raised an exception
└── AgentError
    ├── AgentIterationLimitError  # Agent loop exceeded max iterations
    ├── AgentBudgetExceededError  # Token or cost budget exhausted
    └── AgentEscalationError      # Agent cannot proceed, requires human input
```

Every error carries structured metadata (timestamp, trace context, relevant parameters) for observability integration. Errors are events — when an error occurs, a corresponding trace event is emitted automatically.

## Tech Stack

| Concern | Package | Choice | Notes |
|---|---|---|---|
| Language | SDK | Python 3.11+ | Modern async, structural pattern matching, typing improvements |
| Data validation | SDK | Pydantic v2 | Schemas, serialization, configuration, LLM output |
| Async runtime | SDK | asyncio | Native Python async, no third-party event loop |
| LLM provider | SDK | Anthropic, OpenAI, Mistral, LiteLLM | Abstracted behind LLMClient protocol; each provider installs as an optional extra |
| Containerization | Both | Docker + Docker Compose | Local development infrastructure |
| Task runner | Both | just | Command orchestration for dev workflows |

### Dependency Strategy

**SDK** has minimal dependencies: `pydantic`, with optional extras for specific providers. No framework lock-in.

Anthropic and OpenAI ship as core dependencies. `[anthropic]` and `[openai]` remain as pre-release install aliases and will be dropped at first GA tag.

**SDK optional extras:**
- `nanitics[mistral]` — Mistral provider
- `nanitics[litellm]` — LiteLLM multi-provider adapter
- `nanitics[mcp]` — Model Context Protocol client
- `nanitics[voyage]` — Voyage AI embeddings
- `nanitics[openai-tokenizer]` — `tiktoken` for OpenAI-compatible token counting
- `nanitics[code_execution]` — `DockerSandbox` for sandboxed code execution
- `nanitics[postgres]` — PostgreSQL storage backends (traces, semantic memory, HITL requests)
- `nanitics[api]` — FastAPI for building application servers on top of the SDK
- `nanitics[http-tools]` — HTTP-based built-in tools
- `nanitics[search-tools]` — web-search built-in tools
- `nanitics[tools]` — aggregate extra that pulls in the full built-in tool set

**Core-vs-extra for LLM providers.** A provider ships as a core dependency when both are true: (1) the SDK maintains a native adapter with full error classification, usage/cache-token reporting where the provider supports it, and per-model profile handling; and (2) the provider is one of a small number of industry-default choices adopters expect to work out of the box. All other providers are reachable via `nanitics[litellm]` or a user-authored `LLMClient` implementation. New core additions require both conditions — the native-adapter commitment is what prevents this list from sprawling.

**Development dependencies** are separate per package: `pytest`, `ruff`, `mypy`.
