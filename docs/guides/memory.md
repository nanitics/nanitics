# Memory

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Agents are stateless by default — each LLM call sees only the conversation history accumulated during the current run. Memory systems give agents the ability to maintain structured state within a run, persist knowledge across runs, retrieve information by similarity, learn from past experiences, and coordinate through shared state.

**When you don't need any of this.** If you only need to pass state between N consecutive turns in the same Python process — for example, disambiguating a follow-up question using the prior Q&A — a plain Python variable passed into the next `agent.run(...)` input or system prompt is simpler than anything on this page. The SDK memory types exist for persistence across process restarts, retrieval over large histories by meaning or episode, or coordination between multiple agents. If none of those apply, skip this guide and use the language.

The SDK provides five memory types, each solving a different problem. Choosing the right one (or combining several) is a key design decision.

## Decision Guide

| Memory Type | Persistence | Delivery | Use Case |
|---|---|---|---|
| [Working Memory](#working-memory) | Within a run | Context provider (automatic) | Structured scratchpad — progress tracking, intermediate findings |
| [Long-Term Memory](#long-term-memory) | Across runs | Tools (agent-initiated) | Explicit facts the agent stores and retrieves by key |
| [Semantic Memory](#semantic-memory) | Across runs | Tools or context provider | Knowledge retrieval by meaning — RAG, document search |
| [Episodic Memory](#episodic-memory) | Across runs | Tools or context provider | Learning from past experiences — what worked, what didn't |
| [Shared Memory](#shared-memory) | Within a session | Tools or context provider | Multi-agent coordination through shared state |

**Choosing a memory type:**

- Need structured state during a run (lists, plans, notes)? → **Working Memory**
- Need to store and recall facts by name across runs? → **Long-Term Memory**
- Need to find relevant information by meaning? → **Semantic Memory**
- Need to learn from past successes and failures? → **Episodic Memory**
- Need multiple agents to coordinate through shared artifacts? → **Shared Memory**

## Delivery Mechanisms

Memory content reaches the agent through three mechanisms. Understanding these is essential for configuring memory correctly.

### Context Providers

Context providers inject content automatically before every LLM call. The agent sees the memory content in its context without taking any action. Use context providers when the agent should always have access to the information.

Each memory type has a corresponding provider class (e.g. `WorkingMemoryProvider`, `EpisodicMemoryProvider`, `SharedMemoryProvider`). Wire providers into the agent via the `context_providers` parameter.

**Trade-off:** Automatic injection is convenient but consumes context budget on every call. If the memory content is large or rarely needed, tools may be a better fit.

**Wire shape.** Before the LLM sees a provider's contribution, the SDK wraps the raw `ContextContent.content` in a namespaced `<nanitics:context provider="…" priority="…" protected="…">…</nanitics:context>` block and delivers it as a `role="user"` message inserted just before the latest user turn. The wrapper is the structural signal that tells the LLM "this came from the SDK, not the user" — without it, Anthropic-backed agents in particular tend to treat provider output as untrusted external data and refuse to reference it. Providers return raw strings; the SDK owns the wire shape. When writing a custom `ContextProvider`, you do not need to emit the wrapper yourself — the authoritative spec lives on `Agent._inject_context`'s docstring.

### Tools

Tools let the agent actively store and retrieve information. The agent decides when to read or write memory. Use tools when the agent should control when it accesses memory.

Each memory type has a `create_*_tools()` factory function that returns a set of tools for that store. Pass the returned tools to the agent's `tools` parameter alongside any task-specific tools.

**Trade-off:** The agent must decide to use tools, which adds reasoning overhead. But it avoids wasting context on information the agent doesn't need on a given step.

### System Prompt Contributors

Prompt contributors add static instructions to the system prompt explaining how to use a memory system. They don't deliver data — they teach the agent how to interact with memory. Pass contributors via the `prompt_contributors` parameter.

Some contributors are added automatically — for example, `ReActAgent` adds `WorkingMemoryContributor` whenever `working_memory` is provided. Don't add a contributor manually if the agent already adds it.

### Delivery Support by Memory Type

| Memory Type | Context Provider | Tools | Auto-Contributor |
|---|---|---|---|
| Working Memory | `WorkingMemoryProvider` | — | `WorkingMemoryContributor` (auto-added by `ReActAgent`) |
| Long-Term Memory | — | `create_long_term_memory_tools()` | — |
| Semantic Memory | `SemanticMemoryProvider` | `create_semantic_memory_tools()` | `SemanticMemoryContributor` |
| Episodic Memory | `EpisodicMemoryProvider` | `create_episodic_memory_tools()` | `EpisodicMemoryContributor` |
| Shared Memory | `SharedMemoryProvider` | `create_shared_memory_tools()` | `SharedMemoryContributor` |

Memory types with both provider and tool support can be configured either way — or both simultaneously. The provider gives passive, always-on access; tools give the agent active control.

### Choosing Between Provider and Tools

The decision between context provider and tools depends on how frequently the agent needs the information:

- **Every step** — Use a context provider. The information is always available without the agent needing to decide to look for it. Good for working memory (current state) and episodic memory (past experiences).
- **Occasionally** — Use tools. The agent requests information when it decides it's relevant. Good for long-term memory (specific facts) and semantic memory (search queries).
- **Both** — Use a provider for passive read access and tools for active writes. Common with shared memory: the agent sees the board state automatically but actively writes new entries.

## Working Memory

An in-run structured scratchpad that persists across agent steps. The agent writes structured content using `<working_memory>` blocks in its responses, organized into named sections with `## Section Name` headers. `WorkingMemoryProvider` injects the current state into context before every LLM call.

Working memory solves the problem of structured state tracking over long runs. Without it, the agent's intermediate findings, progress notes, and decision logs exist only in the growing conversation history, which may be truncated or summarized. Working memory provides a stable, structured window that the agent always sees.

**When to use:** The agent needs to maintain structured state across many steps — progress tracking, intermediate findings, decision logs, checklists. Most beneficial for tasks exceeding ~5 steps.

**When not to use:** Short tasks where conversation history suffices. Tasks where the agent doesn't need to track state.

### How It Works

1. `WorkingMemoryContributor` adds instructions to the system prompt explaining the block format
2. The agent includes `<working_memory>` blocks in its responses with `## Section Name` headers
3. The agent loop parses these blocks and updates the store
4. Before each LLM call, `WorkingMemoryProvider` injects the current state as a context block
5. The agent always sees its latest working memory state

Each block is a **full replacement** — if the agent omits a section, that section is lost. The system prompt instructions emphasize this, but less capable models may still drop sections.

You can pre-populate sections with `update()`, but note that `ReActAgent` calls `reset()` at the start of each run, clearing all content. To seed working memory that survives, either override `reset()` in a custom implementation or use a context provider to inject the initial state.

The `WorkingMemory` protocol and `InMemoryWorkingMemory` implementation are documented in their source docstrings.

> **See also:** [`examples/memory/working_memory.py`](../../examples/memory/working_memory.py) — standalone memory operations, provider/contributor integration, and end-to-end agent usage.

## Long-Term Memory

A key-value store that persists across agent runs. Agents interact with it through tools — storing facts under descriptive keys and recalling them later by exact key name. The store supports namespaces for isolating data between different agents or contexts.

Long-term memory is the right choice when agents need to remember explicit, named facts — user preferences, project configurations, learned rules, accumulated knowledge. Keys should be self-documenting since the agent uses key listing to discover what's stored. Keys like `user_preferred_language` work; keys like `data1` don't.

**When to use:** The agent needs to remember explicit facts across conversations — information that can be stored and retrieved by a descriptive name.

**When not to use:** Information that needs similarity-based retrieval (use [Semantic Memory](#semantic-memory)). Information that's better structured as experiences (use [Episodic Memory](#episodic-memory)).

Namespaces isolate memory between different agents or contexts. Pass `namespace` to `create_long_term_memory_tools()` so each agent operates in its own key space without conflicts.

`InMemoryLongTermStore` is useful for testing but loses data when the process ends. For production, implement the `LongTermStore` protocol with database-backed storage. The protocol surface is minimal — four async methods: `store`, `retrieve`, `delete`, and `list_keys`.

> **See also:** [`examples/memory/long_term_memory.py`](../../examples/memory/long_term_memory.py) — store/retrieve operations, namespace isolation, agent tool integration, and multi-run persistence.

## Semantic Memory

A similarity-based knowledge store. Content is embedded into vectors and retrieved by semantic similarity rather than exact key match. Requires an `EmbeddingClient` to convert text into vectors.

Semantic memory enables RAG patterns — the agent searches a knowledge base by meaning rather than knowing exact keys. This is valuable for document retrieval, finding related information across a corpus, or any scenario where the query doesn't match the stored text literally. Results are ranked by cosine similarity.

**When to use:** The agent needs to search a knowledge base by meaning — document retrieval, RAG, finding related information without knowing exact keys.

**When not to use:** Simple key-value lookups (use [Long-Term Memory](#long-term-memory)). Small amounts of data where exact retrieval is sufficient.

The SDK provides two `EmbeddingClient` implementations: `MockEmbeddingClient` for testing (deterministic hash-based vectors, no API calls) and `VoyageEmbeddingClient` for production (requires the `voyage` extra). The quality of similarity retrieval depends entirely on the embedding model — `MockEmbeddingClient` produces semantically meaningless vectors.

For RAG scenarios, pre-load documents into the store before the agent runs, then pass the search tools to the agent.

Semantic memory supports two delivery modes:

- **Tools** — the agent actively decides when to store or search knowledge. Use `create_semantic_memory_tools()` to get store, search, and delete tools.
- **Context provider** — `SemanticMemoryProvider` automatically injects relevant stored knowledge before each LLM call. The provider extracts the query from the most recent user message and searches the store, presenting matches as a `[Semantic Knowledge]` block with content, similarity score, and any non-namespace metadata. Configurable `limit`, `min_score`, and `namespace`. Pair with `SemanticMemoryContributor` to teach the agent how to interpret the block.

> **See also:** [`examples/memory/semantic_memory.py`](../../examples/memory/semantic_memory.py) — semantic store operations, search with metadata, agent integration, and automatic context injection via `SemanticMemoryProvider`.

## Episodic Memory

A system for recording and recalling past experiences. Episodes capture what situation the agent faced, what action it took, and what outcome resulted (success, failure, or partial). Agents learn from past runs by recalling episodes with similar situations.

Episodic memory is most valuable for agents that perform repeated similar tasks and should improve over time. By recalling past episodes, the agent can avoid strategies that failed and favor strategies that succeeded. The `extract_episode()` helper creates episodes from agent run results, automatically inferring outcome from the termination reason and summarizing the action from the message trajectory.

**When to use:** Repeated similar tasks where the agent should learn from past attempts. When past mistakes and successes are relevant to current decision-making.

**When not to use:** One-off tasks where past experience is irrelevant. Tasks where every situation is unique.

Episodic memory supports two delivery modes:

- **Tools** — the agent actively decides when to recall or record episodes. Use `create_episodic_memory_tools()` to get recall, record, and forget tools.
- **Context provider** — `EpisodicMemoryProvider` automatically injects relevant past experiences before each LLM call. The provider extracts the query from the most recent user message and recalls similar episodes, presenting them as a `[Past Experiences]` block with situation, action, outcome, evaluator feedback (when present), and reflection.

The `extract_episode()` helper simplifies episode creation after agent runs. It infers outcome from the termination reason: `"complete"` maps to `SUCCESS`, `"iteration_limit"` / `"cancelled"` / `"evaluation_failed"` to `FAILURE`, and anything else to `PARTIAL`. The action field is summarized from the agent's message trajectory.

Recall supports filtering by outcome type, similarity score threshold, and time range via `RecallFilters`.

Without pruning, the episode store grows indefinitely. Use `max_episodes` to cap it, or call `prune_superseded()` to remove failure episodes once a success is recorded for a similar situation. Pruning removes failures only when a success exists for the same situation — this prevents the agent from seeing obsolete failure patterns.

> **See also:** [`examples/memory/episodic_memory.py`](../../examples/memory/episodic_memory.py) — Episode model, store operations, extract_episode, recall filters, pruning, agent tools, and automatic context injection.

## Shared Memory

A shared state board for multi-agent coordination. Entries are attributed (each entry records who wrote it and when), support scoping by topic, and can be superseded or retracted. Agents coordinate by reading and writing to the shared board rather than communicating directly.

Shared memory models a collaborative artifact board, not a communication channel. Each entry has a lifecycle: it starts as **active**, can be **superseded** (replaced by a newer version, preserving the original for audit) or **retracted** (marked as invalid with a reason). Only the original author can supersede or retract their entries. Default reads show only active entries; use `include_inactive=True` for the full history.

**When to use:** Multiple agents need to coordinate through shared artifacts — collaborative analysis, multi-perspective review, shared findings. Best when agents work asynchronously on a common artifact.

**When not to use:** Single-agent scenarios. When direct agent-to-agent communication (broadcast, message bus) is more appropriate. See [Multi-Agent Foundations](multi-agent-foundations.md) for messaging alternatives.

### Scopes

Scopes organize contributions by topic. An analyst might write to a `"findings"` scope while a reviewer writes to a `"review"` scope. Readers can filter by scope to see only relevant entries. When using `SharedMemoryProvider` for automatic injection, configure the `scopes` parameter to control which topics the agent sees.

### Entry Lifecycle

Entries follow a lifecycle that maintains an audit trail:

1. **Active** — Visible in default reads. Created via `write()`.
2. **Superseded** — Replaced by a newer entry. The original is preserved but hidden from default reads. Created via `supersede()`. Only the original author can supersede.
3. **Retracted** — Marked as invalid with a reason. Preserved but hidden from default reads. Created via `retract()`. Only the original author can retract.

Use `include_inactive=True` in reads to see superseded and retracted entries for audit purposes.

### Delivery Options

Shared memory can be delivered as tools (via `create_shared_memory_tools()`, with automatic author attribution) or as a context provider (`SharedMemoryProvider`) for passive read access. Combine both when the agent needs to write entries and also see the full board state automatically.

> **See also:** [`examples/memory/shared_memory.py`](../../examples/memory/shared_memory.py) — store and entry lifecycle, shared memory tools, two-agent coordination, and context provider/contributor integration.

## Combining Memory Types

Agents often benefit from multiple memory types working together. The delivery mechanism (context provider vs. tools) can differ for each.

### Working Memory + Long-Term Memory

The agent maintains structured state during a run (progress, intermediate results) via working memory as a context provider, and persists key findings for future runs via long-term memory tools. This is the most common combination for agents that do multi-step work and need cross-run persistence.

Typical pattern: working memory tracks "what I'm doing now" (context provider, always visible), long-term memory stores "what I learned" (tools, agent-initiated saves at key moments).

### Episodic Memory + Working Memory

The agent learns from past experiences (injected automatically via `EpisodicMemoryProvider`) and tracks current progress (via `WorkingMemoryProvider`). Useful for agents that repeat similar tasks and need both learning and state tracking.

Typical pattern: episodic memory provides "what worked before" at the start (context provider), working memory tracks "what I'm doing differently this time" throughout the run.

### Shared Memory + Episodic Memory

Multi-agent coordination where agents also learn from past collaborative sessions. Shared memory tools provide the coordination interface; episodic memory (as provider or tools) enables each agent to recall what worked in previous sessions.

Typical pattern: shared memory provides the collaboration surface (tools for writing, optionally provider for reading), episodic memory provides individual learning (provider injects relevant past sessions).

### Semantic Memory + Long-Term Memory

The agent has both explicit fact storage (long-term memory for known keys like `project_config`) and similarity-based search (semantic memory for finding related documents). Use long-term memory for structured data the agent creates, and semantic memory for searching a pre-loaded knowledge base.

### Context Budget Considerations

Each context provider adds content to every LLM call. Multiple memory providers can consume significant context. Monitor `ContextUsage` and consider which providers truly need to be automatic vs. tool-based.

A general principle: if the agent needs the information on most steps, use a context provider. If it needs it occasionally, use tools. When using multiple providers, consider the combined size — working memory content, past episodes, and a shared memory board can collectively take up a large portion of the context window.

See [Context Management](context-management.md) for strategies to manage context budget across all sources.

## Pitfalls

**Working memory is full replacement.** Each `<working_memory>` block replaces the entire content. If the agent omits a section, that section is lost. The system prompt instructions emphasize this, but less capable models may still drop sections.

**Long-term memory keys must be descriptive.** The agent uses `list_memory_keys` to discover what's stored. Keys like `data1`, `temp` are useless — use `user_preferred_language`, `project_tech_stack`.

**Semantic similarity depends on embedding quality.** `MockEmbeddingClient` produces deterministic but semantically meaningless vectors. For real similarity-based retrieval, use `VoyageEmbeddingClient` or another production embedding client.

**Episodic memory accumulates.** Without pruning, the episode store grows indefinitely. Use `max_episodes` to cap it, or call `prune_superseded()` to remove obsolete failure episodes.

**Shared memory is not a message bus.** It's a shared artifact board, not a communication channel. For real-time agent-to-agent messaging, see [Multi-Agent Foundations](multi-agent-foundations.md).

**Context budget.** Each context provider adds content to every LLM call. Multiple memory providers can consume significant context. Monitor `ContextUsage` and consider which providers truly need to be automatic vs. tool-based.

**Working memory reset on run start.** `ReActAgent` calls `working_memory.reset()` at the start of every run. Pre-populated data is lost before the first LLM call unless you override `reset()` in a custom implementation or use a context provider to inject initial state.

## Custom Implementations

All memory types are defined as Python protocols. To build a production-backed store, implement the corresponding protocol with your persistence layer (PostgreSQL, Redis, etc.). The in-memory implementations (`InMemoryWorkingMemory`, `InMemoryLongTermStore`, `InMemorySemanticStore`, `InMemoryEpisodeStore`, `InMemorySharedMemory`) serve as reference implementations and are suitable for testing.

See the protocol definitions and their docstrings in the source code for the exact method signatures and semantics.
