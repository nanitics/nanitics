# Multi-Agent Foundations

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

> **Namespace:** `AgentTool`, the four context-transfer strategies (`RawOutputTransfer`/`TrajectoryTransfer`/`SummaryTransfer`/`CustomTransfer`), and `Broadcast` live in top-level `nanitics`. The structured handoff stack — `HandoffPayload`, `HandoffTransfer`, `HandoffStep`, `create_handoff_chain`, `handoff_sender_instructions`, `handoff_receiver_instructions` — lives in `nanitics.patterns`. The reactive and peer-consultation topologies — `MessageBus` and `PeerNetwork` with their supporting types — live in `nanitics.experimental`.

Multi-agent systems let you split work across specialized agents that communicate and share context. Instead of building one agent that does everything, you compose agents with different expertise, tools, and system prompts — then connect them through communication patterns.

This guide covers the foundational building blocks: agent-as-tool delegation, context transfer between agents, structured handoff protocols, and three communication patterns (broadcast, message bus, peer network). For higher-level coordination patterns like orchestrator, supervisor, blackboard, bidding, debate, and consensus, see [Multi-Agent Coordination](multi-agent-coordination.md).

All primitives emit trace events through `EventEmitter`, making multi-agent interactions visible in the [Observatory](observability.md#observatory) trace viewer. Each section below notes which events are emitted.

## When to Use Multi-Agent

**Use multi-agent when** your task naturally decomposes into distinct domains that benefit from separate system prompts, tools, or reasoning styles. Examples: a researcher agent that feeds findings to a writer agent, a code generator that delegates testing to a review agent, or a team of domain experts that consult each other.

**Don't use multi-agent when** a single agent with the right tools can handle the task. Multi-agent adds communication overhead, context transfer costs, and debugging complexity. Start with one agent and split only when you see clear gains from specialization.

### Pattern Progression

Most multi-agent systems evolve through increasing complexity. Before walking that progression, check the pre-pattern: a single LLM-driven agent emitting typed structured output (e.g., a `ReasoningAgent` with an `output_schema`), followed by deterministic Python that consumes the typed output and dispatches to one of several outcomes. If the second stage is a pure function of its typed input — no further LLM calls, no shared state, no concurrency primitive needed — then no multi-agent pattern is warranted. The agent provides judgment; the dispatcher provides routing. See [`examples/agents/dispatch_over_structured_output.py`](../../examples/agents/dispatch_over_structured_output.py) for the canonical shape.

Otherwise, start with the simplest multi-agent pattern that meets your needs:

1. **Single agent** — one agent with all the tools. Start here.
2. **Agent-as-Tool** — add delegation when a subtask needs different tools or system prompt. The coordinator retains control.
3. **Handoff chain** — move to sequential handoff when agents need to build on each other's work rather than report back to a coordinator.
4. **Broadcast** — add parallel execution when you need multiple perspectives or redundancy on the same task.
5. **Message bus** — switch to pub/sub when the flow of work is reactive and event-driven rather than predetermined.
6. **Peer network** — use peer consultation when agents need bidirectional, ad-hoc access to each other's expertise.

Each step adds communication complexity. Only advance when the simpler pattern genuinely can't express your workflow.

## Decision Guide

The table below compares all six foundational primitives. The coordination patterns (orchestrator, supervisor, etc.) build on top of these — see [Multi-Agent Coordination](multi-agent-coordination.md).

| Need | Pattern | Communication | Direction | State Sharing |
|------|---------|---------------|-----------|---------------|
| One agent delegating subtasks | [Agent-as-Tool](#agent-as-tool) | Synchronous call-response | Caller → delegate → caller | Via `ContextTransferStrategy` |
| Passing state between sequential agents | [Context Transfer](#context-transfer) | Strategy-driven extraction | A → B (one-way) | Configurable fidelity |
| Structured handoff in a workflow | [Handoff Protocol](#handoff-protocol) | Structured payload | A → B → C (chain) | `HandoffPayload` fields |
| Fan-out task to multiple agents | [Broadcast](#broadcast) | Parallel fan-out | One → many | Aggregated via strategy |
| Pub/sub reactive messaging | [Message Bus](#message-bus) | Topic-based async | Many ↔ many | Messages on topics |
| Peer-to-peer consultation | [Peer Network](#peer-network) | Ad-hoc tool calls | Any ↔ any | Shared invocation budget |
| Dynamic delegation to specialists | Orchestrator | [Coordination](multi-agent-coordination.md) | | |
| Runtime quality/budget monitoring | Supervisor | [Coordination](multi-agent-coordination.md) | | |
| Agents coordinating through shared state | Blackboard | [Coordination](multi-agent-coordination.md) | | |
| Competitive task allocation | Bidding | [Coordination](multi-agent-coordination.md) | | |
| Comparative task allocation | JudgeRouter | [Coordination](multi-agent-coordination.md) | | |
| Adversarial reasoning | Debate | [Coordination](multi-agent-coordination.md) | | |
| Collective agreement | Consensus | [Coordination](multi-agent-coordination.md) | | |

## Agent-as-Tool

`AgentTool` wraps an agent behind the `Tool` protocol, letting a calling agent delegate subtasks through normal tool use. The caller describes what it needs, the delegate agent runs autonomously, and the result flows back as a tool response. The delegate receives a single `task` parameter (the caller's description of what to do) and returns a `ToolResult` with the extracted output plus execution metadata.

Use Agent-as-Tool when a coordinator needs on-demand access to a specialist — it's the simplest multi-agent pattern, requiring no workflow setup. The coordinator decides *when* to delegate through its normal reasoning loop, making this pattern ideal for optional or conditional delegation. It differs from handoff in that the calling agent retains control and receives the result within the same conversation turn.

The delegate's output is extracted using a `ContextTransferStrategy` (default: `RawOutputTransfer`). You can swap this for `SummaryTransfer` or `CustomTransfer` to control what flows back to the caller — see [Context Transfer](#context-transfer) for the trade-offs.

`AgentTool` emits a `DelegationEvent` linking caller and delegate in the trace. The delegate runs as a child span, so its full execution (LLM calls, tool use, reasoning steps) is visible in the trace tree.

### Agent-as-Tool vs Handoff

These two patterns solve different problems. Agent-as-Tool is for **on-demand delegation within a conversation**: the caller asks a question, gets an answer, and continues reasoning. Handoff is for **sequential pipeline stages**: each agent does its part and passes everything forward. Choose Agent-as-Tool when the coordinator should retain control and use the delegate's output as one input among many. Choose handoff when you're building a pipeline where each stage transforms or builds on the previous stage's complete output.

> **See also:** [examples/multi_agent/agent_tool.py](../../examples/multi_agent/agent_tool.py)

## Context Transfer

When information flows between agents — through delegation, handoff, or orchestration — a `ContextTransferStrategy` controls what the receiving agent sees. Every strategy takes an `AgentResult` and produces a string for the next agent. The choice of strategy is a trade-off between fidelity and cost.

Context transfer is a cross-cutting concern rather than a standalone pattern. You configure it on `AgentTool`, `HandoffStep`, and other multi-agent constructs. It answers the question: "What does the next agent need to know about what just happened?"

All strategies implement the same protocol: `async def extract(self, result: AgentResult) -> str`. You can create custom strategies by implementing this protocol or by using `CustomTransfer` with a function.

### Strategy Comparison

| Strategy | Cost | Fidelity | Best For |
|----------|------|----------|----------|
| `RawOutputTransfer` | None | Low | Default; when only the final answer matters |
| `TrajectoryTransfer` | None | High | When the receiver needs to understand the full reasoning process |
| `SummaryTransfer` | 1 LLM call | Medium | When trajectory is too large but reasoning context matters |
| `CustomTransfer` | Varies | Custom | Structural extraction (parse JSON, extract specific fields) |

### When to Use Each Strategy

**RawOutputTransfer** passes the agent's final output string. Use this as the default — it's free and sufficient when the next agent only cares about the result, not the reasoning. This is the default on `AgentTool` and the forced strategy for the last step in a handoff chain.

**TrajectoryTransfer** formats the full message history including tool calls and assistant reasoning. Use it when the receiving agent needs to understand *how* the previous agent reached its conclusion — for example, a reviewer agent that needs to verify reasoning steps. Be aware this can be large: a 10-step agent with tool calls may produce a trajectory that consumes a significant portion of the receiver's context window.

**SummaryTransfer** uses an LLM to compress the conversation while preserving key findings and decisions. The compression ratio is significant — a 10-step trajectory might compress to a few paragraphs. Wrapping its LLM client with `InstrumentedLLMClient` makes the summarization call visible in traces (see [Observability](observability.md)). The trade-off is one extra LLM call per transfer.

**CustomTransfer** takes a user-defined extraction function `(AgentResult) -> str` for full control. Use it when you need to extract structured data — for example, parsing JSON from the agent's output, extracting specific fields, or combining output with metadata like step count or tool usage statistics.

> **See also:** [examples/multi_agent/context_transfer.py](../../examples/multi_agent/context_transfer.py)

## Handoff Protocol

The handoff protocol provides structured context transfer between agents in a sequential workflow. While `ContextTransferStrategy` handles raw extraction, the handoff protocol adds structure on top: a `HandoffPayload` data model with semantic fields (`task_state`, `findings`, `decisions`, `open_questions`, `artifacts`, `metadata`), a `HandoffTransfer` strategy that builds payloads from results, workflow integration via `HandoffStep`, and prompt helpers that guide agents to produce and consume structured handoffs.

Use handoff when agents work in sequence and each needs to understand what the previous agent accomplished, decided, and left unresolved. It's more structured than raw context transfer — the receiving agent gets semantically labeled sections rather than unstructured text. This makes handoff ideal for multi-stage pipelines (research → write → review) where each stage builds on the previous one's findings.

### Key Components

- **`HandoffPayload`** — a structured data model that renders to markdown sections. Contains `task_state`, `findings`, `decisions`, `open_questions`, `artifacts`, and `metadata`. Call `.render()` to produce a markdown document for the receiving agent. The semantic fields make it easy for the receiving agent to locate specific information — "what was decided" vs "what's still open" — rather than parsing unstructured text.
- **`HandoffTransfer`** — a `ContextTransferStrategy` implementation that takes a builder function `(AgentResult) → HandoffPayload`, giving you control over how agent results map to structured payloads. Use this when you want the structure of `HandoffPayload` with custom extraction logic.
- **`HandoffStep`** — a workflow `Step` that runs an agent and applies context transfer, emitting a `HandoffEvent` on completion. The event includes `from_agent`, `to_agent`, `payload_fields`, and `payload_size` for trace visibility.
- **`create_handoff_chain`** — a factory that builds a `Sequential` workflow from a list of agents, connecting each pair with a shared transfer strategy. Requires at least 2 agents. The last step always uses `RawOutputTransfer` since its output is the final result.

### Prompt Helpers

`handoff_sender_instructions()` and `handoff_receiver_instructions()` generate system prompt sections that guide agents to produce and consume structured payloads. The sender instructions accept a `payload_fields` parameter to customize which fields the agent should include. Add these to your agent system prompts to teach agents the handoff format without hardcoding it.

These helpers are optional but recommended for handoff chains — without them, agents may not structure their output in a way that maps cleanly to `HandoffPayload` fields.

> **See also:** [examples/multi_agent/handoff.py](../../examples/multi_agent/handoff.py)

## Broadcast

`Broadcast` sends a task to multiple agents in parallel, collects their responses, and aggregates results using a configurable `ResponseStrategy`. Use it when you want multiple perspectives on the same problem, need redundancy, or want to pick the best answer from several attempts.

Broadcast differs from the other patterns in that all agents work on the *same* task independently rather than collaborating or building on each other's work. This independence is its strength — agents can't bias each other, and you get genuinely diverse outputs. Agents that raise exceptions are captured in `failures` rather than crashing the broadcast — callers can inspect `result.failures` to handle partial failures gracefully.

### Response Strategies

The aggregation strategy determines what you get back from the parallel execution:

| Strategy | Behavior |
|----------|----------|
| `CollectAll` | Returns all outputs as a list (default) |
| `SelectBest` | Picks the highest-scoring response via a user-defined scorer (sync or async) |
| `MergeResponses` | Uses an LLM to synthesize all responses into a unified answer |
| `FilterResponses` | Keeps only responses matching a predicate |

Choose `CollectAll` when you need all perspectives (e.g., gathering diverse opinions). Choose `SelectBest` when quality varies and you want the best one — the scorer can be a simple heuristic or an async LLM-based evaluator. Choose `MergeResponses` when you want a synthesis that combines insights from all agents. Choose `FilterResponses` when only some responses meet a quality threshold.

### Eligibility Filters

Not every agent needs to participate in every broadcast. `EligibilityFilter` controls which agents receive the task:

- **`AllEligible`** (default) — every agent participates
- **`CapabilityFilter`** — selects agents whose registered capabilities overlap with the required set, so you can route tasks to agents with relevant expertise without changing the broadcast call site

Broadcast emits `BroadcastStartEvent`, `BroadcastResponseEvent` (per agent), and `BroadcastCompleteEvent`. The complete event includes failure information for any agents that raised exceptions.

> **See also:** [examples/multi_agent/broadcast.py](../../examples/multi_agent/broadcast.py)

## Message Bus

`MessageBus` is a topic-based publish-subscribe communication layer for reactive, event-driven multi-agent systems. Agents subscribe to topics via `TopicSubscription`, and when a message is published, all matching subscribers process it concurrently. Subscribers can publish new messages during processing, creating chains of reactive communication that continue until quiescence or a termination condition fires.

Use message bus when agents need to react to events rather than being explicitly invoked — it's the right pattern for workflows where the flow of control emerges from the data rather than being predetermined. Unlike broadcast (same task, all agents), the message bus routes different messages to different agents based on topic subscriptions. Unlike handoff (linear A → B → C), the bus supports complex, non-linear message flows where any agent can trigger any other.

The message bus is the most complex foundational primitive. Before reaching for it, consider whether a simpler pattern (handoff chain, broadcast) can express your workflow. The bus is justified when you need reactive, event-driven behavior where agents respond to conditions rather than following a predetermined sequence.

### How It Works

1. Seed messages are placed in a queue
2. For each message, the bus finds subscribers for that topic
3. Subscribers execute concurrently, each receiving a `publish_message` tool and a `MessageHistoryProvider` that injects recent messages into context
4. Published messages are added to the queue with incremented depth
5. Processing continues until quiescence, a termination condition fires, or safety bounds are hit

When the bus triggers an agent, it temporarily augments it with the `publish_message` tool, a `MessageHistoryProvider` for conversational context, and a `MessageBusContributor` system prompt section explaining the bus topology. These augmentations are removed after the agent completes, leaving no side effects on agent configuration.

By default, a subscriber does not receive its own publish — a subscriber whose `agent.name` equals the message `author` is excluded from delivery for that message, so an agent that subscribes to a topic it also publishes to will not re-trigger itself. For legitimate reactive patterns where an agent should consume its own publish (polling loops, self-reflection chains, broadcast-to-self echo topologies), construct the bus with `MessageBus(..., allow_self_delivery=True)` to restore the pre-fix delivery shape.

### Safety Bounds and Termination

Without termination conditions, reactive agents can create infinite message chains. The bus provides multiple safety layers:

- **`max_messages`** — hard cap on total messages (default: 100)
- **`max_depth`** — maximum message chain depth (default: 10)
- **`BusTerminationCondition`** — protocol for custom conditions. Built-in: `MaxMessagesTermination`, `MaxExecutionsTermination`, `BusPredicateTermination`
- **`BusCompositeTermination`** — combines multiple conditions with `"any"` or `"all"` mode

`TopicSubscription` supports an optional `MessageFilter` to control which messages on a topic actually trigger the subscriber — useful for routing urgent messages to specific agents or filtering out noise.

Subscribers that raise exceptions are captured in `failed_executions` rather than stopping the bus. The bus result includes `termination_reason` indicating whether it stopped due to quiescence, max messages, or a termination condition.

Each message carries a `depth` field tracking how many generations of reactive messaging produced it. The `parent_message_id` field links each published message back to the message that triggered it, enabling causal chain analysis in traces.

The bus emits `MessageBusStartEvent`, `MessagePublishedEvent`, `MessageDeliveredEvent`, and `MessageBusCompleteEvent`.

> **See also:** [examples/multi_agent/message_bus.py](../../examples/multi_agent/message_bus.py)

## Peer Network

`PeerNetwork` creates a group of agents that can consult each other through peer-to-peer tool calls. Each agent gets `consult_<peer_name>` tools for every other peer, and all peers share an invocation budget that bounds total cross-agent communication. You start execution from a specific agent via `network.run(agent_name, task)`, and that agent can consult peers as needed through its normal reasoning loop.

Use peer network when agents need ad-hoc, bidirectional consultation — where the need for collaboration emerges during reasoning rather than being predetermined. Unlike the message bus (topic-based, reactive), peer consultation is direct and intentional: an agent decides it needs another agent's expertise and asks for it. This makes it ideal for teams of domain experts (financial analyst + legal advisor + technical reviewer) working on problems that cross domain boundaries.

The key advantage of peer network over other patterns is that consultation is **agent-initiated**: the LLM decides when to consult based on its reasoning, rather than being triggered by external events or predetermined workflow steps. This means the consultation pattern adapts to the problem at hand.

### How It Works

1. Each `PeerSpec` defines a peer with its own LLM, prompt, tools, and description. Its `allowed_peers` field declares which peers this one can consult — defaulting to "all other peers" when unset, or narrow it to an explicit list (e.g., `["strategist"]`) to constrain the graph, or set it to `[]` to mark a leaf consultant with no downstream peers
2. `PeerNetwork` creates `ReActAgent` instances, injecting a `consult_<peer>` tool for each allowed peer and augmenting the system prompt with the matching roster
3. When a peer uses a consultation tool, the target peer runs with the consultation message
4. The shared `InvocationBudget` tracks total consultations — when exhausted, `PeerBudgetExceededError` instructs the agent to produce its final answer

The consultation graph is declared structurally via `allowed_peers`, not enforced at the prompt layer. A peer never gets a `consult_<other>` tool unless its `allowed_peers` list (or the default) includes that peer, and a peer is never given a tool to consult itself. This mirrors the SDK's capability-over-prescription principle: the primitive's contract is expressed in the tool-belt the agent can see, not in system-prompt instructions the LLM may or may not follow.

### Budget Control

The `max_invocations` parameter prevents runaway recursion (A consults B, which consults A, which consults B...). This is a shared budget across the entire network — all consultations from all peers count against the same limit.

Set the budget based on expected consultation depth. For a simple two-peer setup where each peer might consult once, 5–10 invocations suffice. For a larger network where agents might need multi-hop consultations (A → B → C → A), set it higher but monitor `PeerConsultationEvent`s to understand actual usage patterns.

The network emits `PeerNetworkStartEvent`, `PeerConsultationEvent`, and `PeerNetworkCompleteEvent`.

> **See also:** [examples/multi_agent/peer_network.py](../../examples/multi_agent/peer_network.py)

## Combining Patterns

These building blocks compose naturally. Some common combinations:

**Agent-as-tool + context transfer** — The most common starting point. Use `AgentTool` with `SummaryTransfer` or `CustomTransfer` when a coordinator delegates to specialists but needs compressed or structured results rather than raw output.

**Handoff chain as pipeline** — `create_handoff_chain` produces a `Sequential` workflow, which you can compose with other workflow patterns from [Orchestration](orchestration.md). Use `SummaryTransfer` between stages to keep context costs manageable in long pipelines.

**Broadcast + SelectBest** — Broadcast a task to multiple agents and pick the best response using an LLM scorer. Effective for tasks where quality varies significantly between attempts, such as creative writing or complex reasoning.

**Broadcast + MergeResponses** — When you want a synthesis rather than selection. Each agent contributes a perspective, and an LLM combines them into a unified answer that's better than any individual response.

**Message bus + peer network** — Use the message bus for structured topic-based communication flow and peer network for ad-hoc consultation within bus-triggered agents. The bus provides the macro-level workflow; peer consultation handles micro-level expertise sharing.

**Agent-as-tool inside orchestration** — Higher-level coordination patterns (orchestrator, supervisor) from [Multi-Agent Coordination](multi-agent-coordination.md) use `AgentTool` internally to delegate to specialists. You can customize the transfer strategy on each `AgentTool` to control how specialist results flow back to the coordinator.

### Pattern Interactions

When combining patterns, be mindful of how they interact:

- **Context transfer compounds** — in a handoff chain where each step uses `SummaryTransfer`, you're paying one LLM call per handoff. In a 5-agent pipeline, that's 4 extra LLM calls. Consider whether `RawOutputTransfer` suffices for intermediate steps.
- **Budget sharing** — peer network budgets are independent of message bus message limits. If you embed a peer network inside a bus subscriber, the bus's `max_messages` and the network's `max_invocations` are separate safety bounds.
- **Event visibility** — all patterns emit events through the same `EventEmitter`. In combined setups, the trace tree shows the full hierarchy: bus message → subscriber execution → peer consultation → delegate agent. This makes debugging feasible even in complex topologies.

## Pitfalls

**Over-decomposition.** Splitting into too many agents wastes tokens on context transfer and coordination. Each agent boundary has cost — the context transfer overhead, the loss of shared reasoning history, and the additional LLM calls. Start with fewer, larger agents and split only when you see clear domain boundaries that benefit from separate system prompts or tools.

**Context loss.** `RawOutputTransfer` loses the reasoning history. If the next agent needs to understand *why* a decision was made, use `TrajectoryTransfer` or `SummaryTransfer`. This is especially important in handoff chains where later agents may need to revisit earlier decisions.

**Runaway message buses.** Without termination conditions, reactive agents can create infinite message chains. Always set `max_messages`, `max_depth`, and ideally a `BusTerminationCondition`. Monitor `MessagePublishedEvent` counts during development to understand message amplification patterns.

**Budget exhaustion in peer networks.** Agents may exhaust the consultation budget before reaching a good answer. Set `max_invocations` based on expected consultation depth and monitor `PeerConsultationEvent`s. If agents consistently hit the budget, the task may need restructuring — either fewer peers, more focused consultation prompts, or a different pattern entirely.

**Agent name collisions.** Peer names, tool names, and bus topics form namespaces. Duplicate names cause routing errors or silent overwrites. Use descriptive, unique names.

**Mixing patterns without purpose.** Each communication pattern adds debugging complexity. Don't use broadcast + message bus + peer network when a simple handoff chain would suffice. Pick the simplest pattern that expresses your workflow (see [Pattern Progression](#pattern-progression)).

## See Also

- [Multi-Agent Coordination](multi-agent-coordination.md) — higher-level patterns (orchestrator, supervisor, blackboard, bidding, debate, consensus) built on these foundations
- [Orchestration](orchestration.md) — workflow composition patterns (sequential, parallel, DAG, loop, map-reduce, conditional)
- [Observability](observability.md) — tracing and event monitoring for multi-agent systems
