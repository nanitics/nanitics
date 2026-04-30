# Architecture Guide

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Designing an agent system means making a sequence of interdependent decisions: which agent type, what tools, how memory works, whether you need multiple agents, and how they coordinate. This guide walks you through those decisions in order, from requirements to working architecture.

See [building-applications.md](building-applications.md) for backend implementation patterns once you have chosen your SDK components. Want to understand how the SDK itself is put together? See [SDK Internals](../architecture/sdk-internals.md).

If you already know what you need, jump to the specific guide. If you're starting from scratch or evaluating whether the SDK fits your use case, start here.

## Starting from Requirements

Before choosing any component, answer these questions about your task:

1. **Single action or multi-step process?** A classification task is one LLM call. A research task might be dozens of tool calls across multiple agents.
2. **One domain or multiple specialized domains?** If the task requires fundamentally different expertise (research vs. writing vs. code review), you likely need multiple agents.
3. **Does the agent need external actions?** Tool calls, API requests, database queries, code execution — or is it pure reasoning?
4. **How important is output quality?** Can you accept first-pass output, or do you need automated quality gates?
5. **Does the agent need to learn from past runs?** Repeated similar tasks benefit from episodic memory.
6. **Does it need persistent state?** Facts, documents, or shared artifacts that survive across runs.
7. **Will it generate or execute code?** Code execution requires sandboxing and a specialized agent type.
8. **Does a human need to approve actions or review output?** High-stakes decisions may require human-in-the-loop integration.
9. **How long will runs take?** Long-running agents need context management, and human approval flows may need durable execution.

Your answers determine which SDK components you need and how they compose.

## Decision Sequence

Work through these decisions in order. Each builds on the previous ones.

### 1. Agent Type

This is the most important decision. See [Agent Types](agent-types.md) for detailed guidance.

**Quick decision flow:**

```
Does the task need tool calls?
├─ No → ReasoningAgent (single LLM call, structured output)
└─ Yes
   ├─ Well-defined steps, minimize LLM calls? → ReWOOAgent
   ├─ Code is the natural expression? → CodeActAgent
   ├─ Repeated task, should learn from mistakes? → ReflexionAgent
   ├─ Single clear solution path? → ReActAgent (the default)
   └─ Multiple approaches worth exploring?
      ├─ Moderate complexity → TreeOfThoughtAgent
      └─ Extreme complexity → LATSAgent
```

**Rule of thumb:** Start with `ReActAgent`. It handles the widest range of tasks. Switch to a specialized type only when you hit a specific limitation — too many LLM calls (ReWOO), need structured output without tools (Reasoning), need learning across runs (Reflexion), or need to explore multiple solution paths (TreeOfThought/LATS).

### 2. Tool Design

Tools are how agents interact with the world. See [Tools](tools.md) for implementation details.

**Design principles:**
- **One tool, one action.** `search_documents` and `update_document` are better than `manage_documents`. The LLM needs clear, distinct options.
- **Descriptive names and descriptions.** The LLM reads tool descriptions to decide when to use them. Vague descriptions lead to wrong tool selections.
- **Return useful errors.** When a tool fails, the error message should help the agent recover. "Document not found: xyz" is better than "Error".
- **Limit output size.** Tools that return thousands of lines overwhelm the context window. Summarize, paginate, or filter.

**How many tools?** Most agents work well with 3–10 tools. More than 15 and the LLM starts making poor selection decisions. If you need more, split across multiple specialized agents.

**Tool state** lets you inject per-run dependencies (database connections, API clients) into tools without globals. Pass a `dict[str, Any]` as `tool_state` and access values via `ctx.state["key"]`:

```python
from nanitics import ReActAgent, tool, ToolContext

@tool(name="query_db", description="Run a database query")
async def query_db(query: str, ctx: ToolContext) -> str:
    result = await ctx.state["db"].execute(query)
    return str(result)

agent = ReActAgent(
    name="analyst",
    llm_client=llm,
    emitter=emitter,
    tools=[query_db],
    tool_state={"db": conn, "api_key": "..."},
)
```

> **Note:** `tool_state` is supported by `ReActAgent` and `LATSAgent` only.

### 3. Memory Architecture

Memory determines what the agent knows beyond the current conversation. See [Memory](memory.md) for the full guide.

**Decision table:**

| Question | Memory Type | Delivery |
|----------|-------------|----------|
| Need structured state during a run? (progress, lists, notes) | Working Memory | Auto-injected via context provider |
| Need to store and retrieve facts by key? | Long-Term Memory | Agent uses tools |
| Need similarity-based search over documents? | Semantic Memory | Agent uses tools |
| Need to learn from past successes and failures? | Episodic Memory | Auto-injected or agent uses tools |
| Multiple agents sharing state? | Shared Memory | Auto-injected or agent uses tools |

**Combining memory types** is common:
- Working Memory + Long-Term: structured state within a run, persistent facts across runs
- Semantic Memory + Episodic Memory: RAG retrieval plus learning from experience
- Shared Memory + Working Memory: multi-agent coordination where each agent also tracks its own state

**When to skip memory:** Simple, stateless tasks that complete in a few steps don't need dedicated memory systems. The conversation history is sufficient.

### 4. Error Handling

Agents fail. Tools return errors, LLMs produce bad output, APIs go down. See [Error Handling](error-handling.md) for implementation details.

**Three error categories drive your strategy:**

| Category | Examples | Behavior with `ErrorHandler()` |
|----------|----------|------------------|
| Retryable | Rate limits, timeouts, transient network errors | Exponential backoff retry |
| Correctable | Bad tool parameters, malformed output | Error fed back to LLM for self-correction |
| Fatal | Authentication failure, missing required resource | Agent stops with error |

**All agents use `ErrorHandler()` by default,** providing retry with backoff, self-correction prompts, and graceful degradation. Pass `ErrorHandler.fail_fast()` to disable resilience during development or testing. Custom error handling is worth it when you need: specific retry policies, custom error classification, graceful degradation to fallback behavior, or integration with external monitoring.

### 5. Context Management

Long-running agents accumulate messages that approach the context window limit. See [Context Management](context-management.md).

**When you need it:** Any agent that might run for more than ~20 steps, or that processes large tool outputs.

**Two strategies, composable:**

| Strategy | Speed | Information Loss | When to Use |
|----------|-------|-----------------|-------------|
| Truncation | Fast | Moderate (drops old messages, preserves system prompt and N most recent) | Quick operations, cost-sensitive |
| Summarization | Slow (LLM call) | Low (compresses, preserves meaning) | Long-running, accuracy-critical |

**When to skip:** Short-lived agents (< 10 steps) with small tool outputs rarely hit context limits.

### 6. Evaluation

Evaluation lets you gate agent output on quality criteria before accepting it. See [Evaluation](evaluation.md).

**Use evaluation when:** Output quality matters and revision can improve it (e.g., generated content, structured data extraction, multi-step analysis).

**Three evaluator types, from cheapest to most capable:**

| Evaluator | Cost | Use For |
|-----------|------|---------|
| `ProgrammaticEvaluator` | Free | Format checks, length constraints, required fields |
| `LLMEvaluator` | 1 LLM call | Quality criteria, coherence, accuracy against rubric |
| `CompositeEvaluator` | Combined | Chain programmatic (fast reject) → LLM (quality gate) |

The evaluation loop: agent produces output → evaluator judges → accept (done), revise (agent tries again with feedback), or reject (fail). Configure `max_revisions` to bound the loop.

**When to skip:** Tasks where first-pass output is acceptable, or where human review replaces automated evaluation.

### 7. Planning

Explicit planning helps agents track progress through multi-step tasks. See [Planning](planning.md).

**Use planning when:** The task has multiple ordered steps, the agent runs long enough to lose track of progress, or you want to track completion programmatically.

**Planning strategies:**

| Strategy | Behavior |
|----------|----------|
| `AdaptivePlanningContributor` | Create plan → execute → revise as needed |
| `UpfrontPlanContributor` | Create complete plan first, then execute |
| `DecompositionContributor` | Break complex problem into subproblems |
| `GoalTrackingContributor` | Maintain a hierarchy of goals and subgoals |

`PlanningCapability` bundles planning tools, context injection, and plan-aware evaluation into a single configuration object.

**When to skip:** Short tasks, tasks where the agent naturally sequences its actions without explicit planning.

### 8. Single vs. Multi-Agent

The most over-engineered decision in agent systems. See [Multi-Agent Foundations](multi-agent-foundations.md) and [Multi-Agent Coordination](multi-agent-coordination.md).

**Start with a single agent.** Split into multiple agents only when you see clear gains from specialization — different domains need different tools, system prompts, or reasoning styles.

**Signs you need multi-agent:**
- The system prompt is trying to cover too many roles
- Tool count exceeds ~15 and the agent picks the wrong ones
- Different parts of the task need fundamentally different reasoning (e.g., code generation vs. code review)
- You need adversarial examination of alternatives (debate)
- You need collective agreement from multiple perspectives (consensus)

**Communication patterns** (how agents exchange information):

| Pattern | Use When |
|---------|----------|
| Agent-as-Tool | One agent delegates subtasks to another |
| Handoff | Sequential agents passing structured context |
| Broadcast | Same task sent to many agents for comparison |
| Message Bus | Pub/sub messaging, loose coupling |
| Peer Network | Agents consulting each other as needed |

**Coordination patterns** (how agent work is managed):

| Pattern | Use When |
|---------|----------|
| Orchestrator | Central coordinator delegates to specialists dynamically |
| Supervisor | Runtime monitoring with quality/budget triggers |
| Blackboard | Agents coordinate through shared state |
| Bidding | Competitive task allocation based on self-assessed capability |
| JudgeRouter | Comparative task allocation via single-call ranking |
| Debate | Adversarial reasoning to examine alternatives |
| Consensus | Collective agreement through voting or deliberation |

### 9. Orchestration

Orchestration composes multiple steps into workflows. See [Orchestration](orchestration.md).

**The 8 workflow patterns:**

| Pattern | Use When |
|---------|----------|
| `Sequential` | Steps must run in order |
| `Parallel` | Independent steps that can run concurrently |
| `Pipeline` | Steps where each output becomes the next input, with type contracts |
| `Conditional` | Branch based on a previous step's result |
| `Loop` | Repeat until a condition is met |
| `MapReduce` | Apply the same step to multiple inputs, then aggregate |
| `DAG` | Complex dependency graphs between steps |
| `plan_to_workflow` | Convert a planning `TaskPlan` into an executable workflow |

**Step adapters** connect agents and functions to workflows:
- `AgentStep` wraps an agent: input becomes the task string, output is the agent's response
- `FunctionStep` wraps an async function: direct input/output mapping

All patterns support cancellation, checkpoint-based suspension/resumption, and observability events.

**When to skip orchestration:** If a single agent can handle the task, don't add workflow structure. Orchestration is for when work naturally decomposes into distinct phases.

### 10. Human-in-the-Loop

HITL integration lets humans participate in agent execution — reviewing output, approving actions, or providing guidance. See [Human-in-the-Loop](human-in-the-loop.md).

**Three levels of human involvement:**

| Level | Mechanism | Use When |
|-------|-----------|----------|
| Agent-initiated | `create_hitl_tools()` — agent asks for input | Agent needs guidance on ambiguous decisions |
| Tool-level approval | `ApprovalWrappedTool` — human approves before execution | Specific high-risk tools need oversight |
| Workflow-level gates | `ApprovalGate` / `RevisionGate` — approval between workflow steps | Checkpoints between phases of work |

**Durable execution:** If a human might take minutes or hours to respond, the process can't block. Durable HITL suspends the agent, saves a checkpoint, and resumes when the response arrives. This requires implementing `CheckpointStore` and `HitlRequestStore` — see [Building Applications](building-applications.md).

### 11. Safety

Safety constraints bound agent execution. See [Safety](safety.md).

**Always configure these in production:**

| Agent Type | Bounding Parameter | Default |
|------------|-------------------|--------|
| `ReActAgent`, `CodeActAgent` | `max_iterations` | 10 |
| `LATSAgent` | `max_iterations` | 20 |
| `ReflexionAgent` | `max_attempts` | 3 |
| `TreeOfThoughtAgent` | `max_depth` + `max_nodes` | 5, 50 |
| `ReWOOAgent`, `ReasoningAgent` | — (bounded by task completion) | — |

All agent types accept `CancellationToken` for external stop signals. `CodeActAgent` requires a `Sandbox` for code isolation.

```python
from nanitics import ReActAgent, CancellationToken

agent = ReActAgent(
    name="bounded-agent",
    llm_client=llm,
    emitter=emitter,
    tools=[...],
    max_iterations=25,
    cancellation_token=CancellationToken(),
)
```

### 12. Observability

Every agent requires an `EventEmitter`. See [Observability](observability.md).

`InMemoryEmitter` collects events for debugging and testing. For production, use `TracedExecutor` to manage run lifecycle with real-time persistence — it composes `InMemoryEmitter`, `TraceCollector`, and `PersistentTraceStore` into a single call. For lower-level control, wire `TraceCollector` directly. Mount `create_trace_router()` for query endpoints. Events are classified into three levels (info, debug, verbose) for filtered streaming and storage.

Events cover the entire lifecycle: agent steps, LLM calls, tool invocations, memory operations, workflow transitions, multi-agent communication, and HITL interactions.

## Example Architectures

### Simple Tool-Calling Agent

**Requirements:** Answer user questions using a search API. Short-lived, no persistence needed.

```python
import asyncio
from nanitics import ReActAgent, AnthropicLLMClient, InMemoryEmitter, tool

@tool(name="search", description="Search for information")
async def search(query: str) -> str:
    # Call your search API
    return f"Results for: {query}"

agent = ReActAgent(
    name="search-agent",
    llm_client=AnthropicLLMClient(model="claude-haiku-4-5-20251001"),
    emitter=InMemoryEmitter(),
    system_prompt="You answer questions by searching for information.",
    tools=[search],
    max_iterations=10,
)

result = asyncio.run(agent.run("What is the capital of France?"))
```

**Decisions made:** ReActAgent (needs tools, single domain), one tool, no memory, default error handling, no context management, no evaluation, no planning, single agent, no orchestration, no HITL, basic safety.

### Research Agent with Memory

**Requirements:** Research topics across runs, remember past findings, use semantic search over a knowledge base.

```python
import asyncio
from nanitics import (
    ReActAgent, AnthropicLLMClient, InMemoryEmitter, MockEmbeddingClient,
    ContextManager, EstimateTokenCounter, TruncationPolicy,
    InMemoryWorkingMemory,
    InMemorySemanticStore, create_semantic_memory_tools,
    InMemoryEpisodeStore, EpisodicMemoryProvider, EpisodicMemoryContributor,
    tool,
)

emitter = InMemoryEmitter()
embedding_client = MockEmbeddingClient()  # Use VoyageEmbeddingClient in production

# Memory systems
working_memory = InMemoryWorkingMemory()
semantic_store = InMemorySemanticStore(embedding_client=embedding_client)
episode_store = InMemoryEpisodeStore(embedding_client=embedding_client)

# Tools: research + semantic memory
@tool(name="web_search", description="Search the web for information")
async def web_search(query: str) -> str:
    return f"Web results for: {query}"

semantic_tools = create_semantic_memory_tools(semantic_store)

agent = ReActAgent(
    name="researcher",
    llm_client=AnthropicLLMClient(model="claude-haiku-4-5-20251001"),
    emitter=emitter,
    system_prompt="You are a research agent. Use semantic memory to store and retrieve knowledge.",
    tools=[web_search, *semantic_tools],
    max_iterations=30,
    context_manager=ContextManager(
        context_limit=80_000,
        token_counter=EstimateTokenCounter(),
        truncation=TruncationPolicy(),
    ),
    # ReActAgent accepts working_memory directly as a simpler alternative
    # to manually creating WorkingMemoryProvider + WorkingMemoryContributor
    working_memory=working_memory,
    context_providers=[
        EpisodicMemoryProvider(episode_store),
    ],
    prompt_contributors=[
        EpisodicMemoryContributor(),
    ],
)

result = asyncio.run(agent.run("Research the latest developments in quantum computing"))
```

**Decisions made:** ReActAgent, web search + semantic memory tools, working + semantic + episodic memory, default error handling, truncation-based context management, no evaluation (research output quality is hard to evaluate automatically), no explicit planning (agent plans implicitly), single agent, no orchestration, no HITL, iteration-limited. Note: `ReActAgent` accepts `working_memory` directly, which auto-creates the provider and contributor internally.

### Multi-Agent Pipeline with Handoff

**Requirements:** Research a topic, write an article, then review it. Three specialized agents with structured context handoff.

```python
import asyncio
from nanitics import (
    ReActAgent, ReasoningAgent, AnthropicLLMClient, InMemoryEmitter,
    ProgrammaticEvaluator, EvaluationCheck,
    HandoffStep, create_handoff_chain, RawOutputTransfer,
    Sequential, AgentStep, tool,
)

emitter = InMemoryEmitter()
llm = AnthropicLLMClient(model="claude-haiku-4-5-20251001")

@tool(name="search", description="Search for information on a topic")
async def search(query: str) -> str:
    return f"Search results for: {query}"

researcher = ReActAgent(
    name="researcher",
    llm_client=llm,
    emitter=emitter,
    system_prompt="Research the given topic thoroughly. Provide detailed findings.",
    tools=[search],
    max_iterations=20,
)

writer = ReActAgent(
    name="writer",
    llm_client=llm,
    emitter=emitter,
    system_prompt="Write a well-structured article based on the research provided.",
    tools=[],
    max_iterations=20,
)

reviewer = ReasoningAgent(
    name="reviewer",
    llm_client=llm,
    emitter=emitter,
    system_prompt="Review the article. Provide a quality score (1-10) and specific feedback.",
)

# Compose into a sequential workflow
workflow = Sequential(
    name="research-pipeline",
    steps=[
        AgentStep(researcher),
        AgentStep(writer),
        AgentStep(reviewer),
    ],
    emitter=emitter,
)

result = asyncio.run(workflow.execute("Write an article about renewable energy trends"))
```

**Decisions made:** Three agents (ReAct for research and writing, Reasoning for review), search tool for researcher only, no memory (single-run pipeline), default error handling, no context management (each agent runs briefly), no evaluation (reviewer provides human-readable feedback), no explicit planning, multi-agent with sequential orchestration, no HITL, iteration-limited.

### Full-Featured System

**Requirements:** Customer support system with dynamic routing to specialists, human approval for refunds, quality monitoring, and episodic learning.

```python
import asyncio
from nanitics import (
    ReActAgent, AnthropicLLMClient, InMemoryEmitter, MockEmbeddingClient,
    CancellationToken,
    ErrorHandler, RetryPolicy,
    ProgrammaticEvaluator, EvaluationCheck,
    InMemoryEpisodeStore, EpisodicMemoryProvider, EpisodicMemoryContributor,
    AgentTool,
    ApprovalWrappedTool,
    CallbackHumanInputProvider, HumanInputResponse, HumanDecision,
    tool,
)

emitter = InMemoryEmitter()
llm = AnthropicLLMClient(model="claude-haiku-4-5-20251001")
embedding_client = MockEmbeddingClient()

# --- Specialist agents ---

@tool(name="lookup_order", description="Look up an order by ID")
async def lookup_order(order_id: str) -> str:
    return f"Order {order_id}: shipped, arriving tomorrow"

@tool(name="process_refund", description="Process a refund for an order")
async def process_refund(order_id: str, amount: float) -> str:
    return f"Refund of ${amount} processed for order {order_id}"

# Wrap refund tool with human approval
hitl_provider = CallbackHumanInputProvider(
    callback=lambda req: HumanInputResponse(
        request_id=req.request_id,
        decision=HumanDecision.APPROVE,
    )  # In production: async HITL flow with durable execution
)
approved_refund = ApprovalWrappedTool(
    tool=process_refund,
    provider=hitl_provider,
    emitter=emitter,
)

order_agent = ReActAgent(
    name="order-specialist",
    llm_client=llm,
    emitter=emitter,
    system_prompt="You handle order inquiries and refunds.",
    tools=[lookup_order, approved_refund],
    max_iterations=15,
)

@tool(name="search_faq", description="Search the FAQ knowledge base")
async def search_faq(query: str) -> str:
    return f"FAQ answer for: {query}"

faq_agent = ReActAgent(
    name="faq-specialist",
    llm_client=llm,
    emitter=emitter,
    system_prompt="You answer general questions using the FAQ.",
    tools=[search_faq],
    max_iterations=10,
)

# --- Coordinator with agent-as-tool delegation ---

episode_store = InMemoryEpisodeStore(embedding_client=embedding_client)

coordinator = ReActAgent(
    name="coordinator",
    llm_client=llm,
    emitter=emitter,
    system_prompt=(
        "You are a customer support coordinator. Route queries to the right specialist. "
        "Use delegate_orders for order/refund issues. Use delegate_faq for general questions."
    ),
    tools=[
        AgentTool(
            agent=order_agent,
            emitter=emitter,
            name="delegate_orders",
            description="Delegate order and refund inquiries",
        ),
        AgentTool(
            agent=faq_agent,
            emitter=emitter,
            name="delegate_faq",
            description="Delegate general FAQ questions",
        ),
    ],
    max_iterations=10,
    context_providers=[
        EpisodicMemoryProvider(episode_store),
    ],
    prompt_contributors=[
        EpisodicMemoryContributor(),
    ],
    error_handler=ErrorHandler(
        retry_policy=RetryPolicy(max_attempts=3, base_delay=1.0),
    ),
    output_evaluator=ProgrammaticEvaluator(
        checks=[
            EvaluationCheck(
                name="non_empty",
                check=lambda output: len(output.strip()) > 20,
                feedback="Response too short — provide a helpful answer.",
            ),
        ],
    ),
    cancellation_token=CancellationToken(),
)

result = asyncio.run(coordinator.run("I need a refund for order #12345"))
```

**Decisions made:** Multi-agent with agent-as-tool delegation, approval-wrapped refund tool, episodic memory on coordinator for cross-run learning, default context management, programmatic evaluation for response quality, custom error handling with retry, supervisor-ready architecture, iteration limits + cancellation token.

## Anti-Patterns

### Over-Engineering

**Problem:** Using multi-agent when a single agent suffices. Adding orchestration, planning, and evaluation to a task that needs one tool call.

**Fix:** Start with the simplest architecture that works. Add complexity only when you hit specific problems. A `ReActAgent` with 5 tools handles most tasks.

### Vague Tool Descriptions

**Problem:** Tools with names like `do_stuff` or descriptions like "Performs an action". The LLM can't choose the right tool if it doesn't understand what each one does.

**Fix:** Write tool descriptions as if explaining to a colleague. Include what the tool does, what input it expects, and what it returns. Test by reading the description and asking: would you know when to use this?

### Missing Safety Bounds

**Problem:** No `max_iterations` configured, letting agents loop indefinitely. No `CancellationToken`, so there's no way to stop a runaway agent.

**Fix:** Always set `max_iterations` (or the agent-type equivalent — see the Safety section) in production. Set `CancellationToken` when external cancellation is possible (user-facing applications, long-running workflows).

### Context Window Overflow

**Problem:** Tools that return entire documents, agents that run for hundreds of steps without context management. The agent either errors out or produces degraded output as important context is pushed out of the window.

**Fix:** Set up `ContextManager` with truncation or summarization for any agent that might run long. Design tools to return concise, relevant output.

### Memory Overload

**Problem:** Enabling every memory type "just in case". Each memory system adds context to every LLM call, consuming tokens and potentially confusing the agent.

**Fix:** Start with no dedicated memory. Add working memory when the agent needs to track structured state. Add other memory types when you have a specific need — semantic search, cross-run learning, multi-agent coordination.

### Monolithic System Prompts

**Problem:** A single agent with a 2000-word system prompt trying to cover research, analysis, writing, code review, and project management.

**Fix:** Split into specialized agents. Each agent has a focused system prompt (~200–500 words) and a targeted tool set. Use agent-as-tool delegation or orchestration to connect them.

## What to Read Next

- Just starting? [Getting Started](getting-started.md)
- Ready to build a production API? [Building Applications](building-applications.md)
- Need to choose an agent type? [Agent Types](agent-types.md)
- Building a multi-agent system? [Multi-Agent Foundations](multi-agent-foundations.md), then [Multi-Agent Coordination](multi-agent-coordination.md)
