# Context Management

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Long-running agents accumulate conversation history — tool calls, results, reasoning steps — that can exceed the LLM's context window. Context management tracks token usage and applies truncation or summarization to keep the conversation within budget while preserving the most important information.

> **See also:** [examples/context/context_management.py](../../examples/context/context_management.py) — runnable example covering truncation, summarization, and combined strategies.

## When You Need It

**Use context management when** your agent might run for many steps (10+), processes large tool outputs, or operates near the context window limit. Without it, the agent will hit a `LLMContextLengthError` and stop.

**Skip it when** your agent runs for a few steps with small outputs. The overhead of token counting and message manipulation isn't worth it for short conversations.

## Choosing a Strategy

The core decision is whether to truncate, summarize, or both. Pass your choice to `ContextManager`, which runs automatically before each LLM call.

| | Truncation | Summarization |
|---|---|---|
| **Mechanism** | Drops old message groups from the middle | Compresses middle messages via an LLM call |
| **Speed** | Fast — no LLM call | Slower — requires an LLM round-trip |
| **Information loss** | Complete — dropped messages are gone | Partial — key information is condensed |
| **Cost** | Free | Uses tokens for the summarization call |
| **Best for** | Intermediate tool calls where old results don't matter | Long-running agents where losing context would hurt quality |

**Truncation only** is the right starting point for most agents. Agents that make many tool calls accumulate history fast, but the intermediate results (search outputs, API responses) rarely matter once processed. Dropping them is cheap and effective.

**Summarization only** makes sense when every piece of context contributes to the final result — research agents synthesizing findings across many sources, or agents building up a complex artifact incrementally.

**Use both** for a layered approach: truncation runs first as a fast pass, and summarization kicks in only if still over budget. When combined, summarization operates on the **original** (pre-truncation) messages, so it has access to the full content that truncation would have discarded. This is the recommended approach for production agents where you want speed without risking information loss.

## How It Works

On every LLM call, `ContextManager` transparently:

1. **Counts tokens** — system prompt + tools + messages
2. **Checks the threshold** — if under budget, returns messages unchanged
3. **Groups messages** — keeps assistant messages together with their tool results (preventing orphaned tool calls)
4. **Truncates** — if configured, drops expendable groups from the middle (oldest first), preserving the first group and N most recent groups
5. **Summarizes** — if still over budget and configured, compresses middle groups via LLM

The agent never sees this process — it happens between the agent loop and the LLM call.

Message grouping is important: a `tool_result` always stays attached to its preceding assistant message with `tool_calls`. Splitting them would produce orphaned tool calls or dangling results, both of which confuse LLMs. The default grouper handles this automatically; provide a custom `MessageGrouper` only if you need different grouping semantics.

## Delta Summarization

The key insight in the summarization design is **delta summarization**. After the first summarization, subsequent calls only process messages accumulated since the last summary:

```
Call 1: Summarize messages 1-20 → "Summary A"
Call 2: Summarize "Summary A" + messages 21-35 → "Summary B"  (delta)
Call 3: Summarize "Summary B" + messages 36-50 → "Summary C"  (delta)
```

This avoids re-summarizing the entire history each time and keeps each summarization call focused on a manageable chunk. The tradeoff is information compression — each delta pass condenses the previous summary further, so details from early in the conversation gradually fade. For most agents this is acceptable; critical information should be marked as protected.

`ContextManager` tracks delta state automatically and resets it between agent runs. If you reuse a `ContextManager` instance manually, call `reset()` to clear the accumulated summary.

## Token Counting

`ContextManager` uses a `TokenCounter` to estimate token usage. The built-in `EstimateTokenCounter` uses character-count heuristics — fast and accurate enough for budget management. If you need exact counting (via tiktoken or a provider's tokenizer), implement the `TokenCounter` protocol.

Token counting covers message content, tool call arguments (JSON-serialized), tool schemas, and per-message framing overhead.

## Protected Messages

Mark messages that should survive both truncation and summarization by setting `metadata={"protected": True}` on any `Message`. Protected messages are never dropped or compressed, regardless of the strategy. Use this for critical instructions or context the agent must retain throughout its run.

## Context Assembly

Context isn't just conversation history — agents can receive injected content from **context providers** before each LLM call. Providers contribute content (retrieved documents, memory, task state) with a priority and optional protection flag. The system assembles these contributions into the message stream alongside conversation history.

Context assembly integrates with capabilities like memory and planning that inject dynamic content into the agent's context window. Each contribution is tracked via `ContextAssemblyEvent` for observability.

This means the effective context budget is shared between conversation history and injected content. An agent with a 200K token limit, 50K of injected memory content, and a 0.9 threshold triggers context management at ~130K tokens of conversation history — not 180K.

> **See also:** [Memory](memory.md), [Planning](planning.md) — capabilities that use context providers.

## Tuning the Threshold

The `threshold` parameter controls when management triggers, expressed as a fraction of the available budget (limit minus reserve). Getting this right matters:

- **0.9 (default)** — conservative, lets the conversation fill most of the window before acting. Good default for most agents.
- **0.7–0.8** — triggers earlier, useful when tool outputs are unpredictably large and you want a buffer before hitting the limit.
- **Below 0.7** — rarely useful. Triggers management too often, wasting summarization calls or discarding context prematurely.

The threshold interacts with `reserve_tokens`: a 200K limit with 8K reserve and 0.9 threshold triggers at ~173K total tokens. If your agent produces long tool outputs in bursts, a lower threshold gives more headroom.

## Composition with Other Capabilities

Context management interacts with several other SDK capabilities:

- **Error handling:** If summarization fails (network error, rate limit), the exception propagates and the agent's LLM call fails. This is by design — context management is on the critical path. Ensure the summarization LLM client has appropriate resilience configuration.
- **Memory:** Episodic and semantic memory inject content via context providers. This injected content counts toward the token budget, so factor it into your `context_limit` and `threshold` settings.
- **Planning:** Active plans inject goal and step context. Like memory, this competes for budget with conversation history.
- **Multi-agent:** Each agent manages its own context independently. Shared context (via context transfer or message bus) arrives as new messages and is subject to the receiving agent's context management. Coordinator agents that receive results from many delegates are especially likely to need context management.
- **Observability:** Truncation emits `ContextTruncationEvent` only when at least one message is dropped, and summarization emits `ContextSummarizationEvent` only when a summary is actually produced — event presence reliably signals that a material reduction happened, so a no-op pass produces no event. Context assembly emits `ContextAssemblyEvent` showing what was injected and by which provider.

> **See also:** [Error Handling](error-handling.md), [Memory](memory.md), [Observability](observability.md), [Multi-Agent Coordination](multi-agent-coordination.md)

## Pitfalls

**No context management with large tool outputs.** If a tool returns 50K tokens and the agent calls it 10 times, you'll hit the context limit fast. Either truncate tool outputs in the tool itself, or configure context management.

**Threshold too low.** A threshold of 0.5 triggers management when only half the context is used, wasting summarization LLM calls. Start with 0.9 (the default) and lower only if you see context errors.

**Summarization without truncation.** Summarization alone works but is slower. Truncation provides a fast first pass that often resolves the issue without an LLM call. Use both when possible.

**`reserve_tokens` too small.** If the LLM's response is cut off, increase `reserve_tokens`. The default of 4096 works for most cases, but agents producing long outputs may need 8192 or more.

**Forgetting injected content.** Memory, planning, and other context providers inject content that counts toward the token budget. If you're hitting the threshold earlier than expected, check how much content is being injected via `ContextAssemblyEvent`.

**Summarization LLM failure.** If the summarization LLM call fails (network error, rate limit), the exception propagates and the agent's LLM call fails. This is by design — context management is on the critical path. Ensure the summarization LLM client has appropriate resilience configuration.

> **See also:** [examples/context/context_management.py](../../examples/context/context_management.py) — runnable example with all strategies
