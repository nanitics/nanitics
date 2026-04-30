# Diagnosing Agent Issues

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

When an agent produces wrong output, the question is not "what instruction would fix this output?" but "which layer failed?" The fix should happen at the right layer. Adding case-specific instructions is the agent equivalent of patching bugs with special-case `if` statements — it works once but compounds into prompt bloat, brittleness, and degraded general performance.

## The Diagnostic Ladder

Work through from top to bottom. Fix at the highest layer that addresses the root cause.

### 1. Prompt Clarity

**Symptom:** The agent misunderstands what to do, applies wrong criteria, or makes inconsistent judgments across similar inputs.

**Diagnosis:** Read the system prompt as if you've never seen this agent before. Is the task clear? Are the judgment criteria explicit? Could a reasonable reader interpret it differently?

**Fix:** Improve the general guidance — clarify definitions, sharpen criteria, add decision frameworks. The change should help with a *class* of inputs, not one specific case.

**Example:** An expense categorization agent misclassifies office supplies. Fix: clarify what "office supplies" means and how it differs from adjacent categories. Don't fix: add "when you see vendor X, use category Y."

### 2. Tools

**Symptom:** The agent reasons correctly but can't act on its reasoning — missing data, wrong data, insufficient search results, no way to verify its conclusions.

**Diagnosis:** Check the tool calls in the trace. Did the agent try to get information it needed? Did the tools return useful results? Is there a tool the agent should have but doesn't?

**Fix:** Improve existing tools (better search, richer data), add missing tools, or fix tool descriptions so the agent knows when and how to use them.

**Example:** A duplicate-detection agent misses duplicates because its search tool only returns exact matches. Fix: improve the search to support fuzzy matching. Don't fix: tell the agent to "also check partial matches manually."

### 3. Context / Information

**Symptom:** The agent lacks domain knowledge or system context needed to make good decisions. It's asked to decide something it doesn't have enough information about.

**Diagnosis:** Would a human with the same prompt and tools produce a good result? If not, what additional context would they need?

**Fix:** Provide missing domain knowledge as reference material, system prompt context contributors, or structured data. This is different from prompt clarity — the instructions may be clear, but the agent needs *facts* to apply them.

**Example:** An invoice processing agent doesn't know the company's VAT rules. Fix: provide VAT rules as reference context. Don't fix: add per-vendor VAT instructions.

### 4. Evaluation

**Symptom:** The agent produces variable-quality output — sometimes good, sometimes bad — and there's no automated check before the result reaches your application.

**Diagnosis:** Is there an evaluator? If yes, is it checking the right things? If no, would an evaluator reliably catch this class of failure?

**Fix:** Add an evaluator to catch quality issues, or improve existing evaluator criteria. The evaluation system provides automatic retry with feedback — the agent self-corrects without prompt changes.

**Example:** A document summarization agent sometimes omits key financial figures. Fix: add an evaluator that checks for the presence of monetary values when the source contains them. Don't fix: tell the agent "always include monetary values" (which masks the real issue — the agent sometimes deprioritizes numbers in long documents).

### 5. Agent Type / Architecture / Model Selection

**Symptom:** The agent's reasoning pattern doesn't match the problem structure. A single ReAct agent is handling too many concerns. The orchestration pattern creates information loss between agents. The problem requires exploration or backtracking that the agent type doesn't support.

**Diagnosis:** Is the agent type right for this task? (See `docs/guides/agent-types.md`.) Is the agent doing too many things? Would splitting into specialists help? Is context lost in handoffs?

**Fix:** Change the agent type, split into specialists, add orchestration, restructure the pipeline, adjust the multi-agent topology.

**Example:** A single agent handles document classification, extraction, and validation — and struggles because it loses focus partway through. Fix: split into specialist agents with an orchestrator. Don't fix: add lengthy instructions about "first do X completely, then do Y completely."

**Model capability — last resort.** If the architecture is right but the agent still fails, the model may not be capable enough for the task. Symptoms: inconsistent failures on complex multi-step reasoning, inability to follow structured output schemas reliably, or hallucinations that persist despite clear prompts and sufficient context. Fix: try a more capable model for this specific agent. Verify all other layers first — model upgrades are the most expensive fix and often mask a cheaper root cause.

### 6. Deterministic Code

**Symptom:** The decision the agent is making is actually rule-based with no ambiguity — the same input should always produce the same output, and the rules can be expressed as code.

**Diagnosis:** Could you write an `if/else` or lookup table that handles this correctly 100% of the time? If yes, it shouldn't be an agentic decision.

**Fix:** Move the logic out of the agent into deterministic code. The agent calls a tool that applies the rules, or the logic runs before/after the agent.

**Example:** An agent decides which tax rate to apply based on a product code. Fix: implement a tax rate lookup table and expose it as a tool or pre-processing step. Don't fix: put the full tax code table in the system prompt.

## Reading the Trace

Before diagnosing root cause, understand what already happened. The trace shows you what the system tried — retries, corrections, evaluations — so you don't misattribute the symptom.

**Error recovery events.** Look for `error.retry`, `error.correction`, and `error.degradation` events. If the system retried 3 times and then degraded, the root cause isn't "the agent made a mistake" — it's "the error isn't transient" or "the correction prompt isn't helping." Contrast this with a trace that shows zero recovery attempts, which means the error handler classified the failure as FATAL or no handler is configured.

**Evaluation events.** `evaluation.result` and `evaluation.revision` show whether output was checked and what happened. An ACCEPT verdict followed by bad downstream results means the evaluator criteria are wrong. A REVISE verdict with feedback that the agent ignored means the feedback isn't actionable. Multiple REVISE cycles ending in `evaluation_failed` means the agent can't self-correct for this class of input — prompt or tool changes won't help if the evaluator keeps failing the output.

**Context events.** `context.truncation` and `context.summarization` reveal whether the agent lost information during the run. If a truncation event drops the messages containing a critical tool result from earlier in the conversation, the agent's final output may be wrong not because it reasoned poorly but because it no longer had access to its own earlier findings.

**Multi-agent spans.** In composed systems, the span tree shows which agent actually failed. A bad orchestrator output might originate from a specialist that returned incomplete data three levels deep. Follow `parent_span_id` links to trace causality before assuming the top-level agent is at fault.

## Anti-Patterns

These are signs that you're applying a quick fix rather than addressing root causes:

- **Accumulating case-specific instructions** — "when you see X, do Y" entries that grow over time
- **Prompt length growing after each bug fix** — each fix adds tokens, overall performance degrades
- **Agent handling known cases well but failing on novel ones** — the agent is pattern-matching instructions, not reasoning
- **Multiple contradictory instructions** — accumulated patches that don't compose cleanly
- **Instructions that reference specific data values** — column names, vendor names, specific numbers

## Common Misdiagnoses

These are patterns where the symptom suggests one layer but the cause lives in another.

**"The agent keeps getting it wrong" → adding prompt instructions.** The most common misdiagnosis. Before touching the prompt, check whether the agent had the information it needed (Layer 3) and whether its tools returned useful results (Layer 2). A prompt fix for a context problem trains the agent to guess rather than reason from evidence.

**"Output is inconsistent" → adding an evaluator.** Evaluators catch quality variation, but if the agent type doesn't match the problem structure (Layer 5), the evaluator just masks a structural mismatch. An evaluator that repeatedly rejects output from a wrong agent type burns revision budget without convergence. If the same evaluator feedback fails to produce improvement across multiple revisions, the issue is upstream.

**"The tool keeps failing" → adding retry logic.** Retry is for transient failures (rate limits, timeouts). If a tool fails deterministically — wrong parameters, missing data, logic error — retrying produces the same failure every time. The SDK's error classification distinguishes RETRYABLE (transient) from CORRECTABLE (agent mistake) from FATAL (unrecoverable) for exactly this reason. Check the error category before assuming retry will help.

**"The deterministic code produced wrong output" → fixing the agent.** When post-agent code (proposal builders, action executors, data transformers) silently falls back to defaults or swallows errors, the symptom looks like the agent made a wrong decision — but the agent's output may have been correct. Check whether the code between the agent and the final result altered, dropped, or misinterpreted the agent's output.

**"The agent lost track of what it was doing" → adding instructions about ordering.** This usually indicates context loss (Layer 3) — the conversation grew long enough that truncation or summarization dropped earlier context. Or it indicates the wrong agent type (Layer 5) — a ReAct agent handling a task that requires structured working memory or a planning capability. Instructions about "do X first, then Y" are fragile and don't address the underlying cause.

**"The model isn't smart enough" → upgrading the model.** Model upgrades are expensive and often mask a cheaper root cause. Before concluding the model is the problem, verify: is the prompt clear (Layer 1)? Does the agent have the right tools and context (Layers 2–3)? Is the architecture appropriate (Layer 5)? A weaker model with clear instructions and good tools often outperforms a stronger model with a vague prompt. Upgrade only when all other layers check out and the task genuinely requires stronger reasoning capability.

## Applying This

When diagnosing an issue:

1. **Read the trace first.** Check what the system already tried — retries, corrections, evaluations, context operations. This prevents misdiagnosis and tells you whether existing recovery mechanisms are working, partially working, or absent.
2. **Identify the actual failure point.** In multi-agent systems, the agent that produced the final bad output may not be the agent that failed. Trace causality through spans to find where things first went wrong.
3. **Walk the ladder from top to bottom** — at which layer does the cause live?
4. **Fix at that layer.** Resist fixing at a lower layer than necessary. A prompt fix is cheaper than a tool fix, but if the tool is the problem, the prompt fix is technical debt.
5. **Verify the fix helps with the general case**, not just the specific instance that triggered the investigation.
