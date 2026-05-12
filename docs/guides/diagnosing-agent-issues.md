# Diagnosing Agent Issues

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

When an agent produces wrong output, the question is not "what instruction would fix this output?" but "where is the failure rooted?" Adding a case-specific instruction is the agent equivalent of patching bugs with special-case `if` statements: it works once, then compounds into prompt bloat, brittleness, and degraded performance on the cases the patch didn't anticipate.

Most agent failures are not model failures. They are context, tool, harness, or coordination failures — in roughly that order of frequency, per the production data available from independent failure-mode studies (Cemri et al.'s MAST taxonomy, Hamel Husain's evaluation work, the Microsoft AI Red Team taxonomy). The model is usually the last thing to blame.

This guide names the **domains** where agent failures live and the **process** for routing a failure to the right domain. It is not an ordered ladder — most production failures sit in one of two domains; the symptom decides which.

## The diagnostic process

1. **Read the trace first.** Before diagnosing, understand what actually happened. Follow `error.retry`, `error.correction`, `evaluation.result`, `context.truncation` events. In multi-agent runs, follow `parent_span_id` to find which sub-agent or step first produced wrong output. The visible bad output is often downstream of the actual failure — cascading is the rule, not the exception.
2. **Classify the symptom.** What shape does the failure take? Common shapes: wrong output on novel inputs, inconsistent output across similar inputs, silent failures (output looks right but isn't), loops or non-termination, cascading errors through a pipeline, tool misuse, goal drift over long runs.
3. **Route to one or more domains.** The symptom-to-domain table below maps shapes to likely roots. Most failures live in one or two domains.
4. **Fix at the root, not the symptom.** Prefer structural fixes over surface patches even when they cost more — the surface patch always fails on the next input class.

## Cross-cutting: error propagation

Failures cascade. The early agent that emits the wrong tool argument poisons every subsequent step that depends on that output. The orchestrator that hands off with truncated context starves the receiving agent. The post-processor that silently substitutes a default value masks an upstream signal.

Two consequences govern every diagnosis:

- **Always trace backwards from the visible bad output to the first wrong step.** That first wrong step is where the cause lives. The downstream effect is noise.
- **The agent that produced the final bad output is often not the agent that failed.** Follow span causality through `parent_span_id` before assigning blame.

This is not a separate domain. It is a property of how every failure looks from the outside, and a discipline you apply while walking the domains below.

## The eight domains

The domains are grouped in three tiers. Group I covers within-agent failures, where most debugging starts. Group II covers architectural failures, where the structure is wrong and no within-agent fix will hold. Group III covers system-level failures outside the agent boundary. A ninth domain — the SDK itself — applies only when you are debugging an application built on top of Nanitics; treat it as a last-resort check.

---

### Group I: within-agent failures

#### 1. Prompt clarity

**Symptom:** The agent misunderstands what to do, applies the wrong criteria, or makes inconsistent judgments across similar inputs.

**Diagnosis:** Read the system prompt as if you'd never seen the agent before. Is the task clear? Are the judgment criteria explicit? Could a reasonable reader interpret it differently?

**Fix:** Improve the *general* guidance — clarify definitions, sharpen criteria, add decision frameworks. The change should help with a *class* of inputs, not one specific case.

**Anti-pattern:** Adding "when you see X, do Y." The prompt grows with every bug fix; performance degrades on the unaffected cases. If you find yourself writing case-specific instructions, the cause lives in another domain.

**Example:** An expense-categorization agent misclassifies office supplies. Fix: clarify what "office supplies" means and how it differs from adjacent categories. Don't fix: add "when you see vendor X, use category Y."

#### 2. Tools

**Symptom:** The agent reasons correctly but can't act on its reasoning. Missing data, wrong data, insufficient search results, no way to verify its conclusions. Or: the agent picks the wrong tool for the situation.

**Diagnosis:** Check tool calls in the trace. Did the agent try to get information it needed? Did the tools return useful results? Is there a tool that should exist but doesn't? Are the tools described well enough for the agent to pick the right one?

**Fix:** Two distinct shapes here.

- **Tool design.** One tool, one action. Descriptive names and parameters. Useful errors. Bounded output (paginate or summarize when output is large). `search_documents` and `update_document` are better than `manage_documents`. `"Document not found: xyz"` beats `"Error"`.
- **Tool selection.** Better descriptions so the agent picks the right tool. Reduce tool surface — an agent with 30 broad tools picks wrong more often than one with 10 specific tools.

See [`tools.md`](tools.md).

**Example:** A duplicate-detection agent misses duplicates because its search tool only returns exact matches. Fix: improve the search to support fuzzy matching. Don't fix: tell the agent to "also check partial matches manually."

#### 3. Context / information

**Symptom:** The agent lacks domain knowledge or system context needed to decide well. It is reasoning correctly with the wrong facts, or with not enough facts. Long-running agents may exhibit goal drift as context truncation drops earlier state.

**Diagnosis:** Would a human with the same prompt, tools, and context produce a good result? If not, what additional context would they need? In long runs: are critical earlier messages still in the window, or have they been truncated or summarized away?

**Fix:** Provide facts as reference material. Pick the right primitive:

- `SystemPromptContributor` for static reference material that should always be visible.
- `WorkingMemoryProvider` for per-run state.
- `SemanticMemoryProvider` for retrieval-augmented context.
- `EpisodicMemoryStore` for memory of past similar runs.

Context engineering — what enters the window, in what order, with what compression — is its own discipline. See [`memory.md`](memory.md) and [`context-management.md`](context-management.md).

**Example:** An invoice-processing agent doesn't know the company's VAT rules. Fix: provide VAT rules as reference context. Don't fix: add per-vendor VAT instructions to the prompt.

#### 4. Evaluation / verification

**Symptom:** Output quality varies run to run. Sometimes correct, sometimes not. No automated check catches the bad runs. Or — worse — the output *looks* correct but isn't (silent failure). Silent failures are particularly dangerous because nothing flags them.

**Diagnosis:** Is there an evaluator? If yes, is it checking the right things? If no, would an evaluator reliably catch this class of failure? If failures are caught but not fixed, is the evaluator's feedback actionable, or does the agent keep ignoring it?

**Fix:** Add or improve an evaluator.

- `ProgrammaticEvaluator` for deterministic checks.
- `LLMEvaluator` for judgment-based checks.
- `CompositeEvaluator` to combine the two.

Wire into a `ReflexionAgent` for revise-on-fail, or a `RevisionGate` for human approval of the verdict.

See [`evaluation.md`](evaluation.md).

**Example:** A document-summarization agent sometimes omits key financial figures. Fix: add an evaluator that checks for the presence of monetary values when the source contains them. Don't fix: tell the agent "always include monetary values" (which masks the real issue — the agent sometimes deprioritizes numbers in long documents).

---

### Group II: architectural failures

When the structure is wrong, no within-agent fix will hold for long.

#### 5. Agent strategy

**Symptom:** The agent's reasoning *pattern* doesn't match the problem shape. The agent loops, backtracks unnecessarily, gets stuck on multi-step planning, or can't recover from a failed step.

**Diagnosis:** Is the agent type right for this task? See [`agent-types.md`](agent-types.md). Common mismatches:

- ReAct (interleaved reason-then-act) for problems that need upfront planning → switch to ReWOO or a planning contributor.
- ReAct for problems where output needs to be revised on failure → switch to `ReflexionAgent`.
- ReAct for problems where code is the natural expression → switch to `CodeActAgent`.
- Tree-of-Thought or LATS for problems with no real benefit from exploration → simplify to ReAct or Reasoning.

**Fix:** Change the agent strategy. Higher-cost than prompt, tool, or context changes; lower-cost than the coordination or model fixes that follow.

**Example:** An agent doing research-then-summarize keeps backtracking because it can't separate the two phases. Switch from a single ReAct agent to a `Pipeline` of two specialists, or to ReWOO so planning and execution are separate.

#### 6. Multi-agent coordination

**Symptom:** Multiple agents are in play. Information is lost between them. The orchestrator routes badly. A specialist returns incomplete output and the next agent runs with garbage input. Handoffs lose context. The visible bad output traces back to a coordination point, not a single agent.

This is a distinct diagnostic shape from agent strategy. A correctly-strategized agent can still fail in a multi-agent system because the *coordination* is wrong.

**Diagnosis:** Walk the span tree. At which handoff or orchestration point does the trajectory go wrong? Is the supervisor seeing the right state? Is the message bus carrying the right messages? Is the blackboard being read at the right granularity? Is the handoff payload carrying enough context?

**Fix:** Restructure the coordination, not the individual agents.

- Change the orchestration topology (handoff vs supervisor vs blackboard vs message bus vs peer network).
- Improve handoff payloads — carry more context, or different context.
- Split a single-agent task that's failing into specialists with explicit handoffs.
- Merge multi-agent overhead that's causing information loss into a single agent if the coordination cost exceeds the benefit.

See [`multi-agent-foundations.md`](multi-agent-foundations.md) and [`multi-agent-coordination.md`](multi-agent-coordination.md) for the trade-offs.

**Example:** A document-classification pipeline fails because the extractor agent returns incomplete output and the validator agent silently accepts it. Fix: add a programmatic check on the handoff payload before the validator sees it. Don't fix: tell the validator to "double-check the extractor's output."

#### 7. Model capability — last resort

**Symptom:** The architecture is right, the context is right, the tools are right, the evaluator is right, and the agent still fails. Failures are about reasoning depth, multi-step inference, or structured-output reliability.

**Diagnosis:** Verify all other domains first. Model upgrades hide architectural problems instead of solving them — when you scale workload, the same problem returns.

**Fix:** Try a more capable model for this specific agent or step. `RoutingLLMClient` with `RuleBasedRouting` or `CostBudgetRouting` lets you use a heavier model only where it matters.

**Example:** A code-synthesis agent consistently produces plausible-looking code that fails on edge cases. Upgrading from a small to a mid-tier model resolves it because the small model was over-pattern-matching. Verified only after prompt, tools, and context were already clean.

---

### Group III: system-level failures

Outside the agent boundary.

#### 8. Harness and deterministic plumbing

**Symptom:** The agent's output looks wrong, but tracing backwards shows the agent did its job correctly. The bug lives in the code BEFORE, AFTER, or AROUND the agent.

Two sub-shapes:

- **Buggy plumbing.** Pre-processing that mis-shapes the agent's input. Post-processing that drops, alters, or silently defaults the agent's output. Orchestration logic that routes incorrectly. The pipeline that calls the agent has a bug — the agent is innocent.
- **Wrong layer of decision-making.** The decision is rule-based with no genuine ambiguity. An `if` statement would do the job 100% of the time, but the code currently calls an LLM. The agent is being asked to do something deterministic.

**Diagnosis:** Check the code path between the agent's output and the visible bad result. Did pre-processing alter the input? Did post-processing silently substitute a default? Is the decision actually ambiguous, or is it rule-based?

**Fix:**

- For buggy plumbing: fix the plumbing.
- For mis-placed decisions: move the logic out of the agent into deterministic code. The agent calls a tool that applies the rules, or the logic runs before or after the agent.

**Example A (plumbing):** A proposal-builder agent's output is correct, but the post-processor that converts the output to JSON silently drops fields whose values contain commas. Fix the post-processor. Don't fix: tell the agent to "avoid commas in values."

**Example B (deterministic):** An agent decides which tax rate to apply based on a product code. Implement a tax-rate lookup table and expose it as a tool or pre-processing step. Don't fix: put the full tax code table in the system prompt.

---

### 9. SDK behaviour (last-resort domain for SDK users)

**Symptom:** The Nanitics primitive itself doesn't behave as documented at the version you pinned. Reading the docstring and the example, the symbol should do X, but it does Y.

**Diagnosis:** Most SDK-shaped problems are misuse. Verify by reading the source on disk (`.venv/lib/python*/site-packages/nanitics/`) and the relevant guide. Confirm your pinned version against the docstring you are reading.

**Fix:**

- If genuinely a bug: file an issue on the Nanitics GitHub repo. Pin to a different version, work around locally, or pause the affected work.
- If a missing feature: same — file an issue or Discussion, work around or pause.

The [deprecation policy](../deprecation-policy.md) covers what the project commits to for public-API stability.

## Symptom → domain routing

The front door for triage. Pick the row that matches what you're seeing.

| Symptom | First check | Second check |
|---|---|---|
| Wrong output on novel inputs | Domain 3 (context) — did it have the info? | Domain 5 (strategy) — wrong reasoning pattern? |
| Inconsistent output across similar inputs | Domain 1 (prompt) — ambiguous criteria? | Domain 4 (evaluation) — no quality gate? |
| Output looks right but is wrong (silent failure) | Domain 4 (evaluation) — no verifier? | Domain 8 (plumbing) — silent default substitution? |
| Agent loops or doesn't terminate | Domain 5 (strategy) — wrong agent type? | Domain 2 (tools) — error not surfacing? |
| Cascading errors through a pipeline | Domain 6 (coordination) — handoff loses context? | Domain 8 (plumbing) — silent failure in between? |
| Tool keeps failing | Domain 2 (tool design) — error category? | Domain 7 (model) — too small to use tools reliably? |
| Goal drift over long runs | Domain 3 (context) — truncation? | Domain 5 (strategy) — wrong agent type for long tasks? |
| "It used to work" | Commit a regression first: SDK pin change? | Plumbing change? Prompt rev? Trace before-after. |

Most failures resolve in the first column. The second column is the typical alternative when the first comes up clean.

## Anti-patterns

Signs you are fixing in the wrong domain:

- **Case-specific patches in the prompt.** "When you see X, do Y." The prompt grows with every fix; performance on unaffected cases degrades.
- **Prompt length growing after each bug fix.** Every fix adds tokens. Overall judgment degrades.
- **Agent handling known cases well but failing on novel ones.** The agent is pattern-matching instructions, not reasoning. The cause is in another domain.
- **Contradictory instructions.** Accumulated patches that no longer compose. Simplify.
- **References to specific data values in the system prompt.** Column names, vendor names, specific numbers. The fact belongs in context (Domain 3), not in the prompt.

When any of these are tempting, the diagnosis is wrong.

## Common misdiagnoses

Patterns where the symptom suggests one domain but the cause lives in another.

**"The agent keeps getting it wrong" → adding prompt instructions.** The most common misdiagnosis. Check whether the agent had the information (Domain 3) and whether its tools returned useful results (Domain 2) before touching the prompt. A prompt fix for a context problem trains the agent to guess rather than reason from evidence.

**"Output is inconsistent" → adding an evaluator.** Evaluators catch quality variation, but if the agent strategy doesn't match the problem (Domain 5), the evaluator masks a structural mismatch. An evaluator that repeatedly rejects output from a wrong agent type burns revision budget without convergence. If the same evaluator feedback fails to produce improvement across multiple revisions, the cause is upstream.

**"The tool keeps failing" → adding retry logic.** Retry is for transient failures (rate limits, timeouts). If a tool fails deterministically — wrong parameters, missing data, logic error — retrying produces the same failure every time. The SDK's error classification distinguishes RETRYABLE (transient) from CORRECTABLE (agent mistake) from FATAL (unrecoverable) for exactly this reason. Check the error category before assuming retry will help.

**"The deterministic code produced wrong output" → fixing the agent.** When post-agent code silently falls back to defaults or swallows errors, the symptom looks like the agent made a wrong decision — but the agent's output may have been correct. Check Domain 8 before assuming the agent is at fault.

**"The agent lost track of what it was doing" → adding ordering instructions.** Usually Domain 3 (context truncation dropped earlier state) or Domain 5 (wrong agent type for the task length). Ordering instructions are fragile and don't address the underlying cause.

**"The model isn't smart enough" → upgrading the model.** Model upgrades are expensive and often mask cheaper root causes. Verify all other domains before concluding the model is the cause. A weaker model with clear prompts, good tools, and the right architecture often outperforms a stronger model with a vague prompt.

**"Looks like an SDK bug" → reporting an issue.** Most SDK-shaped problems are misuse. Read the guide and example for the primitive in question. Reach for an issue only after confirming source behaviour diverges from the docstring at your pinned version.

## Reading the trace

The trace tells you what the system already tried. Reading it first prevents misattributing the symptom.

**Error recovery events.** `error.retry`, `error.correction`, `error.degradation`. If the system retried three times and degraded, the root cause isn't "the agent made a mistake" — it's "the error isn't transient" or "the correction prompt isn't helping." Contrast with a trace that shows zero recovery attempts, which means the error handler classified the failure as FATAL or no handler is configured.

**Evaluation events.** `evaluation.result` and `evaluation.revision` show whether output was checked and what happened. An ACCEPT verdict followed by bad downstream results means the evaluator criteria are wrong. A REVISE verdict with feedback the agent ignored means the feedback isn't actionable. Multiple REVISE cycles ending in `evaluation_failed` means the agent can't self-correct for this class of input — prompt or tool changes won't help if the evaluator keeps failing the output.

**Context events.** `context.truncation` and `context.summarization` reveal whether the agent lost information during the run. If a truncation event drops messages containing a critical tool result from earlier in the conversation, the agent's final output may be wrong not because it reasoned poorly but because it no longer had access to its own earlier findings.

**Multi-agent spans.** In composed systems, the span tree shows which agent actually failed. A bad orchestrator output might originate from a specialist that returned incomplete data three levels deep. Follow `parent_span_id` links to trace causality before assuming the top-level agent is at fault.

## Applying this

When diagnosing an issue:

1. **Read the trace.** Check what the system already tried — retries, corrections, evaluations, context operations.
2. **Trace backwards.** Find the first wrong step. The visible bad output is often downstream of the actual failure.
3. **Classify the symptom.** Use the symptom-to-domain table as the front door.
4. **Walk one or two domains.** Most failures live in one or two. Walk the candidates the table named.
5. **Fix at the root, not the symptom.** Resist surface fixes when the cause is structural — the surface fix will fail on the next input class.
6. **Verify the fix helps the general case,** not just the specific instance that triggered the investigation.

## Further reading

For a comprehensive academic treatment of multi-agent failure modes, see Cemri et al.'s MAST taxonomy (*Why Do Multi-Agent LLM Systems Fail?*), which catalogs 14 failure modes across system design, inter-agent misalignment, and task verification — the empirical basis for elevating coordination to its own domain in this guide. For practitioner-focused evaluation methodology, Hamel Husain's writing on agentic-workflow evaluation is the canonical reference for the trace-first, two-phase (end-to-end then step-level) approach this guide encodes.
