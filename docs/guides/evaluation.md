# Evaluation

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Agents don't always produce correct output on the first attempt. The evaluation system lets you define quality gates that automatically assess agent output and trigger revision when it falls short — before the result reaches your application.

An evaluator inspects the agent's final response, produces a verdict (accept, revise, or reject), and optionally provides feedback. If the verdict is "revise" and the revision budget hasn't been exhausted, the feedback is appended to the conversation and the agent tries again. This creates an automatic quality improvement loop that runs inside the agent, invisible to the caller.

> **See also:** [`examples/evaluation/evaluation.py`](../../examples/evaluation/evaluation.py) for a working example.

## The Evaluation Loop

When an agent has an evaluator, the final response flows through: **evaluate → verdict → accept or revise → re-run**. The agent doesn't know it's being evaluated — it simply receives feedback and tries again, the same way it would respond to any user message.

1. Agent produces output (no more tool calls)
2. Evaluator assesses the output → `EvaluationResult` with verdict `ACCEPT`, `REVISE`, or `REJECT`
3. `ACCEPT` → return output to caller
4. `REVISE` + revisions remaining → append feedback as user message, agent retries
5. `REVISE` + budget exhausted → return output with `termination_reason="evaluation_failed"`
6. `REJECT` → return output with `termination_reason="evaluation_failed"`

The revision budget is controlled by `max_revisions` on the evaluator. Each revision is a full agent iteration — the agent re-enters its main loop, produces new output, and the evaluator runs again. This means revisions consume both the revision budget and the agent's iteration budget.

## Supported Agent Types

Evaluation is available on all agent types, but the parameter name and role differ:

| Agent | Parameter | Required | Behavior |
|-------|-----------|----------|----------|
| `ReActAgent` | `output_evaluator` | Optional | Evaluates final text output (after tool use) |
| `CodeActAgent` | `output_evaluator` | Optional | Evaluates final text answer (not code blocks) |
| `ReasoningAgent` | `output_evaluator` | Optional | Evaluates single-turn reasoning output |
| `ReWOOAgent` | `output_evaluator` | Optional | Evaluates solver's final answer only |
| `TreeOfThoughtAgent` | `node_evaluator` | **Required** | Scores each search node; score drives expansion priority |
| `LATSAgent` | `node_evaluator` | **Required** | Scores MCTS nodes; REJECT prunes nodes |

Tree-search agents (`TreeOfThoughtAgent`, `LATSAgent`) require an evaluator — it guides which branches to explore. Without one, the search has no signal. The other agents use evaluation as an optional quality gate on final output.

## When to Use Evaluation

**Use evaluation when:**

- Output quality matters and automated checks can catch common failures — missing sections, format violations, factual checks, or subjective quality criteria
- You need consistency guarantees before output reaches your application
- The cost of a bad output is higher than the cost of an extra LLM call
- Truncation is a risk and you want automatic retry (truncation handling only activates when an evaluator is present)

**Skip evaluation when:**

- The task is simple enough that the first output is almost always acceptable
- External validation (a human reviewer, downstream system) already handles quality
- Latency is critical and the extra evaluation round-trip is unacceptable
- The agent is part of a multi-agent pipeline where a downstream agent reviews the output

## Choosing an Evaluator Type

The key decision is which evaluator fits your quality requirements. The three built-in types serve different needs:

| Evaluator | Cost | Deterministic | Best for |
|-----------|------|---------------|----------|
| `ProgrammaticEvaluator` | None | Yes | Structural checks — length, keywords, format, regex |
| `LLMEvaluator` | LLM call per eval | No | Subjective quality — coherence, accuracy, tone |
| `CompositeEvaluator` | Varies | Varies | Layered: cheap checks first, expensive checks only if needed |

### ProgrammaticEvaluator

Define predicate functions (`EvaluationCheck`) that run against the output string. All checks run on every evaluation; if any fail, the combined feedback from all failing checks triggers revision. Never returns `REJECT` — it always gives the agent a chance to fix the output.

Use when quality requirements can be expressed as string predicates: minimum length, required keywords, format patterns, absence of placeholder text.

Don't use for subjective assessment — "is this well-written?" can't be reduced to a predicate.

### LLMEvaluator

Describe quality criteria in natural language. The evaluator sends the original task, the agent's output, and your criteria to an LLM. The LLM returns a score (0.0–1.0) and reasoning. Scores below `score_threshold` trigger revision with the reasoning as feedback.

The quality of evaluation depends entirely on the specificity of your criteria. "The response must include at least 3 concrete examples and cite sources for all claims" produces consistent, useful evaluation. "Be good" does not.

Consider using a cheaper, faster model for evaluation than the agent's model. Evaluation requires judgment but not the same generative capability.

### CompositeEvaluator

Chains evaluators sequentially, short-circuiting on the first non-`ACCEPT` verdict. The typical pattern: programmatic checks (free, instant) followed by LLM evaluation (expensive, slow) only when structural checks pass. This avoids wasting LLM calls on output that fails basic requirements.

Set `max_revisions` on the composite, not on individual evaluators — the composite's value is what the agent reads for the revision budget.

### Custom Evaluators

Implement the `OutputEvaluator` protocol for evaluation logic that doesn't fit the built-in types — external API validation, database lookups, multi-step checks. Custom evaluators work standalone or inside a `CompositeEvaluator`.

The protocol hands your `evaluate(output, context)` two things: the final output string and an `EvaluationContext` with `messages` (the full conversation including tool-call and tool-result messages), `task_input` (the original prompt), and tree-search fields for `LATSAgent` / `TreeOfThoughtAgent`. For grounding, verification, or any check that depends on *what the agent did* rather than only *what it said*, walk `context.messages` to reconstruct tool-call history:

```python
from nanitics.evaluation import EvaluationResult
from nanitics.tracing import Message

class SearchBeforeClaimingEvaluator:
    """Fail if the agent produced a factual answer without calling search first."""

    max_revisions = 1

    async def evaluate(self, output: str, context) -> EvaluationResult:
        searched = any(
            tc.name == "web_search"
            for m in context.messages
            if m.role == "assistant"
            for tc in (m.tool_calls or [])
        )
        if not searched and output.strip():
            return EvaluationResult(
                verdict="revise",
                feedback="You produced an answer without calling web_search. Call web_search at least once before answering.",
                score=0.0,
                evaluator_name="search-before-claiming",
            )
        return EvaluationResult(verdict="accept", score=1.0, evaluator_name="search-before-claiming")
```

The same pattern works for inspecting tool *results* (`context.messages` with `role="tool_result"` carries the tool's return value under `content`, keyed back to the invoking call via `tool_call_id`).

## Truncation Handling

When an evaluator is configured and the LLM response is truncated (hits the output token limit), the agent automatically retries before running the evaluator. The truncation retry consumes the revision budget — a truncation retry followed by a normal evaluation revision uses two of the `max_revisions` attempts.

If the revision budget is exhausted by truncation retries, the agent terminates with `termination_reason="evaluation_failed"` without the evaluator ever running.

**Known limitation:** Truncation detection and retry only activate when an evaluator is present. Without an evaluator, truncated output is returned as-is. If your use case is susceptible to truncation, add an evaluator — even a simple `ProgrammaticEvaluator` with a length check will enable the retry mechanism.

## Pitfalls

**Revision retries consume iteration budget.** Each evaluation retry loops back through the agent's main loop, incrementing the iteration counter. If an agent has `max_iterations=5` and `max_revisions=3`, the agent may hit `iteration_limit` before exhausting its revision budget. Set `max_iterations` high enough to accommodate both tool use steps and potential revisions.

**Setting `max_revisions` too high.** Each revision is a full LLM call. With `LLMEvaluator`, each revision also triggers an evaluation LLM call. Two revisions means up to 4 extra LLM calls (2 agent + 2 evaluator). Keep `max_revisions` at 1–2 unless you have a strong reason.

**Vague evaluation criteria.** `LLMEvaluator` is only as good as the criteria you provide. Measurable, specific criteria produce consistent results. Vague criteria produce inconsistent verdicts and waste revision budget on unhelpful feedback.

**Not checking `termination_reason`.** When evaluation fails after exhausting revisions, the agent returns the last output with `termination_reason="evaluation_failed"`. Your application should check this and handle the failure — the output did not pass quality gates.

**Using the same expensive model for evaluation.** Evaluation doesn't require the same generative capability as the agent. A cheaper model often evaluates just as well, at a fraction of the cost and latency.

**Evaluating in multi-agent pipelines.** When agents delegate to other agents, decide where evaluation belongs. Evaluating at every level creates compounding latency and cost. Often it's better to evaluate only at the outermost agent, or at specific critical handoff points.

> **See also:** [Observability](observability.md) — `EvaluationEvent` and `EvaluationRevisionEvent` for tracing evaluation behavior.
