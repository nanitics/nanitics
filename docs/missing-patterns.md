# Missing patterns and primitives

This document captures patterns, primitives, or abstractions that came up during the v1.0 API-surface audit but are not currently in the SDK. The bar for adding to v1.0 is high: a candidate earns its place only when (a) the pattern is overwhelmingly common in real agentic systems, (b) the boilerplate of building it from primitives is non-trivial enough to justify a named symbol, AND (c) the name itself carries semantic weight that helps users recognize the shape.

If none of those hold, the candidate stays out — the substrate is what should be excellent, and compositions belong in guides and examples rather than in `nanitics.__all__`.

## Considered during the audit and rejected

The audit surfaced several candidate additions. Each was considered and rejected against the bar above.

### Cost / budget tracking as a first-class abstraction

The SDK threads token usage through `Usage`, `LLMResponseEvent`, and `RoutingLLMClient` with `CostBudgetRouting`. There is no first-class "Budget" object that tracks lifetime spend across runs, raises when a ceiling is crossed, or attaches to a `Workflow` or `Supervisor` declaratively.

**Why rejected:** `BudgetTrigger` on `Supervisor` and `CostBudgetRouting` on `RoutingLLMClient` already cover the two common shapes (per-run budget enforcement, per-request routing under a budget). A general Budget abstraction would overlap both without obviously beating either. Application code can aggregate `Usage` from `AgentResult` and event streams when lifetime tracking is needed.

### Tool composition primitives

Beyond `ConditionalTool` (specialized), there are no primitives for sequencing two tools, falling back from one tool to another on failure, or composing tools into pipelines.

**Why rejected:** Tools are functions. Python composes functions. A tool that wraps two other tools is two lines. Naming this pattern adds nothing the language doesn't already provide.

### `Skill` / `Capability` abstraction

`BiddableAgent` carries `capabilities: list[str]` as bare strings. There is no formal `Capability` type with structure (description, parameters, examples).

**Why rejected:** Capabilities are LLM-consumed metadata. The string list is read by the bid/judge prompt and that is enough. Formalizing the shape would not change what the LLM sees — it would just add a wrapper.

### Prompt template / prompt composition primitives beyond `SystemPromptBuilder`

`SystemPromptBuilder` and `SystemPromptContributor` cover the system-prompt composition story. There is no equivalent for user-message templates, few-shot example assembly, or chain-of-prompt composition.

**Why rejected:** User messages are strings. F-strings exist. `SystemPromptBuilder` exists because the system prompt is *assembled from many contributors at runtime* (working memory contributor, planning contributor, episodic-memory contributor, etc.) — that composition is non-trivial. User messages don't have the same dynamic-assembly problem.

### Agent introspection / self-description primitive

There is no built-in primitive for asking an agent to describe its own capabilities, tools, or state.

**Why rejected:** This is an agent's *job*, not a primitive. A user who wants self-describing agents writes a system prompt section. Naming it as a primitive would imply a depth that doesn't exist.

### Database / SQL / email built-in tools

The built-in tools are web search, HTTP, file read, code execution. Adopters frequently need database tools, email tools, calendar tools, etc.

**Why rejected:** Each is application-specific (which database, which email provider, which auth). The built-ins that ship are the ones where the *interface* is universal enough to be a single function — HTTP is HTTP, web search has a standard shape behind one of two providers, file read is OS-level. Database and email don't have that shape. MCP tools cover the long tail.

### LatencyTrigger / RateLimiterTrigger on Supervisor

`Supervisor` has `QualityTrigger`, `BudgetTrigger`, `PredicateTrigger`. It does not have a built-in latency trigger or per-second rate limiter.

**Why rejected:** `PredicateTrigger` covers both with a few lines of user code. Three named triggers is already the right number; adding more crowds the namespace without clear payoff.

### "Plan-as-Code" primitive (deterministic plan execution)

`ReWOOAgent` plans a workflow then executes it without re-reasoning. There is no primitive for "user supplies a plan as data, SDK executes it" without an LLM planner stage at all.

**Why rejected:** That is `plan_to_workflow` applied to a user-constructed `TaskPlan`. The capability exists; it just lives in `nanitics.specialized` because the hierarchical-decomposition track is reached for deliberately rather than by default.

## Genuinely flagged for consideration

These were the candidates where the bar is closest to met. They are not in v1.0 today; whether they should be is a judgment call.

*(none flagged at audit completion)*

## Surface-decision notes

A few decisions made during the audit that are worth recording, since they will look arbitrary in retrospect:

- **`BiddableAgent` stays in core, even though `Bidding` moved to specialized.** `JudgeRouter` (core) reuses `BiddableAgent` so adopters can swap `Bidding` ↔ `JudgeRouter` at the call site. The type's `bid_generator` field is unused by `JudgeRouter` — that asymmetry is documented at the field level and is the deliberate cost of keeping the primitives interchangeable.
- **`DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE` is core; `DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE` is specialized.** The two templates have parallel calibration anchors but ship with their respective routing primitive. `JudgeRouter` is core, so the judge template is core; `Bidding` is specialized, so the bid template is specialized.
- **`Plan`/`PlanStep` are core; `TaskPlan`/`TaskNode` are specialized.** Two planning data shapes coexist. The linear `Plan` is what `UpfrontPlanContributor` and `AdaptivePlanningContributor` produce. The hierarchical `TaskPlan` is what `DecompositionContributor` produces and what `plan_to_workflow` consumes. Splitting them tracks the split in their consumers.
- **`MistralLLMClient` is specialized.** `LiteLLMClient` covers Mistral through LiteLLM's translation layer. The native Mistral client exists for adopters who want native error classification and cache-token reporting, but it is not the default reach.
