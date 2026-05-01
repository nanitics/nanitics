# Multi-Agent Coordination

> For API details — signatures, fields, constraints — read the docstrings in your editor, in the source tree under [`nanitics/`](../../nanitics/), or browse them at [docs.nanitics.dev](https://docs.nanitics.dev/). `nanitics.__all__` is the authoritative public surface.

Coordination patterns handle higher-level concerns: dynamic task delegation, quality monitoring, shared-state collaboration, competitive allocation, adversarial reasoning, and collective agreement. These patterns build on the foundations covered in [Multi-Agent Foundations](multi-agent-foundations.md) — read that first.

## When to Use

**Use coordination when** your multi-agent system needs structure beyond simple delegation or communication. Examples: a central coordinator dynamically routing tasks to specialists, automated quality oversight on agent output, multiple experts building a shared artifact, or agents that need to agree on a decision.

**Don't use coordination when** simpler patterns suffice. Agent-as-tool handles basic delegation. Broadcast handles parallel execution. If you find yourself reaching for a coordinator when a sequential workflow would do, step back and reconsider.

## Decision Guide

The table below compares all coordination patterns across key dimensions. Use it to narrow down which pattern fits your problem before reading the detailed sections.

| Need | Pattern | Topology | Communication | Decision-Making | Agent Requirements |
|------|---------|----------|---------------|----------------|--------------------|
| Dynamic delegation to specialists based on task analysis | [Orchestrator](#orchestrator) | Centralized (hub-spoke) | Coordinator → specialist → coordinator | LLM-driven decomposition | Any agent type |
| Runtime quality/budget monitoring with intervention | [Supervisor](#supervisor) | Centralized (wrapper) | Supervisor → agent → supervisor | Trigger-based evaluation | Any agent type |
| Agents coordinating through shared state | [Blackboard](#blackboard) | Decentralized (shared space) | Agent → shared memory → agent | Control strategy + termination condition | `ReActAgent` only (`supports_dynamic_tools`) |
| Competitive task allocation via auction | [Bidding](#bidding) | Centralized (auction) | Coordinator collects bids → selects winner | Allocation strategy on bids | Any agent type |
| Comparative task allocation via single-call ranking | [JudgeRouter](#judge-routed-allocation) | Centralized (judge) | Judge ranks all candidates in one call → top match executes | Confidence threshold on ranking | Any agent type |
| Adversarial reasoning between opposing positions | [Debate](#debate) | Structured (adversarial) | Debaters exchange via transcript | Resolution strategy (judge) | Any agent type |
| Collective agreement through voting or deliberation | [Consensus](#consensus) | Peer (parallel) | Independent or peer-visible responses | Aggregation strategy | Any agent type |

### Topology Comparison

**Centralized patterns** (Orchestrator, Supervisor, Bidding) have a single coordinator that controls flow. They're simpler to reason about and debug but create a single point of failure and can bottleneck on the coordinator's reasoning quality.

**Decentralized patterns** (Blackboard) let agents operate independently on shared state. No coordinator bottleneck, but emergent behavior can be harder to predict and debug.

**Structured patterns** (Debate, Consensus) define a fixed interaction protocol. Agents interact through a prescribed structure (adversarial rounds or voting rounds) rather than through free-form coordination.

### Choosing Between Similar Patterns

**Orchestrator vs Supervisor:** An orchestrator handles task decomposition and delegation — it decides *what work to do*. A supervisor handles quality assurance — it decides *whether work is good enough*. They solve different problems and compose well together.

**Bidding vs JudgeRouter vs Orchestrator:** All three route tasks to agents, but they get there through different mechanisms. **Bidding** is decentralized self-assessment — each agent independently rates its own fit. It's the right shape when agents genuinely have peer-bid semantics (a true auction) and you want every participant to register a self-confidence on the trace. The known failure mode is self-overclaim: independent ratings without comparison tend to converge on uniformly high scores, and strict-tie wins go to first-listed unless you chain a tiebreaker (`HighestConfidence(tiebreaker=LowestCost())`). **JudgeRouter** is centralized comparative judgment — one judge LLM sees every candidate together and returns the full ranking in a single call. Use it when you'd rather pay one judge call than N self-assessment calls and you want the judge to discriminate ("only one candidate should reach 0.9 for this task"). The default `DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE` carries four-tier calibration anchors (0.9 / 0.7 / 0.4 / 0.0) that bite against self-overclaim. **Orchestrator** is coordinator reasoning over a known specialist pool — the coordinator decomposes the task and picks specialists per subtask using its own judgment. Use it when the coordinator has domain knowledge that the specialists themselves can't express, when the work is multi-step decomposition rather than allocation, or when you want the coordinator to synthesize specialist outputs into a final answer. Bidding and JudgeRouter allocate to a single specialist; the orchestrator can interleave several.

**Debate vs Consensus:** Both involve multiple agents on the same question, but they optimize for different things. Debate produces the strongest *arguments* through adversarial pressure. Consensus produces the most *reliable answer* through aggregation. Use debate for analysis; use consensus for decisions.

**Blackboard vs Message Bus:** Both enable multi-agent communication, but blackboard provides shared state with coordination controls (rounds, termination), while message bus provides point-to-point or broadcast messaging without structure. Blackboard is a coordination pattern; message bus is a communication primitive.

## Orchestrator

An orchestrator is a ReAct agent that dynamically analyzes tasks and delegates subtasks to specialist agents. Unlike a fixed workflow, the orchestrator decides at runtime which specialists to invoke, in what order, and how to synthesize their results. The `create_orchestrator` factory wraps specialist agents as tools and generates a system prompt that instructs the orchestrator to decompose, delegate, and synthesize.

The orchestrator receives specialist agents wrapped as `AgentTool` instances. It uses its own LLM reasoning to analyze the incoming task, break it into subtasks, decide which specialist to invoke for each subtask, and then synthesize the results. This makes it fundamentally different from fixed workflows — the decomposition strategy is emergent, not predetermined.

You can override the generated system prompt entirely or use `orchestrator_prompt_section` to get the specialist listing as a standalone section for embedding in your own prompt. The prompt section returns a `(section_name, section_content)` tuple describing all available specialists and their capabilities.

The orchestrator supports all standard ReAct agent options including `cancellation_token` (propagated to specialists), `error_handler`, `context_manager`, `context_providers`, and `output_evaluator`.

By default the coordinator synthesizes. For pipelines where the final specialist produces the deliverable, pass `final_output_strategy=FinalOutputStrategy.RELAY_LAST` — the orchestrator returns that specialist's output verbatim and discards the coordinator's final synthesis turn. Use this when a coordinator rewrite would only compress or paraphrase the specialist's work. `RELAY_LAST` is incompatible with `output_schema` (schema-constrained output is itself a synthesis step) and is rejected at construction. When an `output_evaluator` is supplied alongside `RELAY_LAST`, it runs against the relayed specialist content; `REVISE` is non-actionable (there is no coordinator turn to revise) and the result is marked `evaluation_skipped`.

**When to use:**

- Tasks requiring different skills that a central coordinator can route
- Problems where the decomposition isn't known in advance
- Workflows where specialist ordering depends on intermediate results

**When not to use:**

- Fixed, predictable workflows — use [Sequential or Pipeline](orchestration.md) instead
- When you know exactly which agents to call in what order — orchestration adds unnecessary LLM calls for routing decisions

> **See also:** [examples/multi_agent/orchestrator.py](../../examples/multi_agent/orchestrator.py) — prompt section generation, orchestrator construction, end-to-end delegation with two specialists, custom system prompt override, and the `RELAY_LAST` final-output strategy.

## Supervisor

`Supervisor` wraps an agent run with post-execution monitoring. After the agent completes, triggers evaluate the result and decide whether to accept, retry with feedback, reassign to a different agent, or escalate. Multiple triggers can be composed — the first trigger that fires determines the outcome.

The supervision loop works as follows: the agent runs, triggers evaluate in order, and the first intervention drives the action:

- **ACCEPT** — no trigger fired, result is returned as accepted.
- **RETRY** — agent re-runs with feedback appended to the task. Continues until `max_retries` is exhausted.
- **REASSIGN** — a different agent (from the `agents` registry) takes over. If the target agent isn't registered, the result is returned as not accepted.
- **ESCALATE** — supervision stops immediately, result is returned as not accepted.

If retries exhaust without acceptance, the final result is returned as not accepted.

### Triggers

Three built-in triggers cover common supervision needs:

- **QualityTrigger** — evaluates output quality using an `OutputEvaluator`. Maps ACCEPT → pass, REVISE → retry with feedback, REJECT → escalate. If the agent produces no output, the trigger escalates.
- **BudgetTrigger** — checks token usage against a maximum. Exceeding the budget escalates. Useful for controlling costs on expensive agent runs.
- **PredicateTrigger** — custom logic via a callable that receives the `AgentResult` and task string, returning either `None` (pass) or a `SupervisionDecision` with an action, feedback, and optional reassignment target. Both arguments are keyword-only: `PredicateTrigger(name="length_check", predicate=my_check)`.

Custom triggers implement the `SupervisionTrigger` protocol — see the docstring for the `check` method signature. Triggers evaluate in the order they're listed; the first to fire wins.

A common pattern is composing multiple triggers: a `QualityTrigger` for output quality, a `BudgetTrigger` for cost control, and a `PredicateTrigger` for domain-specific checks (e.g., output length, required keywords, format validation).

### Reassignment

To enable reassignment, provide an `agents` dictionary mapping agent names to agent instances. When a trigger returns a REASSIGN action with `reassign_to` set to a name in this registry, the supervisor switches execution to that agent. This is useful for escalating from a standard agent to a more capable one.

**When to use:**

- Quality assurance on agent output
- Budget enforcement
- Any scenario where you want automated oversight with retry or escalation logic

**When not to use:**

- When you trust the agent's output without review
- When manual human review is preferred — use [HITL](human-in-the-loop.md) instead

The result includes `accepted` (whether all triggers passed), `total_attempts`, `interventions` (all `SupervisionDecision` objects), and `final_agent` (which agent produced the final result — important when reassignment occurred).

Emits `SupervisionEvent` after each trigger check, carrying the supervised agent name, action taken, trigger name, feedback, and attempt number.

> **See also:** [examples/multi_agent/supervisor.py](../../examples/multi_agent/supervisor.py) — all four supervision outcomes (accept, retry, reassign, escalate), three trigger types, and multi-trigger composition.

## Blackboard

`Blackboard` coordinates agents through a shared memory space. Instead of communicating directly, agents read from and write to a `SharedMemory` instance. A control strategy determines which agents run each round, and a termination condition decides when to stop.

The blackboard validates that all agents support dynamic tools at construction time. Before the round loop, it injects shared memory tools (`write_to_shared`, `read_shared`, `supersede_shared`, `retract_shared`) into each agent. Each round, the control strategy selects which agents run and whether they run in parallel or sequentially. Selected agents execute with the task while a listener tracks shared memory events to count contributions. After each round, the termination condition checks whether to stop. When terminated, injected tools are removed from all agents via `finally`.

**Agent requirement:** All agents must have `supports_dynamic_tools == True` (currently `ReActAgent` only). The blackboard protocol requires agents that can reactively read from and write to shared memory mid-execution. Agents that can't use tools reactively (e.g., `ReasoningAgent`, `ReWOOAgent`) are rejected at construction time. See [Coordination Compatibility](agent-types.md#coordination-compatibility) in Agent Types.

### Control Strategies

Three built-in control strategies determine agent selection and execution mode:

- **ScheduledControl** (default) — runs all agents sequentially in order, every round. Simplest strategy for predictable turn-taking.
- **PrioritizedControl** — runs all agents sequentially, ordered by priority (highest first). Accepts a `priorities` dict mapping agent names to numeric priorities. Agents without explicit priorities default to 0.
- **OpportunisticControl** — runs all agents in parallel each round. Best for independent contributions where order doesn't matter.

Custom strategies implement the `ControlStrategy` protocol — `select` receives the agent list and current `BlackboardState` (round number, round contributions, total contributions), returning which agents to run. The `parallel` property controls whether selected agents execute concurrently.

### Termination Conditions

- **NoNewContributions** (default) — stops when a round produces zero shared memory writes, supersedes, or retracts.
- **MaxRoundsTermination** — stops after a fixed number of rounds regardless of progress.
- **BlackboardCompositeTermination** — combines multiple conditions with `"any"` (stop when any fires) or `"all"` (stop when all fire) mode.

Custom conditions implement the `TerminationCondition` protocol — `should_terminate` receives the current `BlackboardState` and the `SharedMemory` instance.

**When to use:**

- Multiple agents building or refining a shared artifact
- Problems where agents need to see each other's work without direct communication
- Iterative refinement where agents react to evolving state

**When not to use:**

- Direct agent-to-agent communication — use [Message Bus](multi-agent-foundations.md#message-bus) or [Peer Network](multi-agent-foundations.md#peer-network)
- Fixed pipelines — use [Sequential](orchestration.md) workflows

The result includes `entries` (all shared memory entries including inactive ones), `rounds_completed`, `termination_reason` (class name of the condition that fired, or `"max_rounds"`), and `agent_contributions` (number of contributions per agent).

Emits `BlackboardStartEvent` before the first round, `BlackboardRoundEvent` after each round (with contribution counts), and `BlackboardCompleteEvent` after termination.

> **See also:** [examples/multi_agent/blackboard.py](../../examples/multi_agent/blackboard.py) — scheduled control, convergence, prioritized ordering, parallel execution, and event trace inspection.

## Bidding

`Bidding` runs a competitive auction where agents bid on tasks based on self-assessed capability. Each agent generates a bid (confidence, capabilities, estimated cost), an allocation strategy selects a winner, and the winning agent executes the task. An optional `min_bid_threshold` rejects winners whose confidence falls below a minimum — if the winning bid's confidence is below the threshold, the result is marked as not allocated.

Agents that raise exceptions during bid generation are excluded from the auction and captured as `AgentFailure` entries in the result (with `agent_name`, `error_type`, and `error_message`). If the winning agent fails during execution, the error is captured in `execution_error` on the result.

### Bid Generators

Each `BiddableAgent` pairs an agent with a bid generator:

- **FixedBidGenerator** — returns a predetermined bid with fixed confidence, capabilities, and estimated cost. Useful when agent capabilities are known in advance and don't vary by task.
- **LLMBidGenerator** — uses an LLM to assess the agent's suitability for the task. The LLM evaluates an `agent_description` against the task and produces a structured bid with confidence, capabilities, cost, and reasoning. Provides more accurate bidding but adds an LLM call per agent per auction.

Custom generators implement the `BidGenerator` protocol:

```python
async def generate(self, agent_name: str, task: str, *, emitter: EventEmitter) -> Bid: ...
```

The `emitter` argument is keyword-only and required — `Bidding.run` passes its own run-scoped emitter so any work done inside `generate` (LLM calls, tool invocations) is traced under the caller's run. Custom generators that call an LLM should wrap the client per-call with `InstrumentedLLMClient(client, emitter=emitter, label="bid")` (or a label of your choosing) before invoking it — this is how `LLMBidGenerator` makes bid-phase LLM spend visible in the run's `summary.total_input_tokens` / `total_output_tokens`. The `label` argument partitions bid-phase events from agent-phase events in the trace, so adopters can separate bid-phase from winner-phase spend when querying `/api/observatory/runs/{run_id}/events`.

### Allocation Strategies

- **HighestConfidence** (default) — selects the bid with the highest confidence score. Straightforward when confidence is the primary signal.
- **LowestCost** — selects the bid with the lowest `estimated_cost`. Ignores bids without cost estimates. Use for cost-sensitive environments.
- **WeightedScore** — computes a weighted score across configurable dimensions (confidence, cost, capabilities). Values are normalized across all bids before weighting. Provides the most nuanced selection when multiple factors matter.

Custom strategies implement the `AllocationStrategy` protocol.

**When to use:**

- Allocating tasks to the most suitable agent from a pool
- When agent capability varies by task type
- Cost-sensitive environments where you want competitive allocation

**When not to use:**

- When you know in advance which agent should handle the task — just call it directly
- With only 2 agents — simpler to use a conditional or orchestrator

The result includes `winning_bid`, `all_bids`, `bid_failures`, `execution_result` (output from the winning agent), `execution_error`, and `allocated` (whether a winner was selected and executed).

Emits `BiddingStartEvent`, `BidReceivedEvent` (per bid), `BidAllocatedEvent` (winner selection with rejection reason if applicable), and `BiddingCompleteEvent`. `LLMBidGenerator` additionally emits one `LLMRequestEvent` + one `LLMResponseEvent` per participant — both with `label="bid"` — so the run's summary counts bid-phase LLM spend alongside the winning agent's calls.

> **See also:** [examples/multi_agent/bidding.py](../../examples/multi_agent/bidding.py) — allocation strategies, basic auction with `FixedBidGenerator`, minimum bid threshold rejection, and event trace inspection.

## Judge-routed allocation

`JudgeRouter` runs a centralised comparative-judgment routing call: a single judge LLM sees every candidate agent together and returns the full ranking in one call. The top-ranked candidate then executes the task. This is the comparative peer to `Bidding` — `Bidding` collects independent self-rated bids, while `JudgeRouter` collects one comparative judgment across all candidates. Both target the same allocation problem; they differ in how the routing decision is reached.

Use `JudgeRouter` when independent self-assessment is fragile. The structural failure mode of independent bidding is **self-overclaim** — each agent rates itself in isolation against no peer signal, and ratings tend to converge on uniformly high confidence. A judge sees the whole pool at once, can discriminate ("billing-specialist is uniquely positioned; the others are adjacent at best"), and produces a properly differentiated ranking in a single LLM call.

`JudgeRouter` reuses `BiddableAgent` so adopters can swap `Bidding` ↔ `JudgeRouter` at the call site without rebuilding agents. The `bid_generator` field is unused by `JudgeRouter` — that's intentional; the type stays shared so primitives are interchangeable.

### Calibration anchors

The default prompt template `DEFAULT_CALIBRATED_JUDGE_PROMPT_TEMPLATE` carries four calibration anchors that the judge anchors confidence against:

- **0.9 — uniquely positioned**: the task falls squarely inside the candidate's stated expertise and no closer specialist is plausible among the candidates.
- **0.7 — capable**: the candidate can handle the task, but a closer specialist exists in the list.
- **0.4 — adjacent**: the candidate has tangentially related expertise; another candidate is clearly better.
- **0.0 — out of scope**: the task is outside the candidate's described scope.

Anchored bands counter the uniformly-high-score failure mode by giving the judge a discrete vocabulary for relative fit. Templates are validated at construction — any custom template missing the required `{participants}` or `{task}` placeholder raises `ValueError` from `__init__` rather than at first call.

The same calibration approach is also available to `LLMBidGenerator` via the `bid_prompt_template=DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE` keyword argument, when you want self-rated bids that nonetheless anchor against the four bands.

### Optional confidence threshold

`min_confidence_threshold` rejects winners below the threshold. When the top-ranked candidate's confidence falls below the threshold, the result is marked as not allocated, the `JudgeAllocatedEvent` carries `rejection_reason="below_threshold"`, and the full ranking is still surfaced on the result so callers can inspect the rejected candidates.

### Failure modes

- **Empty ranking** — the judge returns no candidates. Result is `winner=None`, `allocated=False`, `judge_error="empty_ranking"`.
- **Unknown agent** — the judge names a candidate that isn't in the participant list. Result is `winner=None`, `allocated=False`, `judge_error="unknown_agent: <name>"`.
- **Judge LLM exception** — propagates to the caller. Surface failures, don't mask them.
- **Winning agent execution exception** — captured in `JudgeRouterResult.execution_error`, mirroring `BiddingResult.execution_error`.

**When to use:**

- Pools of specialists with overlapping but distinct scope where comparative discrimination beats independent self-rating
- Routing scenarios where one judge call is preferable to N per-agent bid calls
- Allocation paths where you want a centralised audit trail (the judge's reasoning per candidate is in the ranking events)

**When not to use:**

- Genuine peer-bid auctions where independent self-assessment carries semantic meaning — use [Bidding](#bidding)
- Multi-step decomposition where one specialist isn't enough — use [Orchestrator](#orchestrator)
- 2-agent routing — a `Conditional` workflow is simpler

The result includes `winner` (the top `RankedCandidate` or `None`), `ranking` (the full ordered list of `RankedCandidate`), `execution_result`, `allocated`, `judge_error`, and `execution_error`.

Emits `JudgeRoutingStartEvent`, `JudgeRankingEvent` (one per candidate, carrying rank, confidence, capabilities, optional cost, and reasoning), `JudgeAllocatedEvent` (winner selection with rejection reason if applicable), and `JudgeRoutingCompleteEvent`. The judge call wraps an `InstrumentedLLMClient` with `label="judge"` so judge-phase spend rolls into the run's `summary.total_input_tokens` / `summary.total_output_tokens` alongside the winning agent's calls — parallel to how `LLMBidGenerator` labels bid-phase calls `"bid"`.

> **See also:** [examples/multi_agent/judge_router.py](../../examples/multi_agent/judge_router.py) — comparative judgment, calibration-anchor template injection, below-threshold rejection, full event-trace assertions, and a side-by-side `Bidding` vs `JudgeRouter` trace comparison.

## Debate

`Debate` pits agents against each other in structured adversarial reasoning. Each debater argues an assigned position across multiple rounds, responding to opposing arguments via a shared transcript. A resolution strategy then evaluates the full transcript and produces a verdict with a winner, reasoning, and synthesis of the strongest arguments.

The flow works in rounds: in round 1, each debater presents their opening argument for their assigned position. In subsequent rounds, debaters see the full transcript (all previous arguments from all parties) and respond to opposing arguments. After all rounds complete, the resolution strategy evaluates the debate. Each `Debater` wraps an agent with a `position` string that defines what they're arguing for.

### Resolution Strategies

- **JudgeResolution** — uses a separate agent to judge the debate. The judge receives the formatted transcript and produces a free-form verdict. Gives the judge full agency but the output is unstructured.
- **LLMJudgeResolution** — uses an LLM with structured output to produce a typed verdict (winner, reasoning, synthesis). Optionally accepts evaluation `criteria` that guide the judge's assessment. Preferred when you need a structured, parseable outcome.

Custom strategies implement the `ResolutionStrategy` protocol.

**When to use:**

- Decisions requiring adversarial examination of alternatives
- Exploring trade-offs between competing approaches
- Generating comprehensive pro/con analysis

**When not to use:**

- When you need agreement rather than argumentation — use [Consensus](#consensus)
- Factual questions with clear answers — debate adds unnecessary overhead
- More than 2-3 positions — debate becomes unwieldy with too many cross-responses

The result includes `resolution` (with winner, reasoning, and synthesis), `transcript` (all `Argument` objects in order), `rounds_completed`, and `termination_reason`.

Emits `DebateStartEvent`, `DebateArgumentEvent` (per argument), `DebateResolutionEvent` (judge verdict), and `DebateCompleteEvent`.

> **See also:** [examples/multi_agent/debate.py](../../examples/multi_agent/debate.py) — JudgeResolution, LLMJudgeResolution with criteria, custom resolution strategies, multi-party debate, and event verification.

## Consensus

`Consensus` gathers independent responses from multiple agents and aggregates them into a collective decision. By default, all agents respond in parallel in a single round, and the aggregation strategy produces a result. The result includes an `agreement_level` (0.0–1.0) and a `vote_distribution` showing how responses grouped.

### Deliberation

Enable deliberation by providing a `DeliberationConfig` to run multiple rounds. In round 1, agents respond independently and in parallel. In subsequent rounds, each agent sees the other agents' responses from the previous round and provides a revised answer. After each round, an agreement function measures convergence.

If agreement meets the `agreement_threshold`, the process stops early with `termination_reason="agreement_reached"`. If `max_rounds` is reached without sufficient agreement, the `fallback_strategy` (defaults to `MajorityVoting`) aggregates the final round's responses. You can provide a custom `agreement_fn` to measure convergence differently from the default string-equality grouping.

Diversity in agents — different LLMs, prompts, or perspectives — is essential for deliberation to add value. Identical agents produce identical responses, making multiple rounds pointless.

### Aggregation Strategies

- **MajorityVoting** (default) — groups responses by equality and picks the largest group. Accepts an optional `eq_fn` for custom equality comparison (e.g., comparing only the first word of each response).
- **WeightedVoting** — like majority voting, but each response gets a weight from a `weight_fn`. The group with the highest total weight wins. Useful for weighting by agent speed, confidence, or other metadata.
- **BestOfN** — scores each response individually using a `scorer` function and picks the highest-scoring one. Supports both sync and async scorers.

Custom strategies implement the `AggregationStrategy` protocol.

**When to use:**

- Decisions requiring agreement from multiple perspectives
- Reducing variance by aggregating multiple independent responses
- Quality-critical outputs where majority agreement increases confidence

**When not to use:**

- When one expert is clearly better — use direct delegation instead
- When you need adversarial analysis — use [Debate](#debate)
- With homogeneous agents — you'll get the same answer N times with no added value

The result includes `aggregation` (with `result`, `agreement_level`, `vote_distribution`, and `strategy`), `responses` (all `ConsensusResponse` objects across all rounds), `rounds_completed`, `termination_reason` (`"single_round"`, `"agreement_reached"`, or `"max_rounds"`), and `agents_participated`.

Emits `ConsensusStartEvent`, `ConsensusVoteEvent` (per response), `ConsensusAgreementEvent` (during deliberation rounds), and `ConsensusCompleteEvent`.

> **See also:** [examples/multi_agent/consensus.py](../../examples/multi_agent/consensus.py) — all three aggregation strategies (MajorityVoting, WeightedVoting, BestOfN), deliberation with convergence, and deliberation with max-rounds fallback.

## Combining Coordination Patterns

These patterns compose with each other and with [orchestration](orchestration.md) workflows:

- **Supervisor + Orchestrator**: Wrap an orchestrator in a supervisor to monitor overall coordination quality. The supervisor's triggers can evaluate whether the orchestrator's delegation and synthesis met quality standards.
- **Bidding → Supervised execution**: Select the best agent via bidding, then supervise its execution with quality triggers. Combines capability-based routing with output quality assurance.
- **Blackboard + Consensus**: Use blackboard for iterative shared refinement, then consensus for final agreement on the result. The blackboard builds the artifact; consensus validates it.
- **Debate + Orchestrator**: An orchestrator delegates a contentious decision to a debate, then acts on the resolution. Useful when one subtask benefits from adversarial analysis.

The key principle is that coordination patterns operate on agents, and agents are the universal unit. Any agent — including one produced by a coordination pattern — can be wrapped, supervised, or composed into a larger pattern.

When combining patterns, be mindful of compounding latency and cost. Each layer adds LLM calls. A supervisor wrapping an orchestrator that delegates to debating agents could involve dozens of LLM calls for a single user request. Profile your coordination stack and simplify aggressively.

## Pitfalls

**Over-coordinating.** Every coordination layer adds latency and LLM calls. Use the simplest pattern that solves your problem. Most tasks don't need debate or consensus — start with direct delegation and add coordination only when you have evidence it's needed.

**Supervision loops.** A supervisor that always retries will loop up to `max_retries`. Make sure triggers aren't too strict — the agent may never satisfy an impossible quality bar. Consider using ESCALATE rather than RETRY for borderline cases.

**Blackboard without termination.** Without a proper termination condition, the blackboard runs for `max_rounds` even if no progress is being made. Prefer `BlackboardCompositeTermination` with both `NoNewContributions` and `MaxRoundsTermination` to handle both stagnation and runaway loops.

**Consensus with identical agents.** Deliberation with identical LLM clients and system prompts produces identical responses. Diversity in agents (different LLMs, prompts, or perspectives) is essential for consensus to add value. Without diversity, you're paying for N identical responses.

**Bidding with inaccurate confidence.** `FixedBidGenerator` requires you to set confidence accurately. If agents can't self-assess well, use `LLMBidGenerator` or a custom generator that evaluates capability more robustly. Inaccurate confidence scores defeat the purpose of competitive allocation.

**Debate with too many positions.** Each additional debater multiplies the number of cross-responses per round. With N debaters and R rounds, you get N×R arguments plus the resolution. Keep debates focused with 2-3 positions.

## See Also

- [Multi-Agent Foundations](multi-agent-foundations.md) — agent-as-tool, handoff, context transfer, broadcast, message bus, peer network
- [Orchestration](orchestration.md) — fixed-structure workflows: sequential, parallel, DAG, loop, map-reduce, conditional
- [Agent Types](agent-types.md) — coordination compatibility matrix for each agent type
