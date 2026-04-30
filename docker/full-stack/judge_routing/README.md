# Judge-routed request handling runner

One of the showcase runners inside `docker/full-stack/`.

## What this runner demonstrates

Four tool-using specialist agents — `billing-specialist`,
`technical-specialist`, `account-specialist`, `policy-specialist` — are
routed by a single comparative-judgment LLM call (the *judge*) and the
winning specialist answers using the in-memory fixtures in
[`fixtures.py`](./fixtures.py) and the tools in
[`tools.py`](./tools.py).

This is the tooled counterpart to the
[auction-routing runner](../auction_routing/README.md). Two things are
different:

1. **Comparative judgment, not self-assessment.** The SDK's
   [`JudgeRouter`](../../../nanitics/composition/multi_agent/judge_router.py)
   makes one LLM call that ranks every candidate together — the judge
   sees all four specialists' descriptions side by side and produces a
   full ranking. This counter-balances the self-overclaim bias inherent
   to per-agent independent bids.
2. **Real tools.** Each specialist carries a small bundle of tools
   that operate on an in-memory fixture (invoices, accounts, KB
   articles, policy clauses). The Observatory trace shows the winning
   specialist's actual tool calls — not a single thin LLM step.

A runner-local `_GroundedJudgeRouter` extends the SDK's primitive with
the same cost-grounding pattern as auction-routing: each candidate's
ranking entry includes a 1–5 `complexity` integer, and `estimated_cost`
is `base_rate * complexity`. The judge call is wrapped with
`InstrumentedLLMClient(label="judge")` so judge-phase tokens roll into
the run's `summary.total_input_tokens` /
`summary.total_output_tokens`.

There is no HITL branch on this runner — the judge always allocates to
the top-ranked candidate, and consumers can build a confidence gate
client-side from `ranking[0].confidence` in the response.

## Endpoints

One route mounts under `/runners/judge-routing/`:

| Method | Path | Body | Success | Failure |
|---|---|---|---|---|
| `POST` | `/runners/judge-routing/handle` | `{"request_text": "..."}` | `200` with the envelope below. | `422` when `request_text` is missing or empty (Pydantic schema rejection). `503` when the judge LLM raises or returns an unusable ranking. |

### `/handle` response envelope

```json
{
  "run_id": "...",
  "winner": "billing-specialist",
  "ranking": [
    {"agent_name": "billing-specialist", "confidence": 0.9, "capabilities": ["..."], "estimated_cost": 0.06, "reasoning": "..."},
    ...
  ],
  "answer": "the specialist's answer — or null if the winning agent's run produced no output",
  "trace_url": "/api/observatory/runs/<run-id>"
}
```

Every response returns the full `ranking` list (always four entries —
one per specialist) so the client can see exactly what the judge
produced. The `answer` is the winning specialist's
[`AgentResult.output`](../../../nanitics/core/agents/base.py); a `null`
value means the winning agent's run did not produce a final string.

The `run_id` field is the Observatory run id. It always matches the
tail of `trace_url`.

## Observatory trace shape

Each `/handle` call produces one trace (one `run_id`).

One `JudgeRoutingStartEvent`, four `JudgeRankingEvent`s (one per
candidate, in rank order; each carries the calibrated confidence and
grounded cost), one `JudgeAllocatedEvent` naming the winner, the
winning `ReActAgent`'s full trace (`AgentStartEvent` → tool calls →
LLM steps → `AgentCompleteEvent`), and one `JudgeRoutingCompleteEvent`
with `allocated=True`. The judge LLM call shows up as one
`LLMRequestEvent` + one `LLMResponseEvent`, both with `label="judge"`,
and rolls into `summary.total_input_tokens` /
`summary.total_output_tokens`.

## Adding or replacing a specialist

1. Append (or replace) an entry in `SPECIALIST_SPECS` inside
   [`runner.py`](./runner.py). All six fields are required — the
   calibration anchors, grounded-cost multiplier, and tool bundle all
   need them:

   ```python
   _SpecialistSpec(
       name="shipping-specialist",
       system_prompt=(
           "You are a shipping specialist. Use ``track_shipment`` "
           "and ``contact_carrier`` to investigate. Answer in 2–4 "
           "sentences citing the tracking number and carrier."
       ),
       agent_description=(
           "Shipping specialist for delivery tracking, carrier "
           "escalations, and address changes."
       ),
       out_of_scope=(
           "Does not handle billing/invoicing, account access, "
           "product bugs, or policy/compliance questions."
       ),
       base_rate=0.02,
       tools=shipping_tools(),
   ),
   ```

2. Add the matching tool factories to [`tools.py`](./tools.py) and a
   bundle helper alongside `billing_tools()`,
   `technical_tools()`, `account_tools()`, `policy_tools()`.

3. Restart the stack so `register()` rebuilds the specialist list:

   ```sh
   just full-stack-compose-down && just full-stack-compose
   ```

The `slug` (the `name` field) must be unique across the roster and
kebab-case; the Observatory and the `JudgeRankingEvent` surface it
as-is.
