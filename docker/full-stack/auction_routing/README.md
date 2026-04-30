# Auction-routed request handling runner

One of the showcase runners inside `docker/full-stack/`.

## What this runner demonstrates

Four specialist agents — `billing-specialist`, `technical-specialist`,
`account-specialist`, `policy-specialist` — each self-assess their fit
for every incoming request through a runner-side
[`_GroundedCostBidGenerator`](./runner.py), which extends the SDK's
[`LLMBidGenerator`](../../../nanitics/composition/multi_agent/bidding.py)
with two corrections to the original bidding pattern:

1. **Calibrated confidence** — the bid prompt is built from
   [`DEFAULT_CALIBRATED_BID_PROMPT_TEMPLATE`](../../../nanitics/composition/multi_agent/bidding.py),
   which carries four-tier anchors (0.9 = uniquely positioned; 0.7 =
   capable but a closer specialist may exist; 0.4 = adjacent; 0.0 =
   out of scope). Each specialist's bid description joins its in-scope
   summary with an explicit `out_of_scope` line so the anchors have
   something to bite on.
2. **Grounded cost** — each specialist carries a `base_rate` and the
   bid LLM emits a 1–5 `complexity` integer; `estimated_cost = base_rate *
   complexity`. This replaces the round-number $50/$100 figures the
   uncalibrated prompt used to hallucinate.

[`Bidding`](../../../nanitics/composition/multi_agent/bidding.py) gathers
the four bids in parallel and selects the winner with
`HighestConfidence(tiebreaker=LowestCost())`: the highest calibrated
bid wins, and on a strict tie the bid with the lower grounded cost
wins. There is **no HITL branch** — the auction always allocates to a
specialist; consumers that need a human-handoff gate can build one
client-side around `winner.confidence` from the response envelope.

The showcase is the *routing* — "self-assessed routing that is not a
hardcoded classifier" — paired with the calibration and tiebreaker
machinery the SDK ships for taming self-overclaim and ties. For the
underlying primitives see:

- [`docs/guides/multi-agent-coordination.md#bidding`](../../../docs/guides/multi-agent-coordination.md#bidding)
  — bidding semantics, allocation strategies, calibration anchors,
  tiebreaker chains.

## Endpoints

One route mounts under `/runners/auction-routing/`:

| Method | Path | Body | Success | Failure |
|---|---|---|---|---|
| `POST` | `/runners/auction-routing/handle` | `{"request_text": "..."}` | `200` with the envelope below. | `422` when `request_text` is missing or empty (Pydantic schema rejection). |

### `/handle` response envelope

```json
{
  "run_id": "...",
  "outcome": "specialist_answered",
  "winner": "billing-specialist",
  "bids": [
    {"agent_name": "...", "confidence": 0.9, "capabilities": ["..."], "estimated_cost": 0.06, "reasoning": "..."},
    ...
  ],
  "answer": "the specialist's answer — or null if the winning agent's run produced no output",
  "trace_url": "/api/observatory/runs/<run-id>"
}
```

Every response returns the full `bids` list so the client can see
exactly what the auction produced. The `outcome` literal is always
`"specialist_answered"`; the runner has no other branches.

The `run_id` field is the Observatory run id — the same identifier
adopters pass to `GET /api/observatory/runs/{run_id}` for the trace. It
always matches the tail of `trace_url`.

## Observatory trace shape

Each `/handle` call produces one trace (one `run_id`). The Observatory
UI at `/api/observatory/` renders the span tree.

One `BiddingStartEvent`, four `BidReceivedEvent`s (one per specialist,
in parallel; each carries the calibrated confidence and grounded
cost), one `BidAllocatedEvent` naming the winner, the winning
`ReActAgent`'s full trace (`AgentStartEvent` → tool/LLM steps →
`AgentCompleteEvent`), and one `BiddingCompleteEvent` with
`allocated=True`. Bid-phase LLM calls roll up into the run's
`summary.total_input_tokens` / `summary.total_output_tokens` via
`InstrumentedLLMClient(label="bid")` — the cost-forecasting adopter's
read picks them up alongside the winning agent's calls.

## Adding or replacing a specialist

1. Append (or replace) an entry in `SPECIALIST_SPECS` inside
   [`runner.py`](./runner.py). All five fields are required — the
   calibration anchors and grounded-cost multiplier both need them:

   ```python
   _SpecialistSpec(
       name="shipping-specialist",
       system_prompt=(
           "You are a shipping specialist. Handle delivery "
           "tracking, carrier escalations, and address changes. "
           "Answer in 2–4 sentences."
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
   ),
   ```

2. Restart the stack so `register()` rebuilds the specialist list:

   ```sh
   just full-stack-compose-down && just full-stack-compose
   ```

No other changes are needed. The `slug` (the `name` field) must be
unique across the roster and kebab-case; the Observatory and the
`BidReceivedEvent` surfaces it as-is. Specialists are toolless by
design on this runner — adding tools is a larger change; the
`judge-routing` runner showcases the tooled variant.
