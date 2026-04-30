import { describe, expect, it } from "vitest";
import type { AgentInfo, TraceEvent } from "../../src/types";
import { findRelatedAgents } from "../../src/utils/related-agents";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAgents(...names: string[]): AgentInfo[] {
	return names.map((name) => ({
		agent_name: name,
		agent_type: "react",
		span_id: `span-${name}`,
		capabilities: [],
		stats: {
			llm_calls: 0,
			tool_calls: 0,
			input_tokens: 0,
			output_tokens: 0,
			duration_ms: 0,
			errors: 0,
			iterations: 0,
		},
	}));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("findRelatedAgents", () => {
	it("returns empty array when there are no multi-agent events", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({ event_type: "agent.start" }),
			makeEvent({ event_type: "agent.complete" }),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([]);
	});

	it("returns empty array when current agent is not involved", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: { caller_agent: "bob", delegate_agent: "carol", task: "x", transfer_strategy: "full" },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([]);
	});

	// --- Delegation ---

	it("detects delegation: caller → delegate", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: { caller_agent: "alice", delegate_agent: "bob", task: "research", transfer_strategy: "full" },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([
			{ agentName: "bob", spanId: "span-bob", relationship: "delegated to", eventType: "multi_agent.delegation" },
		]);
	});

	it("detects delegation: delegate → caller (reverse)", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: { caller_agent: "alice", delegate_agent: "bob", task: "research", transfer_strategy: "full" },
			}),
		];

		const result = findRelatedAgents("bob", events, agents);
		expect(result).toEqual([
			{ agentName: "alice", spanId: "span-alice", relationship: "delegated from", eventType: "multi_agent.delegation" },
		]);
	});

	// --- Handoff ---

	it("detects handoff: from → to", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: { from_agent: "alice", to_agent: "bob", payload_fields: [], payload_size: 100 },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([
			{ agentName: "bob", spanId: "span-bob", relationship: "handed off to", eventType: "multi_agent.handoff" },
		]);
	});

	it("detects handoff: to → from (reverse)", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: { from_agent: "alice", to_agent: "bob", payload_fields: [], payload_size: 100 },
			}),
		];

		const result = findRelatedAgents("bob", events, agents);
		expect(result).toEqual([
			{
				agentName: "alice",
				spanId: "span-alice",
				relationship: "received handoff from",
				eventType: "multi_agent.handoff",
			},
		]);
	});

	// --- Supervision ---

	it("detects supervision: supervised agent → reassigned_to", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "alice",
					action: "reassign",
					trigger_name: "quality",
					reassigned_to: "bob",
					attempt: 1,
				},
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([
			{ agentName: "bob", spanId: "span-bob", relationship: "reassigned to", eventType: "multi_agent.supervision" },
		]);
	});

	it("detects supervision: reassigned_to → supervised", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "alice",
					action: "reassign",
					trigger_name: "quality",
					reassigned_to: "bob",
					attempt: 1,
				},
			}),
		];

		const result = findRelatedAgents("bob", events, agents);
		expect(result).toEqual([
			{ agentName: "alice", spanId: "span-alice", relationship: "supervises", eventType: "multi_agent.supervision" },
		]);
	});

	it("ignores supervision without reassigned_to", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "alice",
					action: "warn",
					trigger_name: "quality",
					reassigned_to: null,
					attempt: 1,
				},
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([]);
	});

	// --- Bidding ---

	it("detects bidding participants", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.bidding.start",
				payload: { task: "task", participant_names: ["alice", "bob", "carol"] },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toHaveLength(2);
		expect(result.map((r) => r.agentName).sort()).toEqual(["bob", "carol"]);
		expect(result[0].relationship).toBe("bidding participant");
	});

	// --- Debate ---

	it("detects debate participants", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.debate.start",
				payload: {
					task: "task",
					debater_names: ["alice", "bob"],
					positions: {},
					max_rounds: 3,
					resolution_strategy: "judge",
				},
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([
			{
				agentName: "bob",
				spanId: "span-bob",
				relationship: "debate participant",
				eventType: "multi_agent.debate.start",
			},
		]);
	});

	// --- Consensus ---

	it("detects consensus participants", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.consensus.start",
				payload: {
					task: "task",
					agent_names: ["alice", "bob", "carol"],
					strategy: "voting",
					deliberation_enabled: true,
				},
			}),
		];

		const result = findRelatedAgents("bob", events, agents);
		expect(result).toHaveLength(2);
		expect(result.map((r) => r.agentName).sort()).toEqual(["alice", "carol"]);
		expect(result[0].relationship).toBe("consensus participant");
	});

	// --- Broadcast ---

	it("detects broadcast participants", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.broadcast.start",
				payload: { task: "task", agent_names: ["alice", "bob"], response_strategy: "all" },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([
			{
				agentName: "bob",
				spanId: "span-bob",
				relationship: "broadcast participant",
				eventType: "multi_agent.broadcast.start",
			},
		]);
	});

	// --- Blackboard ---

	it("detects blackboard participants", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "blackboard.start",
				payload: {
					task: "task",
					agent_names: ["alice", "bob", "carol"],
					control_strategy: "round_robin",
					max_rounds: 5,
				},
			}),
		];

		const result = findRelatedAgents("carol", events, agents);
		expect(result).toHaveLength(2);
		expect(result.map((r) => r.agentName).sort()).toEqual(["alice", "bob"]);
		expect(result[0].relationship).toBe("blackboard participant");
	});

	// --- Peer Network ---

	it("detects peer network: entry agent sees peers", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.peer.start",
				payload: {
					task: "task",
					entry_agent: "alice",
					peer_names: ["bob", "carol"],
					peer_descriptions: {},
					max_invocations: 10,
				},
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toHaveLength(2);
		expect(result.map((r) => r.agentName).sort()).toEqual(["bob", "carol"]);
		expect(result[0].relationship).toBe("peer");
	});

	it("detects peer network: peer sees entry agent and other peers", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.peer.start",
				payload: {
					task: "task",
					entry_agent: "alice",
					peer_names: ["bob", "carol"],
					peer_descriptions: {},
					max_invocations: 10,
				},
			}),
		];

		const result = findRelatedAgents("bob", events, agents);
		expect(result.map((r) => r.agentName).sort()).toEqual(["alice", "carol"]);
		const aliceRel = result.find((r) => r.agentName === "alice");
		expect(aliceRel?.relationship).toBe("peer network entry");
	});

	it("detects peer consultation", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.peer.consultation",
				payload: { from_agent: "alice", to_agent: "bob", message: "help", consultation_number: 1, remaining_budget: 5 },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([
			{ agentName: "bob", spanId: "span-bob", relationship: "consulted", eventType: "multi_agent.peer.consultation" },
		]);
	});

	// --- Message Bus ---

	it("detects message bus participants from subscriptions", () => {
		const agents = makeAgents("alice", "bob", "carol");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.bus.start",
				payload: {
					seed_topics: ["topic1"],
					seed_count: 1,
					subscriber_count: 3,
					subscriptions: { topic1: ["alice", "bob", "carol"] },
					max_messages: 10,
					max_depth: 3,
				},
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toHaveLength(2);
		expect(result.map((r) => r.agentName).sort()).toEqual(["bob", "carol"]);
		expect(result[0].relationship).toBe("message bus participant");
	});

	// --- Edge cases ---

	it("does not include agents not in the agents list", () => {
		const agents = makeAgents("alice"); // bob not in agents list
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: { caller_agent: "alice", delegate_agent: "bob", task: "x", transfer_strategy: "full" },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result).toEqual([]);
	});

	it("does not duplicate agents across multiple events", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: { caller_agent: "alice", delegate_agent: "bob", task: "x", transfer_strategy: "full" },
			}),
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: { from_agent: "alice", to_agent: "bob", payload_fields: [], payload_size: 100 },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		// First relationship wins — delegation comes first
		expect(result).toHaveLength(1);
		expect(result[0].agentName).toBe("bob");
		expect(result[0].relationship).toBe("delegated to");
	});

	it("does not include self in results", () => {
		const agents = makeAgents("alice", "bob");
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "multi_agent.consensus.start",
				payload: { task: "task", agent_names: ["alice", "bob"], strategy: "voting", deliberation_enabled: true },
			}),
		];

		const result = findRelatedAgents("alice", events, agents);
		expect(result.every((r) => r.agentName !== "alice")).toBe(true);
	});
});
