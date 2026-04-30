import { describe, expect, it } from "vitest";
import type { SpanTreeNode } from "../../src/types";
import { detectPatterns } from "../../src/utils/pattern-detector";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSpan(overrides: Partial<SpanTreeNode> & { span_id: string }): SpanTreeNode {
	return {
		parent_span_id: null,
		name: "span",
		summary: {
			event_count: 0,
			duration_ms: null,
			has_errors: false,
			agent_name: null,
			agent_type: null,
		},
		events: [],
		children: [],
		...overrides,
	};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("detectPatterns", () => {
	it("returns empty array when tree has no pattern events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({ event_type: "agent.start", span_id: "root" }),
				makeEvent({ event_type: "agent.complete", span_id: "root" }),
			],
		});
		expect(detectPatterns(tree)).toEqual([]);
	});

	// --- Delegation ---

	it("detects a single delegation", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.delegation",
					span_id: "root",
					payload: {
						caller_agent: "Alice",
						delegate_agent: "Bob",
						task: "research",
						transfer_strategy: "full",
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("delegation");
		expect(result[0].label).toBe("Alice → Bob");
		expect(result[0].spanId).toBe("root");
		expect(result[0].events).toHaveLength(1);
	});

	it("detects multiple delegations as separate instances", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.delegation",
					span_id: "root",
					payload: {
						caller_agent: "Alice",
						delegate_agent: "Bob",
						task: "task-1",
						transfer_strategy: "full",
					},
				}),
				makeEvent({
					event_type: "multi_agent.delegation",
					span_id: "root",
					payload: {
						caller_agent: "Alice",
						delegate_agent: "Carol",
						task: "task-2",
						transfer_strategy: "partial",
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(2);
		expect(result[0].label).toBe("Alice → Bob");
		expect(result[1].label).toBe("Alice → Carol");
	});

	// --- Broadcast ---

	it("detects a broadcast pattern with start, responses, and complete", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.broadcast.start",
					span_id: "root",
					payload: {
						task: "analyze",
						agent_names: ["A", "B", "C"],
						response_strategy: "collect_all",
					},
				}),
				makeEvent({
					event_type: "multi_agent.broadcast.response",
					span_id: "root",
					payload: { agent_name: "A", output: "result A", steps: 3 },
				}),
				makeEvent({
					event_type: "multi_agent.broadcast.response",
					span_id: "root",
					payload: { agent_name: "B", output: "result B", steps: 2 },
				}),
				makeEvent({
					event_type: "multi_agent.broadcast.complete",
					span_id: "root",
					payload: {
						total_agents: 3,
						responses_collected: 2,
						response_strategy: "collect_all",
						aggregated_output: "combined",
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("broadcast");
		expect(result[0].label).toBe("Broadcast to 3 agents");
		expect(result[0].events).toHaveLength(4);
	});

	// --- Bidding ---

	it("detects a bidding pattern with all events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.bidding.start",
					span_id: "root",
					payload: {
						task: "process claim",
						participant_names: ["A", "B"],
					},
				}),
				makeEvent({
					event_type: "multi_agent.bidding.bid",
					span_id: "root",
					payload: { agent_name: "A", confidence: 0.9, reasoning: "good fit", estimated_cost: 2.0 },
				}),
				makeEvent({
					event_type: "multi_agent.bidding.bid",
					span_id: "root",
					payload: { agent_name: "B", confidence: 0.6, reasoning: "partial fit", estimated_cost: 1.5 },
				}),
				makeEvent({
					event_type: "multi_agent.bidding.allocated",
					span_id: "root",
					payload: { winner: "A", confidence: 0.9, total_bids: 2 },
				}),
				makeEvent({
					event_type: "multi_agent.bidding.complete",
					span_id: "root",
					payload: { winner: "A", total_participants: 2, allocated: true },
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("bidding");
		expect(result[0].label).toBe("Bidding (2 participants)");
		expect(result[0].events).toHaveLength(5);
	});

	// --- Supervision ---

	it("detects supervision with multiple interventions for same agent", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.supervision",
					span_id: "root",
					payload: {
						supervised_agent: "Agent-A",
						action: "retry",
						trigger_name: "QualityTrigger",
						feedback: "needs more detail",
						attempt: 1,
					},
				}),
				makeEvent({
					event_type: "multi_agent.supervision",
					span_id: "root",
					payload: {
						supervised_agent: "Agent-A",
						action: "reassign",
						trigger_name: "QualityTrigger",
						feedback: "still insufficient",
						reassigned_to: "Agent-B",
						attempt: 2,
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("supervision");
		expect(result[0].label).toBe("Supervision: Agent-A (2 interventions)");
		expect(result[0].events).toHaveLength(2);
	});

	it("groups supervision events by supervised agent", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.supervision",
					span_id: "root",
					payload: { supervised_agent: "Agent-A", action: "retry", trigger_name: "T1", attempt: 1 },
				}),
				makeEvent({
					event_type: "multi_agent.supervision",
					span_id: "root",
					payload: { supervised_agent: "Agent-B", action: "retry", trigger_name: "T2", attempt: 1 },
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(2);
		expect(result.map((p) => p.type)).toEqual(["supervision", "supervision"]);
		const labels = result.map((p) => p.label);
		expect(labels).toContain("Supervision: Agent-A (1 intervention)");
		expect(labels).toContain("Supervision: Agent-B (1 intervention)");
	});

	// --- Handoff ---

	it("detects a handoff chain across spans", () => {
		const tree = makeSpan({
			span_id: "root",
			children: [
				makeSpan({
					span_id: "span-a",
					parent_span_id: "root",
					events: [
						makeEvent({
							event_type: "multi_agent.handoff",
							span_id: "span-a",
							timestamp: "2026-03-05T10:00:00Z",
							payload: {
								from_agent: "Agent-A",
								to_agent: "Agent-B",
								payload_fields: ["output"],
								payload_size: 100,
							},
						}),
					],
				}),
				makeSpan({
					span_id: "span-b",
					parent_span_id: "root",
					events: [
						makeEvent({
							event_type: "multi_agent.handoff",
							span_id: "span-b",
							timestamp: "2026-03-05T10:01:00Z",
							payload: {
								from_agent: "Agent-B",
								to_agent: "Agent-C",
								payload_fields: ["summary"],
								payload_size: 200,
							},
						}),
					],
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("handoff");
		expect(result[0].label).toBe("Agent-A → Agent-B → Agent-C");
		expect(result[0].events).toHaveLength(2);
	});

	it("handles 5-agent handoff chain", () => {
		const agents = ["A", "B", "C", "D", "E"];
		const children = agents.slice(0, -1).map((from, i) => {
			const to = agents[i + 1];
			return makeSpan({
				span_id: `span-${from}`,
				parent_span_id: "root",
				events: [
					makeEvent({
						event_type: "multi_agent.handoff",
						span_id: `span-${from}`,
						timestamp: `2026-03-05T10:0${i}:00Z`,
						payload: {
							from_agent: from,
							to_agent: to,
							payload_fields: ["data"],
							payload_size: 100 * (i + 1),
						},
					}),
				],
			});
		});

		const tree = makeSpan({ span_id: "root", children });
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].label).toBe("A → B → C → D → E");
		expect(result[0].events).toHaveLength(4);
	});

	// --- Mixed patterns ---

	it("detects mixed patterns in one run", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.delegation",
					span_id: "root",
					payload: { caller_agent: "Orchestrator", delegate_agent: "Worker-1", task: "t1", transfer_strategy: "full" },
				}),
				makeEvent({
					event_type: "multi_agent.bidding.start",
					span_id: "root",
					payload: { task: "bid-task", participant_names: ["X", "Y"] },
				}),
				makeEvent({
					event_type: "multi_agent.bidding.complete",
					span_id: "root",
					payload: { winner: "X", total_participants: 2, allocated: true },
				}),
			],
			children: [
				makeSpan({
					span_id: "child",
					parent_span_id: "root",
					events: [
						makeEvent({
							event_type: "multi_agent.broadcast.start",
							span_id: "child",
							payload: { task: "fan-out", agent_names: ["A", "B"], response_strategy: "all" },
						}),
					],
				}),
			],
		});
		const result = detectPatterns(tree);
		// delegation, bidding, broadcast
		expect(result).toHaveLength(3);
		const types = result.map((p) => p.type);
		expect(types).toContain("delegation");
		expect(types).toContain("bidding");
		expect(types).toContain("broadcast");
	});

	// --- Pattern events at different tree depths ---

	it("detects patterns at different tree depths", () => {
		const tree = makeSpan({
			span_id: "root",
			children: [
				makeSpan({
					span_id: "level-1",
					parent_span_id: "root",
					events: [
						makeEvent({
							event_type: "multi_agent.delegation",
							span_id: "level-1",
							payload: { caller_agent: "A", delegate_agent: "B", task: "t1", transfer_strategy: "full" },
						}),
					],
					children: [
						makeSpan({
							span_id: "level-2",
							parent_span_id: "level-1",
							events: [
								makeEvent({
									event_type: "multi_agent.delegation",
									span_id: "level-2",
									payload: { caller_agent: "B", delegate_agent: "C", task: "t2", transfer_strategy: "partial" },
								}),
							],
						}),
					],
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(2);
		expect(result[0].label).toBe("A → B");
		expect(result[1].label).toBe("B → C");
	});

	// --- Edge cases ---

	it("returns empty array for tree with no events", () => {
		const tree = makeSpan({ span_id: "root" });
		expect(detectPatterns(tree)).toEqual([]);
	});

	it("handles tree with only children, no events at root", () => {
		const tree = makeSpan({
			span_id: "root",
			children: [
				makeSpan({
					span_id: "child",
					parent_span_id: "root",
					events: [
						makeEvent({
							event_type: "multi_agent.delegation",
							span_id: "child",
							payload: { caller_agent: "X", delegate_agent: "Y", task: "t", transfer_strategy: "full" },
						}),
					],
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].spanId).toBe("child");
	});

	// --- Debate ---

	it("detects a debate pattern with all events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.debate.start",
					span_id: "root",
					payload: {
						task: "discuss policy",
						debater_names: ["Alice", "Bob"],
						positions: { Alice: "for", Bob: "against" },
						max_rounds: 3,
						resolution_strategy: "judge",
					},
				}),
				makeEvent({
					event_type: "multi_agent.debate.argument",
					span_id: "root",
					payload: { round: 1, agent_name: "Alice", position: "for", argument: "arg1" },
				}),
				makeEvent({
					event_type: "multi_agent.debate.argument",
					span_id: "root",
					payload: { round: 1, agent_name: "Bob", position: "against", argument: "arg2" },
				}),
				makeEvent({
					event_type: "multi_agent.debate.resolution",
					span_id: "root",
					payload: { winner: "Alice", reasoning: "better arguments", rounds_completed: 2 },
				}),
				makeEvent({
					event_type: "multi_agent.debate.complete",
					span_id: "root",
					payload: { winner: "Alice", rounds_completed: 2, total_arguments: 4, termination_reason: "resolved" },
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("debate");
		expect(result[0].label).toBe("Debate (2 debaters, 2 rounds)");
		expect(result[0].events).toHaveLength(5);
	});

	it("detects debate without complete event (in progress)", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.debate.start",
					span_id: "root",
					payload: {
						task: "discuss",
						debater_names: ["A", "B", "C"],
						positions: {},
						max_rounds: 5,
						resolution_strategy: "vote",
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].label).toBe("Debate (3 debaters)");
	});

	// --- Consensus ---

	it("detects a consensus pattern with all events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.consensus.start",
					span_id: "root",
					payload: {
						task: "evaluate options",
						agent_names: ["A", "B", "C"],
						strategy: "MajorityVoting",
						deliberation_enabled: false,
					},
				}),
				makeEvent({
					event_type: "multi_agent.consensus.vote",
					span_id: "root",
					payload: { agent_name: "A", output: "option 1", round: 1 },
				}),
				makeEvent({
					event_type: "multi_agent.consensus.vote",
					span_id: "root",
					payload: { agent_name: "B", output: "option 1", round: 1 },
				}),
				makeEvent({
					event_type: "multi_agent.consensus.agreement",
					span_id: "root",
					payload: { round: 1, agreement_level: 0.8, converged: true },
				}),
				makeEvent({
					event_type: "multi_agent.consensus.complete",
					span_id: "root",
					payload: {
						strategy: "MajorityVoting",
						rounds_completed: 1,
						final_agreement: 0.8,
						agents_participated: 3,
						termination_reason: "converged",
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("consensus");
		expect(result[0].label).toBe("Consensus (3 agents, MajorityVoting)");
		expect(result[0].events).toHaveLength(5);
	});

	// --- Blackboard ---

	it("detects a blackboard pattern with all events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "blackboard.start",
					span_id: "root",
					payload: {
						task: "solve problem",
						agent_names: ["A", "B"],
						control_strategy: "ScheduledControl",
						max_rounds: 5,
					},
				}),
				makeEvent({
					event_type: "blackboard.round",
					span_id: "root",
					payload: { round_number: 1, agents_activated: ["A"], contributions: 2, total_contributions: 2 },
				}),
				makeEvent({
					event_type: "blackboard.round",
					span_id: "root",
					payload: { round_number: 2, agents_activated: ["B"], contributions: 1, total_contributions: 3 },
				}),
				makeEvent({
					event_type: "blackboard.complete",
					span_id: "root",
					payload: {
						rounds_completed: 2,
						termination_reason: "converged",
						total_contributions: 3,
						agent_contributions: { A: 2, B: 1 },
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("blackboard");
		expect(result[0].label).toBe("Blackboard (2 agents, ScheduledControl)");
		expect(result[0].events).toHaveLength(4);
	});

	// --- Peer Network ---

	it("detects a peer network pattern with all events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.peer.start",
					span_id: "root",
					payload: {
						task: "research topic",
						entry_agent: "Coordinator",
						peer_names: ["Expert-A", "Expert-B", "Expert-C"],
						peer_descriptions: { "Expert-A": "domain A", "Expert-B": "domain B", "Expert-C": "domain C" },
						max_invocations: 10,
					},
				}),
				makeEvent({
					event_type: "multi_agent.peer.consultation",
					span_id: "root",
					payload: {
						from_agent: "Coordinator",
						to_agent: "Expert-A",
						message: "help",
						consultation_number: 1,
						remaining_budget: 9,
					},
				}),
				makeEvent({
					event_type: "multi_agent.peer.consultation",
					span_id: "root",
					payload: {
						from_agent: "Expert-A",
						to_agent: "Expert-B",
						message: "need input",
						consultation_number: 2,
						remaining_budget: 8,
					},
				}),
				makeEvent({
					event_type: "multi_agent.peer.complete",
					span_id: "root",
					payload: {
						entry_agent: "Coordinator",
						total_consultations: 2,
						invocations_used: 3,
						agents_consulted: ["Expert-A", "Expert-B"],
						termination_reason: "completed",
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("peer_network");
		expect(result[0].label).toBe("Peer network from Coordinator (3 peers)");
		expect(result[0].events).toHaveLength(4);
	});

	// --- Message Bus ---

	it("detects a message bus pattern with all events", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.bus.start",
					span_id: "root",
					payload: {
						seed_topics: ["topic-A", "topic-B"],
						seed_count: 2,
						subscriber_count: 3,
						subscriptions: { "topic-A": ["Agent-1"], "topic-B": ["Agent-2", "Agent-3"] },
						max_messages: 100,
						max_depth: 5,
					},
				}),
				makeEvent({
					event_type: "multi_agent.bus.published",
					span_id: "root",
					payload: { message_id: "msg-1", topic: "topic-A", author: "seed", content: "initial", depth: 0 },
				}),
				makeEvent({
					event_type: "multi_agent.bus.delivered",
					span_id: "root",
					payload: {
						message_id: "msg-1",
						topic: "topic-A",
						agent_name: "Agent-1",
						output: "processed",
						steps: 2,
						messages_published: 1,
					},
				}),
				makeEvent({
					event_type: "multi_agent.bus.published",
					span_id: "root",
					payload: {
						message_id: "msg-2",
						topic: "topic-B",
						author: "Agent-1",
						content: "derived",
						depth: 1,
						parent_message_id: "msg-1",
					},
				}),
				makeEvent({
					event_type: "multi_agent.bus.complete",
					span_id: "root",
					payload: {
						total_messages: 2,
						total_executions: 3,
						max_depth_reached: 1,
						termination_reason: "all_processed",
						agent_execution_counts: { "Agent-1": 1, "Agent-2": 1, "Agent-3": 1 },
					},
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].type).toBe("message_bus");
		expect(result[0].label).toBe("Message bus (2 topics, 2 messages)");
		expect(result[0].events).toHaveLength(5);
	});

	it("message bus label uses published count when no complete event", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				makeEvent({
					event_type: "multi_agent.bus.start",
					span_id: "root",
					payload: {
						seed_topics: ["t1"],
						seed_count: 1,
						subscriber_count: 1,
						subscriptions: {},
						max_messages: 50,
						max_depth: 3,
					},
				}),
				makeEvent({
					event_type: "multi_agent.bus.published",
					span_id: "root",
					payload: { message_id: "m1", topic: "t1", author: "seed", content: "x", depth: 0 },
				}),
				makeEvent({
					event_type: "multi_agent.bus.published",
					span_id: "root",
					payload: { message_id: "m2", topic: "t1", author: "A", content: "y", depth: 1, parent_message_id: "m1" },
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(1);
		expect(result[0].label).toBe("Message bus (1 topics, 2 messages)");
	});

	// --- 5C patterns co-occurring with 5B patterns ---

	it("detects all 5C patterns together with 5B patterns", () => {
		const tree = makeSpan({
			span_id: "root",
			events: [
				// 5B: delegation
				makeEvent({
					event_type: "multi_agent.delegation",
					span_id: "root",
					payload: { caller_agent: "O", delegate_agent: "W", task: "t", transfer_strategy: "full" },
				}),
				// 5C: debate
				makeEvent({
					event_type: "multi_agent.debate.start",
					span_id: "root",
					payload: { task: "d", debater_names: ["A", "B"], positions: {}, max_rounds: 3, resolution_strategy: "judge" },
				}),
			],
			children: [
				makeSpan({
					span_id: "child-1",
					parent_span_id: "root",
					events: [
						// 5C: consensus
						makeEvent({
							event_type: "multi_agent.consensus.start",
							span_id: "child-1",
							payload: { task: "c", agent_names: ["X", "Y"], strategy: "WeightedVoting", deliberation_enabled: true },
						}),
					],
				}),
				makeSpan({
					span_id: "child-2",
					parent_span_id: "root",
					events: [
						// 5C: blackboard
						makeEvent({
							event_type: "blackboard.start",
							span_id: "child-2",
							payload: { task: "b", agent_names: ["P"], control_strategy: "Opportunistic", max_rounds: 10 },
						}),
					],
				}),
			],
		});
		const result = detectPatterns(tree);
		const types = result.map((p) => p.type);
		expect(types).toContain("delegation");
		expect(types).toContain("debate");
		expect(types).toContain("consensus");
		expect(types).toContain("blackboard");
	});

	// --- 5C patterns nested at different depths ---

	it("detects 5C patterns at different tree depths", () => {
		const tree = makeSpan({
			span_id: "root",
			children: [
				makeSpan({
					span_id: "level-1",
					parent_span_id: "root",
					events: [
						makeEvent({
							event_type: "multi_agent.peer.start",
							span_id: "level-1",
							payload: {
								task: "t",
								entry_agent: "E",
								peer_names: ["P1", "P2"],
								peer_descriptions: {},
								max_invocations: 5,
							},
						}),
					],
					children: [
						makeSpan({
							span_id: "level-2",
							parent_span_id: "level-1",
							events: [
								makeEvent({
									event_type: "multi_agent.bus.start",
									span_id: "level-2",
									payload: {
										seed_topics: ["t"],
										seed_count: 1,
										subscriber_count: 1,
										subscriptions: {},
										max_messages: 10,
										max_depth: 2,
									},
								}),
							],
						}),
					],
				}),
			],
		});
		const result = detectPatterns(tree);
		expect(result).toHaveLength(2);
		const types = result.map((p) => p.type);
		expect(types).toContain("peer_network");
		expect(types).toContain("message_bus");
		expect(result.find((p) => p.type === "peer_network")?.spanId).toBe("level-1");
		expect(result.find((p) => p.type === "message_bus")?.spanId).toBe("level-2");
	});
});
