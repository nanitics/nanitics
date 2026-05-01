import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { EventDetailPanel } from "../../src/components/event-detail/event-detail-panel";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { createDefaultRegistrations, createDefaultRegistry } from "../../src/registry/default-renderers";
import type { TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderEvent(event: TraceEvent) {
	const client = new ObservatoryClient("/test");
	const registry = createDefaultRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<EventDetailPanel event={event} />
		</ObservatoryProvider>,
	);
}

function getSummary(event: TraceEvent): string {
	const registrations = createDefaultRegistrations();
	for (const reg of registrations) {
		if (reg.matches(event.event_type) && reg.summary) {
			return reg.summary(event);
		}
	}
	return event.event_type;
}

// ---------------------------------------------------------------------------
// Workflow Event Renderers
// ---------------------------------------------------------------------------

describe("Workflow event renderers", () => {
	describe("workflow.start", () => {
		it("renders workflow name and type", () => {
			renderEvent(
				makeEvent({
					event_type: "workflow.start",
					payload: { workflow_name: "research", workflow_type: "sequential", step_count: 5 },
				}),
			);
			expect(screen.getByText("research")).toBeInTheDocument();
			expect(screen.getByText("sequential")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "workflow.start",
					payload: { workflow_name: "research", workflow_type: "sequential", step_count: 5 },
				}),
			);
			expect(summary).toBe("Started workflow 'research' (sequential, 5 steps)");
		});

		it("handles missing optional fields", () => {
			renderEvent(makeEvent({ event_type: "workflow.start", payload: {} }));
			expect(screen.getByText("workflow.start")).toBeInTheDocument();
		});
	});

	describe("workflow.structure", () => {
		it("renders step table", () => {
			renderEvent(
				makeEvent({
					event_type: "workflow.structure",
					payload: {
						workflow_name: "analysis",
						steps: [
							{ name: "fetch", step_type: "agent", depends_on: [] },
							{ name: "process", step_type: "function", depends_on: ["fetch"] },
						],
					},
				}),
			);
			expect(screen.getAllByText("fetch").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("process")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "workflow.structure",
					payload: { steps: [{ name: "a" }, { name: "b" }, { name: "c" }] },
				}),
			);
			expect(summary).toBe("Workflow structure: 3 steps");
		});
	});

	describe("workflow.step.complete", () => {
		it("renders step name and duration", () => {
			renderEvent(
				makeEvent({
					event_type: "workflow.step.complete",
					payload: { step_name: "fetch", step_index: 0, step_duration_ms: 1200 },
				}),
			);
			expect(screen.getByText("fetch")).toBeInTheDocument();
			expect(screen.getAllByText("1.2s").length).toBeGreaterThanOrEqual(1);
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "workflow.step.complete",
					payload: { step_name: "fetch", step_duration_ms: 500 },
				}),
			);
			expect(summary).toBe("Step 'fetch' completed (500ms)");
		});
	});

	describe("workflow.complete", () => {
		it("renders workflow name and steps executed", () => {
			renderEvent(
				makeEvent({
					event_type: "workflow.complete",
					payload: { workflow_name: "analysis", workflow_type: "dag", total_steps_executed: 4 },
				}),
			);
			expect(screen.getByText("analysis")).toBeInTheDocument();
			expect(screen.getAllByText("4").length).toBeGreaterThanOrEqual(1);
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "workflow.complete",
					payload: { workflow_name: "analysis", total_steps_executed: 4 },
				}),
			);
			expect(summary).toBe("Workflow 'analysis' completed (4 steps)");
		});
	});

	describe("workflow.error", () => {
		it("renders error details", () => {
			renderEvent(
				makeEvent({
					event_type: "workflow.error",
					payload: {
						workflow_name: "analysis",
						error_type: "TimeoutError",
						error_message: "Step timed out",
						failed_step: "fetch",
					},
				}),
			);
			expect(screen.getByText("TimeoutError")).toBeInTheDocument();
			expect(screen.getByText("Step timed out")).toBeInTheDocument();
			expect(screen.getByText("fetch")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "workflow.error",
					payload: { workflow_name: "analysis", error_type: "TimeoutError" },
				}),
			);
			expect(summary).toBe("Workflow 'analysis' failed: TimeoutError");
		});
	});
});

// ---------------------------------------------------------------------------
// Delegation / Handoff / Supervision
// ---------------------------------------------------------------------------

describe("Delegation/Handoff/Supervision renderers", () => {
	describe("multi_agent.delegation", () => {
		it("renders caller and delegate", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.delegation",
					payload: {
						caller_agent: "orchestrator",
						delegate_agent: "researcher",
						task: "Find data",
						transfer_strategy: "full",
					},
				}),
			);
			expect(screen.getByText("orchestrator")).toBeInTheDocument();
			expect(screen.getByText("researcher")).toBeInTheDocument();
			expect(screen.getByText("full")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.delegation",
					payload: { delegate_agent: "researcher", transfer_strategy: "full" },
				}),
			);
			expect(summary).toBe("Delegated to researcher (full)");
		});
	});

	describe("multi_agent.handoff", () => {
		it("renders from/to agents and payload size", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.handoff",
					payload: { from_agent: "A", to_agent: "B", payload_fields: ["context", "results"], payload_size: 4096 },
				}),
			);
			expect(screen.getByText("A")).toBeInTheDocument();
			expect(screen.getByText("B")).toBeInTheDocument();
			expect(screen.getByText("4096 bytes")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.handoff",
					payload: { from_agent: "A", to_agent: "B", payload_size: 4096 },
				}),
			);
			expect(summary).toBe("Handoff: A → B (4096 bytes)");
		});
	});

	describe("multi_agent.supervision", () => {
		it("renders action and supervised agent", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.supervision",
					payload: { supervised_agent: "worker", action: "retry", trigger_name: "quality_check", attempt: 2 },
				}),
			);
			expect(screen.getByText("worker")).toBeInTheDocument();
			expect(screen.getByText("retry")).toBeInTheDocument();
			expect(screen.getByText("quality_check")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.supervision",
					payload: { action: "retry", supervised_agent: "worker" },
				}),
			);
			expect(summary).toBe("Supervision: retry on worker");
		});
	});
});

// ---------------------------------------------------------------------------
// Bidding
// ---------------------------------------------------------------------------

describe("Bidding renderers", () => {
	describe("multi_agent.bidding.start", () => {
		it("renders participant names", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bidding.start",
					payload: { task: "Classify document", participant_names: ["agent-a", "agent-b", "agent-c"] },
				}),
			);
			expect(screen.getByText("agent-a")).toBeInTheDocument();
			expect(screen.getByText("agent-b")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bidding.start",
					payload: { participant_names: ["a", "b"] },
				}),
			);
			expect(summary).toBe("Bidding started (2 participants)");
		});
	});

	describe("multi_agent.bidding.bid", () => {
		it("renders agent name and confidence", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bidding.bid",
					payload: { agent_name: "expert", confidence: 0.85, reasoning: "High relevance" },
				}),
			);
			expect(screen.getByText("expert")).toBeInTheDocument();
			expect(screen.getByText("85%")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bidding.bid",
					payload: { agent_name: "expert", confidence: 0.85 },
				}),
			);
			expect(summary).toBe("Bid from expert: 85%");
		});
	});

	describe("multi_agent.bidding.allocated", () => {
		it("renders winner", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bidding.allocated",
					payload: { winner: "expert", confidence: 0.9, total_bids: 3 },
				}),
			);
			expect(screen.getByText("expert")).toBeInTheDocument();
			expect(screen.getByText("winner")).toBeInTheDocument();
		});

		it("renders rejection when no winner", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bidding.allocated",
					payload: { winner: null, total_bids: 3, rejection_reason: "Low confidence" },
				}),
			);
			expect(screen.getByText("no winner")).toBeInTheDocument();
			expect(screen.getByText("Low confidence")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			expect(
				getSummary(makeEvent({ event_type: "multi_agent.bidding.allocated", payload: { winner: "expert" } })),
			).toBe("Allocated to expert");
			expect(getSummary(makeEvent({ event_type: "multi_agent.bidding.allocated", payload: { winner: null } }))).toBe(
				"All bids rejected",
			);
		});
	});

	describe("multi_agent.bidding.complete", () => {
		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bidding.complete",
					payload: { winner: "expert", total_participants: 3, allocated: true },
				}),
			);
			expect(summary).toBe("Bidding complete: expert");
		});
	});
});

// ---------------------------------------------------------------------------
// Debate
// ---------------------------------------------------------------------------

describe("Debate renderers", () => {
	describe("multi_agent.debate.start", () => {
		it("renders debater names and max rounds", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.debate.start",
					payload: {
						task: "Best approach",
						debater_names: ["optimist", "pessimist"],
						positions: { optimist: "pro", pessimist: "con" },
						max_rounds: 3,
						resolution_strategy: "judge",
					},
				}),
			);
			expect(screen.getByText("optimist")).toBeInTheDocument();
			expect(screen.getByText("pessimist")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.debate.start",
					payload: { debater_names: ["a", "b"], max_rounds: 5 },
				}),
			);
			expect(summary).toBe("Debate started (2 debaters, max 5 rounds)");
		});
	});

	describe("multi_agent.debate.argument", () => {
		it("renders round, agent and position", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.debate.argument",
					payload: { round: 1, agent_name: "optimist", position: "pro", argument: "I believe..." },
				}),
			);
			expect(screen.getByText("optimist")).toBeInTheDocument();
			expect(screen.getByText("pro")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.debate.argument",
					payload: { round: 2, agent_name: "pessimist", position: "con" },
				}),
			);
			expect(summary).toBe("Round 2: pessimist argues (con)");
		});
	});

	describe("multi_agent.debate.resolution", () => {
		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.debate.resolution",
					payload: { winner: "optimist", rounds_completed: 3 },
				}),
			);
			expect(summary).toBe("Resolved: optimist (3 rounds)");
		});
	});

	describe("multi_agent.debate.complete", () => {
		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.debate.complete",
					payload: { winner: "optimist", total_arguments: 6 },
				}),
			);
			expect(summary).toBe("Debate complete: optimist (6 arguments)");
		});
	});
});

// ---------------------------------------------------------------------------
// Consensus
// ---------------------------------------------------------------------------

describe("Consensus renderers", () => {
	describe("multi_agent.consensus.start", () => {
		it("renders agent names and strategy", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.consensus.start",
					payload: {
						task: "Agree on plan",
						agent_names: ["a1", "a2", "a3"],
						strategy: "majority",
						deliberation_enabled: true,
					},
				}),
			);
			expect(screen.getByText("a1")).toBeInTheDocument();
			expect(screen.getByText("majority")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.consensus.start",
					payload: { agent_names: ["a", "b", "c"], strategy: "unanimous" },
				}),
			);
			expect(summary).toBe("Consensus started (3 agents, unanimous)");
		});
	});

	describe("multi_agent.consensus.vote", () => {
		it("renders agent name and round", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.consensus.vote",
					payload: { agent_name: "a1", output: "I agree", round: 1 },
				}),
			);
			expect(screen.getByText("a1")).toBeInTheDocument();
		});

		it("renders error styling when error present", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.consensus.vote",
					payload: { agent_name: "a1", round: 1, error: "Timed out" },
				}),
			);
			expect(screen.getByText("Timed out")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.consensus.vote",
					payload: { agent_name: "a1", round: 2 },
				}),
			);
			expect(summary).toBe("Vote from a1 (round 2)");
		});
	});

	describe("multi_agent.consensus.agreement", () => {
		it("renders agreement level as percentage", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.consensus.agreement",
					payload: { round: 2, agreement_level: 0.75, converged: false },
				}),
			);
			expect(screen.getByText("75%")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.consensus.agreement",
					payload: { agreement_level: 0.85, round: 3 },
				}),
			);
			expect(summary).toBe("Agreement: 85% (round 3)");
		});
	});

	describe("multi_agent.consensus.complete", () => {
		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.consensus.complete",
					payload: { final_agreement: 0.92, strategy: "majority" },
				}),
			);
			expect(summary).toBe("Consensus: 92% (majority)");
		});
	});
});

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

describe("Broadcast renderers", () => {
	describe("multi_agent.broadcast.start", () => {
		it("renders agent list and strategy", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.broadcast.start",
					payload: { task: "Analyze data", agent_names: ["a1", "a2"], response_strategy: "all" },
				}),
			);
			expect(screen.getByText("a1")).toBeInTheDocument();
			expect(screen.getByText("all")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.broadcast.start",
					payload: { agent_names: ["a", "b", "c"], response_strategy: "first" },
				}),
			);
			expect(summary).toBe("Broadcast to 3 agents (first)");
		});
	});

	describe("multi_agent.broadcast.response", () => {
		it("renders agent name and steps", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.broadcast.response",
					payload: { agent_name: "analyst", output: "Results here", steps: 5 },
				}),
			);
			expect(screen.getByText("analyst")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.broadcast.response",
					payload: { agent_name: "analyst", steps: 5 },
				}),
			);
			expect(summary).toBe("Response from analyst (5 steps)");
		});
	});

	describe("multi_agent.broadcast.complete", () => {
		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.broadcast.complete",
					payload: { responses_collected: 3, total_agents: 4 },
				}),
			);
			expect(summary).toBe("Broadcast complete (3/4 responses)");
		});
	});
});

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------

describe("Blackboard renderers", () => {
	describe("blackboard.start", () => {
		it("renders agent names and control strategy", () => {
			renderEvent(
				makeEvent({
					event_type: "blackboard.start",
					payload: {
						task: "Solve problem",
						agent_names: ["expert1", "expert2"],
						control_strategy: "round-robin",
						max_rounds: 5,
					},
				}),
			);
			expect(screen.getByText("expert1")).toBeInTheDocument();
			expect(screen.getByText("round-robin")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "blackboard.start",
					payload: { agent_names: ["a", "b"], max_rounds: 10 },
				}),
			);
			expect(summary).toBe("Blackboard started (2 agents, max 10 rounds)");
		});
	});

	describe("blackboard.round", () => {
		it("renders round number and contributions", () => {
			renderEvent(
				makeEvent({
					event_type: "blackboard.round",
					payload: { round_number: 3, agents_activated: ["expert1"], contributions: 2, total_contributions: 8 },
				}),
			);
			expect(screen.getByText("Round 3")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "blackboard.round",
					payload: { round_number: 3, contributions: 2 },
				}),
			);
			expect(summary).toBe("Round 3: 2 contributions");
		});
	});

	describe("blackboard.complete", () => {
		it("renders per-agent contributions", () => {
			renderEvent(
				makeEvent({
					event_type: "blackboard.complete",
					payload: {
						rounds_completed: 5,
						termination_reason: "converged",
						total_contributions: 12,
						agent_contributions: { expert1: 7, expert2: 5 },
					},
				}),
			);
			expect(screen.getByText("expert1")).toBeInTheDocument();
			expect(screen.getByText("expert2")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "blackboard.complete",
					payload: { rounds_completed: 5, total_contributions: 12 },
				}),
			);
			expect(summary).toBe("Blackboard complete (5 rounds, 12 contributions)");
		});
	});
});

// ---------------------------------------------------------------------------
// Peer Network
// ---------------------------------------------------------------------------

describe("Peer Network renderers", () => {
	describe("multi_agent.peer.start", () => {
		it("renders entry agent and peers", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.peer.start",
					payload: {
						task: "Research topic",
						entry_agent: "coordinator",
						peer_names: ["expert1", "expert2"],
						peer_descriptions: { expert1: "Domain expert", expert2: "Analyst" },
						max_invocations: 10,
					},
				}),
			);
			expect(screen.getByText("coordinator")).toBeInTheDocument();
			expect(screen.getByText("expert1")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.peer.start",
					payload: { entry_agent: "coordinator" },
				}),
			);
			expect(summary).toBe("Peer network started (entry: coordinator)");
		});
	});

	describe("multi_agent.peer.consultation", () => {
		it("renders from/to agents", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.peer.consultation",
					payload: {
						from_agent: "A",
						to_agent: "B",
						message: "Need help",
						consultation_number: 1,
						remaining_budget: 9,
					},
				}),
			);
			expect(screen.getByText("A")).toBeInTheDocument();
			expect(screen.getByText("B")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.peer.consultation",
					payload: { from_agent: "A", to_agent: "B" },
				}),
			);
			expect(summary).toBe("Consultation: A → B");
		});
	});

	describe("multi_agent.peer.complete", () => {
		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.peer.complete",
					payload: { total_consultations: 7 },
				}),
			);
			expect(summary).toBe("Peer network complete (7 consultations)");
		});
	});
});

// ---------------------------------------------------------------------------
// Message Bus
// ---------------------------------------------------------------------------

describe("Message Bus renderers", () => {
	describe("multi_agent.bus.start", () => {
		it("renders topics and subscriber count", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bus.start",
					payload: {
						seed_topics: ["analysis", "review"],
						seed_count: 2,
						subscriber_count: 3,
						subscriptions: { agent1: ["analysis"], agent2: ["review"] },
						max_messages: 50,
						max_depth: 5,
					},
				}),
			);
			expect(screen.getByText("analysis")).toBeInTheDocument();
			expect(screen.getByText("review")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bus.start",
					payload: { seed_topics: ["a", "b"], subscriber_count: 3 },
				}),
			);
			expect(summary).toBe("Message bus started (2 topics, 3 subscribers)");
		});
	});

	describe("multi_agent.bus.published", () => {
		it("renders topic, author and depth", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bus.published",
					payload: { message_id: "msg-1", topic: "analysis", author: "agent1", content: "Results", depth: 0 },
				}),
			);
			expect(screen.getByText("analysis")).toBeInTheDocument();
			expect(screen.getByText("agent1")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bus.published",
					payload: { topic: "analysis", author: "agent1" },
				}),
			);
			expect(summary).toBe("Published to analysis by agent1");
		});
	});

	describe("multi_agent.bus.delivered", () => {
		it("renders topic and agent", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bus.delivered",
					payload: {
						message_id: "msg-1",
						topic: "analysis",
						agent_name: "agent2",
						output: "Processed",
						steps: 3,
						messages_published: 1,
					},
				}),
			);
			expect(screen.getByText("agent2")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bus.delivered",
					payload: { topic: "analysis", agent_name: "agent2" },
				}),
			);
			expect(summary).toBe("Delivered analysis to agent2");
		});
	});

	describe("multi_agent.bus.complete", () => {
		it("renders per-agent execution counts", () => {
			renderEvent(
				makeEvent({
					event_type: "multi_agent.bus.complete",
					payload: {
						total_messages: 15,
						total_executions: 20,
						max_depth_reached: 3,
						termination_reason: "max_messages",
						agent_execution_counts: { agent1: 10, agent2: 10 },
					},
				}),
			);
			expect(screen.getByText("agent1")).toBeInTheDocument();
			expect(screen.getByText("agent2")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "multi_agent.bus.complete",
					payload: { total_messages: 15, max_depth_reached: 3 },
				}),
			);
			expect(summary).toBe("Bus complete (15 messages, depth 3)");
		});
	});
});

// ---------------------------------------------------------------------------
// Cross-cutting: all renderers match correctly
// ---------------------------------------------------------------------------

describe("Workflow renderer matching", () => {
	const allEventTypes = [
		"workflow.start",
		"workflow.structure",
		"workflow.step.complete",
		"workflow.complete",
		"workflow.error",
		"multi_agent.delegation",
		"multi_agent.handoff",
		"multi_agent.supervision",
		"multi_agent.bidding.start",
		"multi_agent.bidding.bid",
		"multi_agent.bidding.allocated",
		"multi_agent.bidding.complete",
		"multi_agent.debate.start",
		"multi_agent.debate.argument",
		"multi_agent.debate.resolution",
		"multi_agent.debate.complete",
		"multi_agent.consensus.start",
		"multi_agent.consensus.vote",
		"multi_agent.consensus.agreement",
		"multi_agent.consensus.complete",
		"multi_agent.broadcast.start",
		"multi_agent.broadcast.response",
		"multi_agent.broadcast.complete",
		"blackboard.start",
		"blackboard.round",
		"blackboard.complete",
		"multi_agent.peer.start",
		"multi_agent.peer.consultation",
		"multi_agent.peer.complete",
		"multi_agent.bus.start",
		"multi_agent.bus.published",
		"multi_agent.bus.delivered",
		"multi_agent.bus.complete",
	];

	const registrations = createDefaultRegistrations();

	it.each(allEventTypes)("has a dedicated renderer for %s", (eventType) => {
		const match = registrations.find((r) => r.matches(eventType) && r.priority >= 0);
		expect(match).toBeDefined();
		expect(match?.summary).toBeDefined();
	});

	it("all 33 workflow/multi-agent event types have renderers", () => {
		expect(allEventTypes.length).toBe(33);
	});
});
