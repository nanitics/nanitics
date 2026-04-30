import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvaluationPanel } from "../../src/components/panels/evaluation-panel";
import { HITLPanel } from "../../src/components/panels/hitl-panel";
import { MemoryInspectorPanel } from "../../src/components/panels/memory-inspector-panel";
import { PlanningPanel } from "../../src/components/panels/planning-panel";
import { createDefaultPanelRegistry } from "../../src/registry/default-panels";
import type { AgentInfo, SpanTreeNode } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "test-agent",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: ["tool_use", "memory"],
	stats: {
		llm_calls: 3,
		tool_calls: 2,
		input_tokens: 1000,
		output_tokens: 500,
		duration_ms: 5000,
		errors: 0,
		iterations: 3,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "test-agent",
	summary: {
		event_count: 10,
		duration_ms: 5000,
		has_errors: false,
		agent_name: "test-agent",
		agent_type: "react",
	},
	events: [],
	children: [],
};

// ---------------------------------------------------------------------------
// MemoryInspectorPanel
// ---------------------------------------------------------------------------

describe("MemoryInspectorPanel", () => {
	it("renders empty state when no memory events", () => {
		render(<MemoryInspectorPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);
		expect(screen.getByText("No memory events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders correct sub-tabs based on event types", () => {
		const events = [
			makeEvent({
				event_type: "memory.working.update",
				payload: { source: "llm", new_content: "new" },
			}),
			makeEvent({
				event_type: "memory.semantic.search",
				payload: { results_count: 3, top_score: 0.95 },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Working Memory")).toBeInTheDocument();
		expect(screen.getByText("Semantic")).toBeInTheDocument();
		expect(screen.getByText("Timeline")).toBeInTheDocument();
		// Episodic, Long-Term, Shared should not appear
		expect(screen.queryByText("Episodic")).not.toBeInTheDocument();
		expect(screen.queryByText("Long-Term")).not.toBeInTheDocument();
		expect(screen.queryByText("Shared")).not.toBeInTheDocument();
	});

	it("shows event count", () => {
		const events = [
			makeEvent({ event_type: "memory.working.update", payload: { source: "llm" } }),
			makeEvent({ event_type: "memory.working.read", payload: { token_count: 500 } }),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);
		expect(screen.getByText("2 memory events")).toBeInTheDocument();
	});

	it("renders working memory updates with source", () => {
		const events = [
			makeEvent({
				event_type: "memory.working.update",
				payload: { source: "llm_response", previous_content: "old", new_content: "new" },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Source: llm_response/)).toBeInTheDocument();
	});

	it("renders semantic search results", () => {
		const events = [
			makeEvent({
				event_type: "memory.semantic.search",
				payload: { query: "test query", results_count: 5, top_score: 0.89 },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Click Semantic tab
		fireEvent.click(screen.getByText("Semantic"));
		expect(screen.getByText(/Search: 5 results/)).toBeInTheDocument();
	});

	it("renders episodic events", () => {
		const events = [
			makeEvent({
				event_type: "memory.episode.record",
				payload: { episode_id: "ep-1", situation: "Agent tried X" },
			}),
			makeEvent({
				event_type: "memory.episode.recall",
				payload: { query: "similar tasks", results_count: 2, top_score: 0.78 },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Episodic")).toBeInTheDocument();
		expect(screen.getByText(/Episode recorded/)).toBeInTheDocument();
	});

	it("renders long-term memory operations", () => {
		const events = [
			makeEvent({
				event_type: "memory.longterm.store",
				payload: { key: "user_pref", value: "dark_mode" },
			}),
			makeEvent({
				event_type: "memory.longterm.retrieve",
				payload: { key: "user_pref", found: true, value: "dark_mode" },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Long-Term")).toBeInTheDocument();
		expect(screen.getByText(/Stored: user_pref/)).toBeInTheDocument();
	});

	it("renders shared memory operations", () => {
		const events = [
			makeEvent({
				event_type: "memory.shared.write",
				payload: { author: "agent-a", content: "shared data", scope: "global" },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Shared")).toBeInTheDocument();
		expect(screen.getByText(/Write by agent-a/)).toBeInTheDocument();
	});

	it("renders timeline sub-tab with all events", () => {
		const events = [
			makeEvent({
				event_type: "memory.working.update",
				payload: { source: "llm" },
			}),
			makeEvent({
				event_type: "memory.semantic.store",
				payload: { entry_id: "e1" },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		fireEvent.click(screen.getByText("Timeline"));
		expect(screen.getByText("update")).toBeInTheDocument();
		expect(screen.getByText("store")).toBeInTheDocument();
	});

	it("handles empty sub-tab gracefully", () => {
		const events = [
			makeEvent({
				event_type: "memory.working.update",
				payload: { source: "llm" },
			}),
		];

		render(<MemoryInspectorPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Working Memory tab has data, but if we click Timeline it should also work
		fireEvent.click(screen.getByText("Timeline"));
		expect(screen.getByText("update")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// PlanningPanel
// ---------------------------------------------------------------------------

describe("PlanningPanel", () => {
	it("renders empty state when no planning events", () => {
		render(<PlanningPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);
		expect(screen.getByText("No planning events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders plan overview from planning.plan.created", () => {
		const events = [
			makeEvent({
				event_type: "planning.plan.created",
				payload: {
					plan_id: "plan-1",
					plan_name: "Research Plan",
					step_count: 3,
					goal_count: 1,
					steps: [
						{ step_id: "s1", description: "Search for papers" },
						{ step_id: "s2", description: "Analyze results" },
						{ step_id: "s3", description: "Write summary" },
					],
				},
			}),
		];

		render(<PlanningPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Plan: Research Plan")).toBeInTheDocument();
		expect(screen.getByText("3 steps")).toBeInTheDocument();
		expect(screen.getByText("Search for papers")).toBeInTheDocument();
		expect(screen.getByText("Analyze results")).toBeInTheDocument();
		expect(screen.getByText("Write summary")).toBeInTheDocument();
	});

	it("renders step status timeline", () => {
		const events = [
			makeEvent({
				event_type: "planning.step.updated",
				payload: {
					step_id: "s1",
					step_description: "Search for papers",
					previous_status: "not_started",
					new_status: "in_progress",
				},
			}),
			makeEvent({
				event_type: "planning.step.updated",
				payload: {
					step_id: "s1",
					step_description: "Search for papers",
					previous_status: "in_progress",
					new_status: "completed",
				},
			}),
		];

		render(<PlanningPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Step Status Timeline")).toBeInTheDocument();
		expect(screen.getByText("Search for papers")).toBeInTheDocument();
		expect(screen.getByText("in_progress")).toBeInTheDocument();
		expect(screen.getAllByText("completed").length).toBeGreaterThanOrEqual(1);
	});

	it("renders plan revision cards", () => {
		const events = [
			makeEvent({
				event_type: "planning.plan.revised",
				payload: {
					reason: "New information discovered",
					steps_before: 3,
					steps_after: 5,
					steps_preserved: 2,
				},
			}),
		];

		render(<PlanningPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Plan Revisions")).toBeInTheDocument();
		expect(screen.getByText("Revision 1")).toBeInTheDocument();
		expect(screen.getByText("3 → 5 steps")).toBeInTheDocument();
		expect(screen.getByText("New information discovered")).toBeInTheDocument();
	});

	it("renders goal tracking", () => {
		const events = [
			makeEvent({
				event_type: "planning.goal.status_changed",
				payload: {
					goal_id: "g1",
					goal_description: "Find relevant papers",
					previous_status: "in_progress",
					new_status: "achieved",
				},
			}),
		];

		render(<PlanningPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Goal Tracking")).toBeInTheDocument();
		expect(screen.getByText("Find relevant papers")).toBeInTheDocument();
		expect(screen.getAllByText("achieved").length).toBeGreaterThanOrEqual(1);
	});

	it("renders all sections together", () => {
		const events = [
			makeEvent({
				event_type: "planning.plan.created",
				payload: {
					plan_name: "Full Plan",
					step_count: 2,
					steps: [
						{ step_id: "s1", description: "Step one" },
						{ step_id: "s2", description: "Step two" },
					],
				},
			}),
			makeEvent({
				event_type: "planning.step.updated",
				payload: {
					step_id: "s1",
					step_description: "Step one",
					previous_status: "not_started",
					new_status: "completed",
				},
			}),
			makeEvent({
				event_type: "planning.plan.revised",
				payload: { reason: "Adjustment", steps_before: 2, steps_after: 3, steps_preserved: 1 },
			}),
			makeEvent({
				event_type: "planning.goal.status_changed",
				payload: {
					goal_id: "g1",
					goal_description: "Main goal",
					previous_status: "not_started",
					new_status: "in_progress",
				},
			}),
		];

		render(<PlanningPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Plan: Full Plan")).toBeInTheDocument();
		expect(screen.getByText("Step Status Timeline")).toBeInTheDocument();
		expect(screen.getByText("Plan Revisions")).toBeInTheDocument();
		expect(screen.getByText("Goal Tracking")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// EvaluationPanel
// ---------------------------------------------------------------------------

describe("EvaluationPanel", () => {
	it("renders empty state when no evaluation events", () => {
		render(<EvaluationPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);
		expect(screen.getByText("No evaluation events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders score progression with multiple evaluations", () => {
		const events = [
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "quality_check",
					verdict: "REVISE",
					score: 0.4,
					feedback: "Needs improvement",
					revision_attempt: 1,
				},
			}),
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "quality_check",
					verdict: "REVISE",
					score: 0.7,
					feedback: "Getting better",
					revision_attempt: 2,
				},
			}),
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "quality_check",
					verdict: "ACCEPT",
					score: 0.95,
					feedback: "Excellent",
					revision_attempt: 3,
				},
			}),
		];

		render(<EvaluationPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("3 evaluations")).toBeInTheDocument();
		expect(screen.getByText("Score Progression")).toBeInTheDocument();
		expect(screen.getByText("Final score: 0.95")).toBeInTheDocument();
	});

	it("renders per-attempt evaluation cards", () => {
		const events = [
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "quality_check",
					verdict: "ACCEPT",
					score: 0.9,
					feedback: "Good work",
					revision_attempt: 1,
				},
			}),
		];

		render(<EvaluationPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("1 evaluation")).toBeInTheDocument();
		expect(screen.getAllByText("ACCEPT").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("Score: 0.90")).toBeInTheDocument();
		expect(screen.getAllByText("quality_check").length).toBeGreaterThanOrEqual(1);

		// Expand to see details
		fireEvent.click(screen.getByText("1 of 1"));
		expect(screen.getByText("Good work")).toBeInTheDocument();
	});

	it("renders reflection cards", () => {
		const events = [
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "eval",
					verdict: "REVISE",
					score: 0.5,
					feedback: "Fix errors",
					revision_attempt: 1,
				},
			}),
			makeEvent({
				event_type: "reflection.generated",
				payload: {
					reflection_text: "I should have checked more carefully",
					evaluation_feedback: "Fix errors",
					episode_id: "ep-123",
					attempt_number: 1,
					max_attempts: 3,
				},
			}),
		];

		render(<EvaluationPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Expand the card to see the reflection
		fireEvent.click(screen.getByText("1 of 1"));
		expect(screen.getByText("reflection")).toBeInTheDocument();
		expect(screen.getByText("I should have checked more carefully")).toBeInTheDocument();
	});

	it("renders revision request within evaluation card", () => {
		const events = [
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "eval",
					verdict: "REVISE",
					score: 0.3,
					feedback: "Too brief",
					revision_attempt: 1,
				},
			}),
			makeEvent({
				event_type: "evaluation.revision",
				payload: {
					feedback: "Please elaborate",
					revision_attempt: 1,
					max_revisions: 3,
				},
			}),
		];

		render(<EvaluationPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		fireEvent.click(screen.getByText("1 of 1"));
		expect(screen.getByText("Revision requested")).toBeInTheDocument();
		expect(screen.getByText("(attempt 1/3)")).toBeInTheDocument();
	});

	it("displays final verdict badge", () => {
		const events = [
			makeEvent({
				event_type: "evaluation.result",
				payload: {
					evaluator_name: "eval",
					verdict: "ACCEPT",
					score: 0.9,
					feedback: "Good",
					revision_attempt: 1,
				},
			}),
		];

		render(<EvaluationPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// The verdict badge in the summary
		const acceptBadges = screen.getAllByText("ACCEPT");
		expect(acceptBadges.length).toBeGreaterThanOrEqual(1);
	});
});

// ---------------------------------------------------------------------------
// HITLPanel
// ---------------------------------------------------------------------------

describe("HITLPanel", () => {
	it("renders empty state when no HITL events", () => {
		render(<HITLPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);
		expect(screen.getByText("No human-in-the-loop events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders paired request/response interactions", () => {
		const events = [
			makeEvent({
				event_type: "hitl.request",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					request_id: "req-1",
					request_type: "approval",
					prompt: "Approve this action?",
					agent_name: "test-agent",
					tool_name: "delete_file",
				},
			}),
			makeEvent({
				event_type: "hitl.response",
				timestamp: "2026-03-05T10:00:30Z",
				payload: {
					request_id: "req-1",
					decision: "approve",
					has_content: false,
					wait_duration_ms: 30000,
				},
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Total")).toBeInTheDocument();
		expect(screen.getAllByText("1").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText(/Approve this action/)).toBeInTheDocument();
		expect(screen.getAllByText("approve").length).toBeGreaterThanOrEqual(1);
	});

	it("highlights pending requests", () => {
		const events = [
			makeEvent({
				event_type: "hitl.request",
				payload: {
					request_id: "req-2",
					request_type: "question",
					prompt: "What should I do next?",
				},
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("1 pending request")).toBeInTheDocument();
		expect(screen.getByText("pending")).toBeInTheDocument();
	});

	it("shows summary statistics", () => {
		const events = [
			makeEvent({
				event_type: "hitl.request",
				payload: {
					request_id: "req-1",
					request_type: "approval",
					prompt: "Approve?",
				},
			}),
			makeEvent({
				event_type: "hitl.response",
				payload: {
					request_id: "req-1",
					decision: "approve",
					has_content: false,
					wait_duration_ms: 5000,
				},
			}),
			makeEvent({
				event_type: "hitl.request",
				payload: {
					request_id: "req-2",
					request_type: "approval",
					prompt: "Approve again?",
				},
			}),
			makeEvent({
				event_type: "hitl.response",
				payload: {
					request_id: "req-2",
					decision: "reject",
					has_content: false,
					wait_duration_ms: 3000,
				},
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Total")).toBeInTheDocument();
		expect(screen.getAllByText("2").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("Approval rate")).toBeInTheDocument();
		expect(screen.getByText("50%")).toBeInTheDocument();
		expect(screen.getByText("Avg wait")).toBeInTheDocument();
	});

	it("expands interaction to show details", () => {
		const events = [
			makeEvent({
				event_type: "hitl.request",
				payload: {
					request_id: "req-1",
					request_type: "approval",
					prompt: "Should I proceed with deletion?",
					agent_name: "cleanup-agent",
					tool_name: "delete_file",
					context: "File: /tmp/old-data.csv",
				},
			}),
			makeEvent({
				event_type: "hitl.response",
				payload: {
					request_id: "req-1",
					decision: "approve",
					has_content: true,
					content: "Yes, proceed",
					wait_duration_ms: 10000,
				},
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Click to expand
		fireEvent.click(screen.getByText(/Should I proceed/));
		expect(screen.getByText("cleanup-agent")).toBeInTheDocument();
		expect(screen.getByText("delete_file")).toBeInTheDocument();
		expect(screen.getByText("File: /tmp/old-data.csv")).toBeInTheDocument();
		expect(screen.getByText("Yes, proceed")).toBeInTheDocument();
	});

	it("shows different request type badges", () => {
		const events = [
			makeEvent({
				event_type: "hitl.request",
				payload: {
					request_id: "req-1",
					request_type: "question",
					prompt: "What is the target database?",
				},
			}),
			makeEvent({
				event_type: "hitl.response",
				payload: {
					request_id: "req-1",
					decision: "answer",
					has_content: true,
					content: "production-db",
					wait_duration_ms: 2000,
				},
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getAllByText("question").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("answer").length).toBeGreaterThanOrEqual(1);
	});
});

// ---------------------------------------------------------------------------
// Panel visibility logic (via registry)
// ---------------------------------------------------------------------------

describe("Panel visibility", () => {
	it("Memory panel is visible when memory events exist", () => {
		const registry = createDefaultPanelRegistry();

		const eventsWithMemory = [makeEvent({ event_type: "memory.working.update", payload: {} })];
		const eventsWithout = [makeEvent({ event_type: "llm.request", payload: {} })];

		const panelsWithMemory = registry.getPanels(mockAgent, eventsWithMemory);
		const panelsWithout = registry.getPanels(mockAgent, eventsWithout);

		expect(panelsWithMemory.some((p) => p.id === "memory")).toBe(true);
		expect(panelsWithout.some((p) => p.id === "memory")).toBe(false);
	});

	it("Planning panel is visible with planning capability or events", () => {
		const registry = createDefaultPanelRegistry();

		const agentWithPlanning = { ...mockAgent, capabilities: ["planning"] };
		const emptyEvents = [makeEvent({ event_type: "llm.request", payload: {} })];
		const planningEvents = [makeEvent({ event_type: "planning.plan.created", payload: {} })];

		// Visible via capability
		const panels1 = registry.getPanels(agentWithPlanning, emptyEvents);
		expect(panels1.some((p) => p.id === "planning")).toBe(true);

		// Visible via events
		const panels2 = registry.getPanels(mockAgent, planningEvents);
		expect(panels2.some((p) => p.id === "planning")).toBe(true);

		// Not visible without either
		const panels3 = registry.getPanels(mockAgent, emptyEvents);
		expect(panels3.some((p) => p.id === "planning")).toBe(false);
	});

	it("Evaluation panel is visible when evaluation or reflection events exist", () => {
		const registry = createDefaultPanelRegistry();

		const evalEvents = [makeEvent({ event_type: "evaluation.result", payload: {} })];
		const reflectionEvents = [makeEvent({ event_type: "reflection.generated", payload: {} })];
		const noEvents = [makeEvent({ event_type: "llm.request", payload: {} })];

		expect(registry.getPanels(mockAgent, evalEvents).some((p) => p.id === "evaluation")).toBe(true);
		expect(registry.getPanels(mockAgent, reflectionEvents).some((p) => p.id === "evaluation")).toBe(true);
		expect(registry.getPanels(mockAgent, noEvents).some((p) => p.id === "evaluation")).toBe(false);
	});

	it("HITL panel is visible when hitl events exist", () => {
		const registry = createDefaultPanelRegistry();

		const hitlEvents = [makeEvent({ event_type: "hitl.request", payload: {} })];
		const noEvents = [makeEvent({ event_type: "llm.request", payload: {} })];

		expect(registry.getPanels(mockAgent, hitlEvents).some((p) => p.id === "hitl")).toBe(true);
		expect(registry.getPanels(mockAgent, noEvents).some((p) => p.id === "hitl")).toBe(false);
	});
});
