import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ReWOOAgentView } from "../../src/components/agent-views/rewoo-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "rewoo-planner",
	agent_type: "rewoo",
	span_id: "agent-1",
	capabilities: ["planning", "tools"],
	stats: {
		llm_calls: 4,
		tool_calls: 3,
		input_tokens: 2000,
		output_tokens: 1000,
		duration_ms: 8000,
		errors: 0,
		iterations: 5,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "rewoo-planner",
	summary: {
		event_count: 15,
		duration_ms: 8000,
		has_errors: false,
		agent_name: "rewoo-planner",
		agent_type: "rewoo",
	},
	events: [],
	children: [],
};

function renderView(events: TraceEvent[]) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<ReWOOAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeReWOOEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "agent.start",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:00Z",
			payload: { agent_name: "rewoo-planner" },
		}),
		// Plan phase
		makeEvent({
			event_type: "planning.plan.created",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:01Z",
			payload: {
				plan_id: "plan-1",
				plan_name: "research-plan",
				step_count: 3,
				goal_count: 0,
				steps: [
					{
						step_id: "s1",
						description: "#1: Search for climate data",
						metadata: {
							tool: "web_search",
							args: { query: "climate change data 2025" },
							variable: "#1",
							depends_on: [],
							execution_level: 0,
						},
					},
					{
						step_id: "s2",
						description: "#2: Search for economic impact",
						metadata: {
							tool: "web_search",
							args: { query: "economic impact of climate change" },
							variable: "#2",
							depends_on: [],
							execution_level: 0,
						},
					},
					{
						step_id: "s3",
						description: "#3: Analyze results from #1 and #2",
						metadata: {
							tool: "analyze",
							args: { data: "#1, #2" },
							variable: "#3",
							depends_on: [1, 2],
							execution_level: 1,
						},
					},
				],
			},
		}),
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:02Z",
			payload: { step: 1, thought: "Planning complete" },
		}),
		// Execute phase
		makeEvent({
			event_type: "planning.step.updated",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:03Z",
			payload: {
				plan_id: "plan-1",
				step_id: "s1",
				step_description: "Search for climate data",
				previous_status: "not_started",
				new_status: "completed",
				has_result: true,
			},
		}),
		makeEvent({
			event_type: "tool.invoke",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:03.5Z",
			payload: { tool_name: "web_search", arguments: { query: "climate change data 2025" } },
		}),
		makeEvent({
			event_type: "tool.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:04Z",
			payload: { tool_name: "web_search", success: true, result: "Climate data results..." },
		}),
		makeEvent({
			event_type: "planning.step.updated",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:05Z",
			payload: {
				plan_id: "plan-1",
				step_id: "s2",
				step_description: "Search for economic impact",
				previous_status: "not_started",
				new_status: "completed",
				has_result: true,
			},
		}),
		makeEvent({
			event_type: "planning.step.updated",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:07Z",
			payload: {
				plan_id: "plan-1",
				step_id: "s3",
				step_description: "Analyze results",
				previous_status: "not_started",
				new_status: "completed",
				has_result: true,
			},
		}),
		// Solve phase
		makeEvent({
			event_type: "llm.request",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:08Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "llm.response",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:09Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:10Z",
			payload: { step: 5, thought: "Based on #1 and #2, the analysis shows..." },
		}),
		makeEvent({
			event_type: "agent.complete",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:11Z",
			payload: { termination_reason: "complete", total_steps: 5 },
		}),
	];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ReWOOAgentView", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders plan section with step count", () => {
		renderView(makeReWOOEvents());
		expect(screen.getByText("Plan")).toBeInTheDocument();
		expect(screen.getByText("3 steps")).toBeInTheDocument();
		expect(screen.getByText("research-plan")).toBeInTheDocument();
	});

	it("renders plan steps with variable references and tool names", () => {
		renderView(makeReWOOEvents());

		// Variable references
		expect(screen.getAllByText("#1").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("#2").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("#3").length).toBeGreaterThanOrEqual(1);

		// Tool names
		expect(screen.getAllByText("web_search").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("analyze").length).toBeGreaterThanOrEqual(1);
	});

	it("renders execution section with dependency-level grouping", () => {
		renderView(makeReWOOEvents());
		expect(screen.getByText("Execution")).toBeInTheDocument();
		// Level labels
		expect(screen.getAllByText("Level 0").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("Level 1").length).toBeGreaterThanOrEqual(1);
	});

	it("shows step status badges", () => {
		renderView(makeReWOOEvents());
		// All steps completed
		const completedBadges = screen.getAllByText("completed");
		expect(completedBadges.length).toBe(3);
	});

	it("renders solver section", () => {
		renderView(makeReWOOEvents());
		expect(screen.getByText("Synthesis")).toBeInTheDocument();
	});

	it("shows completion section", () => {
		renderView(makeReWOOEvents());
		expect(screen.getByText("Completed")).toBeInTheDocument();
		expect(screen.getByText("(complete)")).toBeInTheDocument();
		expect(screen.getByText("5 steps")).toBeInTheDocument();
	});

	it("renders parallel execution indicator for same-level steps", () => {
		renderView(makeReWOOEvents());
		// Steps s1 and s2 are at level 0 (parallel)
		expect(screen.getByText("parallel")).toBeInTheDocument();
	});

	it("shows dependency information in plan steps", () => {
		renderView(makeReWOOEvents());
		// Step 3 depends on steps 1 and 2
		expect(screen.getByText(/depends on/)).toBeInTheDocument();
	});

	it("renders with evaluation result in solver section", () => {
		const events = makeReWOOEvents();
		// Add evaluation result before completion
		events.splice(
			-1,
			0,
			makeEvent({
				event_type: "evaluation.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:10.5Z",
				payload: {
					evaluator_name: "quality-check",
					verdict: "accept",
					score: 0.92,
					feedback: "Good analysis",
					revision_attempt: 0,
				},
			}),
		);

		renderView(events);
		expect(screen.getByText("Synthesis")).toBeInTheDocument();
		// Evaluation verdict shown in synthesis header
		expect(screen.getAllByText("accept").length).toBeGreaterThanOrEqual(1);
	});

	it("handles empty plan (no planning.plan.created event)", () => {
		const events = [
			makeEvent({
				event_type: "agent.start",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {},
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "Direct output" },
			}),
			makeEvent({
				event_type: "agent.complete",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { termination_reason: "complete" },
			}),
		];

		renderView(events);
		// Should not render plan section
		expect(screen.queryByText("Plan")).not.toBeInTheDocument();
		// But should still show completion
		expect(screen.getByText("Completed")).toBeInTheDocument();
	});

	it("handles failed step status", () => {
		const events = [
			makeEvent({
				event_type: "planning.plan.created",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: {
					plan_id: "plan-1",
					plan_name: "test-plan",
					step_count: 1,
					steps: [
						{
							step_id: "s1",
							description: "Failing step",
							metadata: {
								tool: "flaky_tool",
								args: {},
								variable: "#1",
								depends_on: [],
								execution_level: 0,
							},
						},
					],
				},
			}),
			makeEvent({
				event_type: "planning.step.updated",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: {
					plan_id: "plan-1",
					step_id: "s1",
					step_description: "Failing step",
					previous_status: "not_started",
					new_status: "failed",
					has_result: false,
				},
			}),
		];

		renderView(events);
		expect(screen.getByText("failed")).toBeInTheDocument();
	});
});
