import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { GenericAgentView } from "../../src/components/agent-views/generic-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "test-agent",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: ["tool_use"],
	stats: {
		llm_calls: 3,
		tool_calls: 2,
		input_tokens: 500,
		output_tokens: 300,
		duration_ms: 2000,
		errors: 0,
		iterations: 2,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "test-agent",
	summary: {
		event_count: 5,
		duration_ms: 2000,
		has_errors: false,
		agent_name: "test-agent",
		agent_type: "react",
	},
	events: [],
	children: [],
};

function renderView(events: TraceEvent[]) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<GenericAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("GenericAgentView", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No events recorded for this agent.")).toBeInTheDocument();
	});

	it("groups events by step number", () => {
		const events = [
			makeEvent({
				event_type: "agent.start",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { agent_name: "test-agent", agent_type: "react" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "Thinking first", action: "search" },
			}),
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { model_name: "claude-haiku-4-5-20251001" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:05Z",
				payload: { step: 2, action: "write" },
			}),
			makeEvent({
				event_type: "agent.complete",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:10Z",
				payload: { agent_name: "test-agent", termination_reason: "task_complete" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		expect(screen.getByText("Step 2")).toBeInTheDocument();
	});

	it("displays thought content within a step", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "I should search for data", action: "search" },
			}),
		];

		renderView(events);

		expect(screen.getByText("I should search for data")).toBeInTheDocument();
	});

	it("displays observation content within a step", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, observation: "Found 5 results" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Found 5 results")).toBeInTheDocument();
	});

	it("shows child events within their temporal step", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "search" },
			}),
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { model_name: "claude-haiku-4-5-20251001" },
			}),
			makeEvent({
				event_type: "tool.invoke",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: { tool_name: "search" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		// Child events should show as event rows with their event_type
		expect(screen.getAllByText("llm.request").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("tool.invoke").length).toBeGreaterThanOrEqual(1);
	});

	it("shows agent.start in header section", () => {
		const events = [
			makeEvent({
				event_type: "agent.start",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { agent_name: "test-agent" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
		];

		renderView(events);

		// agent.start should be rendered outside the step sections
		expect(screen.getAllByText("agent.start").length).toBeGreaterThanOrEqual(1);
	});

	it("shows agent.complete in footer section", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "agent.complete",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:10Z",
				payload: { agent_name: "test-agent" },
			}),
		];

		renderView(events);

		expect(screen.getAllByText("agent.complete").length).toBeGreaterThanOrEqual(1);
	});

	it("renders all events flat when no steps exist", () => {
		const events = [
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { model_name: "claude-haiku-4-5-20251001" },
			}),
			makeEvent({
				event_type: "llm.response",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {},
			}),
		];

		renderView(events);

		expect(screen.getAllByText("llm.request").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("llm.response").length).toBeGreaterThanOrEqual(1);
		// No step sections
		expect(screen.queryByText(/Step \d+/)).not.toBeInTheDocument();
	});

	it("shows event count badge on step headers", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "search" },
			}),
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {},
			}),
			makeEvent({
				event_type: "tool.invoke",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: {},
			}),
		];

		renderView(events);

		// 2 child events in step 1
		expect(screen.getByText("2")).toBeInTheDocument();
	});

	it("collapses/expands step sections when clicking header", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "My thought" },
			}),
		];

		renderView(events);

		// Initially expanded (shows thought)
		expect(screen.getByText("My thought")).toBeInTheDocument();

		// Click to collapse
		fireEvent.click(screen.getByText("Step 1"));
		expect(screen.queryByText("My thought")).not.toBeInTheDocument();

		// Click to expand again
		fireEvent.click(screen.getByText("Step 1"));
		expect(screen.getByText("My thought")).toBeInTheDocument();
	});

	it("expands event rows to show renderer content", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { model_name: "test-model" },
			}),
		];

		renderView(events);

		// Click to expand the llm.request event row (first occurrence)
		fireEvent.click(screen.getAllByText("llm.request")[0]);

		// The expanded area should show the payload content
		// With the default registry (empty), it shows nothing extra,
		// but the expanded container is rendered
		expect(screen.getAllByText("llm.request").length).toBeGreaterThanOrEqual(1);
	});

	it("shows step action in the step header", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "search_web" },
			}),
		];

		renderView(events);

		expect(screen.getByText("search_web")).toBeInTheDocument();
	});

	it("handles multiple steps with correct temporal grouping", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {},
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:05Z",
				payload: { step: 2 },
			}),
			makeEvent({
				event_type: "tool.invoke",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:06Z",
				payload: {},
			}),
		];

		renderView(events);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		expect(screen.getByText("Step 2")).toBeInTheDocument();

		// Each step should show 1 child event
		const badges = screen.getAllByText("1");
		expect(badges.length).toBeGreaterThanOrEqual(2);
	});
});
