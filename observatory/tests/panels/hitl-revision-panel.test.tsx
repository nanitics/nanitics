import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HITLPanel } from "../../src/components/panels/hitl-panel";
import { createDefaultPanelRegistry } from "../../src/registry/default-panels";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "test-agent",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: [],
	stats: {
		llm_calls: 2,
		tool_calls: 1,
		input_tokens: 500,
		output_tokens: 300,
		duration_ms: 5000,
		errors: 0,
		iterations: 2,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "test-agent",
	summary: {
		event_count: 6,
		duration_ms: 5000,
		has_errors: false,
		agent_name: "test-agent",
		agent_type: "react",
	},
	events: [],
	children: [],
};

// ---------------------------------------------------------------------------
// Panel visibility with revision events
// ---------------------------------------------------------------------------

describe("HITL panel visibility with revision events", () => {
	it("is visible when revision.start events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "revision.start",
				payload: { step_name: "review", worker_count: 2, max_revisions: 5 },
			}),
		];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "hitl");
		expect(panel).toBeDefined();
	});

	it("is visible when revision.complete events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "revision.complete",
				payload: { step_name: "review", final_decision: "approve", total_attempts: 1 },
			}),
		];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "hitl");
		expect(panel).toBeDefined();
	});

	it("is hidden when no hitl or revision events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [makeEvent({ event_type: "llm.request", payload: {} })];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "hitl");
		expect(panel).toBeUndefined();
	});
});

// ---------------------------------------------------------------------------
// Revision workflows in HITL panel
// ---------------------------------------------------------------------------

describe("HITL panel revision workflows", () => {
	it("renders revision workflows section when revision events present", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "revision.start",
				payload: { step_name: "review_step", worker_count: 2, max_revisions: 5 },
			}),
			makeEvent({
				event_type: "revision.attempt",
				payload: { step_name: "review_step", attempt_number: 1, feedback: "Needs more detail" },
			}),
			makeEvent({
				event_type: "revision.complete",
				payload: { step_name: "review_step", final_decision: "approve", total_attempts: 1 },
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Revision Workflows")).toBeInTheDocument();
		expect(screen.getByText("review_step")).toBeInTheDocument();
		expect(screen.getByText("approve")).toBeInTheDocument();
		expect(screen.getByText("1 attempt")).toBeInTheDocument();
	});

	it("shows revision count in summary pills", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "revision.start",
				payload: { step_name: "step_a", worker_count: 1, max_revisions: 3 },
			}),
			makeEvent({
				event_type: "revision.complete",
				payload: { step_name: "step_a", final_decision: "approve", total_attempts: 0 },
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Revisions")).toBeInTheDocument();
	});

	it("does not show revision section when only hitl events present", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "hitl.request",
				payload: { request_id: "r1", request_type: "approval", prompt: "Approve?" },
			}),
			makeEvent({
				event_type: "hitl.response",
				payload: { request_id: "r1", decision: "approve" },
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.queryByText("Revision Workflows")).not.toBeInTheDocument();
	});

	it("expands revision workflow card to show details", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "revision.start",
				payload: { step_name: "review_step", worker_count: 2, max_revisions: 5 },
			}),
			makeEvent({
				event_type: "revision.attempt",
				payload: { step_name: "review_step", attempt_number: 1, feedback: "Add sources" },
			}),
			makeEvent({
				event_type: "revision.complete",
				payload: { step_name: "review_step", final_decision: "approve", total_attempts: 1 },
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Click to expand
		const expandButton = screen.getByText("review_step").closest("button")!;
		fireEvent.click(expandButton);

		expect(screen.getByText("Attempts")).toBeInTheDocument();
		expect(screen.getByText("Add sources")).toBeInTheDocument();
		expect(screen.getByText("Outcome")).toBeInTheDocument();
	});

	it("shows in progress badge when revision has no completion", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "revision.start",
				payload: { step_name: "pending_review", worker_count: 1, max_revisions: 3 },
			}),
		];

		render(<HITLPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("in progress")).toBeInTheDocument();
	});

	it("renders empty state when no events at all", () => {
		render(<HITLPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No human-in-the-loop events recorded for this agent.")).toBeInTheDocument();
	});
});
