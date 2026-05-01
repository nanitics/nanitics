import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DurabilityPanel } from "../../src/components/panels/durability-panel";
import { createDefaultPanelRegistry } from "../../src/registry/default-panels";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "durable-agent",
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
	name: "durable-agent",
	summary: {
		event_count: 6,
		duration_ms: 5000,
		has_errors: false,
		agent_name: "durable-agent",
		agent_type: "react",
	},
	events: [],
	children: [],
};

// ---------------------------------------------------------------------------
// Panel visibility
// ---------------------------------------------------------------------------

describe("Durability panel registration", () => {
	it("is visible when execution.suspended events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "execution.suspended",
				payload: { suspension_id: "sus-1", suspension_type: "hitl" },
			}),
		];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "durability");
		expect(panel).toBeDefined();
		expect(panel?.label).toBe("Durability");
	});

	it("is visible when checkpoint.saved events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "checkpoint.saved",
				payload: { checkpoint_id: "cp-1", checkpoint_type: "orchestration" },
			}),
		];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "durability");
		expect(panel).toBeDefined();
	});

	it("is visible when execution.resumed events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "execution.resumed",
				payload: { checkpoint_id: "cp-1", suspension_id: "sus-1" },
			}),
		];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "durability");
		expect(panel).toBeDefined();
	});

	it("is hidden when no durability events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [makeEvent({ event_type: "llm.request", payload: {} })];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "durability");
		expect(panel).toBeUndefined();
	});

	it("is ordered after HITL (70)", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "execution.suspended",
				payload: { suspension_id: "sus-1" },
			}),
			makeEvent({ event_type: "hitl.request", payload: { request_id: "r1" } }),
		];

		const panels = registry.getPanels(mockAgent, events);
		const ids = panels.map((p) => p.id);
		const hitlIdx = ids.indexOf("hitl");
		const durabilityIdx = ids.indexOf("durability");

		expect(hitlIdx).toBeLessThan(durabilityIdx);
	});
});

// ---------------------------------------------------------------------------
// Panel rendering
// ---------------------------------------------------------------------------

describe("DurabilityPanel", () => {
	it("renders empty state when no durability events", () => {
		render(<DurabilityPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No durability events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders summary statistics", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "checkpoint.saved",
				payload: {
					checkpoint_id: "cp-1",
					checkpoint_type: "orchestration",
					run_id: "run-1",
				},
			}),
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-1",
					suspension_type: "hitl",
					checkpoint_id: "cp-1",
					step_name: "review",
					agent_name: "reviewer",
				},
			}),
			makeEvent({
				event_type: "execution.resumed",
				payload: {
					checkpoint_id: "cp-1",
					suspension_id: "sus-1",
					resumed_from_step: "review",
				},
			}),
		];

		render(<DurabilityPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Summary pills
		expect(screen.getByText("Checkpoints")).toBeInTheDocument();
		expect(screen.getByText("Suspensions")).toBeInTheDocument();
		expect(screen.getByText("Pending")).toBeInTheDocument();
		expect(screen.getByText("Completed")).toBeInTheDocument();
	});

	it("correlates events by checkpoint_id", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "checkpoint.saved",
				payload: {
					checkpoint_id: "cp-1",
					checkpoint_type: "orchestration",
					run_id: "run-1",
				},
			}),
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-1",
					suspension_type: "hitl",
					checkpoint_id: "cp-1",
					step_name: "review",
				},
			}),
			makeEvent({
				event_type: "execution.resumed",
				payload: {
					checkpoint_id: "cp-1",
					suspension_id: "sus-1",
					resumed_from_step: "review",
				},
			}),
		];

		render(<DurabilityPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Should show Resumed status badge
		expect(screen.getByText("Resumed")).toBeInTheDocument();
		// Step context shown
		expect(screen.getByText("at review")).toBeInTheDocument();
	});

	it("shows Pending status for suspended without resumption", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-1",
					suspension_type: "hitl",
					checkpoint_id: "cp-1",
					step_name: "approval",
				},
			}),
		];

		render(<DurabilityPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// "Pending" appears in both the summary pill and the status badge
		expect(screen.getAllByText("Pending").length).toBeGreaterThanOrEqual(1);
	});

	it("handles standalone suspensions with empty checkpoint_id", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-standalone",
					suspension_type: "hitl",
					checkpoint_id: "",
					step_name: "agent_step",
				},
			}),
		];

		render(<DurabilityPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Standalone renders as "unchecked"
		expect(screen.getByText("unchecked")).toBeInTheDocument();
		expect(screen.getAllByText("Pending").length).toBeGreaterThanOrEqual(1);
	});

	it("expands lifecycle card to show details", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "checkpoint.saved",
				payload: {
					checkpoint_id: "cp-expand",
					checkpoint_type: "orchestration",
					run_id: "run-x",
				},
			}),
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-expand",
					suspension_type: "hitl",
					checkpoint_id: "cp-expand",
					step_name: "step-expand",
					agent_name: "agent-expand",
				},
			}),
		];

		render(<DurabilityPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Expand the card
		const expandButton = screen.getByRole("button");
		fireEvent.click(expandButton);

		// Expanded details visible
		expect(screen.getByText("Checkpoint")).toBeInTheDocument();
		expect(screen.getByText("Suspension")).toBeInTheDocument();
		expect(screen.getByText("cp-expand")).toBeInTheDocument();
		expect(screen.getByText("run-x")).toBeInTheDocument();
		expect(screen.getByText("step-expand")).toBeInTheDocument();
		expect(screen.getByText("agent-expand")).toBeInTheDocument();
		expect(screen.getByText("Awaiting resumption…")).toBeInTheDocument();
	});

	it("shows multiple suspensions per checkpoint", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-a",
					suspension_type: "hitl",
					checkpoint_id: "cp-multi",
					step_name: "step-1",
					agent_name: "agent-a",
				},
			}),
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-b",
					suspension_type: "hitl",
					checkpoint_id: "cp-multi",
					step_name: "step-1",
					agent_name: "agent-b",
				},
			}),
		];

		render(<DurabilityPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Expand card
		const expandButton = screen.getByRole("button");
		fireEvent.click(expandButton);

		// Both suspension IDs visible
		expect(screen.getByText("sus-a")).toBeInTheDocument();
		expect(screen.getByText("sus-b")).toBeInTheDocument();
		// Numbered suspensions
		expect(screen.getByText("Suspension 1")).toBeInTheDocument();
		expect(screen.getByText("Suspension 2")).toBeInTheDocument();
	});
});
