import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ReActAgentView } from "../../src/components/agent-views/react-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "research-assistant",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: ["tools"],
	stats: {
		llm_calls: 5,
		tool_calls: 3,
		input_tokens: 1200,
		output_tokens: 800,
		duration_ms: 3500,
		errors: 2,
		iterations: 3,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "research-assistant",
	summary: {
		event_count: 10,
		duration_ms: 3500,
		has_errors: true,
		agent_name: "research-assistant",
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
			<ReActAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ReActAgentView", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No events recorded for this agent.")).toBeInTheDocument();
	});

	it("displays step with thought, action, and observation labels", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: {
					step: 1,
					thought: "I need to search for data",
					action: "web_search",
					observation: "Found 8 results",
				},
			}),
		];

		renderView(events);

		// Step header
		expect(screen.getByText("Step 1")).toBeInTheDocument();

		// Labeled sections with lucide glyphs (icons are aria-hidden; assert on the label text)
		expect(screen.getByText("Thought")).toBeInTheDocument();
		expect(screen.getByText("I need to search for data")).toBeInTheDocument();

		expect(screen.getByText("Action")).toBeInTheDocument();
		// Action appears in both step header and action section
		expect(screen.getAllByText("web_search").length).toBeGreaterThanOrEqual(1);

		expect(screen.getByText("Observation")).toBeInTheDocument();
		expect(screen.getByText("Found 8 results")).toBeInTheDocument();
	});

	it("groups multiple steps chronologically", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "First thought", action: "search" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:05Z",
				payload: { step: 2, thought: "Second thought", action: "read" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:10Z",
				payload: { step: 3, thought: "Third thought", action: "answer" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		expect(screen.getByText("Step 2")).toBeInTheDocument();
		expect(screen.getByText("Step 3")).toBeInTheDocument();
		expect(screen.getByText("First thought")).toBeInTheDocument();
		expect(screen.getByText("Second thought")).toBeInTheDocument();
		expect(screen.getByText("Third thought")).toBeInTheDocument();
	});

	it("shows child events (LLM calls, tool invocations) as expandable", () => {
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
				event_type: "llm.response",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: { model_name: "claude-haiku-4-5-20251001" },
			}),
			makeEvent({
				event_type: "tool.invoke",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:04Z",
				payload: { tool_name: "search" },
			}),
		];

		renderView(events);

		// Child events should show a toggle (collapsed by default)
		expect(screen.getByText(/3 events \(LLM calls, tool invocations\)/)).toBeInTheDocument();

		// Events should NOT be visible initially
		expect(screen.queryByText("llm.request")).not.toBeInTheDocument();

		// Click to expand
		fireEvent.click(screen.getByText(/3 events \(LLM calls, tool invocations\)/));

		// Now events are visible
		expect(screen.getAllByText("llm.request").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("llm.response").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("tool.invoke").length).toBeGreaterThanOrEqual(1);
	});

	it("displays error recovery events inline with visual distinction", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "api_call" },
			}),
			makeEvent({
				event_type: "error.retry",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: {
					attempt: 1,
					max_attempts: 3,
					category: "transient",
					error: "TimeoutError",
					delay_ms: 1000,
				},
			}),
			makeEvent({
				event_type: "error.correction",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:04Z",
				payload: {
					attempt: 1,
					error_type: "ValidationError",
					correction_prompt: "Please fix the output",
				},
			}),
		];

		renderView(events);

		// Error events are always visible (not hidden behind toggle)
		expect(screen.getByText("Retry")).toBeInTheDocument();
		expect(screen.getByText(/Attempt 1\/3/)).toBeInTheDocument();
		expect(screen.getByText("Correction")).toBeInTheDocument();
		expect(screen.getByText(/Please fix the output/)).toBeInTheDocument();

		// Error events render RecoveryIcon (tested via testid); at least one per event
		expect(screen.getAllByTestId("recovery-icon").length).toBeGreaterThanOrEqual(2);
	});

	it("displays degradation events with warning styling", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "complex_analysis" },
			}),
			makeEvent({
				event_type: "error.degradation",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:05Z",
				payload: {
					original_approach: "deep analysis",
					fallback: "shallow analysis",
					reason: "Persistent MemoryError after 2 retries",
				},
			}),
		];

		renderView(events);

		expect(screen.getByText("Degradation")).toBeInTheDocument();
		expect(screen.getAllByTestId("recovery-icon").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText(/Persistent MemoryError after 2 retries/)).toBeInTheDocument();
	});

	it("shows failed tool results as error events", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "api_call" },
			}),
			makeEvent({
				event_type: "tool.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: {
					tool_name: "api_call",
					success: false,
					error: "Connection timeout",
				},
			}),
		];

		renderView(events);

		// Failed tool result displayed with error styling
		expect(screen.getByText("api_call failed")).toBeInTheDocument();
		expect(screen.getByText("Connection timeout")).toBeInTheDocument();
	});

	it("shows final output section from agent.complete event", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "Done", action: "answer" },
			}),
			makeEvent({
				event_type: "agent.complete",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:10Z",
				payload: {
					agent_name: "research-assistant",
					termination_reason: "task_complete",
					total_steps: 3,
				},
			}),
		];

		renderView(events);

		expect(screen.getByText("Completed")).toBeInTheDocument();
		expect(screen.getByText("(task_complete)")).toBeInTheDocument();
		expect(screen.getByText("3 steps")).toBeInTheDocument();
	});

	it("shows agent.start in header section", () => {
		const events = [
			makeEvent({
				event_type: "agent.start",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { agent_name: "research-assistant", agent_type: "react" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
		];

		renderView(events);

		expect(screen.getAllByText("agent.start").length).toBeGreaterThanOrEqual(1);
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

		// Initially expanded
		expect(screen.getByText("My thought")).toBeInTheDocument();

		// Click to collapse
		fireEvent.click(screen.getByText("Step 1"));
		expect(screen.queryByText("My thought")).not.toBeInTheDocument();

		// Click to expand
		fireEvent.click(screen.getByText("Step 1"));
		expect(screen.getByText("My thought")).toBeInTheDocument();
	});

	it("shows error count badge on step header when errors present", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "api_call" },
			}),
			makeEvent({
				event_type: "error.retry",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: { attempt: 1, max_attempts: 3, error: "Timeout" },
			}),
			makeEvent({
				event_type: "error.retry",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:04Z",
				payload: { attempt: 2, max_attempts: 3, error: "Timeout" },
			}),
		];

		renderView(events);

		expect(screen.getByText("2 errors")).toBeInTheDocument();
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
		expect(screen.queryByText(/Step \d+/)).not.toBeInTheDocument();
	});

	it("separates error events from regular child events", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, action: "api_call" },
			}),
			// Regular child event
			makeEvent({
				event_type: "llm.request",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { model_name: "claude-haiku-4-5-20251001" },
			}),
			// Error event — should be inline, not in expandable section
			makeEvent({
				event_type: "error.correction",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:03Z",
				payload: { attempt: 1, correction_prompt: "Fix the output" },
			}),
			// Successful tool result — regular child event
			makeEvent({
				event_type: "tool.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:04Z",
				payload: { tool_name: "api_call", success: true },
			}),
		];

		renderView(events);

		// Error event should be visible inline (not hidden behind toggle)
		expect(screen.getByText("Correction")).toBeInTheDocument();

		// Regular events should show event count (LLM request + successful tool result = 2)
		expect(screen.getByText(/2 events \(LLM calls, tool invocations\)/)).toBeInTheDocument();
	});

	it("handles step with only thought (no action or observation)", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "Just thinking..." },
			}),
		];

		renderView(events);

		expect(screen.getByText("Just thinking...")).toBeInTheDocument();
		expect(screen.queryByText("Action")).not.toBeInTheDocument();
		expect(screen.queryByText("Observation")).not.toBeInTheDocument();
	});

	it("handles completion without termination_reason gracefully", () => {
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
				payload: { agent_name: "research-assistant" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Completed")).toBeInTheDocument();
		expect(screen.queryByText("(")).not.toBeInTheDocument();
	});
});
