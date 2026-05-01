import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { CodeActAgentView } from "../../src/components/agent-views/codeact-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "data-analyst",
	agent_type: "codeact",
	span_id: "agent-1",
	capabilities: ["code_execution"],
	stats: {
		llm_calls: 5,
		tool_calls: 0,
		input_tokens: 3000,
		output_tokens: 600,
		duration_ms: 9000,
		errors: 1,
		iterations: 5,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "data-analyst",
	summary: {
		event_count: 25,
		duration_ms: 9000,
		has_errors: true,
		agent_name: "data-analyst",
		agent_type: "codeact",
	},
	events: [],
	children: [],
};

function renderView(events: TraceEvent[]) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<CodeActAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CodeActAgentView", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
		// Mock clipboard for CodeBlock copy button
		Object.assign(navigator, {
			clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
		});
	});

	it("shows empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No events recorded for this agent.")).toBeInTheDocument();
	});

	it("displays code blocks within steps", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "Load the data" },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { code: "import pandas as pd\ndf = pd.read_csv('data.csv')", language: "python" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.300Z",
				payload: { success: true, stdout: "OK", stderr: "", return_value: null, duration_ms: 100 },
			}),
		];

		renderView(events);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		expect(screen.getByText("Load the data")).toBeInTheDocument();
		expect(screen.getByText("import pandas as pd")).toBeInTheDocument();
		expect(screen.getByText("success")).toBeInTheDocument();
		expect(screen.getByText("100ms")).toBeInTheDocument();
	});

	it("displays failed execution results", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { code: "plt.figure(figsize=(10, 6)", language: "python" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.200Z",
				payload: {
					success: false,
					stdout: "",
					stderr: "",
					return_value: null,
					error: "SyntaxError: '(' was never closed",
					duration_ms: 50,
				},
			}),
		];

		renderView(events);

		expect(screen.getAllByText("failed").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("1 failed")).toBeInTheDocument();
		expect(screen.getByText("SyntaxError: '(' was never closed")).toBeInTheDocument();
	});

	it("shows stdout output", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { code: "print('hello world')" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.200Z",
				payload: { success: true, stdout: "hello world", stderr: "", return_value: null, duration_ms: 10 },
			}),
		];

		renderView(events);

		expect(screen.getByText("stdout:")).toBeInTheDocument();
		expect(screen.getByText("hello world")).toBeInTheDocument();
	});

	it("shows stderr in red styling", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { code: "import warnings; warnings.warn('test')" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.200Z",
				payload: { success: true, stdout: "", stderr: "UserWarning: test", return_value: null, duration_ms: 10 },
			}),
		];

		renderView(events);

		expect(screen.getByText("stderr:")).toBeInTheDocument();
		expect(screen.getByText("UserWarning: test")).toBeInTheDocument();
	});

	it("shows return_value when present", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { code: "df.head()" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.200Z",
				payload: { success: true, stdout: "", stderr: "", return_value: "DataFrame(5 rows)", duration_ms: 12 },
			}),
		];

		renderView(events);

		expect(screen.getByText("return_value:")).toBeInTheDocument();
		expect(screen.getByText("DataFrame(5 rows)")).toBeInTheDocument();
	});

	it("handles multiple code blocks per step", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { code: "x = 1" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.200Z",
				payload: { success: true, stdout: "", stderr: "", return_value: null, duration_ms: 5 },
			}),
			makeEvent({
				event_type: "code.execution",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.300Z",
				payload: { code: "y = x + 1" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01.400Z",
				payload: { success: true, stdout: "", stderr: "", return_value: null, duration_ms: 3 },
			}),
		];

		renderView(events);

		// Both code blocks render with labels
		expect(screen.getByText("Code Block 1")).toBeInTheDocument();
		expect(screen.getByText("Code Block 2")).toBeInTheDocument();
		expect(screen.getByText("2 code blocks")).toBeInTheDocument();
		expect(screen.getByText("x = 1")).toBeInTheDocument();
		expect(screen.getByText("y = x + 1")).toBeInTheDocument();
	});

	it("groups multiple steps chronologically", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "First step" },
			}),
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:05Z",
				payload: { step: 2, thought: "Second step" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		expect(screen.getByText("Step 2")).toBeInTheDocument();
		expect(screen.getByText("First step")).toBeInTheDocument();
		expect(screen.getByText("Second step")).toBeInTheDocument();
	});

	it("shows agent.start header events", () => {
		const events = [
			makeEvent({
				event_type: "agent.start",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { agent_name: "data-analyst", agent_type: "codeact" },
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
		expect(screen.getByText("Step 1")).toBeInTheDocument();
	});

	it("shows completion section from agent.complete event", () => {
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
				payload: {
					agent_name: "data-analyst",
					termination_reason: "goal_reached",
					total_steps: 5,
				},
			}),
		];

		renderView(events);

		expect(screen.getByText("Completed")).toBeInTheDocument();
		expect(screen.getByText("(goal_reached)")).toBeInTheDocument();
		expect(screen.getByText("5 steps")).toBeInTheDocument();
	});

	it("shows observation in step", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, observation: "Dataset has 1500 rows" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Observation")).toBeInTheDocument();
		expect(screen.getByText("Dataset has 1500 rows")).toBeInTheDocument();
	});

	it("shows error recovery events inline", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1 },
			}),
			makeEvent({
				event_type: "error.correction",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { attempt: 1, correction_prompt: "Fix the syntax error" },
			}),
		];

		renderView(events);

		expect(screen.getByText("Correction")).toBeInTheDocument();
		expect(screen.getByText(/Fix the syntax error/)).toBeInTheDocument();
	});

	it("collapses step when header is clicked", () => {
		const events = [
			makeEvent({
				event_type: "agent.step",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { step: 1, thought: "Some thought" },
			}),
		];

		renderView(events);

		// Thought is visible (expanded by default)
		expect(screen.getByText("Some thought")).toBeInTheDocument();

		// Click step header to collapse
		fireEvent.click(screen.getByText("Step 1"));

		// Thought should no longer be visible
		expect(screen.queryByText("Some thought")).not.toBeInTheDocument();
	});

	it("shows child events expandable under step", () => {
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
				timestamp: "2026-03-05T10:00:01.100Z",
				payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 200 },
			}),
			makeEvent({
				event_type: "llm.response",
				span_id: "agent-1",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { model_name: "claude-haiku-4-5-20251001", usage: { input_tokens: 200, output_tokens: 100 } },
			}),
		];

		renderView(events);

		// Child events section shows count
		expect(screen.getByText(/2 events/)).toBeInTheDocument();

		// Child events are hidden by default
		expect(screen.queryByText("llm.request")).not.toBeInTheDocument();

		// Click to expand
		fireEvent.click(screen.getByText(/2 events/));

		// Now visible
		expect(screen.getAllByText("llm.request").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("llm.response").length).toBeGreaterThanOrEqual(1);
	});
});
