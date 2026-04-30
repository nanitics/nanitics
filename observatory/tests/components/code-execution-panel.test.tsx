import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CodeExecutionPanel } from "../../src/components/panels/code-execution-panel";
import { createDefaultPanelRegistry } from "../../src/registry/default-panels";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "code-agent",
	agent_type: "codeact",
	span_id: "agent-1",
	capabilities: ["code_execution"],
	stats: {
		llm_calls: 2,
		tool_calls: 0,
		input_tokens: 500,
		output_tokens: 300,
		duration_ms: 3000,
		errors: 0,
		iterations: 2,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "code-agent",
	summary: {
		event_count: 6,
		duration_ms: 3000,
		has_errors: false,
		agent_name: "code-agent",
		agent_type: "codeact",
	},
	events: [],
	children: [],
};

// ---------------------------------------------------------------------------
// Panel visibility
// ---------------------------------------------------------------------------

describe("Code Execution panel registration", () => {
	it("is visible when code.execution events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [makeEvent({ event_type: "code.execution", payload: { step_number: 1 } })];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "code-execution");
		expect(panel).toBeDefined();
		expect(panel?.label).toBe("Code Execution");
	});

	it("is hidden when no code.execution events exist", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [makeEvent({ event_type: "llm.request", payload: {} })];

		const panels = registry.getPanels(mockAgent, events);
		const panel = panels.find((p) => p.id === "code-execution");
		expect(panel).toBeUndefined();
	});

	it("is ordered between Tools (20) and Errors (30)", () => {
		const registry = createDefaultPanelRegistry();
		const events: TraceEvent[] = [
			makeEvent({ event_type: "code.execution", payload: { step_number: 1 } }),
			makeEvent({ event_type: "tool.invoke", payload: { tool_name: "search" } }),
		];
		const agentWithTools = { ...mockAgent, stats: { ...mockAgent.stats, tool_calls: 1 } };

		const panels = registry.getPanels(agentWithTools, events);
		const ids = panels.map((p) => p.id);
		const toolsIdx = ids.indexOf("tools");
		const codeIdx = ids.indexOf("code-execution");

		expect(toolsIdx).toBeLessThan(codeIdx);
	});
});

// ---------------------------------------------------------------------------
// Panel rendering
// ---------------------------------------------------------------------------

describe("CodeExecutionPanel", () => {
	it("renders empty state when no code execution events", () => {
		render(<CodeExecutionPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No code executions recorded for this agent.")).toBeInTheDocument();
	});

	it("renders overview stats with correct counts", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 1, code: "x = 1", agent_name: "code-agent" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: true,
					duration_ms: 100,
					stdout: "1",
					stderr: "",
					return_value: null,
					error: null,
				},
			}),
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 2, code: "y = foo()", agent_name: "code-agent" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 2,
					success: false,
					duration_ms: 50,
					stdout: "",
					stderr: "Traceback...",
					return_value: null,
					error: "NameError: name 'foo' is not defined",
				},
			}),
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 3, code: "z = 3", agent_name: "code-agent" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 3,
					success: true,
					duration_ms: 80,
					stdout: "",
					stderr: "",
					return_value: "3",
					error: null,
				},
			}),
		];

		render(<CodeExecutionPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Overview bar
		expect(screen.getByText("Executions")).toBeInTheDocument();
		expect(screen.getByText("3")).toBeInTheDocument();
		expect(screen.getByText("Success")).toBeInTheDocument();
		expect(screen.getByText("2")).toBeInTheDocument();
		expect(screen.getByText("Failed")).toBeInTheDocument();
		expect(screen.getByText("1")).toBeInTheDocument();
	});

	it("renders execution timeline with step numbers", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 1, code: "print('hello')", agent_name: "code-agent" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: true,
					duration_ms: 25,
					stdout: "hello",
					stderr: "",
					return_value: null,
					error: null,
				},
			}),
		];

		render(<CodeExecutionPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Execution Timeline")).toBeInTheDocument();
		expect(screen.getByText("#1")).toBeInTheDocument();
		expect(screen.getByText("✓")).toBeInTheDocument();
	});

	it("highlights failed executions", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 1, code: "bad_code()", agent_name: "code-agent" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: false,
					duration_ms: 10,
					stdout: "",
					stderr: "",
					return_value: null,
					error: "SyntaxError",
				},
			}),
		];

		render(<CodeExecutionPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("✗")).toBeInTheDocument();
	});

	it("expands execution row to show code and output", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 1, code: "result = 2 + 2\nprint(result)", agent_name: "code-agent" },
			}),
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: true,
					duration_ms: 15,
					stdout: "4",
					stderr: "",
					return_value: "4",
					error: null,
				},
			}),
		];

		render(<CodeExecutionPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Click to expand
		const row = screen.getByText(/result = 2 \+ 2/);
		fireEvent.click(row);

		// Should show full code block and output
		expect(screen.getByText("print(result)")).toBeInTheDocument();
		expect(screen.getByText("stdout:")).toBeInTheDocument();
		expect(screen.getByText("return_value:")).toBeInTheDocument();
	});
});
