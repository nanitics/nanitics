import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ErrorRecoveryPanel } from "../../src/components/panels/error-recovery-panel";
import { LLMCallsPanel } from "../../src/components/panels/llm-calls-panel";
import { ToolAnalyticsPanel } from "../../src/components/panels/tool-analytics-panel";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "test-agent",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: ["tool_use"],
	stats: {
		llm_calls: 3,
		tool_calls: 4,
		input_tokens: 1000,
		output_tokens: 500,
		duration_ms: 5000,
		errors: 2,
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
		has_errors: true,
		agent_name: "test-agent",
		agent_type: "react",
	},
	events: [],
	children: [],
};

// ---------------------------------------------------------------------------
// LLMCallsPanel
// ---------------------------------------------------------------------------

describe("LLMCallsPanel", () => {
	it("renders empty state when no LLM events", () => {
		render(<LLMCallsPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No LLM calls recorded for this agent.")).toBeInTheDocument();
	});

	it("renders LLM call cards from request/response pairs", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "llm.request",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					model_name: "claude-haiku-4-5-20251001",
					input_tokens: 200,
					messages_count: 3,
				},
			}),
			makeEvent({
				event_type: "llm.response",
				timestamp: "2026-03-05T10:00:01Z",
				payload: {
					model_name: "claude-haiku-4-5-20251001",
					usage: { input_tokens: 200, output_tokens: 150 },
					stop_reason: "end_turn",
				},
			}),
			makeEvent({
				event_type: "llm.request",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {
					model_name: "claude-haiku-4-5-20251001",
					input_tokens: 400,
					messages_count: 5,
				},
			}),
			makeEvent({
				event_type: "llm.response",
				timestamp: "2026-03-05T10:00:03Z",
				payload: {
					model_name: "claude-haiku-4-5-20251001",
					usage: { input_tokens: 400, output_tokens: 200 },
					stop_reason: "end_turn",
				},
			}),
		];

		render(<LLMCallsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("2 LLM calls")).toBeInTheDocument();
		expect(screen.getByText("Call 1 of 2")).toBeInTheDocument();
		expect(screen.getByText("Call 2 of 2")).toBeInTheDocument();
	});

	it("shows context events as banners before LLM calls", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "context.assembly",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { contributions: ["system_prompt", "memory"], total_injected: 800 },
			}),
			makeEvent({
				event_type: "llm.request",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 200 },
			}),
			makeEvent({
				event_type: "llm.response",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {
					usage: { input_tokens: 200, output_tokens: 100 },
					stop_reason: "end_turn",
				},
			}),
		];

		render(<LLMCallsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Context assembled from 2 providers/)).toBeInTheDocument();
		expect(screen.getByText(/800 tokens injected/)).toBeInTheDocument();
	});

	it("shows context truncation banner", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "context.truncation",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					messages_before: 20,
					messages_after: 10,
					tokens_before: 5000,
					tokens_after: 3000,
				},
			}),
			makeEvent({
				event_type: "llm.request",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 200 },
			}),
		];

		render(<LLMCallsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Context truncated: 20→10 messages/)).toBeInTheDocument();
	});

	it("expands LLM call to show request details", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "llm.request",
				payload: {
					model_name: "claude-haiku-4-5-20251001",
					input_tokens: 300,
					messages_count: 4,
				},
			}),
			makeEvent({
				event_type: "llm.response",
				payload: {
					usage: { input_tokens: 300, output_tokens: 150 },
					stop_reason: "end_turn",
				},
			}),
		];

		render(<LLMCallsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Click to expand
		fireEvent.click(screen.getByText("Call 1 of 1"));

		expect(screen.getByText("Request")).toBeInTheDocument();
		expect(screen.getByText("Response")).toBeInTheDocument();
		expect(screen.getByText("end_turn")).toBeInTheDocument();
	});

	it("handles unpaired request (no response)", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "llm.request",
				payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 200 },
			}),
		];

		render(<LLMCallsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("1 LLM call")).toBeInTheDocument();

		// Expand to see "no response"
		fireEvent.click(screen.getByText("Call 1 of 1"));
		expect(screen.getByText("No response recorded")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// ToolAnalyticsPanel
// ---------------------------------------------------------------------------

describe("ToolAnalyticsPanel", () => {
	it("renders empty state when no tool events", () => {
		render(<ToolAnalyticsPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No tool calls recorded for this agent.")).toBeInTheDocument();
	});

	it("renders summary stats and per-tool table", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.invoke",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { tool_name: "search", parameters: { query: "AI" } },
			}),
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:01Z",
				payload: {
					tool_name: "search",
					success: true,
					result: "Found results",
					duration_ms: 450,
				},
			}),
			makeEvent({
				event_type: "tool.invoke",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {
					tool_name: "write_file",
					parameters: { path: "/tmp/out.txt" },
				},
			}),
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:03Z",
				payload: {
					tool_name: "write_file",
					success: false,
					error: "Permission denied",
					duration_ms: 120,
				},
			}),
			makeEvent({
				event_type: "tool.invoke",
				timestamp: "2026-03-05T10:00:04Z",
				payload: { tool_name: "search", parameters: { query: "agents" } },
			}),
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:05Z",
				payload: {
					tool_name: "search",
					success: true,
					result: "More results",
					duration_ms: 500,
				},
			}),
		];

		render(<ToolAnalyticsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Summary
		expect(screen.getByText("Total calls")).toBeInTheDocument();
		expect(screen.getByText("3")).toBeInTheDocument(); // total
		expect(screen.getByText("Success rate")).toBeInTheDocument();
		expect(screen.getByText("67%")).toBeInTheDocument();

		// Table
		const table = screen.getByTestId("tool-stats-table");
		expect(table).toBeInTheDocument();
		expect(screen.getAllByText("search").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("write_file").length).toBeGreaterThanOrEqual(1);
	});

	it("expands tool call to show parameters and result", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.invoke",
				payload: { tool_name: "search", parameters: { query: "test query" } },
			}),
			makeEvent({
				event_type: "tool.result",
				payload: {
					tool_name: "search",
					success: true,
					result: "Search completed",
					duration_ms: 300,
				},
			}),
		];

		render(<ToolAnalyticsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Click Call Detail row to expand
		const callRows = screen.getAllByText("search");
		// Click the one in the call detail list (not the stats table)
		fireEvent.click(callRows[callRows.length - 1]);

		expect(screen.getByText("Parameters:")).toBeInTheDocument();
		expect(screen.getByText("Result:")).toBeInTheDocument();
	});

	it("shows error details for failed tool calls", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.invoke",
				payload: { tool_name: "write_file", parameters: { path: "/etc/config" } },
			}),
			makeEvent({
				event_type: "tool.result",
				payload: {
					tool_name: "write_file",
					success: false,
					error: "Permission denied",
					duration_ms: 50,
				},
			}),
		];

		render(<ToolAnalyticsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// Expand
		const callRows = screen.getAllByText("write_file");
		fireEvent.click(callRows[callRows.length - 1]);

		expect(screen.getByText("Error:")).toBeInTheDocument();
		expect(screen.getByText("Permission denied")).toBeInTheDocument();
	});

	it("shows 100% success rate when all calls succeed", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.invoke",
				payload: { tool_name: "search", parameters: {} },
			}),
			makeEvent({
				event_type: "tool.result",
				payload: { tool_name: "search", success: true, duration_ms: 100 },
			}),
		];

		render(<ToolAnalyticsPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getAllByText("100%").length).toBeGreaterThanOrEqual(1);
	});
});

// ---------------------------------------------------------------------------
// ErrorRecoveryPanel
// ---------------------------------------------------------------------------

describe("ErrorRecoveryPanel", () => {
	it("renders empty state when no errors", () => {
		render(<ErrorRecoveryPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No errors recorded for this agent.")).toBeInTheDocument();
	});

	it("renders error chains from failed tool results and recovery events", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					tool_name: "search_web",
					success: false,
					error: "Missing required parameter: query",
				},
			}),
			makeEvent({
				event_type: "error.correction",
				timestamp: "2026-03-05T10:00:01Z",
				payload: {
					attempt: 1,
					correction_prompt: "Tool 'search_web' rejected parameters",
				},
			}),
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { tool_name: "search_web", success: true, result: "OK" },
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Total errors")).toBeInTheDocument();
		expect(screen.getByText("Error Chain 1")).toBeInTheDocument();
		expect(screen.getAllByText(/Corrected/).length).toBeGreaterThanOrEqual(1);
	});

	it("displays degradation outcome when error degrades", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					tool_name: "write_file",
					success: false,
					error: "Permission denied",
				},
			}),
			makeEvent({
				event_type: "error.degradation",
				timestamp: "2026-03-05T10:00:01Z",
				payload: {
					reason: "write_file is unavailable",
					fallback: "Proceed without writing",
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Error Chain 1")).toBeInTheDocument();
		expect(screen.getByText(/Degraded/)).toBeInTheDocument();
		expect(screen.getByText("Degradations")).toBeInTheDocument();
	});

	it("shows summary header with totals", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:00Z",
				payload: { tool_name: "t1", success: false, error: "err" },
			}),
			makeEvent({
				event_type: "error.correction",
				timestamp: "2026-03-05T10:00:01Z",
				payload: { attempt: 1 },
			}),
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:02Z",
				payload: { tool_name: "t1", success: true },
			}),
			makeEvent({
				event_type: "tool.result",
				timestamp: "2026-03-05T10:00:03Z",
				payload: { tool_name: "t2", success: false, error: "err2" },
			}),
			makeEvent({
				event_type: "error.degradation",
				timestamp: "2026-03-05T10:00:04Z",
				payload: { reason: "unavailable" },
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// 2 error chains
		expect(screen.getByText("Error Chain 1")).toBeInTheDocument();
		expect(screen.getByText("Error Chain 2")).toBeInTheDocument();

		// Summary
		expect(screen.getByText("Total errors")).toBeInTheDocument();
		expect(screen.getByText("Recovery rate")).toBeInTheDocument();
	});

	it("renders retry events within error chain", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "error.retry",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					attempt: 1,
					max_attempts: 3,
					error_type: "ConnectionError",
					error_message: "Connection timeout",
					delay_ms: 1000,
				},
			}),
			makeEvent({
				event_type: "error.retry",
				timestamp: "2026-03-05T10:00:02Z",
				payload: {
					attempt: 2,
					max_attempts: 3,
					error_type: "ConnectionError",
					error_message: "Connection timeout",
					delay_ms: 2000,
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Error Chain 1")).toBeInTheDocument();
		// First retry becomes the initial error, second becomes a recovery step
		expect(screen.getAllByText("Connection timeout").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText(/Retry \(attempt 2\/3\)/)).toBeInTheDocument();
	});

	it("handles agent.error as initial error", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				timestamp: "2026-03-05T10:00:00Z",
				payload: {
					error_type: "ToolParameterError",
					error_message: "Invalid parameter format",
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("ToolParameterError")).toBeInTheDocument();
		expect(screen.getByText("Invalid parameter format")).toBeInTheDocument();
	});
});
