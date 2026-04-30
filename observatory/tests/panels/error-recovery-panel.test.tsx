import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ErrorRecoveryPanel } from "../../src/components/panels/error-recovery-panel";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "error-agent",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: [],
	stats: {
		llm_calls: 2,
		tool_calls: 1,
		input_tokens: 500,
		output_tokens: 300,
		duration_ms: 5000,
		errors: 1,
		iterations: 2,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "error-agent",
	summary: {
		event_count: 4,
		duration_ms: 5000,
		has_errors: true,
		agent_name: "error-agent",
		agent_type: "react",
	},
	events: [],
	children: [],
};

// ---------------------------------------------------------------------------
// Summary counts
// ---------------------------------------------------------------------------

describe("ErrorRecoveryPanel summary", () => {
	it("renders summary with correct counts from a chain of error events", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "ValueError", error_message: "bad input" },
			}),
			makeEvent({
				event_type: "error.retry",
				payload: {
					error_type: "ValueError",
					error_message: "bad input",
					attempt: 1,
					max_attempts: 3,
					delay_ms: 1000,
					category: "transient",
				},
			}),
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "APIError", error_message: "timeout" },
			}),
			makeEvent({
				event_type: "error.degradation",
				payload: {
					error_type: "APIError",
					error_message: "timeout",
					degradation_message: "Using cached result",
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Total errors")).toBeInTheDocument();
		expect(screen.getByText("Corrections")).toBeInTheDocument();
		expect(screen.getByText("Degradations")).toBeInTheDocument();
		expect(screen.getByText("Recovery rate")).toBeInTheDocument();
	});

	it("renders empty state when no errors", () => {
		render(<ErrorRecoveryPanel agent={mockAgent} events={[]} spanTree={mockSpanTree} />);

		expect(screen.getByText("No errors recorded for this agent.")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// ErrorNode field access
// ---------------------------------------------------------------------------

describe("ErrorNode field access", () => {
	it("correctly reads error_message from agent.error events", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: {
					agent_name: "test-agent",
					error_type: "ValueError",
					error_message: "Invalid parameter value",
					step_number: 3,
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("ValueError")).toBeInTheDocument();
		expect(screen.getByText("Invalid parameter value")).toBeInTheDocument();
		expect(screen.getByText("Step 3")).toBeInTheDocument();
	});

	it("reads error_type and error_message from error.retry used as initial error", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "error.retry",
				payload: {
					error_type: "ConnectionError",
					error_message: "Connection refused",
					attempt: 1,
					max_attempts: 3,
					delay_ms: 500,
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		// The initial error node should use error_type/error_message
		expect(screen.getByText("ConnectionError")).toBeInTheDocument();
		expect(screen.getByText("Connection refused")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// RecoveryNode field access
// ---------------------------------------------------------------------------

describe("RecoveryNode field access", () => {
	it("correctly reads degradation_message from error.degradation events", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "APIError", error_message: "service down" },
			}),
			makeEvent({
				event_type: "error.degradation",
				payload: {
					error_type: "APIError",
					error_message: "service down",
					degradation_message: "Falling back to cached data",
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Falling back to cached data")).toBeInTheDocument();
		expect(screen.getByText("Degradation")).toBeInTheDocument();
	});

	it("displays category badge for retry events", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "Timeout", error_message: "request timed out" },
			}),
			makeEvent({
				event_type: "error.retry",
				payload: {
					error_type: "Timeout",
					error_message: "request timed out",
					attempt: 1,
					max_attempts: 3,
					delay_ms: 2000,
					category: "transient",
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("transient")).toBeInTheDocument();
		expect(screen.getByText("delay: 2000ms")).toBeInTheDocument();
	});

	it("displays max_attempts for correction events", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "ParseError", error_message: "invalid JSON" },
			}),
			makeEvent({
				event_type: "error.correction",
				payload: {
					error_type: "ParseError",
					error_message: "invalid JSON",
					correction_prompt: "Please output valid JSON",
					attempt: 2,
					max_attempts: 3,
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText("Correction (attempt 2/3)")).toBeInTheDocument();
		expect(screen.getByText("Please output valid JSON")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// Chain classification
// ---------------------------------------------------------------------------

describe("Error chain classification", () => {
	it("classifies chain with degradation as degraded", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "Error", error_message: "fail" },
			}),
			makeEvent({
				event_type: "error.degradation",
				payload: {
					error_type: "Error",
					error_message: "fail",
					degradation_message: "Graceful fallback",
				},
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Degraded/)).toBeInTheDocument();
	});

	it("classifies chain with correction as corrected", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "Error", error_message: "fail" },
			}),
			makeEvent({
				event_type: "error.correction",
				payload: {
					error_type: "Error",
					error_message: "fail",
					correction_prompt: "Fix it",
					attempt: 1,
					max_attempts: 2,
				},
			}),
			// Finalized because next error starts new chain
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "Other", error_message: "other fail" },
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Corrected/)).toBeInTheDocument();
	});

	it("classifies chain with only retries as retried", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "Error", error_message: "fail" },
			}),
			makeEvent({
				event_type: "error.retry",
				payload: {
					error_type: "Error",
					error_message: "fail",
					attempt: 1,
					max_attempts: 3,
					delay_ms: 500,
				},
			}),
			// Finalized because next error starts new chain
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "Other", error_message: "other" },
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Retried/)).toBeInTheDocument();
	});

	it("classifies chain without recovery as unresolved", () => {
		const events: TraceEvent[] = [
			makeEvent({
				event_type: "agent.error",
				payload: { error_type: "FatalError", error_message: "crash" },
			}),
		];

		render(<ErrorRecoveryPanel agent={mockAgent} events={events} spanTree={mockSpanTree} />);

		expect(screen.getByText(/Unresolved/)).toBeInTheDocument();
	});
});
