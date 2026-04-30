import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SupervisionPanel } from "../../src/components/patterns/supervision-panel";
import type { AgentInfo } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

function makeAgents(...names: string[]): AgentInfo[] {
	return names.map((name) => ({
		agent_name: name,
		agent_type: "react",
		span_id: `span-${name}`,
		capabilities: [],
		stats: {
			llm_calls: 0,
			tool_calls: 0,
			input_tokens: 0,
			output_tokens: 0,
			duration_ms: 0,
			errors: 0,
			iterations: 0,
		},
	}));
}

describe("SupervisionPanel", () => {
	it("renders intervention sequence in order", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "retry",
					trigger_name: "QualityTrigger",
					feedback: "needs more detail",
					attempt: 1,
				},
			}),
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "reassign",
					trigger_name: "QualityTrigger",
					feedback: "still insufficient",
					reassigned_to: "Agent-B",
					attempt: 2,
				},
			}),
		];
		render(<SupervisionPanel events={events} agents={makeAgents("Agent-A", "Agent-B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("Attempt 1")).toBeInTheDocument();
		expect(screen.getByText("Attempt 2")).toBeInTheDocument();
		expect(screen.getByText("needs more detail")).toBeInTheDocument();
		expect(screen.getByText("still insufficient")).toBeInTheDocument();
	});

	it("shows action badges with correct colors", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "retry",
					trigger_name: "T1",
					attempt: 1,
				},
			}),
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "escalate",
					trigger_name: "T2",
					attempt: 2,
				},
			}),
		];
		render(<SupervisionPanel events={events} agents={makeAgents("Agent-A")} onNavigateToAgent={vi.fn()} />);
		const retryBadge = screen.getByText("retry");
		expect(retryBadge).toHaveClass("bg-warning-muted");
		const escalateBadge = screen.getByText("escalate");
		expect(escalateBadge).toHaveClass("bg-destructive-muted");
	});

	it("retry shows feedback", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "retry",
					trigger_name: "QualityTrigger",
					feedback: "Output lacks specificity",
					attempt: 1,
				},
			}),
		];
		render(<SupervisionPanel events={events} agents={makeAgents("Agent-A")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("Output lacks specificity")).toBeInTheDocument();
	});

	it("reassign shows target agent link", () => {
		const onNavigate = vi.fn();
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "reassign",
					trigger_name: "T1",
					feedback: "routing to better agent",
					reassigned_to: "Agent-B",
					attempt: 1,
				},
			}),
		];
		render(
			<SupervisionPanel events={events} agents={makeAgents("Agent-A", "Agent-B")} onNavigateToAgent={onNavigate} />,
		);
		// Agent-B should be a clickable link
		fireEvent.click(screen.getByText("Agent-B"));
		expect(onNavigate).toHaveBeenCalledWith("span-Agent-B");
	});

	it("handles single intervention", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Worker",
					action: "retry",
					trigger_name: "Accuracy",
					feedback: "try again",
					attempt: 1,
				},
			}),
		];
		render(<SupervisionPanel events={events} agents={makeAgents("Worker")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("1 attempt, final action: retry")).toBeInTheDocument();
	});

	it("handles multi-intervention with reassign summary", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "retry",
					trigger_name: "T1",
					attempt: 1,
				},
			}),
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "reassign",
					trigger_name: "T1",
					reassigned_to: "Agent-B",
					attempt: 2,
				},
			}),
		];
		render(<SupervisionPanel events={events} agents={makeAgents("Agent-A", "Agent-B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("2 attempts, reassigned to Agent-B")).toBeInTheDocument();
	});

	it("shows trigger name badges", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.supervision",
				payload: {
					supervised_agent: "Agent-A",
					action: "retry",
					trigger_name: "QualityTrigger",
					attempt: 1,
				},
			}),
		];
		render(<SupervisionPanel events={events} agents={makeAgents("Agent-A")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("QualityTrigger")).toBeInTheDocument();
	});
});
