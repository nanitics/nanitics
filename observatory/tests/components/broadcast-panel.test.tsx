import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BroadcastPanel } from "../../src/components/patterns/broadcast-panel";
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

function makeBroadcastEvents(withError = false) {
	return [
		makeEvent({
			event_type: "multi_agent.broadcast.start",
			payload: {
				task: "evaluate market conditions",
				agent_names: ["Agent-A", "Agent-B", "Agent-C"],
				response_strategy: "collect_all",
			},
		}),
		makeEvent({
			event_type: "multi_agent.broadcast.response",
			payload: { agent_name: "Agent-A", output: "The market is bullish", steps: 5, error: null },
		}),
		makeEvent({
			event_type: "multi_agent.broadcast.response",
			payload: { agent_name: "Agent-B", output: "Based on analysis", steps: 3, error: null },
		}),
		makeEvent({
			event_type: "multi_agent.broadcast.response",
			payload: {
				agent_name: "Agent-C",
				output: "Analysis failed",
				steps: 4,
				error: withError ? "timeout" : null,
			},
		}),
		makeEvent({
			event_type: "multi_agent.broadcast.complete",
			payload: {
				total_agents: 3,
				responses_collected: withError ? 2 : 3,
				response_strategy: "collect_all",
				aggregated_output: "Combined analysis result",
			},
		}),
	];
}

describe("BroadcastPanel", () => {
	it("renders response table with correct counts", () => {
		render(
			<BroadcastPanel
				events={makeBroadcastEvents()}
				agents={makeAgents("Agent-A", "Agent-B", "Agent-C")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("3/3 responses")).toBeInTheDocument();
		expect(screen.getByText("Agent-A")).toBeInTheDocument();
		expect(screen.getByText("Agent-B")).toBeInTheDocument();
		expect(screen.getByText("Agent-C")).toBeInTheDocument();
	});

	it("shows strategy badge", () => {
		render(
			<BroadcastPanel
				events={makeBroadcastEvents()}
				agents={makeAgents("Agent-A", "Agent-B", "Agent-C")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("collect_all")).toBeInTheDocument();
	});

	it("handles partial responses with errors", () => {
		render(
			<BroadcastPanel
				events={makeBroadcastEvents(true)}
				agents={makeAgents("Agent-A", "Agent-B", "Agent-C")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("2/3 responses")).toBeInTheDocument();
		expect(screen.getByText("✗")).toBeInTheDocument();
	});

	it("triggers navigation on agent click", () => {
		const onNavigate = vi.fn();
		render(
			<BroadcastPanel
				events={makeBroadcastEvents()}
				agents={makeAgents("Agent-A", "Agent-B", "Agent-C")}
				onNavigateToAgent={onNavigate}
			/>,
		);
		fireEvent.click(screen.getByText("Agent-A"));
		expect(onNavigate).toHaveBeenCalledWith("span-Agent-A");
	});

	it("shows aggregated output", () => {
		render(
			<BroadcastPanel
				events={makeBroadcastEvents()}
				agents={makeAgents("Agent-A", "Agent-B", "Agent-C")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("Combined analysis result")).toBeInTheDocument();
	});

	it("shows task text", () => {
		render(
			<BroadcastPanel
				events={makeBroadcastEvents()}
				agents={makeAgents("Agent-A", "Agent-B", "Agent-C")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("evaluate market conditions")).toBeInTheDocument();
	});
});
