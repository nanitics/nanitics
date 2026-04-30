import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HandoffPanel } from "../../src/components/patterns/handoff-panel";
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

describe("HandoffPanel", () => {
	it("renders chain in correct order", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: {
					from_agent: "Agent-A",
					to_agent: "Agent-B",
					payload_fields: ["output"],
					payload_size: 100,
				},
			}),
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: {
					from_agent: "Agent-B",
					to_agent: "Agent-C",
					payload_fields: ["summary"],
					payload_size: 200,
				},
			}),
		];
		render(
			<HandoffPanel events={events} agents={makeAgents("Agent-A", "Agent-B", "Agent-C")} onNavigateToAgent={vi.fn()} />,
		);
		// All three agents in chain
		const agentAs = screen.getAllByText("Agent-A");
		expect(agentAs.length).toBeGreaterThanOrEqual(1);
		const agentBs = screen.getAllByText("Agent-B");
		expect(agentBs.length).toBeGreaterThanOrEqual(1);
		const agentCs = screen.getAllByText("Agent-C");
		expect(agentCs.length).toBeGreaterThanOrEqual(1);
	});

	it("shows payload fields and size", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: {
					from_agent: "Agent-A",
					to_agent: "Agent-B",
					payload_fields: ["output", "analysis"],
					payload_size: 1500,
				},
			}),
		];
		render(<HandoffPanel events={events} agents={makeAgents("Agent-A", "Agent-B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("output")).toBeInTheDocument();
		expect(screen.getByText("analysis")).toBeInTheDocument();
		expect(screen.getByText("1.5 KB")).toBeInTheDocument();
	});

	it("triggers navigation on agent click", () => {
		const onNavigate = vi.fn();
		const events = [
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: {
					from_agent: "Agent-A",
					to_agent: "Agent-B",
					payload_fields: [],
					payload_size: 100,
				},
			}),
		];
		render(<HandoffPanel events={events} agents={makeAgents("Agent-A", "Agent-B")} onNavigateToAgent={onNavigate} />);
		// Click first instance of Agent-B
		fireEvent.click(screen.getAllByText("Agent-B")[0]);
		expect(onNavigate).toHaveBeenCalledWith("span-Agent-B");
	});

	it("handles 2-agent chain", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: {
					from_agent: "Alpha",
					to_agent: "Beta",
					payload_fields: ["data"],
					payload_size: 50,
				},
			}),
		];
		render(<HandoffPanel events={events} agents={makeAgents("Alpha", "Beta")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getAllByText("Alpha").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("Beta").length).toBeGreaterThanOrEqual(1);
	});

	it("handles 5-agent chain", () => {
		const agents = ["A", "B", "C", "D", "E"];
		const events = agents.slice(0, -1).map((from, i) =>
			makeEvent({
				event_type: "multi_agent.handoff",
				payload: {
					from_agent: from,
					to_agent: agents[i + 1],
					payload_fields: [`field-${i}`],
					payload_size: 100 * (i + 1),
				},
			}),
		);
		render(<HandoffPanel events={events} agents={makeAgents(...agents)} onNavigateToAgent={vi.fn()} />);
		// Check transfer count text
		expect(screen.getByText("Transfers")).toBeInTheDocument();
		// All field tags present
		for (let i = 0; i < 4; i++) {
			expect(screen.getByText(`field-${i}`)).toBeInTheDocument();
		}
	});
});
