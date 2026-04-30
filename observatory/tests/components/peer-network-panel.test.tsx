import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PeerNetworkPanel } from "../../src/components/patterns/peer-network-panel";
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

function makePeerNetworkEvents() {
	return [
		makeEvent({
			event_type: "multi_agent.peer.start",
			payload: {
				task: "Research question",
				entry_agent: "Coordinator",
				peer_names: ["Coordinator", "Analyst", "Researcher"],
				peer_descriptions: {
					Coordinator: "Coordinates work",
					Analyst: "Analyzes data",
					Researcher: "Researches topics",
				},
				max_invocations: 10,
			},
		}),
		makeEvent({
			event_type: "multi_agent.peer.consultation",
			payload: {
				from_agent: "Coordinator",
				to_agent: "Analyst",
				message: "Please analyze the data",
				consultation_number: 1,
				remaining_budget: 9,
			},
		}),
		makeEvent({
			event_type: "multi_agent.peer.consultation",
			payload: {
				from_agent: "Analyst",
				to_agent: "Researcher",
				message: "Need more context",
				consultation_number: 2,
				remaining_budget: 8,
			},
		}),
		makeEvent({
			event_type: "multi_agent.peer.consultation",
			payload: {
				from_agent: "Coordinator",
				to_agent: "Analyst",
				message: "Follow up question",
				consultation_number: 3,
				remaining_budget: 7,
			},
		}),
		makeEvent({
			event_type: "multi_agent.peer.complete",
			payload: {
				entry_agent: "Coordinator",
				total_consultations: 3,
				invocations_used: 3,
				agents_consulted: ["Analyst", "Researcher"],
				termination_reason: "Task completed",
			},
		}),
	];
}

describe("PeerNetworkPanel", () => {
	it("renders graph with correct number of agent nodes", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("peer-network-panel")).toBeInTheDocument();
		expect(screen.getByTestId("peer-nodes")).toBeInTheDocument();
		expect(screen.getByTestId("peer-node-Coordinator")).toBeInTheDocument();
		expect(screen.getByTestId("peer-node-Analyst")).toBeInTheDocument();
		expect(screen.getByTestId("peer-node-Researcher")).toBeInTheDocument();
	});

	it("distinguishes entry agent from peers", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		const entryNode = screen.getByTestId("peer-node-Coordinator");
		const entryCircle = entryNode.querySelector("circle");
		expect(entryCircle?.getAttribute("class")).toContain("stroke-primary");

		const peerNode = screen.getByTestId("peer-node-Analyst");
		const peerCircle = peerNode.querySelector("circle");
		expect(peerCircle?.getAttribute("class")).toContain("stroke-border");
	});

	it("renders consultation edges", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("peer-edges")).toBeInTheDocument();
		// Coordinator → Analyst (2 consultations), Analyst → Researcher (1)
		expect(screen.getByTestId("edge-Coordinator-Analyst")).toBeInTheDocument();
		expect(screen.getByTestId("edge-Analyst-Researcher")).toBeInTheDocument();
	});

	it("shows edge count label for repeated consultations", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		// Coordinator → Analyst consulted 2 times
		expect(screen.getByText("×2")).toBeInTheDocument();
	});

	it("renders budget indicator with correct values", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("budget-indicator")).toBeInTheDocument();
		expect(screen.getByText("3 / 10 invocations")).toBeInTheDocument();
	});

	it("fires navigation on agent node click", () => {
		const onNavigate = vi.fn();
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={onNavigate}
			/>,
		);
		fireEvent.click(screen.getByTestId("peer-node-Analyst"));
		expect(onNavigate).toHaveBeenCalledWith("span-Analyst");
	});

	it("shows consultation log when toggled", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		// Log is collapsed initially
		expect(screen.queryByText("Please analyze the data")).not.toBeInTheDocument();

		// Expand
		fireEvent.click(screen.getByTestId("consultation-log-toggle"));
		expect(screen.getByText(/Please analyze the data/)).toBeInTheDocument();
		expect(screen.getByText(/Need more context/)).toBeInTheDocument();
		expect(screen.getByText(/Follow up question/)).toBeInTheDocument();
	});

	it("renders footer with consultation stats", () => {
		render(
			<PeerNetworkPanel
				events={makePeerNetworkEvents()}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("3 consultations")).toBeInTheDocument();
		expect(screen.getByText("· 2 agents consulted")).toBeInTheDocument();
		expect(screen.getByText("· Task completed")).toBeInTheDocument();
	});

	it("handles partial event set (no complete event)", () => {
		const events = makePeerNetworkEvents().filter((e) => e.event_type !== "multi_agent.peer.complete");
		render(
			<PeerNetworkPanel
				events={events}
				agents={makeAgents("Coordinator", "Analyst", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		// Graph still renders
		expect(screen.getByTestId("peer-nodes")).toBeInTheDocument();
		// Budget uses consultation count as fallback
		expect(screen.getByText("3 / 10 invocations")).toBeInTheDocument();
	});

	it("handles bidirectional consultations", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.peer.start",
				payload: {
					task: "Discuss",
					entry_agent: "A",
					peer_names: ["A", "B"],
					peer_descriptions: { A: "Agent A", B: "Agent B" },
					max_invocations: 10,
				},
			}),
			makeEvent({
				event_type: "multi_agent.peer.consultation",
				payload: {
					from_agent: "A",
					to_agent: "B",
					message: "Hello B",
					consultation_number: 1,
					remaining_budget: 9,
				},
			}),
			makeEvent({
				event_type: "multi_agent.peer.consultation",
				payload: {
					from_agent: "B",
					to_agent: "A",
					message: "Hello A",
					consultation_number: 2,
					remaining_budget: 8,
				},
			}),
		];
		render(<PeerNetworkPanel events={events} agents={makeAgents("A", "B")} onNavigateToAgent={vi.fn()} />);
		// Both edges should exist
		expect(screen.getByTestId("edge-A-B")).toBeInTheDocument();
		expect(screen.getByTestId("edge-B-A")).toBeInTheDocument();
	});
});
