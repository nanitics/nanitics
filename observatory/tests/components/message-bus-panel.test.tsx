import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MessageBusPanel } from "../../src/components/patterns/message-bus-panel";
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

function makeSmallMessageBusEvents() {
	return [
		makeEvent({
			event_type: "multi_agent.bus.start",
			payload: {
				seed_topics: ["analysis", "research"],
				seed_count: 2,
				subscriber_count: 3,
				subscriptions: {
					Analyzer: ["analysis"],
					Researcher: ["research"],
					Summarizer: ["analysis", "research"],
				},
				max_messages: 50,
				max_depth: 5,
			},
		}),
		makeEvent({
			event_type: "multi_agent.bus.published",
			payload: {
				message_id: "msg-1",
				topic: "analysis",
				author: "Coordinator",
				content: "Initial analysis request",
				depth: 0,
				parent_message_id: null,
			},
		}),
		makeEvent({
			event_type: "multi_agent.bus.published",
			payload: {
				message_id: "msg-2",
				topic: "research",
				author: "Coordinator",
				content: "Initial research request",
				depth: 0,
				parent_message_id: null,
			},
		}),
		makeEvent({
			event_type: "multi_agent.bus.published",
			payload: {
				message_id: "msg-3",
				topic: "analysis",
				author: "Analyzer",
				content: "Analysis result",
				depth: 1,
				parent_message_id: "msg-1",
			},
		}),
		makeEvent({
			event_type: "multi_agent.bus.delivered",
			payload: {
				message_id: "msg-1",
				topic: "analysis",
				agent_name: "Analyzer",
				output: "Processed analysis",
				steps: 3,
				messages_published: 1,
				error: null,
			},
		}),
		makeEvent({
			event_type: "multi_agent.bus.delivered",
			payload: {
				message_id: "msg-2",
				topic: "research",
				agent_name: "Researcher",
				output: "Research findings",
				steps: 5,
				messages_published: 0,
				error: null,
			},
		}),
		makeEvent({
			event_type: "multi_agent.bus.complete",
			payload: {
				total_messages: 3,
				total_executions: 5,
				max_depth_reached: 1,
				termination_reason: "All messages processed",
				agent_execution_counts: {
					Analyzer: 2,
					Researcher: 1,
					Summarizer: 2,
				},
			},
		}),
	];
}

function makeLargeMessageBusEvents() {
	const events = [
		makeEvent({
			event_type: "multi_agent.bus.start",
			payload: {
				seed_topics: ["topic-a", "topic-b"],
				seed_count: 2,
				subscriber_count: 3,
				subscriptions: { A: ["topic-a"], B: ["topic-b"], C: ["topic-a", "topic-b"] },
				max_messages: 100,
				max_depth: 10,
			},
		}),
	];

	// Create 45 published events (above threshold)
	for (let i = 0; i < 45; i++) {
		events.push(
			makeEvent({
				event_type: "multi_agent.bus.published",
				payload: {
					message_id: `msg-${i}`,
					topic: i % 2 === 0 ? "topic-a" : "topic-b",
					author: i % 3 === 0 ? "A" : i % 3 === 1 ? "B" : "C",
					content: `Message ${i} content`,
					depth: Math.floor(i / 5),
					parent_message_id: i > 1 ? `msg-${i - 2}` : null,
				},
			}),
		);
	}

	events.push(
		makeEvent({
			event_type: "multi_agent.bus.complete",
			payload: {
				total_messages: 45,
				total_executions: 30,
				max_depth_reached: 8,
				termination_reason: "Max depth reached",
				agent_execution_counts: { A: 15, B: 10, C: 5 },
			},
		}),
	);

	return events;
}

describe("MessageBusPanel", () => {
	describe("DAG view (≤40 messages)", () => {
		it("renders DAG view for small message count", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			expect(screen.getByTestId("message-bus-panel")).toBeInTheDocument();
			expect(screen.getByTestId("message-dag")).toBeInTheDocument();
		});

		it("renders correct number of message nodes", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			expect(screen.getByTestId("bus-node-msg-1")).toBeInTheDocument();
			expect(screen.getByTestId("bus-node-msg-2")).toBeInTheDocument();
			expect(screen.getByTestId("bus-node-msg-3")).toBeInTheDocument();
		});

		it("renders parent→child edges", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			// msg-1 → msg-3 edge
			expect(screen.getByTestId("bus-edge-msg-1-msg-3")).toBeInTheDocument();
		});

		it("shows header with topic count and message stats", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			expect(screen.getByText("2 topics")).toBeInTheDocument();
			expect(screen.getByText("· 3 messages")).toBeInTheDocument();
			expect(screen.getByText("· max depth 1")).toBeInTheDocument();
		});

		it("shows expanded message details on node click", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			// Click on msg-1
			fireEvent.click(screen.getByTestId("bus-node-msg-1"));
			expect(screen.getByTestId("expanded-message-details")).toBeInTheDocument();
			// Text appears in both the SVG node and the expanded panel
			const matches = screen.getAllByText("Initial analysis request");
			expect(matches.length).toBeGreaterThanOrEqual(2);
			// Delivery info
			expect(screen.getByText(/Delivered to 1 agent/)).toBeInTheDocument();
		});

		it("closes expanded details on close button", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			fireEvent.click(screen.getByTestId("bus-node-msg-1"));
			expect(screen.getByTestId("expanded-message-details")).toBeInTheDocument();
			fireEvent.click(screen.getByText("✕"));
			expect(screen.queryByTestId("expanded-message-details")).not.toBeInTheDocument();
		});
	});

	describe("Summary view (>40 messages)", () => {
		it("renders summary view for large message count", () => {
			render(
				<MessageBusPanel
					events={makeLargeMessageBusEvents()}
					agents={makeAgents("A", "B", "C")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			expect(screen.getByTestId("message-bus-panel")).toBeInTheDocument();
			expect(screen.getByTestId("message-summary")).toBeInTheDocument();
			expect(screen.queryByTestId("message-dag")).not.toBeInTheDocument();
		});

		it("renders topic breakdown table", () => {
			render(
				<MessageBusPanel
					events={makeLargeMessageBusEvents()}
					agents={makeAgents("A", "B", "C")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			expect(screen.getByTestId("topic-breakdown")).toBeInTheDocument();
			expect(screen.getByText("topic-a")).toBeInTheDocument();
			expect(screen.getByText("topic-b")).toBeInTheDocument();
		});
	});

	describe("Agent activity table", () => {
		it("renders agent activity table from complete event", () => {
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={vi.fn()}
				/>,
			);
			expect(screen.getByTestId("agent-activity")).toBeInTheDocument();
			// Analyzer has 2 executions
			const rows = screen.getByTestId("agent-activity").querySelectorAll("tbody tr");
			expect(rows.length).toBeGreaterThanOrEqual(3);
		});

		it("triggers navigation on agent click in activity table", () => {
			const onNavigate = vi.fn();
			render(
				<MessageBusPanel
					events={makeSmallMessageBusEvents()}
					agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
					onNavigateToAgent={onNavigate}
				/>,
			);
			// Analyzer may appear in the DAG and in the activity table — click the one in the table
			const activityTable = screen.getByTestId("agent-activity");
			const analyzerButton = activityTable.querySelector("button")!;
			fireEvent.click(analyzerButton);
			expect(onNavigate).toHaveBeenCalledWith("span-Analyzer");
		});
	});

	it("handles partial event set (no complete event)", () => {
		const events = makeSmallMessageBusEvents().filter((e) => e.event_type !== "multi_agent.bus.complete");
		render(
			<MessageBusPanel
				events={events}
				agents={makeAgents("Coordinator", "Analyzer", "Researcher")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		// Panel still renders with DAG
		expect(screen.getByTestId("message-bus-panel")).toBeInTheDocument();
		expect(screen.getByTestId("message-dag")).toBeInTheDocument();
	});

	it("renders termination reason in header", () => {
		render(
			<MessageBusPanel
				events={makeSmallMessageBusEvents()}
				agents={makeAgents("Coordinator", "Analyzer", "Researcher", "Summarizer")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("· All messages processed")).toBeInTheDocument();
	});
});
