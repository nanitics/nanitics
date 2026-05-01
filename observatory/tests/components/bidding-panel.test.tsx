import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BiddingPanel } from "../../src/components/patterns/bidding-panel";
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

function makeBiddingEvents(winner: string | null = "Agent-A") {
	return [
		makeEvent({
			event_type: "multi_agent.bidding.start",
			payload: { task: "process claim", participant_names: ["Agent-A", "Agent-B"] },
		}),
		makeEvent({
			event_type: "multi_agent.bidding.bid",
			payload: { agent_name: "Agent-A", confidence: 0.92, reasoning: "Best suited for this", estimated_cost: 2.1 },
		}),
		makeEvent({
			event_type: "multi_agent.bidding.bid",
			payload: { agent_name: "Agent-B", confidence: 0.65, reasoning: "Can handle it", estimated_cost: 1.5 },
		}),
		makeEvent({
			event_type: "multi_agent.bidding.allocated",
			payload: {
				winner,
				confidence: winner ? 0.92 : null,
				total_bids: 2,
				rejection_reason: winner ? null : "Below threshold",
			},
		}),
		makeEvent({
			event_type: "multi_agent.bidding.complete",
			payload: { winner, total_participants: 2, allocated: !!winner },
		}),
	];
}

describe("BiddingPanel", () => {
	it("renders all bids in table", () => {
		render(
			<BiddingPanel
				events={makeBiddingEvents()}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		// Agent-A appears in table + winner summary
		expect(screen.getAllByText("Agent-A").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("Agent-B")).toBeInTheDocument();
		expect(screen.getByText("92%")).toBeInTheDocument();
		expect(screen.getByText("65%")).toBeInTheDocument();
	});

	it("winner row is distinguished", () => {
		render(
			<BiddingPanel
				events={makeBiddingEvents("Agent-A")}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		const winnerRow = screen.getByTestId("winner-row");
		expect(winnerRow).toBeInTheDocument();
		expect(winnerRow).toHaveTextContent("Agent-A");
	});

	it("confidence bars are proportional", () => {
		render(
			<BiddingPanel
				events={makeBiddingEvents()}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		const bars = screen.getAllByTestId("confidence-bar");
		expect(bars).toHaveLength(2);
		// Agent-A = 92%, Agent-B = 65% (sorted by confidence, A first)
		expect(bars[0]).toHaveStyle({ width: "92%" });
		expect(bars[1]).toHaveStyle({ width: "65%" });
	});

	it("handles no-winner case (rejection)", () => {
		render(
			<BiddingPanel
				events={makeBiddingEvents(null)}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("Below threshold")).toBeInTheDocument();
		expect(screen.queryByTestId("winner-row")).not.toBeInTheDocument();
	});

	it("triggers navigation on agent click", () => {
		const onNavigate = vi.fn();
		render(
			<BiddingPanel
				events={makeBiddingEvents()}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={onNavigate}
			/>,
		);
		fireEvent.click(screen.getAllByText("Agent-A")[0]);
		expect(onNavigate).toHaveBeenCalledWith("span-Agent-A");
	});

	it("renders task text", () => {
		render(
			<BiddingPanel
				events={makeBiddingEvents()}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("process claim")).toBeInTheDocument();
	});

	it("shows cost values", () => {
		render(
			<BiddingPanel
				events={makeBiddingEvents()}
				agents={makeAgents("Agent-A", "Agent-B")}
				onNavigateToAgent={vi.fn()}
			/>,
		);
		expect(screen.getByText("$2.1")).toBeInTheDocument();
		expect(screen.getByText("$1.5")).toBeInTheDocument();
	});
});
