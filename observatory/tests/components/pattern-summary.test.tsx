import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PatternSummary } from "../../src/components/patterns/pattern-summary";
import type { AgentInfo } from "../../src/types";
import type { DetectedPattern } from "../../src/utils/pattern-detector";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function makePattern(overrides: Partial<DetectedPattern>): DetectedPattern {
	return {
		type: "delegation",
		events: [makeEvent({ event_type: "multi_agent.delegation" })],
		spanId: "span-root",
		label: "A → B",
		...overrides,
	};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PatternSummary", () => {
	it("renders nothing when no patterns", () => {
		const { container } = render(<PatternSummary patterns={[]} agents={[]} onNavigateToAgent={vi.fn()} />);
		expect(container.firstChild).toBeNull();
	});

	it("shows collapsed state with pattern count", () => {
		const patterns = [
			makePattern({ type: "delegation", label: "A → B" }),
			makePattern({ type: "bidding", label: "Bidding (2)" }),
		];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A", "B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("2 patterns detected")).toBeInTheDocument();
	});

	it("shows correct pattern type badges", () => {
		const patterns = [
			makePattern({ type: "delegation" }),
			makePattern({ type: "broadcast" }),
			makePattern({ type: "bidding" }),
		];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A", "B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("Delegation")).toBeInTheDocument();
		expect(screen.getByText("Broadcast")).toBeInTheDocument();
		expect(screen.getByText("Bidding")).toBeInTheDocument();
	});

	it("expands on click to show pattern cards", () => {
		const patterns = [makePattern({ type: "delegation", label: "Alice → Bob" })];
		render(<PatternSummary patterns={patterns} agents={makeAgents("Alice", "Bob")} onNavigateToAgent={vi.fn()} />);

		// Initially collapsed — no label visible in expanded area
		expect(screen.queryByText("Alice → Bob")).not.toBeInTheDocument();

		// Click to expand
		fireEvent.click(screen.getByText("1 pattern detected"));
		expect(screen.getByText("Alice → Bob")).toBeInTheDocument();
	});

	it("shows single pattern text", () => {
		const patterns = [makePattern({ type: "handoff" })];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("1 pattern detected")).toBeInTheDocument();
	});

	it("de-duplicates type badges when multiple patterns of same type", () => {
		const patterns = [
			makePattern({ type: "delegation", label: "A → B" }),
			makePattern({ type: "delegation", label: "A → C" }),
		];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A", "B", "C")} onNavigateToAgent={vi.fn()} />);
		// Should show "Delegation" badge only once
		const badges = screen.getAllByText("Delegation");
		expect(badges).toHaveLength(1);
	});

	// --- 5C pattern type badges ---

	it("shows 5C pattern type badges", () => {
		const patterns = [
			makePattern({ type: "debate", label: "Debate (2 debaters)" }),
			makePattern({ type: "consensus", label: "Consensus (3 agents)" }),
			makePattern({ type: "blackboard", label: "Blackboard (2 agents)" }),
			makePattern({ type: "peer_network", label: "Peer network from E" }),
			makePattern({ type: "message_bus", label: "Message bus (2 topics)" }),
		];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A", "B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("Debate")).toBeInTheDocument();
		expect(screen.getByText("Consensus")).toBeInTheDocument();
		expect(screen.getByText("Blackboard")).toBeInTheDocument();
		expect(screen.getByText("Peer Network")).toBeInTheDocument();
		expect(screen.getByText("Message Bus")).toBeInTheDocument();
		expect(screen.getByText("5 patterns detected")).toBeInTheDocument();
	});

	it("dispatches 5C patterns to correct panels when expanded", () => {
		const patterns = [
			makePattern({
				type: "debate",
				label: "Debate (2 debaters)",
				events: [
					makeEvent({
						event_type: "multi_agent.debate.start",
						payload: {
							task: "Test debate task",
							debater_names: ["A", "B"],
							positions: { A: "pro", B: "con" },
							max_rounds: 3,
							resolution_strategy: "JudgeResolution",
						},
					}),
					makeEvent({
						event_type: "multi_agent.debate.argument",
						payload: {
							round: 1,
							agent_name: "A",
							position: "pro",
							argument: "Argument from A",
						},
					}),
				],
			}),
			makePattern({
				type: "consensus",
				label: "Consensus (3 agents)",
				events: [
					makeEvent({
						event_type: "multi_agent.consensus.start",
						payload: {
							task: "Test consensus task",
							agent_names: ["A", "B", "C"],
							strategy: "MajorityVoting",
							deliberation_enabled: false,
						},
					}),
				],
			}),
		];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A", "B", "C")} onNavigateToAgent={vi.fn()} />);
		fireEvent.click(screen.getByText("2 patterns detected"));
		// Debate panel renders strategy badge and initial positions
		expect(screen.getByText("JudgeResolution")).toBeInTheDocument();
		expect(screen.getByText("Initial positions")).toBeInTheDocument();
		// Consensus panel renders strategy badge and agent count
		expect(screen.getByText("MajorityVoting")).toBeInTheDocument();
		expect(screen.getByText("3 agents")).toBeInTheDocument();
	});

	it("renders mix of 5B and 5C pattern badges", () => {
		const patterns = [
			makePattern({ type: "delegation", label: "A → B" }),
			makePattern({ type: "debate", label: "Debate (2 debaters)" }),
			makePattern({ type: "blackboard", label: "Blackboard (3 agents)" }),
		];
		render(<PatternSummary patterns={patterns} agents={makeAgents("A", "B")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("Delegation")).toBeInTheDocument();
		expect(screen.getByText("Debate")).toBeInTheDocument();
		expect(screen.getByText("Blackboard")).toBeInTheDocument();
		expect(screen.getByText("3 patterns detected")).toBeInTheDocument();
	});
});
