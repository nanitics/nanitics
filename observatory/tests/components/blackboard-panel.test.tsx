import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BlackboardPanel } from "../../src/components/patterns/blackboard-panel";
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

function makeRoundEvent(
	roundNumber: number,
	agents: string[],
	contributions: number,
	totalContributions: number,
	roundEntries?: Array<{
		operation: string;
		author: string;
		content?: string;
		scope?: string | null;
		entry_id: string;
		original_entry_id?: string | null;
		retract_reason?: string | null;
	}>,
) {
	return makeEvent({
		event_type: "blackboard.round",
		payload: {
			round_number: roundNumber,
			agents_activated: agents,
			contributions,
			total_contributions: totalContributions,
			...(roundEntries !== undefined ? { round_entries: roundEntries } : {}),
		},
	});
}

describe("BlackboardPanel", () => {
	it("renders round cards without entries (backward compat)", () => {
		const events = [
			makeEvent({
				event_type: "blackboard.start",
				payload: { task: "analyze", agent_names: ["a1"], control_strategy: "ScheduledControl", max_rounds: 3 },
			}),
			makeRoundEvent(1, ["a1"], 2, 2),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByTestId("blackboard-round-1")).toBeInTheDocument();
		expect(screen.queryByTestId("round-1-entries")).not.toBeInTheDocument();
	});

	it("renders write entries in round cards", () => {
		const events = [
			makeRoundEvent(1, ["writer"], 1, 1, [
				{
					operation: "write",
					author: "writer",
					content: "Some analysis content",
					entry_id: "entry-1",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("writer")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByTestId("round-1-entries")).toBeInTheDocument();
		expect(screen.getAllByTestId("round-entry")).toHaveLength(1);
		expect(screen.getAllByText("writer").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("write")).toBeInTheDocument();
	});

	it("renders supersede entries with original entry indicator", () => {
		const events = [
			makeRoundEvent(1, ["editor"], 1, 1, [
				{
					operation: "supersede",
					author: "editor",
					content: "Updated analysis",
					entry_id: "entry-2",
					original_entry_id: "entry-1-abcdef01",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("editor")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("supersede")).toBeInTheDocument();
		// Original entry ID should be shown (truncated)
		expect(screen.getByText("← entry-1-…")).toBeInTheDocument();
	});

	it("renders retract entries with reason", () => {
		const events = [
			makeRoundEvent(1, ["reviewer"], 1, 1, [
				{
					operation: "retract",
					author: "reviewer",
					content: "",
					entry_id: "entry-1",
					retract_reason: "Outdated information",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("reviewer")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("retract")).toBeInTheDocument();
		expect(screen.getByText("Outdated information")).toBeInTheDocument();
	});

	it("renders scope tags on entries", () => {
		const events = [
			makeRoundEvent(1, ["a1"], 1, 1, [
				{
					operation: "write",
					author: "a1",
					content: "scoped content",
					scope: "findings",
					entry_id: "entry-1",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("findings")).toBeInTheDocument();
	});

	it("content previews are collapsed by default and expandable", () => {
		const events = [
			makeRoundEvent(1, ["a1"], 1, 1, [
				{
					operation: "write",
					author: "a1",
					content: "Detailed analysis of the market",
					entry_id: "entry-1",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1")} onNavigateToAgent={vi.fn()} />);
		// Content not visible by default
		expect(screen.queryByTestId("content-preview")).not.toBeInTheDocument();

		// Click toggle to expand
		fireEvent.click(screen.getByTestId("toggle-preview"));
		expect(screen.getByTestId("content-preview")).toBeInTheDocument();
		expect(screen.getByText("Detailed analysis of the market")).toBeInTheDocument();
	});

	it("renders multiple entries from different agents", () => {
		const events = [
			makeRoundEvent(1, ["a1", "a2"], 2, 2, [
				{
					operation: "write",
					author: "a1",
					content: "from a1",
					entry_id: "entry-1",
				},
				{
					operation: "write",
					author: "a2",
					content: "from a2",
					entry_id: "entry-2",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1", "a2")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getAllByTestId("round-entry")).toHaveLength(2);
	});

	it("renders mixed operation types in same round", () => {
		const events = [
			makeRoundEvent(1, ["a1"], 3, 3, [
				{ operation: "write", author: "a1", content: "initial", entry_id: "e1" },
				{ operation: "supersede", author: "a1", content: "updated", entry_id: "e2", original_entry_id: "e1" },
				{ operation: "retract", author: "a1", content: "", entry_id: "e3", retract_reason: "wrong" },
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1")} onNavigateToAgent={vi.fn()} />);
		const badges = screen.getAllByTestId("operation-badge");
		expect(badges).toHaveLength(3);
		expect(badges[0].textContent).toBe("write");
		expect(badges[1].textContent).toBe("supersede");
		expect(badges[2].textContent).toBe("retract");
	});

	it("shows full content when expanded", () => {
		const fullContent = "This is the full content that is much longer than the 200 char preview";
		const events = [
			makeRoundEvent(1, ["a1"], 1, 1, [
				{
					operation: "write",
					author: "a1",
					content: fullContent,
					entry_id: "entry-1",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1")} onNavigateToAgent={vi.fn()} />);

		// Click the row to expand
		fireEvent.click(screen.getByTestId("toggle-preview"));
		expect(screen.getByTestId("content-preview")).toBeInTheDocument();
		// Should show the full content, not the preview
		expect(screen.getByText(fullContent)).toBeInTheDocument();
	});

	it("entire entry row is clickable to toggle content", () => {
		const events = [
			makeRoundEvent(1, ["a1"], 1, 1, [
				{
					operation: "write",
					author: "a1",
					content: "Some content",
					entry_id: "entry-1",
				},
			]),
		];
		render(<BlackboardPanel events={events} agents={makeAgents("a1")} onNavigateToAgent={vi.fn()} />);

		// The toggle-preview should be a button wrapping the whole row
		const toggleButton = screen.getByTestId("toggle-preview");
		expect(toggleButton.tagName).toBe("BUTTON");

		// Click it to expand
		fireEvent.click(toggleButton);
		expect(screen.getByTestId("content-preview")).toBeInTheDocument();

		// Click again to collapse
		fireEvent.click(toggleButton);
		expect(screen.queryByTestId("content-preview")).not.toBeInTheDocument();
	});
});
