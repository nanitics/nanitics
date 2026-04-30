import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DelegationPanel } from "../../src/components/patterns/delegation-panel";
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

describe("DelegationPanel", () => {
	it("renders agent names", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: {
					caller_agent: "Alice",
					delegate_agent: "Bob",
					task: "research topic",
					transfer_strategy: "full",
				},
			}),
		];
		render(<DelegationPanel events={events} agents={makeAgents("Alice", "Bob")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("Alice")).toBeInTheDocument();
		expect(screen.getByText("Bob")).toBeInTheDocument();
	});

	it("renders task text", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: {
					caller_agent: "Alice",
					delegate_agent: "Bob",
					task: "analyze the quarterly report",
					transfer_strategy: "full",
				},
			}),
		];
		render(<DelegationPanel events={events} agents={makeAgents("Alice", "Bob")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("analyze the quarterly report")).toBeInTheDocument();
	});

	it("renders strategy badge", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: {
					caller_agent: "Alice",
					delegate_agent: "Bob",
					task: "task",
					transfer_strategy: "FullTransfer",
				},
			}),
		];
		render(<DelegationPanel events={events} agents={makeAgents("Alice", "Bob")} onNavigateToAgent={vi.fn()} />);
		expect(screen.getByText("FullTransfer")).toBeInTheDocument();
	});

	it("triggers navigation on agent click", () => {
		const onNavigate = vi.fn();
		const events = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: {
					caller_agent: "Alice",
					delegate_agent: "Bob",
					task: "task",
					transfer_strategy: "full",
				},
			}),
		];
		render(<DelegationPanel events={events} agents={makeAgents("Alice", "Bob")} onNavigateToAgent={onNavigate} />);
		fireEvent.click(screen.getByText("Bob"));
		expect(onNavigate).toHaveBeenCalledWith("span-Bob");
	});

	it("renders multiple delegation instances separately", () => {
		const events = [
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: {
					caller_agent: "Alice",
					delegate_agent: "Bob",
					task: "task-1",
					transfer_strategy: "full",
				},
			}),
			makeEvent({
				event_type: "multi_agent.delegation",
				payload: {
					caller_agent: "Alice",
					delegate_agent: "Carol",
					task: "task-2",
					transfer_strategy: "partial",
				},
			}),
		];
		render(
			<DelegationPanel events={events} agents={makeAgents("Alice", "Bob", "Carol")} onNavigateToAgent={vi.fn()} />,
		);
		expect(screen.getByText("Bob")).toBeInTheDocument();
		expect(screen.getByText("Carol")).toBeInTheDocument();
		expect(screen.getByText("task-1")).toBeInTheDocument();
		expect(screen.getByText("task-2")).toBeInTheDocument();
	});
});
