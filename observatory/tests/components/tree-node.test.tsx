import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { TreeNode } from "../../src/components/trace-tree/tree-node";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeNode(overrides: Partial<SpanTreeNode> = {}): SpanTreeNode {
	return {
		span_id: "span-1",
		parent_span_id: null,
		name: "test-span",
		summary: {
			event_count: 2,
			duration_ms: 1500,
			has_errors: false,
			agent_name: null,
			agent_type: null,
		},
		events: [
			makeEvent({ id: 901, event_type: "llm.request", level: "debug" }),
			makeEvent({ id: 902, event_type: "tool.invoke", level: "debug" }),
		],
		children: [],
		...overrides,
	};
}

function renderNode(
	nodeOverrides: Partial<SpanTreeNode> = {},
	props: {
		isExpanded?: boolean;
		selectedEvent?: TraceEvent | null;
		maxDuration?: number;
	} = {},
) {
	const node = makeNode(nodeOverrides);
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<TreeNode
				node={node}
				depth={0}
				maxDuration={props.maxDuration ?? 3000}
				isExpanded={props.isExpanded ?? false}
				selectedEvent={props.selectedEvent ?? null}
				onToggle={vi.fn()}
				onSelectEvent={vi.fn()}
				renderChildren={() => null}
			/>
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("TreeNode", () => {
	it("displays span name as summary", () => {
		renderNode({ name: "my-span" });
		expect(screen.getByText("my-span")).toBeInTheDocument();
	});

	it("displays agent name and type when present", () => {
		renderNode({
			summary: {
				event_count: 1,
				duration_ms: 1000,
				has_errors: false,
				agent_name: "research-bot",
				agent_type: "react",
			},
		});
		expect(screen.getByText("research-bot (react)")).toBeInTheDocument();
	});

	it("shows duration bar when duration is set", () => {
		renderNode({
			events: [],
			summary: {
				event_count: 0,
				duration_ms: 1500,
				has_errors: false,
				agent_name: null,
				agent_type: null,
			},
		});
		expect(screen.getByText("1.5s")).toBeInTheDocument();
	});

	it("shows error indicator when has_errors is true", () => {
		renderNode({
			summary: {
				event_count: 1,
				duration_ms: 1000,
				has_errors: true,
				agent_name: null,
				agent_type: null,
			},
		});
		expect(screen.getByText("●")).toBeInTheDocument();
	});

	it("does not show error indicator when has_errors is false", () => {
		renderNode({ events: [] });
		expect(screen.queryByText("●")).not.toBeInTheDocument();
	});

	it("shows expand toggle when node has children or events", () => {
		renderNode({ events: [makeEvent({ id: 910, event_type: "agent.start" })] });
		expect(screen.getByText("▸")).toBeInTheDocument();
	});

	it("shows expanded toggle when isExpanded", () => {
		renderNode({}, { isExpanded: true });
		expect(screen.getByText("▾")).toBeInTheDocument();
	});

	it("shows events when expanded", () => {
		renderNode({}, { isExpanded: true });
		// Events should have their type displayed via registry summary (falls back to event_type)
		expect(screen.getByText("llm.request")).toBeInTheDocument();
		expect(screen.getByText("tool.invoke")).toBeInTheDocument();
	});

	it("hides events when collapsed", () => {
		renderNode({}, { isExpanded: false });
		expect(screen.queryByText("llm.request")).not.toBeInTheDocument();
		expect(screen.queryByText("tool.invoke")).not.toBeInTheDocument();
	});

	it("shows collapsed descendant count when collapsed", () => {
		renderNode(
			{ events: [makeEvent({ id: 920 }), makeEvent({ id: 921 }), makeEvent({ id: 922 })] },
			{ isExpanded: false },
		);
		expect(screen.getByText("3")).toBeInTheDocument();
	});

	it("calls onToggle when span row is clicked", () => {
		const onToggle = vi.fn();
		const node = makeNode({ name: "unique-toggle-span", events: [makeEvent({ id: 930 })] });
		const client = new ObservatoryClient("/test");
		const registry = new EventRendererRegistry();

		render(
			<ObservatoryProvider client={client} registry={registry}>
				<TreeNode
					node={node}
					depth={0}
					maxDuration={3000}
					isExpanded={false}
					selectedEvent={null}
					onToggle={onToggle}
					onSelectEvent={vi.fn()}
					renderChildren={() => null}
				/>
			</ObservatoryProvider>,
		);

		fireEvent.click(screen.getByText("unique-toggle-span"));
		expect(onToggle).toHaveBeenCalledWith("span-1");
	});

	it("calls onSelectEvent when an event row is clicked", () => {
		const onSelectEvent = vi.fn();
		const event = makeEvent({ event_type: "tool.invoke" });
		const node = makeNode({ events: [event] });
		const client = new ObservatoryClient("/test");
		const registry = new EventRendererRegistry();

		render(
			<ObservatoryProvider client={client} registry={registry}>
				<TreeNode
					node={node}
					depth={0}
					maxDuration={3000}
					isExpanded={true}
					selectedEvent={null}
					onToggle={vi.fn()}
					onSelectEvent={onSelectEvent}
					renderChildren={() => null}
				/>
			</ObservatoryProvider>,
		);

		fireEvent.click(screen.getByText("tool.invoke"));
		expect(onSelectEvent).toHaveBeenCalledWith(event);
	});

	it("uses registry summary for event display text", () => {
		const client = new ObservatoryClient("/test");
		const registry = new EventRendererRegistry();
		registry.register({
			matches: (t) => t === "llm.response",
			priority: 0,
			component: () => null,
			summary: () => "150+100 tokens",
		});

		const event = makeEvent({ event_type: "llm.response" });
		const node = makeNode({ events: [event] });

		render(
			<ObservatoryProvider client={client} registry={registry}>
				<TreeNode
					node={node}
					depth={0}
					maxDuration={3000}
					isExpanded={true}
					selectedEvent={null}
					onToggle={vi.fn()}
					onSelectEvent={vi.fn()}
					renderChildren={() => null}
				/>
			</ObservatoryProvider>,
		);

		expect(screen.getByText("150+100 tokens")).toBeInTheDocument();
	});

	it("shows navigate icon on agent spans when onNavigateToAgent is provided", () => {
		const onNavigateToAgent = vi.fn();
		const node = makeNode({
			summary: {
				event_count: 1,
				duration_ms: 1000,
				has_errors: false,
				agent_name: "my-agent",
				agent_type: "react",
			},
		});
		const client = new ObservatoryClient("/test");
		const registry = new EventRendererRegistry();

		render(
			<ObservatoryProvider client={client} registry={registry}>
				<TreeNode
					node={node}
					depth={0}
					maxDuration={3000}
					isExpanded={false}
					selectedEvent={null}
					onToggle={vi.fn()}
					onSelectEvent={vi.fn()}
					onNavigateToAgent={onNavigateToAgent}
					renderChildren={() => null}
				/>
			</ObservatoryProvider>,
		);

		const navButton = screen.getByTitle("View agent details");
		expect(navButton).toBeInTheDocument();
		fireEvent.click(navButton);
		expect(onNavigateToAgent).toHaveBeenCalledWith("span-1");
	});

	it("does not show navigate icon on non-agent spans", () => {
		const onNavigateToAgent = vi.fn();
		const node = makeNode({
			summary: {
				event_count: 1,
				duration_ms: 1000,
				has_errors: false,
				agent_name: null,
				agent_type: null,
			},
		});
		const client = new ObservatoryClient("/test");
		const registry = new EventRendererRegistry();

		render(
			<ObservatoryProvider client={client} registry={registry}>
				<TreeNode
					node={node}
					depth={0}
					maxDuration={3000}
					isExpanded={false}
					selectedEvent={null}
					onToggle={vi.fn()}
					onSelectEvent={vi.fn()}
					onNavigateToAgent={onNavigateToAgent}
					renderChildren={() => null}
				/>
			</ObservatoryProvider>,
		);

		expect(screen.queryByTitle("View agent details")).not.toBeInTheDocument();
	});

	it("does not show navigate icon when onNavigateToAgent is not provided", () => {
		const node = makeNode({
			summary: {
				event_count: 1,
				duration_ms: 1000,
				has_errors: false,
				agent_name: "my-agent",
				agent_type: "react",
			},
		});
		const client = new ObservatoryClient("/test");
		const registry = new EventRendererRegistry();

		render(
			<ObservatoryProvider client={client} registry={registry}>
				<TreeNode
					node={node}
					depth={0}
					maxDuration={3000}
					isExpanded={false}
					selectedEvent={null}
					onToggle={vi.fn()}
					onSelectEvent={vi.fn()}
					renderChildren={() => null}
				/>
			</ObservatoryProvider>,
		);

		expect(screen.queryByTitle("View agent details")).not.toBeInTheDocument();
	});
});
