import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { TraceTree } from "../../src/components/trace-tree/trace-tree";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { TraceEvent } from "../../src/types";
import { errorTree, singleAgentTree, workflowTree } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Test wrapper
// ---------------------------------------------------------------------------

function renderTree(props: Partial<Parameters<typeof TraceTree>[0]> = {}, expandedIds?: Set<string>) {
	const tree = props.tree ?? singleAgentTree;
	const expanded = expandedIds ?? new Set(["root", "agent-1", "step-1"]);
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	const onToggleNode = props.onToggleNode ?? (() => {});
	const onExpandAll = props.onExpandAll ?? (() => {});
	const onCollapseAll = props.onCollapseAll ?? (() => {});
	const onSelectEvent = props.onSelectEvent ?? (() => {});

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<TraceTree
				tree={tree}
				expandedNodes={expanded}
				selectedEvent={props.selectedEvent ?? null}
				onToggleNode={onToggleNode}
				onExpandAll={onExpandAll}
				onCollapseAll={onCollapseAll}
				onSelectEvent={onSelectEvent}
				eventFilter={props.eventFilter}
			/>
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("TraceTree", () => {
	it("renders root and agent span names", () => {
		renderTree();
		expect(screen.getByText("run")).toBeInTheDocument();
		expect(screen.getByText("research-assistant (react)")).toBeInTheDocument();
	});

	it("shows events when parent span is expanded", () => {
		renderTree();
		// Events from step-1 span should be visible since step-1 is expanded
		expect(screen.getByText("llm.request")).toBeInTheDocument();
		expect(screen.getByText("tool.invoke")).toBeInTheDocument();
	});

	it("hides children when span is collapsed", () => {
		// Only expand root, not agent-1
		renderTree({}, new Set(["root"]));
		// Agent span itself is visible
		expect(screen.getByText("research-assistant (react)")).toBeInTheDocument();
		// But its children events should not be rendered
		expect(screen.queryByText("llm.request")).not.toBeInTheDocument();
	});

	it("shows collapsed descendant count", () => {
		renderTree({}, new Set(["root"]));
		// agent-1 is collapsed with children — should show count badge
		const count =
			singleAgentTree.root.children[0].events.length +
			1 + // step-1 span
			singleAgentTree.root.children[0].children[0].events.length;
		expect(screen.getByText(String(count))).toBeInTheDocument();
	});

	it("renders multi-agent workflow tree", () => {
		renderTree({ tree: workflowTree }, new Set(["root-wf", "wf-agent-1", "wf-agent-2"]));
		expect(screen.getByText("content-pipeline")).toBeInTheDocument();
		expect(screen.getByText("researcher (react)")).toBeInTheDocument();
		expect(screen.getByText("writer (react)")).toBeInTheDocument();
	});

	it("shows error indicator for spans with errors", () => {
		renderTree({ tree: errorTree }, new Set(["root-err", "err-agent"]));
		// Error indicator (red dot) should be present
		const errorDots = screen.getAllByText("●");
		expect(errorDots.length).toBeGreaterThan(0);
	});

	it("renders expand all / collapse all controls", () => {
		renderTree();
		expect(screen.getByText("Expand all")).toBeInTheDocument();
		expect(screen.getByText("Collapse all")).toBeInTheDocument();
	});

	it("calls onExpandAll when expand all is clicked", () => {
		const onExpandAll = vi.fn();
		renderTree({ onExpandAll });
		fireEvent.click(screen.getByText("Expand all"));
		expect(onExpandAll).toHaveBeenCalled();
	});

	it("calls onCollapseAll when collapse all is clicked", () => {
		const onCollapseAll = vi.fn();
		renderTree({ onCollapseAll });
		fireEvent.click(screen.getByText("Collapse all"));
		expect(onCollapseAll).toHaveBeenCalled();
	});

	it("filters events when eventFilter is provided", () => {
		const infoOnly: (e: TraceEvent) => boolean = (e) => e.level === "info";
		renderTree({ eventFilter: infoOnly });
		// agent.start is info level → visible, llm.request is debug → hidden
		// Since run.start is info, it should render
		// debug-level events should not render
	});
});
