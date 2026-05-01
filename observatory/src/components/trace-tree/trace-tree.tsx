import type { RefObject } from "react";
import type { SpanTreeNode, SpanTreeResponse, TraceEvent } from "../../types";
import { TreeControls } from "./tree-controls";
import { TreeNode } from "./tree-node";

/** Find the maximum duration across all spans for proportional bar sizing. */
function findMaxDuration(node: SpanTreeNode): number {
	let max = node.summary.duration_ms ?? 0;
	for (const child of node.children) {
		max = Math.max(max, findMaxDuration(child));
	}
	return max;
}

interface TraceTreeProps {
	tree: SpanTreeResponse;
	expandedNodes: Set<string>;
	selectedEvent: TraceEvent | null;
	onToggleNode: (spanId: string) => void;
	onExpandAll: () => void;
	onCollapseAll: () => void;
	onSelectEvent: (event: TraceEvent) => void;
	/** Optional filter predicate for events — when provided, only matching events are shown. */
	eventFilter?: (event: TraceEvent) => boolean;
	/** Ref for the scrollable tree container (used for auto-scroll). */
	treeContainerRef?: RefObject<HTMLDivElement | null>;
	/** Scroll handler for the tree container (used to detect manual scrolling). */
	onTreeScroll?: () => void;
	/** Optional callback for navigating to an agent detail page. */
	onNavigateToAgent?: (spanId: string) => void;
}

export function TraceTree({
	tree,
	expandedNodes,
	selectedEvent,
	onToggleNode,
	onExpandAll,
	onCollapseAll,
	onSelectEvent,
	eventFilter,
	treeContainerRef,
	onTreeScroll,
	onNavigateToAgent,
}: TraceTreeProps) {
	const maxDuration = findMaxDuration(tree.root);

	function renderNode(node: SpanTreeNode, depth: number) {
		return (
			<TreeNode
				key={node.span_id}
				node={node}
				depth={depth}
				maxDuration={maxDuration}
				isExpanded={expandedNodes.has(node.span_id)}
				selectedEvent={selectedEvent}
				onToggle={onToggleNode}
				onSelectEvent={onSelectEvent}
				eventFilter={eventFilter}
				onNavigateToAgent={onNavigateToAgent}
				renderChildren={(parent, parentDepth) => parent.children.map((child) => renderNode(child, parentDepth + 1))}
			/>
		);
	}

	return (
		<div className="flex flex-col h-full">
			<TreeControls onExpandAll={onExpandAll} onCollapseAll={onCollapseAll} />
			<div className="flex-1 overflow-y-auto py-1" ref={treeContainerRef} onScroll={onTreeScroll}>
				{renderNode(tree.root, 0)}
			</div>
		</div>
	);
}
