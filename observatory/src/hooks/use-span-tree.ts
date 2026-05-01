import { useCallback, useEffect, useMemo, useState } from "react";
import { useObservatory } from "../context/observatory-context";
import type { SpanTreeNode, SpanTreeResponse, TraceEvent, TraceLevel } from "../types";

interface UseSpanTreeOptions {
	minLevel?: TraceLevel;
}

interface UseSpanTreeResult {
	tree: SpanTreeResponse | null;
	expandedNodes: Set<string>;
	toggleNode: (spanId: string) => void;
	expandAll: () => void;
	collapseAll: () => void;
	selectedEvent: TraceEvent | null;
	selectEvent: (event: TraceEvent | null) => void;
	isLoading: boolean;
	error: string | null;
	/** Insert a streamed event into the tree. */
	addStreamedEvent: (event: TraceEvent) => void;
}

/** Collect all span_ids from a tree up to a given depth. */
function collectSpanIds(node: SpanTreeNode, depth: number): string[] {
	if (depth <= 0) return [];
	const ids = [node.span_id];
	for (const child of node.children) {
		ids.push(...collectSpanIds(child, depth - 1));
	}
	return ids;
}

/** Collect every span_id in the tree. */
function collectAllSpanIds(node: SpanTreeNode): string[] {
	const ids = [node.span_id];
	for (const child of node.children) {
		ids.push(...collectAllSpanIds(child));
	}
	return ids;
}

export function useSpanTree(runId: string, options?: UseSpanTreeOptions): UseSpanTreeResult {
	const { client } = useObservatory();
	const minLevel = options?.minLevel;
	const [tree, setTree] = useState<SpanTreeResponse | null>(null);
	const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
	const [selectedEvent, setSelectedEvent] = useState<TraceEvent | null>(null);
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		setIsLoading(true);
		setError(null);

		client
			.getSpanTree(runId, { minLevel })
			.then((data) => {
				if (cancelled) return;
				setTree(data);
				// Expand to depth 2 by default
				const initial = collectSpanIds(data.root, 2);
				setExpandedNodes(new Set(initial));
			})
			.catch((err) => {
				if (cancelled) return;
				setError(String(err));
			})
			.finally(() => {
				if (!cancelled) setIsLoading(false);
			});

		return () => {
			cancelled = true;
		};
	}, [client, runId, minLevel]);

	const toggleNode = useCallback((spanId: string) => {
		setExpandedNodes((prev) => {
			const next = new Set(prev);
			if (next.has(spanId)) {
				next.delete(spanId);
			} else {
				next.add(spanId);
			}
			return next;
		});
	}, []);

	const expandAll = useCallback(() => {
		if (!tree) return;
		setExpandedNodes(new Set(collectAllSpanIds(tree.root)));
	}, [tree]);

	const collapseAll = useCallback(() => {
		setExpandedNodes(new Set());
	}, []);

	const selectEvent = useCallback((event: TraceEvent | null) => {
		setSelectedEvent(event);
	}, []);

	/** Find a span node by span_id in the tree recursively. */
	const findNode = useCallback((node: SpanTreeNode, spanId: string): SpanTreeNode | null => {
		if (node.span_id === spanId) return node;
		for (const child of node.children) {
			const found = findNode(child, spanId);
			if (found) return found;
		}
		return null;
	}, []);

	/** Deep-clone a tree, inserting the event into the correct span node. */
	const insertEvent = useCallback((root: SpanTreeNode, event: TraceEvent): SpanTreeNode => {
		// Check if this event's span_id matches the current node
		if (root.span_id === event.span_id) {
			// Check for duplicate event ids
			if (root.events.some((e) => e.id === event.id)) return root;
			return {
				...root,
				events: [...root.events, event],
				summary: {
					...root.summary,
					event_count: root.summary.event_count + 1,
					has_errors: root.summary.has_errors || (event.level === "info" && event.event_type.includes("error")),
				},
			};
		}
		// Recurse into children
		const updatedChildren = root.children.map((child) => insertEvent(child, event));
		// If the event's parent_span_id matches this node but no child has the
		// event's span_id, create a new child span node for it.
		if (
			event.parent_span_id === root.span_id &&
			!root.children.some((c) => c.span_id === event.span_id) &&
			!updatedChildren.some((c) => c.span_id === event.span_id)
		) {
			const newChild: SpanTreeNode = {
				span_id: event.span_id,
				parent_span_id: event.parent_span_id,
				name: event.event_type,
				summary: {
					event_count: 1,
					duration_ms: null,
					has_errors: false,
					agent_name: null,
					agent_type: null,
				},
				events: [event],
				children: [],
			};
			return { ...root, children: [...updatedChildren, newChild] };
		}
		// Only return a new object if children actually changed
		if (updatedChildren === root.children) return root;
		return { ...root, children: updatedChildren };
	}, []);

	const addStreamedEvent = useCallback(
		(event: TraceEvent) => {
			setTree((prev) => {
				if (!prev) return prev;
				const updatedRoot = insertEvent(prev.root, event);
				if (updatedRoot === prev.root) return prev;
				return { ...prev, root: updatedRoot };
			});
		},
		[insertEvent],
	);

	return useMemo(
		() => ({
			tree,
			expandedNodes,
			toggleNode,
			expandAll,
			collapseAll,
			selectedEvent,
			selectEvent,
			isLoading,
			error,
			addStreamedEvent,
		}),
		[
			tree,
			expandedNodes,
			toggleNode,
			expandAll,
			collapseAll,
			selectedEvent,
			selectEvent,
			isLoading,
			error,
			addStreamedEvent,
		],
	);
}
