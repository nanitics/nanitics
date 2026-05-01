import { useCallback, useMemo } from "react";
import type { ContentBounds } from "../../hooks/use-svg-viewport";
import { useSVGViewport } from "../../hooks/use-svg-viewport";
import type { DAGEdge, DAGLayout, DAGNode } from "../../types/dag-types";

interface DAGVisualizationProps {
	layout: DAGLayout;
	renderNodeContent: (node: DAGNode, isSelected: boolean) => React.ReactNode;
	nodeWidth: number;
	nodeHeight: number;
	selectedNodeId?: string;
	onNodeSelect?: (nodeId: string) => void;
	showCriticalPath?: boolean;
}

/** Compute edge path from source bottom-center to target top-center. */
function edgePath(source: DAGNode, target: DAGNode, nodeHeight: number): string {
	const sx = source.x ?? 0;
	const sy = (source.y ?? 0) + nodeHeight / 2;
	const tx = target.x ?? 0;
	const ty = (target.y ?? 0) - nodeHeight / 2;
	const midY = (sy + ty) / 2;
	return `M ${sx} ${sy} C ${sx} ${midY}, ${tx} ${midY}, ${tx} ${ty}`;
}

/** Find all nodes transitively reachable from error nodes. */
function findBlockedNodes(layout: DAGLayout): Set<string> {
	const errorNodeIds = new Set(layout.nodes.filter((n) => n.status === "error").map((n) => n.id));
	if (errorNodeIds.size === 0) return new Set();

	const outEdges = new Map<string, string[]>();
	for (const edge of layout.edges) {
		const targets = outEdges.get(edge.source) ?? [];
		targets.push(edge.target);
		outEdges.set(edge.source, targets);
	}

	const blocked = new Set<string>();
	const queue = [...errorNodeIds];
	while (queue.length > 0) {
		const id = queue.shift()!;
		for (const target of outEdges.get(id) ?? []) {
			if (!blocked.has(target) && !errorNodeIds.has(target)) {
				blocked.add(target);
				queue.push(target);
			}
		}
	}
	return blocked;
}

export function DAGVisualization({
	layout,
	renderNodeContent,
	nodeWidth,
	nodeHeight,
	selectedNodeId,
	onNodeSelect,
	showCriticalPath = false,
}: DAGVisualizationProps) {
	const nodeMap = useMemo(() => new Map(layout.nodes.map((n) => [n.id, n])), [layout.nodes]);

	const blockedNodes = useMemo(() => findBlockedNodes(layout), [layout]);

	// Content bounds from layout positions
	const contentBounds = useMemo<ContentBounds | null>(() => {
		if (layout.nodes.length === 0) return null;
		let minX = Infinity,
			maxX = -Infinity,
			minY = Infinity,
			maxY = -Infinity;
		for (const n of layout.nodes) {
			const x = n.x ?? 0;
			const y = n.y ?? 0;
			minX = Math.min(minX, x - nodeWidth / 2);
			maxX = Math.max(maxX, x + nodeWidth / 2);
			minY = Math.min(minY, y - nodeHeight / 2);
			maxY = Math.max(maxY, y + nodeHeight / 2);
		}
		return { minX, minY, maxX, maxY };
	}, [layout.nodes, nodeWidth, nodeHeight]);

	const { svgRef, viewBox, isDragging, zoomIn, zoomOut, fitToContent, svgHandlers } = useSVGViewport(contentBounds);

	const handleNodeClick = useCallback(
		(nodeId: string) => {
			onNodeSelect?.(nodeId);
		},
		[onNodeSelect],
	);

	// Edge styling
	function edgeClass(edge: DAGEdge): string {
		const sourceNode = nodeMap.get(edge.source);
		const isSourceError = sourceNode?.status === "error";
		const isTargetBlocked = blockedNodes.has(edge.target);

		if (isSourceError || isTargetBlocked) {
			return "stroke-muted-foreground/30 stroke-1";
		}
		if (showCriticalPath && edge.isCriticalPath) {
			return "stroke-primary stroke-[3]";
		}
		return "stroke-muted-foreground/40 stroke-[1.5]";
	}

	function edgeDashArray(edge: DAGEdge): string | undefined {
		const sourceNode = nodeMap.get(edge.source);
		if (sourceNode?.status === "error" || blockedNodes.has(edge.target)) {
			return "6 4";
		}
		return undefined;
	}

	if (layout.nodes.length === 0) {
		return (
			<div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
				No workflow data to display.
			</div>
		);
	}

	return (
		<div className="relative w-full h-full min-h-[400px] overflow-hidden select-none" data-testid="dag-visualization">
			{/* Zoom controls */}
			<div className="absolute top-2 right-2 z-10 flex gap-1">
				<button
					type="button"
					onClick={zoomIn}
					className="p-1 rounded bg-background border border-border text-muted-foreground hover:text-foreground text-xs"
					aria-label="Zoom in"
				>
					+
				</button>
				<button
					type="button"
					onClick={zoomOut}
					className="p-1 rounded bg-background border border-border text-muted-foreground hover:text-foreground text-xs"
					aria-label="Zoom out"
				>
					−
				</button>
				<button
					type="button"
					onClick={fitToContent}
					className="p-1 rounded bg-background border border-border text-muted-foreground hover:text-foreground text-xs"
					aria-label="Fit to content"
				>
					⊞
				</button>
			</div>

			<svg
				ref={svgRef}
				viewBox={viewBox}
				className={`w-full h-full ${isDragging ? "cursor-grabbing" : "cursor-grab"}`}
				{...svgHandlers}
			>
				{/* Arrowhead marker definition */}
				<defs>
					<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
						<polygon points="0 0, 8 3, 0 6" className="fill-muted-foreground/40" />
					</marker>
					<marker id="arrowhead-critical" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
						<polygon points="0 0, 8 3, 0 6" className="fill-primary" />
					</marker>
					<marker id="arrowhead-blocked" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
						<polygon points="0 0, 8 3, 0 6" className="fill-muted-foreground/30" />
					</marker>
				</defs>

				{/* Edges */}
				<g data-testid="dag-edges">
					{layout.edges.map((edge) => {
						const source = nodeMap.get(edge.source);
						const target = nodeMap.get(edge.target);
						if (!source || !target) return null;

						const isBlocked = source.status === "error" || blockedNodes.has(edge.target);
						const isCritical = showCriticalPath && edge.isCriticalPath;
						const markerId = isBlocked
							? "url(#arrowhead-blocked)"
							: isCritical
								? "url(#arrowhead-critical)"
								: "url(#arrowhead)";

						return (
							<path
								key={`${edge.source}-${edge.target}`}
								d={edgePath(source, target, nodeHeight)}
								fill="none"
								className={edgeClass(edge)}
								strokeDasharray={edgeDashArray(edge)}
								markerEnd={markerId}
								data-testid={`edge-${edge.source}-${edge.target}`}
							/>
						);
					})}
				</g>

				{/* Nodes */}
				<g data-testid="dag-nodes">
					{layout.nodes.map((node) => {
						const x = (node.x ?? 0) - nodeWidth / 2;
						const y = (node.y ?? 0) - nodeHeight / 2;
						const isSelected = selectedNodeId === node.id;
						const isBlocked = blockedNodes.has(node.id);

						return (
							<g
								key={node.id}
								transform={`translate(${x}, ${y})`}
								data-dag-node
								data-testid={`node-${node.id}`}
								onClick={(e) => {
									e.stopPropagation();
									handleNodeClick(node.id);
								}}
								className={`cursor-pointer ${isBlocked ? "opacity-40" : ""}`}
							>
								<foreignObject width={nodeWidth} height={nodeHeight}>
									<div className="w-full h-full">{renderNodeContent(node, isSelected)}</div>
								</foreignObject>
							</g>
						);
					})}
				</g>
			</svg>
		</div>
	);
}
