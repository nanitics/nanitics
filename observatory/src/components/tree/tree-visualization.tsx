import { tree as d3Tree, hierarchy } from "d3-hierarchy";
import { useCallback, useMemo } from "react";
import type { ContentBounds } from "../../hooks/use-svg-viewport";
import { useSVGViewport } from "../../hooks/use-svg-viewport";
import type { TreeVisualizationConfig, VisualTreeNode } from "../../types/tree-types";

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

interface LayoutNode {
	data: VisualTreeNode;
	x: number;
	y: number;
	parent: LayoutNode | null;
	children?: LayoutNode[];
}

function computeLayout(root: VisualTreeNode, nodeWidth: number, nodeHeight: number): LayoutNode[] {
	const h = hierarchy(root, (d) => d.children);

	const treeLayout = d3Tree<VisualTreeNode>().nodeSize([nodeWidth + 24, nodeHeight + 48]);

	const laid = treeLayout(h);
	return laid.descendants() as unknown as LayoutNode[];
}

function curvedPath(parent: { x: number; y: number }, child: { x: number; y: number }): string {
	const midY = (parent.y + child.y) / 2;
	return `M ${parent.x} ${parent.y} C ${parent.x} ${midY}, ${child.x} ${midY}, ${child.x} ${child.y}`;
}

// ---------------------------------------------------------------------------
// TreeVisualization
// ---------------------------------------------------------------------------

interface TreeVisualizationProps {
	root: VisualTreeNode;
	config: TreeVisualizationConfig;
}

export function TreeVisualization({ root, config }: TreeVisualizationProps) {
	const {
		renderNodeContent,
		nodeWidth,
		nodeHeight,
		highlightedNodeIds,
		highlightStyle = "path",
		dimmedNodeIds,
		selectedNodeId,
		onNodeSelect,
	} = config;

	// Layout computation
	const layoutNodes = useMemo(() => computeLayout(root, nodeWidth, nodeHeight), [root, nodeWidth, nodeHeight]);

	// Edges derived from layout
	const edges = useMemo(() => {
		const result: { key: string; path: string; sourceId: string; targetId: string }[] = [];
		for (const node of layoutNodes) {
			if (node.parent) {
				result.push({
					key: `${node.parent.data.id}-${node.data.id}`,
					path: curvedPath(node.parent, node),
					sourceId: node.parent.data.id,
					targetId: node.data.id,
				});
			}
		}
		return result;
	}, [layoutNodes]);

	// Compute content bounds from layout nodes
	const contentBounds = useMemo<ContentBounds | null>(() => {
		if (layoutNodes.length === 0) return null;
		let minX = Infinity,
			maxX = -Infinity,
			minY = Infinity,
			maxY = -Infinity;
		for (const n of layoutNodes) {
			minX = Math.min(minX, n.x - nodeWidth / 2);
			maxX = Math.max(maxX, n.x + nodeWidth / 2);
			minY = Math.min(minY, n.y - nodeHeight / 2);
			maxY = Math.max(maxY, n.y + nodeHeight / 2);
		}
		return { minX, minY, maxX, maxY };
	}, [layoutNodes, nodeWidth, nodeHeight]);

	// Shared viewport hook for zoom/pan
	const { svgRef, viewBox, isDragging, zoomIn, zoomOut, fitToContent, svgHandlers } = useSVGViewport(contentBounds);

	// Node click
	const handleNodeClick = useCallback(
		(nodeId: string) => {
			onNodeSelect?.(nodeId);
		},
		[onNodeSelect],
	);

	// Edge styling
	function edgeClass(sourceId: string, targetId: string): string {
		const bothHighlighted = highlightedNodeIds?.has(sourceId) && highlightedNodeIds?.has(targetId);
		const eitherDimmed = dimmedNodeIds?.has(sourceId) || dimmedNodeIds?.has(targetId);

		if (bothHighlighted) return "stroke-primary stroke-2";
		if (eitherDimmed) return "stroke-muted-foreground/20 stroke-1";
		return "stroke-muted-foreground/40 stroke-[1.5]";
	}

	// Node container styling
	function nodeContainerClass(nodeId: string): string {
		const isHighlighted = highlightedNodeIds?.has(nodeId);
		const isDimmed = dimmedNodeIds?.has(nodeId);

		if (isDimmed) return "opacity-40";
		if (isHighlighted && highlightStyle === "glow") return "drop-shadow-[0_0_6px_var(--color-primary)]";
		if (isHighlighted && highlightStyle === "outline") return "[&>*]:ring-2 [&>*]:ring-primary";
		return "";
	}

	if (layoutNodes.length === 0) {
		return (
			<div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
				No tree data to display.
			</div>
		);
	}

	return (
		<div className="relative w-full h-full min-h-[400px] overflow-hidden select-none" data-testid="tree-visualization">
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
				{/* Edges */}
				<g data-testid="tree-edges">
					{edges.map((edge) => (
						<path
							key={edge.key}
							d={edge.path}
							fill="none"
							className={edgeClass(edge.sourceId, edge.targetId)}
							data-testid={`edge-${edge.sourceId}-${edge.targetId}`}
						/>
					))}
				</g>

				{/* Nodes */}
				<g data-testid="tree-nodes">
					{layoutNodes.map((layoutNode) => {
						const node = layoutNode.data;
						const isSelected = selectedNodeId === node.id;

						return (
							<g
								key={node.id}
								transform={`translate(${layoutNode.x - nodeWidth / 2}, ${layoutNode.y - nodeHeight / 2})`}
								data-tree-node
								data-testid={`node-${node.id}`}
								onClick={(e) => {
									e.stopPropagation();
									handleNodeClick(node.id);
								}}
								className={`cursor-pointer ${nodeContainerClass(node.id)}`}
							>
								<foreignObject width={nodeWidth} height={nodeHeight} overflow="visible">
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
