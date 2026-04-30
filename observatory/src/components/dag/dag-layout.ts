import dagre from "@dagrejs/dagre";
import type { WorkflowStep } from "../../types";
import type { DAGEdge, DAGLayout, DAGNode } from "../../types/dag-types";

export interface DAGLayoutOptions {
	/** Node width in pixels (default: 220) */
	nodeWidth?: number;
	/** Node height in pixels (default: 80) */
	nodeHeight?: number;
	/** Horizontal spacing between nodes (default: 40) */
	rankSep?: number;
	/** Vertical spacing between nodes (default: 60) */
	nodeSep?: number;
}

/**
 * Compute a DAG layout from workflow steps using dagre.
 *
 * Returns positioned nodes, edges, and overall graph dimensions.
 */
export function computeDAGLayout(steps: WorkflowStep[], options?: DAGLayoutOptions): DAGLayout {
	const { nodeWidth = 220, nodeHeight = 80, rankSep = 60, nodeSep = 40 } = options ?? {};

	if (steps.length === 0) {
		return { nodes: [], edges: [], width: 0, height: 0 };
	}

	// Build dagre graph
	const g = new dagre.graphlib.Graph();
	g.setGraph({ rankdir: "TB", ranksep: rankSep, nodesep: nodeSep });
	g.setDefaultEdgeLabel(() => ({}));

	// Build nodes
	const dagNodes: DAGNode[] = steps.map((step) => ({
		id: step.name,
		label: step.name,
		status: step.status as DAGNode["status"],
		stepType: step.step_type,
		durationMs: step.duration_ms,
		agentSpanId: step.agent_span_id,
		parallelGroup: step.parallel_group,
		metadata: step.metadata,
	}));

	for (const node of dagNodes) {
		g.setNode(node.id, { width: nodeWidth, height: nodeHeight });
	}

	// Build edges from depends_on
	const edges: DAGEdge[] = [];
	for (const step of steps) {
		for (const dep of step.depends_on) {
			edges.push({ source: dep, target: step.name });
			g.setEdge(dep, step.name);
		}
	}

	// Run dagre layout
	dagre.layout(g);

	// Extract positions
	for (const node of dagNodes) {
		const dagreNode = g.node(node.id);
		if (dagreNode) {
			node.x = dagreNode.x;
			node.y = dagreNode.y;
			node.width = nodeWidth;
			node.height = nodeHeight;
		}
	}

	// Compute critical path
	const criticalEdges = computeCriticalPath(dagNodes, edges);
	for (const edge of edges) {
		const key = `${edge.source}->${edge.target}`;
		edge.isCriticalPath = criticalEdges.has(key);
	}

	// Compute overall dimensions
	const graphInfo = g.graph();
	const width = graphInfo.width ?? 0;
	const height = graphInfo.height ?? 0;

	return { nodes: dagNodes, edges, width, height };
}

/**
 * Compute the critical path through a DAG — the longest weighted path.
 *
 * Uses forward pass (longest path to each node) weighted by duration_ms.
 * Returns edge keys in "source->target" format.
 */
function computeCriticalPath(nodes: DAGNode[], edges: DAGEdge[]): Set<string> {
	if (nodes.length === 0) return new Set();

	const nodeMap = new Map(nodes.map((n) => [n.id, n]));
	const inEdges = new Map<string, DAGEdge[]>();
	const outEdges = new Map<string, DAGEdge[]>();

	for (const edge of edges) {
		const ins = inEdges.get(edge.target) ?? [];
		ins.push(edge);
		inEdges.set(edge.target, ins);

		const outs = outEdges.get(edge.source) ?? [];
		outs.push(edge);
		outEdges.set(edge.source, outs);
	}

	// Topological sort (Kahn's algorithm)
	const inDegree = new Map<string, number>();
	for (const node of nodes) inDegree.set(node.id, 0);
	for (const edge of edges) {
		inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
	}

	const queue: string[] = [];
	for (const [id, deg] of inDegree) {
		if (deg === 0) queue.push(id);
	}

	const topoOrder: string[] = [];
	while (queue.length > 0) {
		const id = queue.shift()!;
		topoOrder.push(id);
		for (const edge of outEdges.get(id) ?? []) {
			const newDeg = (inDegree.get(edge.target) ?? 1) - 1;
			inDegree.set(edge.target, newDeg);
			if (newDeg === 0) queue.push(edge.target);
		}
	}

	// Forward pass: compute longest path to each node
	const dist = new Map<string, number>();
	const predecessor = new Map<string, string | null>();
	for (const id of topoOrder) {
		dist.set(id, 0);
		predecessor.set(id, null);
	}

	for (const id of topoOrder) {
		const currentDist = dist.get(id) ?? 0;
		const node = nodeMap.get(id);
		const nodeDuration = node?.durationMs ?? 0;
		const endDist = currentDist + nodeDuration;

		for (const edge of outEdges.get(id) ?? []) {
			const targetDist = dist.get(edge.target) ?? 0;
			if (endDist >= targetDist) {
				dist.set(edge.target, endDist);
				predecessor.set(edge.target, id);
			}
		}
	}

	// Find the terminal node with the longest path
	let maxDist = -1;
	let terminalId: string | null = null;
	for (const id of topoOrder) {
		const d = (dist.get(id) ?? 0) + (nodeMap.get(id)?.durationMs ?? 0);
		if (d >= maxDist) {
			maxDist = d;
			terminalId = id;
		}
	}

	// Trace back predecessor chain to build critical path edges
	const criticalEdges = new Set<string>();
	let current = terminalId;
	while (current !== null) {
		const pred = predecessor.get(current) ?? null;
		if (pred !== null) {
			criticalEdges.add(`${pred}->${current}`);
		}
		current = pred;
	}

	return criticalEdges;
}
