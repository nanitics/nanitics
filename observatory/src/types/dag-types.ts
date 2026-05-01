export interface DAGNode {
	id: string;
	label: string;
	status: "pending" | "running" | "completed" | "error" | "skipped";
	/** Step type from the workflow definition */
	stepType: string;
	/** Agent type (react, codeact, etc.) if this step runs an agent */
	agentType?: string;
	/** Duration in milliseconds, null if not yet completed */
	durationMs: number | null;
	/** Span ID for agent click-through navigation */
	agentSpanId: string | null;
	/** Parallel group name for visual grouping */
	parallelGroup: string | null;
	/** Step output preview text */
	outputPreview?: string;
	/** Step metadata from workflow definition */
	metadata: Record<string, unknown>;
	/** Populated by dagre layout */
	x?: number;
	y?: number;
	width?: number;
	height?: number;
}

export interface DAGEdge {
	source: string;
	target: string;
	/** Whether this edge is on the critical path */
	isCriticalPath?: boolean;
}

export interface DAGLayout {
	nodes: DAGNode[];
	edges: DAGEdge[];
	/** Overall graph dimensions after layout */
	width: number;
	height: number;
}
