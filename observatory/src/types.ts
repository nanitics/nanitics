/** Trace event severity level. */
export type TraceLevel = "info" | "debug" | "verbose";

/** Run execution status. */
export type RunStatus = "running" | "completed" | "failed" | "suspended";

/** Sort options for the run list. */
export type RunSortOption = "started_at_desc" | "started_at_asc" | "duration_desc" | "duration_asc";

/** A single trace event as returned by the API. */
export interface TraceEvent {
	id: number;
	event_type: string;
	level: TraceLevel;
	trace_id: string;
	span_id: string;
	parent_span_id: string | null;
	timestamp: string;
	payload: Record<string, unknown>;
}

/** Serialized run record. */
export interface RunResponse {
	id: string;
	trace_id: string;
	status: RunStatus;
	started_at: string;
	completed_at: string | null;
	metadata: Record<string, unknown>;
	error: string | null;
	result: string | null;
}

/** A run paired with its summary statistics. */
export interface RunListItem {
	run: RunResponse;
	summary: TraceSummaryResponse;
}

/** Paginated list of runs with inline summaries. */
export interface RunListResponse {
	runs: RunListItem[];
	total: number;
}

/** Aggregated statistics for trace events under a parent. */
export interface TraceSummaryResponse {
	total_events: number;
	events_by_level: Record<string, number>;
	llm_calls: number;
	tool_calls: number;
	total_input_tokens: number;
	total_output_tokens: number;
	total_duration_ms: number | null;
	agent_names: string[];
	errors: number;
	cache_creation_tokens: number;
	cache_read_tokens: number;
}

/** Run record with summary statistics. */
export interface RunDetailResponse {
	run: RunResponse;
	summary: TraceSummaryResponse;
}

/** Per-span aggregated statistics. */
export interface SpanSummary {
	event_count: number;
	duration_ms: number | null;
	has_errors: boolean;
	agent_name: string | null;
	agent_type: string | null;
}

/** A node in the span tree with events, children, and summary. */
export interface SpanTreeNode {
	span_id: string;
	parent_span_id: string | null;
	name: string;
	summary: SpanSummary;
	events: TraceEvent[];
	children: SpanTreeNode[];
}

/** Wrapper for the full span tree. */
export interface SpanTreeResponse {
	trace_id: string;
	root: SpanTreeNode;
}

/** Events within a specific span. */
export interface SpanEventsResponse {
	span_id: string;
	events: TraceEvent[];
}

/** Per-agent computed statistics. */
export interface AgentStats {
	llm_calls: number;
	tool_calls: number;
	input_tokens: number;
	output_tokens: number;
	duration_ms: number | null;
	errors: number;
	iterations: number;
}

/** Agent metadata and stats. */
export interface AgentInfo {
	agent_name: string;
	agent_type: string | null;
	span_id: string;
	capabilities: string[];
	stats: AgentStats;
}

/** List of agents in a run. */
export interface AgentListResponse {
	agents: AgentInfo[];
}

/** Detailed agent view with events and span subtree. */
export interface AgentDetailResponse {
	agent: AgentInfo;
	events: TraceEvent[];
	span_tree: SpanTreeNode;
}

/** A workflow step with definition and runtime status. */
export interface WorkflowStep {
	name: string;
	step_type: string;
	index: number | null;
	depends_on: string[];
	parallel_group: string | null;
	status: string;
	duration_ms: number | null;
	agent_span_id: string | null;
	metadata: Record<string, unknown>;
}

/** Complete workflow DAG structure. */
export interface WorkflowDAGResponse {
	workflow_name: string;
	workflow_type: string;
	steps: WorkflowStep[];
}

/** Paginated event list. */
export interface EventListResponse {
	events: TraceEvent[];
	has_more: boolean;
}
