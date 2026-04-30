import type { RunResponse, SpanTreeNode, SpanTreeResponse, TraceEvent, TraceSummaryResponse } from "../../src/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let nextEventId = 1;

export function makeEvent(overrides: Partial<TraceEvent> = {}): TraceEvent {
	return {
		id: nextEventId++,
		event_type: "test.event",
		level: "info",
		trace_id: "trace-1",
		span_id: "span-root",
		parent_span_id: null,
		timestamp: "2026-03-05T10:00:00Z",
		payload: {},
		...overrides,
	};
}

export function makeRun(overrides: Partial<RunResponse> = {}): RunResponse {
	return {
		id: "run-1",
		trace_id: "trace-1",
		status: "completed",
		started_at: "2026-03-05T10:00:00Z",
		completed_at: "2026-03-05T10:05:00Z",
		metadata: {},
		error: null,
		result: null,
		...overrides,
	};
}

export function makeSummary(overrides: Partial<TraceSummaryResponse> = {}): TraceSummaryResponse {
	return {
		total_events: 25,
		events_by_level: { info: 5, debug: 15, verbose: 5 },
		llm_calls: 6,
		tool_calls: 4,
		total_input_tokens: 1200,
		total_output_tokens: 800,
		total_duration_ms: 5000,
		agent_names: ["research-assistant"],
		errors: 0,
		cache_creation_tokens: 0,
		cache_read_tokens: 0,
		...overrides,
	};
}

function makeSpanNode(overrides: Partial<SpanTreeNode> & { span_id: string }): SpanTreeNode {
	return {
		parent_span_id: null,
		name: "span",
		summary: {
			event_count: 0,
			duration_ms: null,
			has_errors: false,
			agent_name: null,
			agent_type: null,
		},
		events: [],
		children: [],
		...overrides,
	};
}

// ---------------------------------------------------------------------------
// Single Agent Scenario
// ---------------------------------------------------------------------------

export const singleAgentTree: SpanTreeResponse = {
	trace_id: "trace-single",
	root: makeSpanNode({
		span_id: "root",
		name: "run",
		summary: {
			event_count: 2,
			duration_ms: 5000,
			has_errors: false,
			agent_name: null,
			agent_type: null,
		},
		events: [
			makeEvent({
				event_type: "run.start",
				span_id: "root",
				trace_id: "trace-single",
			}),
		],
		children: [
			makeSpanNode({
				span_id: "agent-1",
				parent_span_id: "root",
				name: "research-assistant",
				summary: {
					event_count: 8,
					duration_ms: 4500,
					has_errors: false,
					agent_name: "research-assistant",
					agent_type: "react",
				},
				events: [
					makeEvent({
						event_type: "agent.start",
						span_id: "agent-1",
						level: "info",
						payload: {
							agent_name: "research-assistant",
							agent_type: "react",
							capabilities: ["web_search", "summarize"],
							tools_available: ["search", "read_page"],
						},
					}),
					makeEvent({
						event_type: "agent.step",
						span_id: "agent-1",
						level: "debug",
						payload: { step: 1, thought: "Search for topic", action: "search" },
					}),
				],
				children: [
					makeSpanNode({
						span_id: "step-1",
						parent_span_id: "agent-1",
						name: "step-1",
						summary: {
							event_count: 4,
							duration_ms: 2000,
							has_errors: false,
							agent_name: null,
							agent_type: null,
						},
						events: [
							makeEvent({
								event_type: "llm.request",
								span_id: "step-1",
								level: "debug",
								payload: {
									model_name: "claude-haiku-4-5-20251001",
									input_tokens: 200,
									messages_count: 3,
								},
							}),
							makeEvent({
								event_type: "llm.response",
								span_id: "step-1",
								level: "debug",
								payload: {
									model_name: "claude-haiku-4-5-20251001",
									usage: { input_tokens: 200, output_tokens: 150 },
									stop_reason: "end_turn",
								},
							}),
							makeEvent({
								event_type: "tool.invoke",
								span_id: "step-1",
								level: "debug",
								payload: {
									tool_name: "search",
									parameters: { query: "AI agents" },
								},
							}),
							makeEvent({
								event_type: "tool.result",
								span_id: "step-1",
								level: "debug",
								payload: {
									tool_name: "search",
									success: true,
									result: "Found 10 results",
								},
							}),
						],
					}),
				],
			}),
		],
	}),
};

// ---------------------------------------------------------------------------
// Multi-Agent Workflow Scenario
// ---------------------------------------------------------------------------

export const workflowTree: SpanTreeResponse = {
	trace_id: "trace-workflow",
	root: makeSpanNode({
		span_id: "root-wf",
		name: "content-pipeline",
		summary: {
			event_count: 3,
			duration_ms: 12000,
			has_errors: false,
			agent_name: null,
			agent_type: null,
		},
		events: [
			makeEvent({
				event_type: "workflow.start",
				span_id: "root-wf",
				trace_id: "trace-workflow",
				payload: { workflow_name: "content-pipeline", workflow_type: "dag" },
			}),
		],
		children: [
			makeSpanNode({
				span_id: "wf-agent-1",
				parent_span_id: "root-wf",
				name: "researcher",
				summary: {
					event_count: 5,
					duration_ms: 4000,
					has_errors: false,
					agent_name: "researcher",
					agent_type: "react",
				},
				events: [
					makeEvent({
						event_type: "agent.start",
						span_id: "wf-agent-1",
						payload: { agent_name: "researcher", agent_type: "react" },
					}),
				],
			}),
			makeSpanNode({
				span_id: "wf-agent-2",
				parent_span_id: "root-wf",
				name: "writer",
				summary: {
					event_count: 5,
					duration_ms: 5000,
					has_errors: false,
					agent_name: "writer",
					agent_type: "react",
				},
				events: [
					makeEvent({
						event_type: "agent.start",
						span_id: "wf-agent-2",
						payload: { agent_name: "writer", agent_type: "react" },
					}),
				],
			}),
		],
	}),
};

// ---------------------------------------------------------------------------
// Capability Events Scenario
// ---------------------------------------------------------------------------

export const capabilityEvents: TraceEvent[] = [
	makeEvent({
		event_type: "context.assembly",
		level: "debug",
		payload: {
			sources: ["system_prompt", "working_memory", "tool_results"],
			total_tokens: 1500,
		},
	}),
	makeEvent({
		event_type: "planning.plan.created",
		level: "info",
		payload: {
			plan_id: "plan-1",
			goal: "Write a report",
			steps: [
				{ index: 0, description: "Research topic", status: "pending" },
				{ index: 1, description: "Write draft", status: "pending" },
			],
		},
	}),
	makeEvent({
		event_type: "memory.working.update",
		level: "verbose",
		payload: { key: "research_notes", action: "set", summary: "Added notes" },
	}),
	makeEvent({
		event_type: "memory.semantic.search",
		level: "verbose",
		payload: {
			collection: "knowledge",
			query: "AI agent architectures",
			results_count: 5,
		},
	}),
	makeEvent({
		event_type: "evaluation.result",
		level: "info",
		payload: {
			evaluator: "quality_check",
			score: 0.85,
			passed: true,
		},
	}),
];

// ---------------------------------------------------------------------------
// Error Scenario
// ---------------------------------------------------------------------------

export const errorTree: SpanTreeResponse = {
	trace_id: "trace-errors",
	root: makeSpanNode({
		span_id: "root-err",
		name: "resilient-worker",
		summary: {
			event_count: 5,
			duration_ms: 8000,
			has_errors: true,
			agent_name: null,
			agent_type: null,
		},
		events: [],
		children: [
			makeSpanNode({
				span_id: "err-agent",
				parent_span_id: "root-err",
				name: "worker",
				summary: {
					event_count: 10,
					duration_ms: 7000,
					has_errors: true,
					agent_name: "worker",
					agent_type: "react",
				},
				events: [
					makeEvent({
						event_type: "agent.start",
						span_id: "err-agent",
						payload: { agent_name: "worker", agent_type: "react" },
					}),
					makeEvent({
						event_type: "tool.result",
						span_id: "err-agent",
						level: "debug",
						payload: {
							tool_name: "fetch_data",
							success: false,
							error: "Connection timeout",
						},
					}),
					makeEvent({
						event_type: "error.retry",
						span_id: "err-agent",
						level: "debug",
						payload: {
							attempt: 1,
							max_attempts: 3,
							category: "transient",
							error: "Connection timeout",
							delay_ms: 1000,
						},
					}),
				],
			}),
		],
	}),
};

// ---------------------------------------------------------------------------
// HITL Scenario
// ---------------------------------------------------------------------------

export const hitlEvents: TraceEvent[] = [
	makeEvent({
		event_type: "hitl.request",
		level: "info",
		payload: {
			request_type: "approval",
			message: "Deploy to production?",
			options: ["approve", "reject"],
		},
	}),
	makeEvent({
		event_type: "hitl.response",
		level: "info",
		payload: {
			response: "approve",
			responded_by: "admin",
		},
	}),
];

// ---------------------------------------------------------------------------
// Streaming Scenario Events
// ---------------------------------------------------------------------------

export const streamingEvents: TraceEvent[] = [
	makeEvent({
		event_type: "agent.start",
		level: "info",
		timestamp: "2026-03-05T10:00:00Z",
		payload: { agent_name: "live-analyzer", agent_type: "react" },
	}),
	makeEvent({
		event_type: "llm.request",
		level: "debug",
		timestamp: "2026-03-05T10:00:02Z",
		payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 150 },
	}),
	makeEvent({
		event_type: "llm.response",
		level: "debug",
		timestamp: "2026-03-05T10:00:04Z",
		payload: {
			model_name: "claude-haiku-4-5-20251001",
			usage: { input_tokens: 150, output_tokens: 100 },
			stop_reason: "end_turn",
		},
	}),
	makeEvent({
		event_type: "agent.complete",
		level: "info",
		timestamp: "2026-03-05T10:00:06Z",
		payload: { agent_name: "live-analyzer", termination_reason: "goal_reached" },
	}),
];

// ---------------------------------------------------------------------------
// Run list fixtures
// ---------------------------------------------------------------------------

export const sampleRuns: RunResponse[] = [
	makeRun({
		id: "run-1",
		status: "completed",
		metadata: { description: "Single agent run" },
	}),
	makeRun({
		id: "run-2",
		status: "completed",
		metadata: { description: "Multi-agent workflow" },
	}),
	makeRun({
		id: "run-3",
		status: "completed",
		metadata: { description: "Capability test" },
	}),
	makeRun({
		id: "run-4",
		status: "failed",
		metadata: { description: "Error scenario" },
		error: "Max retries exceeded",
	}),
	makeRun({
		id: "run-5",
		status: "suspended",
		metadata: { description: "HITL scenario" },
		completed_at: null,
	}),
	makeRun({
		id: "run-6",
		status: "running",
		metadata: { description: "Streaming scenario" },
		completed_at: null,
	}),
];

// ---------------------------------------------------------------------------
// CodeAct Scenario — "Data Analyst"
// ---------------------------------------------------------------------------

export const codeactTree: SpanTreeResponse = {
	trace_id: "trace-codeact",
	root: makeSpanNode({
		span_id: "root-ca",
		name: "data-analysis",
		summary: {
			event_count: 2,
			duration_ms: 10000,
			has_errors: true,
			agent_name: null,
			agent_type: null,
		},
		events: [
			makeEvent({
				event_type: "run.start",
				span_id: "root-ca",
				trace_id: "trace-codeact",
			}),
		],
		children: [
			makeSpanNode({
				span_id: "ca-agent",
				parent_span_id: "root-ca",
				name: "data-analyst",
				summary: {
					event_count: 25,
					duration_ms: 9000,
					has_errors: true,
					agent_name: "data-analyst",
					agent_type: "codeact",
				},
				events: [
					// agent.start
					makeEvent({
						event_type: "agent.start",
						span_id: "ca-agent",
						level: "info",
						timestamp: "2026-03-05T10:00:00Z",
						payload: {
							agent_name: "data-analyst",
							agent_type: "codeact",
							capabilities: ["code_execution"],
							tools_available: [],
						},
					}),

					// --- Step 1: Import and load data ---
					makeEvent({
						event_type: "agent.step",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:01Z",
						payload: {
							step: 1,
							thought: "Let me import the necessary libraries and load the dataset.",
						},
					}),
					makeEvent({
						event_type: "llm.request",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:01.100Z",
						payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 300, messages_count: 2 },
					}),
					makeEvent({
						event_type: "llm.response",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:02Z",
						payload: {
							model_name: "claude-haiku-4-5-20251001",
							usage: { input_tokens: 300, output_tokens: 80 },
							stop_reason: "end_turn",
						},
					}),
					makeEvent({
						event_type: "code.execution",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:02.100Z",
						payload: {
							code: 'import pandas as pd\n\ndf = pd.read_csv("data.csv")\nprint(f"Loaded {len(df)} rows")',
							language: "python",
						},
					}),
					makeEvent({
						event_type: "code.execution.result",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:02.300Z",
						payload: {
							success: true,
							stdout: "Loaded 1500 rows",
							stderr: "",
							return_value: null,
							duration_ms: 150,
						},
					}),

					// --- Step 2: Compute statistics ---
					makeEvent({
						event_type: "agent.step",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:03Z",
						payload: {
							step: 2,
							thought: "Now I'll compute descriptive statistics for the dataset.",
						},
					}),
					makeEvent({
						event_type: "llm.request",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:03.100Z",
						payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 450, messages_count: 4 },
					}),
					makeEvent({
						event_type: "llm.response",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:04Z",
						payload: {
							model_name: "claude-haiku-4-5-20251001",
							usage: { input_tokens: 450, output_tokens: 120 },
							stop_reason: "end_turn",
						},
					}),
					makeEvent({
						event_type: "code.execution",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:04.100Z",
						payload: {
							code: 'summary = df.describe()\nprint(summary.to_string())\n\nmissing = df.isnull().sum()\nprint(f"\\nMissing values:\\n{missing}")',
							language: "python",
						},
					}),
					makeEvent({
						event_type: "code.execution.result",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:04.400Z",
						payload: {
							success: true,
							stdout:
								"       col1    col2    col3\ncount  1500   1500    1500\nmean   42.3   18.7    0.85\nstd    12.1    5.4    0.12\nmin     1.0    0.0    0.31\n25%    34.2   15.1    0.78\n50%    41.8   18.5    0.86\n75%    50.1   22.3    0.92\nmax    89.7   39.8    0.99\n\nMissing values:\ncol1    0\ncol2    3\ncol3    0",
							stderr: "",
							return_value: null,
							duration_ms: 250,
						},
					}),

					// --- Step 3: Attempt plot with syntax error ---
					makeEvent({
						event_type: "agent.step",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:05Z",
						payload: {
							step: 3,
							thought: "I'll create a visualization of the distribution.",
						},
					}),
					makeEvent({
						event_type: "llm.request",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:05.100Z",
						payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 600, messages_count: 6 },
					}),
					makeEvent({
						event_type: "llm.response",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:06Z",
						payload: {
							model_name: "claude-haiku-4-5-20251001",
							usage: { input_tokens: 600, output_tokens: 100 },
							stop_reason: "end_turn",
						},
					}),
					makeEvent({
						event_type: "code.execution",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:06.100Z",
						payload: {
							code: "import matplotlib.pyplot as plt\n\nplt.figure(figsize=(10, 6)\nplt.hist(df['col1'], bins=30)\nplt.title('Distribution of col1')\nplt.savefig('plot.png')",
							language: "python",
						},
					}),
					makeEvent({
						event_type: "code.execution.result",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:06.200Z",
						payload: {
							success: false,
							stdout: "",
							stderr: "",
							return_value: null,
							error: "SyntaxError: '(' was never closed (line 3)",
							duration_ms: 50,
						},
					}),

					// --- Step 4: Fix and re-execute ---
					makeEvent({
						event_type: "agent.step",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:07Z",
						payload: {
							step: 4,
							thought: "The previous code had a syntax error — missing closing parenthesis. Let me fix it.",
						},
					}),
					makeEvent({
						event_type: "llm.request",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:07.100Z",
						payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 750, messages_count: 8 },
					}),
					makeEvent({
						event_type: "llm.response",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:08Z",
						payload: {
							model_name: "claude-haiku-4-5-20251001",
							usage: { input_tokens: 750, output_tokens: 110 },
							stop_reason: "end_turn",
						},
					}),
					makeEvent({
						event_type: "code.execution",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:08.100Z",
						payload: {
							code: "import matplotlib.pyplot as plt\n\nplt.figure(figsize=(10, 6))\nplt.hist(df['col1'], bins=30)\nplt.title('Distribution of col1')\nplt.savefig('plot.png')\nprint('Plot saved to plot.png')",
							language: "python",
						},
					}),
					makeEvent({
						event_type: "code.execution.result",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:08.500Z",
						payload: {
							success: true,
							stdout: "Plot saved to plot.png",
							stderr: "",
							return_value: null,
							duration_ms: 350,
						},
					}),

					// --- Step 5: Summarize findings (text only, no code) ---
					makeEvent({
						event_type: "agent.step",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:09Z",
						payload: {
							step: 5,
							thought: "Let me summarize the findings.",
							observation:
								"The dataset has 1500 rows with 3 columns. col2 has 3 missing values. Distributions look normally distributed. A histogram has been saved.",
						},
					}),
					makeEvent({
						event_type: "llm.request",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:09.100Z",
						payload: { model_name: "claude-haiku-4-5-20251001", input_tokens: 900, messages_count: 10 },
					}),
					makeEvent({
						event_type: "llm.response",
						span_id: "ca-agent",
						level: "debug",
						timestamp: "2026-03-05T10:00:09.500Z",
						payload: {
							model_name: "claude-haiku-4-5-20251001",
							usage: { input_tokens: 900, output_tokens: 200 },
							stop_reason: "end_turn",
						},
					}),

					// agent.complete
					makeEvent({
						event_type: "agent.complete",
						span_id: "ca-agent",
						level: "info",
						timestamp: "2026-03-05T10:00:10Z",
						payload: {
							agent_name: "data-analyst",
							termination_reason: "goal_reached",
							total_steps: 5,
						},
					}),
				],
				children: [],
			}),
		],
	}),
};

// ---------------------------------------------------------------------------
// Tree of Thought Scenario
// ---------------------------------------------------------------------------

export const treeOfThoughtEvents: TraceEvent[] = [
	// Agent start
	makeEvent({
		event_type: "agent.start",
		span_id: "tot-agent",
		level: "info",
		timestamp: "2026-03-05T10:00:00Z",
		payload: { agent_name: "tot-solver", agent_type: "tree_of_thought", search_strategy: "best_first" },
	}),

	// Root node
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:01Z",
		payload: {
			node_id: "tot-root",
			parent_id: null,
			depth: 0,
			content:
				"Let me decompose this problem into sub-tasks. First, I need to identify the key variables, then establish relationships between them.",
			node_type: "thought",
			is_terminal: false,
			is_failed: false,
		},
	}),

	// Depth-1 children (3 branches)
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:02Z",
		payload: {
			node_id: "tot-1a",
			parent_id: "tot-root",
			depth: 1,
			content:
				"Using algebraic manipulation, we can substitute x = 2y into the second equation to get 3(2y) - y = 10, which simplifies to 5y = 10, so y = 2.",
			node_type: "thought",
		},
	}),
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:02.100Z",
		payload: {
			node_id: "tot-1b",
			parent_id: "tot-root",
			depth: 1,
			content:
				"Plotting both equations on a graph, we can find the intersection point. Line 1: y = x/2, Line 2: y = 3x - 10. Setting equal: x/2 = 3x - 10.",
			node_type: "thought",
		},
	}),
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:02.200Z",
		payload: {
			node_id: "tot-1c",
			parent_id: "tot-root",
			depth: 1,
			content:
				"Setting up the augmented matrix [[1, -2, 0], [3, -1, 10]] and performing row reduction to solve the system.",
			node_type: "thought",
		},
	}),

	// Evaluations for depth-1
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03Z",
		payload: { node_id: "tot-1a", score: 0.85, is_terminal: false },
	}),
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03.100Z",
		payload: { node_id: "tot-1b", score: 0.45, is_terminal: false },
	}),
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03.200Z",
		payload: { node_id: "tot-1c", score: 0.3, is_terminal: false },
	}),

	// Prune low-scoring branch
	makeEvent({
		event_type: "tree_search.node.pruned",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03.300Z",
		payload: { node_id: "tot-1c", reason: "Score below threshold (0.30 < 0.40)" },
	}),

	// Depth-2 children of best branch (tot-1a)
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:04Z",
		payload: {
			node_id: "tot-2a",
			parent_id: "tot-1a",
			depth: 2,
			content: "If y = 2, then x = 2(2) = 4. Let me verify: 3(4) - 2 = 12 - 2 = 10 ✓. The solution is (4, 2).",
			node_type: "thought",
			is_terminal: true,
		},
	}),

	// Terminal evaluation
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "tot-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:05Z",
		payload: { node_id: "tot-2a", score: 0.95, is_terminal: true },
	}),

	// Search complete
	makeEvent({
		event_type: "tree_search.complete",
		span_id: "tot-agent",
		level: "info",
		timestamp: "2026-03-05T10:00:06Z",
		payload: {
			selected_node_id: "tot-2a",
			termination_reason: "solution_found",
			total_nodes: 5,
			max_depth_reached: 2,
			search_strategy: "best_first",
		},
	}),
];

export const treeOfThoughtTree: SpanTreeResponse = {
	trace_id: "trace-tot",
	root: makeSpanNode({
		span_id: "root",
		name: "run",
		summary: { event_count: 1, duration_ms: 6000, has_errors: false, agent_name: null, agent_type: null },
		events: [makeEvent({ event_type: "run.start", span_id: "root", trace_id: "trace-tot" })],
		children: [
			makeSpanNode({
				span_id: "tot-agent",
				parent_span_id: "root",
				name: "tot-solver",
				summary: {
					event_count: treeOfThoughtEvents.length,
					duration_ms: 6000,
					has_errors: false,
					agent_name: "tot-solver",
					agent_type: "tree_of_thought",
				},
				events: treeOfThoughtEvents,
				children: [],
			}),
		],
	}),
};

// ---------------------------------------------------------------------------
// LATS Scenario
// ---------------------------------------------------------------------------

export const latsEvents: TraceEvent[] = [
	// Agent start
	makeEvent({
		event_type: "agent.start",
		span_id: "lats-agent",
		level: "info",
		timestamp: "2026-03-05T10:00:00Z",
		payload: { agent_name: "lats-solver", agent_type: "lats", exploration_constant: Math.SQRT2 },
	}),

	// Root node
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:01Z",
		payload: {
			node_id: "lats-root",
			parent_id: null,
			depth: 0,
			content: "The task is to find the current population of Paris. I need to search for this information.",
			node_type: "thought",
			action: null,
			observation: null,
			is_terminal: false,
			is_failed: false,
		},
	}),

	// Iteration 1: expand root
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:02Z",
		payload: {
			node_id: "lats-1a",
			parent_id: "lats-root",
			depth: 1,
			content: "Using web_search to find the current population of Paris, France.",
			node_type: "action",
			action: "web_search",
			observation: "Paris, the capital of France, has a population of approximately 2.16 million in the city proper...",
			is_terminal: false,
			is_failed: false,
		},
	}),
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:02.100Z",
		payload: {
			node_id: "lats-1b",
			parent_id: "lats-root",
			depth: 1,
			content: "Searching Wikipedia for detailed population statistics of Paris.",
			node_type: "action",
			action: "wiki_search",
			observation: "Paris — Wikipedia article about the capital city of France. Population (2024): 2,102,650...",
			is_terminal: false,
			is_failed: false,
		},
	}),

	// Evaluate nodes
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03Z",
		payload: { node_id: "lats-1a", score: 0.7, is_terminal: false },
	}),
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03.100Z",
		payload: { node_id: "lats-1b", score: 0.8, is_terminal: false },
	}),

	// MCTS iteration 1
	makeEvent({
		event_type: "mcts.iteration",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03.200Z",
		payload: {
			iteration_number: 1,
			selected_node_id: "lats-1b",
			selection_path: ["lats-root", "lats-1b"],
			expanded_count: 2,
			best_value_so_far: 0.8,
			node_values: { "lats-root": 0.75, "lats-1a": 0.7, "lats-1b": 0.8 },
		},
	}),

	// Backpropagation for iteration 1
	makeEvent({
		event_type: "mcts.backpropagation",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:03.300Z",
		payload: {
			propagated_value: 0.8,
			path_length: 2,
			updated_node_ids: ["lats-1b", "lats-root"],
		},
	}),

	// Iteration 2: expand lats-1b further
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:04Z",
		payload: {
			node_id: "lats-2a",
			parent_id: "lats-1b",
			depth: 2,
			content:
				"Based on Wikipedia data, Paris has a population of 2,102,650 (2024 census). The metropolitan area has about 12.4 million people.",
			node_type: "thought",
			action: null,
			observation: null,
			is_terminal: true,
			is_failed: false,
		},
	}),
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:04.100Z",
		payload: {
			node_id: "lats-2b",
			parent_id: "lats-1b",
			depth: 2,
			content: "Verifying with French government statistical agency (INSEE) for the most recent population figure.",
			node_type: "action",
			action: "web_search",
			observation: "INSEE: Population de Paris au 1er janvier 2024: 2,102,650 habitants...",
			is_terminal: false,
			is_failed: false,
		},
	}),

	// Evaluate iteration 2 nodes
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:05Z",
		payload: { node_id: "lats-2a", score: 0.92, is_terminal: true },
	}),
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:05.100Z",
		payload: { node_id: "lats-2b", score: 0.75, is_terminal: false },
	}),

	// MCTS iteration 2
	makeEvent({
		event_type: "mcts.iteration",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:05.200Z",
		payload: {
			iteration_number: 2,
			selected_node_id: "lats-2a",
			selection_path: ["lats-root", "lats-1b", "lats-2a"],
			expanded_count: 2,
			best_value_so_far: 0.92,
			node_values: {
				"lats-root": 0.79,
				"lats-1a": 0.7,
				"lats-1b": 0.83,
				"lats-2a": 0.92,
				"lats-2b": 0.75,
			},
		},
	}),

	// Backpropagation for iteration 2
	makeEvent({
		event_type: "mcts.backpropagation",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:05.300Z",
		payload: {
			propagated_value: 0.92,
			path_length: 3,
			updated_node_ids: ["lats-2a", "lats-1b", "lats-root"],
		},
	}),

	// Iteration 3: expand lats-1a (exploration)
	makeEvent({
		event_type: "tree_search.node.created",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:06Z",
		payload: {
			node_id: "lats-3a",
			parent_id: "lats-1a",
			depth: 2,
			content: "Attempted to search for additional population sources but the API returned a rate limit error.",
			node_type: "action",
			action: "web_search",
			observation: "Error: Rate limit exceeded. Please try again in 60 seconds.",
			is_terminal: false,
			is_failed: true,
		},
	}),

	// Evaluate failed node
	makeEvent({
		event_type: "tree_search.node.evaluated",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:06.500Z",
		payload: { node_id: "lats-3a", score: 0.1, is_terminal: false },
	}),

	// MCTS iteration 3
	makeEvent({
		event_type: "mcts.iteration",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:06.600Z",
		payload: {
			iteration_number: 3,
			selected_node_id: "lats-1a",
			selection_path: ["lats-root", "lats-1a"],
			expanded_count: 1,
			best_value_so_far: 0.92,
			node_values: {
				"lats-root": 0.69,
				"lats-1a": 0.4,
				"lats-1b": 0.83,
				"lats-2a": 0.92,
				"lats-2b": 0.75,
				"lats-3a": 0.1,
			},
		},
	}),

	// Backpropagation for iteration 3
	makeEvent({
		event_type: "mcts.backpropagation",
		span_id: "lats-agent",
		level: "debug",
		timestamp: "2026-03-05T10:00:06.700Z",
		payload: {
			propagated_value: 0.1,
			path_length: 3,
			updated_node_ids: ["lats-3a", "lats-1a", "lats-root"],
		},
	}),

	// Search complete
	makeEvent({
		event_type: "tree_search.complete",
		span_id: "lats-agent",
		level: "info",
		timestamp: "2026-03-05T10:00:07Z",
		payload: {
			selected_node_id: "lats-2a",
			termination_reason: "solution_found",
			total_nodes: 7,
			max_depth_reached: 2,
		},
	}),
];

export const latsTree: SpanTreeResponse = {
	trace_id: "trace-lats",
	root: makeSpanNode({
		span_id: "root",
		name: "run",
		summary: { event_count: 1, duration_ms: 7000, has_errors: false, agent_name: null, agent_type: null },
		events: [makeEvent({ event_type: "run.start", span_id: "root", trace_id: "trace-lats" })],
		children: [
			makeSpanNode({
				span_id: "lats-agent",
				parent_span_id: "root",
				name: "lats-solver",
				summary: {
					event_count: latsEvents.length,
					duration_ms: 7000,
					has_errors: false,
					agent_name: "lats-solver",
					agent_type: "lats",
				},
				events: latsEvents,
				children: [],
			}),
		],
	}),
};
