import { useState } from "react";
import { PayloadViewer } from "../components/event-detail/payload-viewer";
import { CodeBlock } from "../components/primitives/code-block";
import { StatusBadge } from "../components/primitives/status-badge";
import { TokenUsage } from "../components/primitives/token-usage";
import type { TraceEvent } from "../types";
import type { AgentViewRegistry } from "./agent-view-registry";
import type { CapabilityPanelRegistry } from "./capability-panel-registry";
import { createDefaultAgentViewRegistry } from "./default-agent-views";
import { createDefaultPanelRegistry } from "./default-panels";
import type { EventDetailProps, EventRendererRegistration } from "./renderer-registry";
import { EventRendererRegistry } from "./renderer-registry";
import { createRunRegistrations } from "./run-renderers";
import { createWorkflowRegistrations } from "./workflow-renderers";

// --- Generic Fallback ---

function GenericPayloadRenderer({ event }: EventDetailProps) {
	return (
		<div className="space-y-2">
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- LLM Renderers ---

function LLMRequestRenderer({ event }: EventDetailProps) {
	const { model_name, input_tokens, messages_count } = event.payload as {
		model_name?: string;
		input_tokens?: number;
		messages_count?: number;
	};

	return (
		<div className="space-y-3">
			{model_name && (
				<div className="flex items-center gap-2">
					<span className="text-xs text-muted-foreground">Model</span>
					<span className="text-sm font-mono">{model_name}</span>
				</div>
			)}
			<div className="flex items-center gap-4 text-xs">
				{input_tokens != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Input tokens</span>
						<span className="font-mono tabular-nums">{input_tokens.toLocaleString()}</span>
					</div>
				)}
				{messages_count != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Messages</span>
						<span className="font-mono tabular-nums">{messages_count}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function LLMResponseRenderer({ event }: EventDetailProps) {
	const { model_name, usage, stop_reason } = event.payload as {
		model_name?: string;
		usage?: { input_tokens: number; output_tokens: number };
		stop_reason?: string;
	};

	return (
		<div className="space-y-3">
			{model_name && (
				<div className="flex items-center gap-2">
					<span className="text-xs text-muted-foreground">Model</span>
					<span className="text-sm font-mono">{model_name}</span>
				</div>
			)}
			{usage && <TokenUsage inputTokens={usage.input_tokens} outputTokens={usage.output_tokens} />}
			{stop_reason && (
				<div className="flex items-center gap-2 text-xs">
					<span className="text-muted-foreground">Stop reason</span>
					<span className="font-mono">{stop_reason}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Tool Renderers ---

function ToolInvokeRenderer({ event }: EventDetailProps) {
	const { tool_name, parameters } = event.payload as {
		tool_name?: string;
		parameters?: Record<string, unknown>;
	};

	return (
		<div className="space-y-3">
			{tool_name && (
				<div className="flex items-center gap-2">
					<span className="text-xs text-muted-foreground">Tool</span>
					<span className="text-sm font-mono font-medium">{tool_name}</span>
				</div>
			)}
			{parameters && Object.keys(parameters).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Parameters</span>
					<PayloadViewer payload={parameters} />
				</div>
			)}
			{(!parameters || Object.keys(parameters).length === 0) && <PayloadViewer payload={event.payload} />}
		</div>
	);
}

function ToolResultRenderer({ event }: EventDetailProps) {
	const { tool_name, success, result_summary, error } = event.payload as {
		tool_name?: string;
		success?: boolean;
		result_summary?: string;
		error?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{tool_name && <span className="text-sm font-mono font-medium">{tool_name}</span>}
				{success != null && (
					<span
						className={`text-xs px-1.5 py-0.5 rounded ${success ? "bg-success-muted text-success" : "bg-destructive-muted text-destructive"}`}
					>
						{success ? "success" : "failed"}
					</span>
				)}
			</div>
			{result_summary && <div className="text-sm text-foreground">{result_summary}</div>}
			{error && (
				<div className="text-sm text-destructive-muted-foreground bg-destructive-muted rounded-md p-2">{error}</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Agent Renderers ---

function AgentStartRenderer({ event, onNavigateToAgent }: EventDetailProps) {
	const { agent_name, agent_type, capabilities, tools_available } = event.payload as {
		agent_name?: string;
		agent_type?: string;
		capabilities?: string[];
		tools_available?: string[];
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{agent_name && <span className="text-sm font-medium">{agent_name}</span>}
				{agent_type && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{agent_type}</span>}
			</div>
			{onNavigateToAgent && (
				<button
					type="button"
					className="text-xs text-primary hover:text-primary/80 transition-colors"
					onClick={() => onNavigateToAgent(event.span_id)}
				>
					View agent details →
				</button>
			)}
			{capabilities && capabilities.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Capabilities</span>
					<div className="flex flex-wrap gap-1">
						{capabilities.map((cap) => (
							<span key={cap} className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
								{cap}
							</span>
						))}
					</div>
				</div>
			)}
			{tools_available && tools_available.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Tools</span>
					<div className="flex flex-wrap gap-1">
						{tools_available.map((tool) => (
							<span key={tool} className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-mono text-muted-foreground">
								{tool}
							</span>
						))}
					</div>
				</div>
			)}
		</div>
	);
}

function AgentStepRenderer({ event }: EventDetailProps) {
	const { step, thought, action, observation } = event.payload as {
		step?: number;
		thought?: string;
		action?: string;
		observation?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{step != null && <span className="text-xs px-1.5 py-0.5 rounded bg-muted font-mono">Step {step}</span>}
				{action && <span className="text-sm font-mono text-muted-foreground">{action}</span>}
			</div>
			{thought && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Thought</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{thought}</div>
				</div>
			)}
			{observation && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Observation</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{observation}</div>
				</div>
			)}
		</div>
	);
}

function AgentCompleteRenderer({ event }: EventDetailProps) {
	const { agent_name, termination_reason, total_steps } = event.payload as {
		agent_name?: string;
		termination_reason?: string;
		total_steps?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{agent_name && <span className="text-sm font-medium">{agent_name}</span>}
				{termination_reason && (
					<StatusBadge status={termination_reason === "task_complete" ? "completed" : termination_reason} />
				)}
			</div>
			{total_steps != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Steps</span>
					<span className="font-mono tabular-nums">{total_steps}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Memory Renderers ---

function MemoryWorkingReadRenderer({ event }: EventDetailProps) {
	const { content, token_count } = event.payload as {
		content?: string;
		token_count?: number;
	};

	return (
		<div className="space-y-3">
			{token_count != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Tokens</span>
					<span className="font-mono tabular-nums">{token_count.toLocaleString()}</span>
				</div>
			)}
			{content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Content</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">{content}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryWorkingUpdateRenderer({ event }: EventDetailProps) {
	const { source, previous_content, new_content } = event.payload as {
		source?: string;
		previous_content?: string;
		new_content?: string;
	};

	return (
		<div className="space-y-3">
			{source && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Source</span>
					<span className="font-mono">{source}</span>
				</div>
			)}
			{previous_content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Previous</span>
					<div className="text-sm bg-destructive-muted rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">
						{previous_content}
					</div>
				</div>
			)}
			{new_content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Updated</span>
					<div className="text-sm bg-success-muted rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">
						{new_content}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySemanticStoreRenderer({ event }: EventDetailProps) {
	const { content, entry_id, namespace } = event.payload as {
		content?: string;
		entry_id?: string;
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{entry_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Entry</span>
						<span className="font-mono">{entry_id}</span>
					</div>
				)}
				{namespace && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Namespace</span>
						<span className="font-mono">{namespace}</span>
					</div>
				)}
			</div>
			{content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Content</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">{content}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySemanticSearchRenderer({ event }: EventDetailProps) {
	const { query, results_count, top_score, namespace } = event.payload as {
		query?: string;
		results_count?: number;
		top_score?: number;
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{results_count != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Results</span>
						<span className="font-mono tabular-nums">{results_count}</span>
					</div>
				)}
				{top_score != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Top score</span>
						<span className="font-mono tabular-nums">{top_score.toFixed(3)}</span>
					</div>
				)}
				{namespace && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Namespace</span>
						<span className="font-mono">{namespace}</span>
					</div>
				)}
			</div>
			{query && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Query</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{query}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySemanticDeleteRenderer({ event }: EventDetailProps) {
	const { entry_id, namespace } = event.payload as {
		entry_id?: string;
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{entry_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Entry</span>
						<span className="font-mono">{entry_id}</span>
					</div>
				)}
				{namespace && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Namespace</span>
						<span className="font-mono">{namespace}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryEpisodeRecordRenderer({ event }: EventDetailProps) {
	const { episode_id, situation, outcome, has_reflection } = event.payload as {
		episode_id?: string;
		situation?: string;
		outcome?: string;
		has_reflection?: boolean;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{episode_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Episode</span>
						<span className="font-mono">{episode_id}</span>
					</div>
				)}
				{has_reflection && (
					<span className="px-1.5 py-0.5 rounded bg-accent-status-muted text-accent-status-muted-foreground text-[10px]">
						has reflection
					</span>
				)}
			</div>
			{situation && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Situation</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{situation}</div>
				</div>
			)}
			{outcome && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Outcome</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{outcome}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryEpisodeRecallRenderer({ event }: EventDetailProps) {
	const { query, results_count, top_score } = event.payload as {
		query?: string;
		results_count?: number;
		top_score?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{results_count != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Results</span>
						<span className="font-mono tabular-nums">{results_count}</span>
					</div>
				)}
				{top_score != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Top score</span>
						<span className="font-mono tabular-nums">{top_score.toFixed(3)}</span>
					</div>
				)}
			</div>
			{query && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Query</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{query}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryEpisodeForgetRenderer({ event }: EventDetailProps) {
	const { episode_id } = event.payload as { episode_id?: string };

	return (
		<div className="space-y-3">
			{episode_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Episode</span>
					<span className="font-mono">{episode_id}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryLongtermStoreRenderer({ event }: EventDetailProps) {
	const { key, value, namespace } = event.payload as {
		key?: string;
		value?: string;
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{key && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Key</span>
						<span className="font-mono font-medium">{key}</span>
					</div>
				)}
				{namespace && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Namespace</span>
						<span className="font-mono">{namespace}</span>
					</div>
				)}
			</div>
			{value && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Value</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">{value}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryLongtermRetrieveRenderer({ event }: EventDetailProps) {
	const { key, value, found, namespace } = event.payload as {
		key?: string;
		value?: string;
		found?: boolean;
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{key && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Key</span>
						<span className="font-mono font-medium">{key}</span>
					</div>
				)}
				{found != null && (
					<span
						className={`px-1.5 py-0.5 rounded text-[10px] ${found ? "bg-success-muted text-success" : "bg-warning-muted text-warning"}`}
					>
						{found ? "found" : "not found"}
					</span>
				)}
				{namespace && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Namespace</span>
						<span className="font-mono">{namespace}</span>
					</div>
				)}
			</div>
			{value && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Value</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">{value}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryLongtermDeleteRenderer({ event }: EventDetailProps) {
	const { key, namespace } = event.payload as {
		key?: string;
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{key && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Key</span>
						<span className="font-mono font-medium">{key}</span>
					</div>
				)}
				{namespace && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Namespace</span>
						<span className="font-mono">{namespace}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemoryLongtermListRenderer({ event }: EventDetailProps) {
	const { keys, namespace } = event.payload as {
		keys?: string[];
		namespace?: string;
	};

	return (
		<div className="space-y-3">
			{namespace && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Namespace</span>
					<span className="font-mono">{namespace}</span>
				</div>
			)}
			{keys && keys.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Keys ({keys.length})</span>
					<div className="flex flex-wrap gap-1">
						{keys.map((k) => (
							<span key={k} className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-mono text-muted-foreground">
								{k}
							</span>
						))}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySharedWriteRenderer({ event }: EventDetailProps) {
	const { author, content, scope, entry_id } = event.payload as {
		author?: string;
		content?: string;
		scope?: string;
		entry_id?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{author && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Author</span>
						<span className="font-mono">{author}</span>
					</div>
				)}
				{scope && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Scope</span>
						<span className="font-mono">{scope}</span>
					</div>
				)}
				{entry_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Entry</span>
						<span className="font-mono">{entry_id}</span>
					</div>
				)}
			</div>
			{content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Content</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{content}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySharedReadRenderer({ event }: EventDetailProps) {
	const { scope, author_filter, entries_returned } = event.payload as {
		scope?: string;
		author_filter?: string;
		entries_returned?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{entries_returned != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Entries</span>
						<span className="font-mono tabular-nums">{entries_returned}</span>
					</div>
				)}
				{scope && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Scope</span>
						<span className="font-mono">{scope}</span>
					</div>
				)}
				{author_filter && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Author filter</span>
						<span className="font-mono">{author_filter}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySharedSupersedeRenderer({ event }: EventDetailProps) {
	const { author, original_entry_id, new_entry_id, content, scope } = event.payload as {
		author?: string;
		original_entry_id?: string;
		new_entry_id?: string;
		content?: string;
		scope?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{author && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Author</span>
						<span className="font-mono">{author}</span>
					</div>
				)}
				{scope && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Scope</span>
						<span className="font-mono">{scope}</span>
					</div>
				)}
			</div>
			<div className="flex items-center gap-2 text-xs">
				{original_entry_id && <span className="font-mono text-destructive">{original_entry_id}</span>}
				<span className="text-muted-foreground">→</span>
				{new_entry_id && <span className="font-mono text-success">{new_entry_id}</span>}
			</div>
			{content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">New content</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{content}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MemorySharedRetractRenderer({ event }: EventDetailProps) {
	const { entry_id, author, reason, scope } = event.payload as {
		entry_id?: string;
		author?: string;
		reason?: string;
		scope?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{author && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Author</span>
						<span className="font-mono">{author}</span>
					</div>
				)}
				{entry_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Entry</span>
						<span className="font-mono">{entry_id}</span>
					</div>
				)}
				{scope && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Scope</span>
						<span className="font-mono">{scope}</span>
					</div>
				)}
			</div>
			{reason && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Reason</span>
					<div className="text-sm bg-warning-muted rounded-md p-2">{reason}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Planning Renderers ---

function PlanCreatedRenderer({ event }: EventDetailProps) {
	const { plan_name, step_count, goal_count, steps } = event.payload as {
		plan_name?: string;
		step_count?: number;
		goal_count?: number;
		steps?: Array<{ step_id: string; description: string }>;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{plan_name && <span className="text-sm font-medium">{plan_name}</span>}
				{step_count != null && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{step_count} steps</span>
				)}
				{goal_count != null && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{goal_count} goals</span>
				)}
			</div>
			{steps && steps.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Steps</span>
					<div className="space-y-0.5">
						{steps.map((s, i) => (
							<div key={s.step_id} className="text-sm flex items-baseline gap-2">
								<span className="text-xs text-muted-foreground font-mono tabular-nums w-4 text-right shrink-0">
									{i + 1}.
								</span>
								<span>{s.description}</span>
							</div>
						))}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function PlanStepUpdatedRenderer({ event }: EventDetailProps) {
	const { step_description, previous_status, new_status, has_result } = event.payload as {
		step_description?: string;
		previous_status?: string;
		new_status?: string;
		has_result?: boolean;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{step_description && <span className="text-sm">{step_description}</span>}
				{has_result && (
					<span className="text-[10px] px-1.5 py-0.5 rounded bg-success-muted text-success">has result</span>
				)}
			</div>
			{previous_status && new_status && (
				<div className="flex items-center gap-2 text-xs">
					<StatusBadge status={previous_status} />
					<span className="text-muted-foreground">→</span>
					<StatusBadge status={new_status} />
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function PlanRevisedRenderer({ event }: EventDetailProps) {
	const { steps_before, steps_after, steps_preserved, revision_reason } = event.payload as {
		steps_before?: number;
		steps_after?: number;
		steps_preserved?: number;
		revision_reason?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{steps_before != null && steps_after != null && (
					<div className="flex items-center gap-1.5">
						<span className="font-mono tabular-nums">{steps_before}</span>
						<span className="text-muted-foreground">→</span>
						<span className="font-mono tabular-nums">{steps_after}</span>
						<span className="text-muted-foreground">steps</span>
					</div>
				)}
				{steps_preserved != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Preserved</span>
						<span className="font-mono tabular-nums">{steps_preserved}</span>
					</div>
				)}
			</div>
			{revision_reason && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Reason</span>
					<div className="text-sm bg-warning-muted rounded-md p-2">{revision_reason}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function GoalStatusChangedRenderer({ event }: EventDetailProps) {
	const { goal_description, previous_status, new_status } = event.payload as {
		goal_description?: string;
		previous_status?: string;
		new_status?: string;
	};

	return (
		<div className="space-y-3">
			{goal_description && <div className="text-sm">{goal_description}</div>}
			{previous_status && new_status && (
				<div className="flex items-center gap-2 text-xs">
					<StatusBadge status={previous_status} />
					<span className="text-muted-foreground">→</span>
					<StatusBadge status={new_status} />
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Evaluation & Reflection Renderers ---

function EvaluationResultRenderer({ event }: EventDetailProps) {
	const { evaluator_name, verdict, score, feedback, revision_attempt } = event.payload as {
		evaluator_name?: string;
		verdict?: string;
		score?: number;
		feedback?: string;
		revision_attempt?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{evaluator_name && <span className="text-sm font-medium">{evaluator_name}</span>}
				{verdict && (
					<span
						className={`text-xs px-1.5 py-0.5 rounded ${verdict === "ACCEPT" ? "bg-success-muted text-success" : "bg-warning-muted text-warning"}`}
					>
						{verdict}
					</span>
				)}
				{score != null && <span className="text-xs font-mono tabular-nums">{score.toFixed(2)}</span>}
				{revision_attempt != null && <span className="text-xs text-muted-foreground">attempt {revision_attempt}</span>}
			</div>
			{feedback && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Feedback</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{feedback}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function EvaluationRevisionRenderer({ event }: EventDetailProps) {
	const { feedback, revision_attempt, max_revisions } = event.payload as {
		feedback?: string;
		revision_attempt?: number;
		max_revisions?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{revision_attempt != null && max_revisions != null && (
					<span className="text-muted-foreground">
						Attempt {revision_attempt} of {max_revisions}
					</span>
				)}
			</div>
			{feedback && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Feedback</span>
					<div className="text-sm bg-warning-muted rounded-md p-2">{feedback}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function ReflectionGeneratedRenderer({ event }: EventDetailProps) {
	const { attempt_number, max_attempts, reflection_text, evaluation_feedback, episode_id } = event.payload as {
		attempt_number?: number;
		max_attempts?: number;
		reflection_text?: string;
		evaluation_feedback?: string;
		episode_id?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{attempt_number != null && max_attempts != null && (
					<span className="text-muted-foreground">
						Attempt {attempt_number} of {max_attempts}
					</span>
				)}
				{episode_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Episode</span>
						<span className="font-mono">{episode_id}</span>
					</div>
				)}
			</div>
			{reflection_text && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Reflection</span>
					<div className="text-sm bg-accent-status-muted rounded-md p-2 whitespace-pre-wrap">{reflection_text}</div>
				</div>
			)}
			{evaluation_feedback && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Triggered by feedback</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{evaluation_feedback}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Revision Renderers ---

function RevisionStartRenderer({ event }: EventDetailProps) {
	const { step_name, worker_count, max_revisions } = event.payload as {
		step_name?: string;
		worker_count?: number;
		max_revisions?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">Revision workflow</span>
				{step_name && <span className="text-sm font-medium">{step_name}</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{worker_count != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Workers</span>
						<span className="font-mono tabular-nums">{worker_count}</span>
					</div>
				)}
				{max_revisions != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Max revisions</span>
						<span className="font-mono tabular-nums">{max_revisions}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function RevisionAttemptRenderer({ event }: EventDetailProps) {
	const { attempt_number, feedback } = event.payload as {
		attempt_number?: number;
		feedback?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2 text-xs">
				{attempt_number != null && <span className="text-muted-foreground">Attempt {attempt_number}</span>}
			</div>
			{feedback && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Feedback</span>
					<div className="text-sm bg-warning-muted rounded-md p-2">{feedback}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function RevisionCompleteRenderer({ event }: EventDetailProps) {
	const { final_decision, total_attempts } = event.payload as {
		final_decision?: string;
		total_attempts?: number;
	};

	const decisionColor =
		final_decision === "approve"
			? "bg-success-muted text-success"
			: final_decision === "reject"
				? "bg-destructive-muted text-destructive"
				: final_decision === "max_revisions_exceeded"
					? "bg-warning-muted text-warning"
					: "bg-muted text-muted-foreground";

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{final_decision && <span className={`text-xs px-1.5 py-0.5 rounded ${decisionColor}`}>{final_decision}</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{total_attempts != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Total attempts</span>
						<span className="font-mono tabular-nums">{total_attempts}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- HITL Renderers ---

function HITLRequestRenderer({ event }: EventDetailProps) {
	const { request_id, request_type, prompt, context, agent_name, tool_name } = event.payload as {
		request_id?: string;
		request_type?: string;
		prompt?: string;
		context?: string;
		agent_name?: string;
		tool_name?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{request_type && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{request_type}</span>}
				{request_id && <span className="text-xs font-mono text-muted-foreground">{request_id}</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{agent_name && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Agent</span>
						<span className="font-mono">{agent_name}</span>
					</div>
				)}
				{tool_name && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Tool</span>
						<span className="font-mono">{tool_name}</span>
					</div>
				)}
			</div>
			{prompt && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Prompt</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{prompt}</div>
				</div>
			)}
			{context && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Context</span>
					<div className="text-sm bg-muted/50 rounded-md p-2">{context}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function HITLResponseRenderer({ event }: EventDetailProps) {
	const { request_id, decision, has_content, wait_duration_ms } = event.payload as {
		request_id?: string;
		decision?: string;
		has_content?: boolean;
		wait_duration_ms?: number;
	};

	const decisionColor =
		decision === "approve"
			? "bg-success-muted text-success"
			: decision === "reject"
				? "bg-destructive-muted text-destructive"
				: "bg-warning-muted text-warning";

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{decision && <span className={`text-xs px-1.5 py-0.5 rounded ${decisionColor}`}>{decision}</span>}
				{request_id && <span className="text-xs font-mono text-muted-foreground">{request_id}</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{wait_duration_ms != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Wait</span>
						<span className="font-mono tabular-nums">
							{wait_duration_ms < 1000 ? `${wait_duration_ms}ms` : `${(wait_duration_ms / 1000).toFixed(1)}s`}
						</span>
					</div>
				)}
				{has_content != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Content</span>
						<span>{has_content ? "yes" : "no"}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Tree Search Renderers ---

function TreeSearchNodeCreatedRenderer({ event }: EventDetailProps) {
	const { node_id, parent_id, depth, content, node_type, action, observation, is_terminal, is_failed } =
		event.payload as {
			node_id?: string;
			parent_id?: string | null;
			depth?: number;
			content?: string;
			node_type?: string;
			action?: string;
			observation?: string;
			is_terminal?: boolean;
			is_failed?: boolean;
		};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{node_type && <span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{node_type}</span>}
				{is_terminal && <span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">terminal</span>}
				{is_failed && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-destructive-muted text-destructive">failed</span>
				)}
				{depth != null && <span className="text-xs text-muted-foreground">depth {depth}</span>}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{node_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Node</span>
						<span className="font-mono">{node_id}</span>
					</div>
				)}
				{parent_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Parent</span>
						<span className="font-mono">{parent_id}</span>
					</div>
				)}
			</div>
			{action && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Action</span>
					<span className="font-mono font-medium">{action}</span>
				</div>
			)}
			{content && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Content</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-60 overflow-auto">{content}</div>
				</div>
			)}
			{observation && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Observation</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap max-h-40 overflow-auto">
						{observation}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function TreeSearchNodeEvaluatedRenderer({ event }: EventDetailProps) {
	const { node_id, score, is_terminal } = event.payload as {
		node_id?: string;
		score?: number;
		is_terminal?: boolean;
	};

	const scoreColor =
		score != null
			? score >= 0.7
				? "bg-success-muted text-success"
				: score >= 0.4
					? "bg-warning-muted text-warning"
					: "bg-destructive-muted text-destructive"
			: "";

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{score != null && (
					<span className={`text-xs px-1.5 py-0.5 rounded font-mono tabular-nums ${scoreColor}`}>
						{score.toFixed(2)}
					</span>
				)}
				{is_terminal && <span className="text-xs px-1.5 py-0.5 rounded bg-success-muted text-success">terminal</span>}
			</div>
			{node_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Node</span>
					<span className="font-mono">{node_id}</span>
				</div>
			)}
			{score != null && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Score</span>
					<div className="h-2 bg-muted rounded-full overflow-hidden">
						<div
							className={`h-full rounded-full ${
								score >= 0.7 ? "bg-success" : score >= 0.4 ? "bg-warning" : "bg-destructive"
							}`}
							style={{ width: `${Math.round(score * 100)}%` }}
						/>
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function TreeSearchNodePrunedRenderer({ event }: EventDetailProps) {
	const { node_id, reason } = event.payload as {
		node_id?: string;
		reason?: string;
	};

	return (
		<div className="space-y-3">
			{node_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Node</span>
					<span className="font-mono">{node_id}</span>
				</div>
			)}
			{reason && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Reason</span>
					<div className="text-sm bg-warning-muted rounded-md p-2">{reason}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function TreeSearchCompleteRenderer({ event }: EventDetailProps) {
	const { total_nodes, max_depth_reached, selected_node_id, termination_reason, search_strategy } = event.payload as {
		total_nodes?: number;
		max_depth_reached?: number;
		selected_node_id?: string;
		termination_reason?: string;
		search_strategy?: string;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{search_strategy && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info">{search_strategy}</span>
				)}
				{termination_reason && (
					<StatusBadge status={termination_reason === "solution_found" ? "completed" : termination_reason} />
				)}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{total_nodes != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Nodes</span>
						<span className="font-mono tabular-nums">{total_nodes}</span>
					</div>
				)}
				{max_depth_reached != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Max depth</span>
						<span className="font-mono tabular-nums">{max_depth_reached}</span>
					</div>
				)}
				{selected_node_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Selected</span>
						<span className="font-mono">{selected_node_id}</span>
					</div>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MCTSIterationRenderer({ event }: EventDetailProps) {
	const { iteration_number, selected_node_id, selection_path, expanded_count, best_value_so_far, node_values } =
		event.payload as {
			iteration_number?: number;
			selected_node_id?: string;
			selection_path?: string[];
			expanded_count?: number;
			best_value_so_far?: number;
			node_values?: Record<string, number>;
		};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{iteration_number != null && (
					<span className="text-xs px-1.5 py-0.5 rounded bg-info-muted text-info font-mono">#{iteration_number}</span>
				)}
				{best_value_so_far != null && (
					<span className="text-xs font-mono tabular-nums">best={best_value_so_far.toFixed(2)}</span>
				)}
			</div>
			<div className="flex items-center gap-4 text-xs">
				{expanded_count != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Expanded</span>
						<span className="font-mono tabular-nums">{expanded_count}</span>
					</div>
				)}
				{selected_node_id && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Selected</span>
						<span className="font-mono">{selected_node_id}</span>
					</div>
				)}
			</div>
			{selection_path && selection_path.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Selection path</span>
					<div className="flex flex-wrap items-center gap-1">
						{selection_path.map((nodeId, i) => (
							<span key={nodeId} className="flex items-center gap-1">
								<span className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-mono text-muted-foreground">
									{nodeId}
								</span>
								{i < selection_path.length - 1 && <span className="text-muted-foreground">→</span>}
							</span>
						))}
					</div>
				</div>
			)}
			{node_values && Object.keys(node_values).length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Node values ({Object.keys(node_values).length})</span>
					<div className="grid grid-cols-2 gap-1 text-xs">
						{Object.entries(node_values).map(([nodeId, value]) => (
							<div key={nodeId} className="flex items-center gap-1.5">
								<span className="font-mono text-muted-foreground">{nodeId}</span>
								<span className="font-mono tabular-nums">{value.toFixed(2)}</span>
							</div>
						))}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function MCTSBackpropagationRenderer({ event }: EventDetailProps) {
	const { propagated_value, path_length, updated_node_ids } = event.payload as {
		propagated_value?: number;
		path_length?: number;
		updated_node_ids?: string[];
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-4 text-xs">
				{propagated_value != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Value</span>
						<span className="font-mono tabular-nums">{propagated_value.toFixed(2)}</span>
					</div>
				)}
				{path_length != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Path length</span>
						<span className="font-mono tabular-nums">{path_length}</span>
					</div>
				)}
			</div>
			{updated_node_ids && updated_node_ids.length > 0 && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Updated nodes</span>
					<div className="flex flex-wrap items-center gap-1">
						{updated_node_ids.map((nodeId, i) => (
							<span key={nodeId} className="flex items-center gap-1">
								<span className="text-[10px] px-1.5 py-0.5 rounded bg-muted font-mono text-muted-foreground">
									{nodeId}
								</span>
								{i < updated_node_ids.length - 1 && <span className="text-muted-foreground">→</span>}
							</span>
						))}
					</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Span Renderers ---

function SpanRenderer({ event }: EventDetailProps) {
	const { name, span_id, duration_ms } = event.payload as {
		name?: string;
		span_id?: string;
		duration_ms?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{name && <span className="text-sm font-medium">{name}</span>}
				{event.event_type === "span.end" && duration_ms != null && (
					<span className="text-xs text-muted-foreground font-mono tabular-nums">
						{duration_ms < 1000 ? `${Math.round(duration_ms)}ms` : `${(duration_ms / 1000).toFixed(1)}s`}
					</span>
				)}
			</div>
			{span_id && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Span ID</span>
					<span className="font-mono">{span_id}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Code Execution Renderers ---

function CodeExecutionRenderer({ event }: EventDetailProps) {
	const { agent_name, code, step_number } = event.payload as {
		agent_name?: string;
		code?: string;
		step_number?: number;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{step_number != null && (
					<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">Step {step_number}</span>
				)}
				{agent_name && <span className="text-sm font-mono">{agent_name}</span>}
			</div>
			{code && <CodeBlock code={code} language="python" maxHeight={400} />}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function CodeExecutionResultRenderer({ event }: EventDetailProps) {
	const { stdout, stderr, return_value, success, error, duration_ms, step_number } = event.payload as {
		agent_name?: string;
		stdout?: string;
		stderr?: string;
		return_value?: string | null;
		success?: boolean;
		error?: string | null;
		duration_ms?: number;
		step_number?: number;
	};

	const [stdoutExpanded, setStdoutExpanded] = useState(false);
	const isLongStdout = (stdout?.length ?? 0) > 200;

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				{step_number != null && (
					<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">Step {step_number}</span>
				)}
				{success != null && (
					<span
						className={`text-xs px-1.5 py-0.5 rounded ${
							success ? "bg-success-muted text-success" : "bg-destructive-muted text-destructive"
						}`}
					>
						{success ? "success" : "failed"}
					</span>
				)}
				{duration_ms != null && (
					<span className="text-xs text-muted-foreground font-mono tabular-nums">
						{duration_ms < 1000 ? `${Math.round(duration_ms)}ms` : `${(duration_ms / 1000).toFixed(1)}s`}
					</span>
				)}
			</div>

			{/* stdout */}
			{stdout && (
				<div className="space-y-0.5">
					<div className="flex items-center gap-1">
						<span className="text-xs text-muted-foreground font-mono">stdout:</span>
						{isLongStdout && (
							<button
								type="button"
								className="text-[10px] text-info-muted-foreground hover:underline"
								onClick={() => setStdoutExpanded(!stdoutExpanded)}
							>
								{stdoutExpanded ? "collapse" : "expand"}
							</button>
						)}
					</div>
					<pre
						className={`text-xs font-mono bg-muted/50 rounded p-2 whitespace-pre-wrap ${
							!stdoutExpanded && isLongStdout ? "max-h-20 overflow-hidden" : ""
						}`}
					>
						{stdout}
					</pre>
				</div>
			)}

			{/* stderr */}
			{stderr && (
				<div className="space-y-0.5">
					<span className="text-xs text-destructive font-mono">stderr:</span>
					<pre className="text-xs font-mono bg-destructive-muted text-destructive-muted-foreground rounded p-2 whitespace-pre-wrap">
						{stderr}
					</pre>
				</div>
			)}

			{/* return_value */}
			{return_value != null && (
				<div className="space-y-0.5">
					<span className="text-xs text-muted-foreground font-mono">return_value:</span>
					<pre className="text-xs font-mono bg-muted/50 rounded p-2 whitespace-pre-wrap">{return_value}</pre>
				</div>
			)}

			{/* error */}
			{error && (
				<div className="space-y-0.5">
					<span className="text-xs text-destructive font-mono">error:</span>
					<pre className="text-xs font-mono bg-destructive-muted text-destructive-muted-foreground rounded p-2 whitespace-pre-wrap">
						{error}
					</pre>
				</div>
			)}

			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Error Recovery Renderers ---

function ErrorRetryRenderer({ event }: EventDetailProps) {
	const { error_type, error_message, attempt, max_attempts, delay_ms, category } = event.payload as {
		error_type?: string;
		error_message?: string;
		attempt?: number;
		max_attempts?: number;
		delay_ms?: number;
		category?: string;
	};

	return (
		<div className="space-y-3">
			{(error_type || error_message) && (
				<div className="text-sm">
					{error_type && <span className="font-mono font-medium">{error_type}</span>}
					{error_type && error_message && <span className="text-muted-foreground">: </span>}
					{error_message && <span className="text-muted-foreground">{error_message}</span>}
				</div>
			)}
			<div className="flex items-center gap-3 text-xs">
				{attempt != null && max_attempts != null && (
					<span className="px-1.5 py-0.5 rounded bg-warning-muted text-warning">
						attempt {attempt}/{max_attempts}
					</span>
				)}
				{delay_ms != null && (
					<div className="flex items-center gap-1.5">
						<span className="text-muted-foreground">Delay</span>
						<span className="font-mono tabular-nums">{delay_ms}ms</span>
					</div>
				)}
				{category && (
					<span className="px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-mono text-[10px]">{category}</span>
				)}
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function ErrorCorrectionRenderer({ event }: EventDetailProps) {
	const { error_type, error_message, attempt, max_attempts, correction_prompt } = event.payload as {
		error_type?: string;
		error_message?: string;
		attempt?: number;
		max_attempts?: number;
		correction_prompt?: string;
	};

	return (
		<div className="space-y-3">
			{(error_type || error_message) && (
				<div className="text-sm">
					{error_type && <span className="font-mono font-medium">{error_type}</span>}
					{error_type && error_message && <span className="text-muted-foreground">: </span>}
					{error_message && <span className="text-muted-foreground">{error_message}</span>}
				</div>
			)}
			{attempt != null && max_attempts != null && (
				<span className="text-xs px-1.5 py-0.5 rounded bg-warning-muted text-warning">
					attempt {attempt}/{max_attempts}
				</span>
			)}
			{correction_prompt && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Correction prompt</span>
					<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap">{correction_prompt}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function ErrorDegradationRenderer({ event }: EventDetailProps) {
	const { error_type, error_message, degradation_message } = event.payload as {
		error_type?: string;
		error_message?: string;
		degradation_message?: string;
	};

	return (
		<div className="space-y-3">
			{(error_type || error_message) && (
				<div className="text-sm">
					{error_type && <span className="font-mono font-medium">{error_type}</span>}
					{error_type && error_message && <span className="text-muted-foreground">: </span>}
					{error_message && <span className="text-muted-foreground">{error_message}</span>}
				</div>
			)}
			{degradation_message && (
				<div className="space-y-1">
					<span className="text-xs text-muted-foreground">Degradation</span>
					<div className="text-sm bg-warning-muted rounded-md p-2">{degradation_message}</div>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Safety Renderers ---

function SafetyIterationLimitRenderer({ event }: EventDetailProps) {
	const { agent_name, current_iteration, max_iterations, step_number } = event.payload as {
		agent_name?: string;
		current_iteration?: number;
		max_iterations?: number;
		step_number?: number;
	};

	const pct =
		current_iteration != null && max_iterations != null && max_iterations > 0
			? Math.min(100, Math.round((current_iteration / max_iterations) * 100))
			: null;

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<StatusBadge status="suspended" />
				<span className="text-sm font-medium">Iteration limit reached</span>
			</div>
			{agent_name && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Agent</span>
					<span className="font-mono">{agent_name}</span>
				</div>
			)}
			{current_iteration != null && max_iterations != null && (
				<div className="space-y-1.5">
					<div className="flex items-center justify-between text-xs">
						<span className="text-muted-foreground">Iterations</span>
						<span className="font-mono tabular-nums">
							{current_iteration} / {max_iterations}
						</span>
					</div>
					{pct != null && (
						<div className="h-1.5 w-full rounded-full bg-muted">
							<div className="h-full rounded-full bg-warning" style={{ width: `${pct}%` }} />
						</div>
					)}
				</div>
			)}
			{step_number != null && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">At step</span>
					<span className="font-mono tabular-nums">{step_number}</span>
				</div>
			)}
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

function SafetyCancellationRenderer({ event }: EventDetailProps) {
	const { agent_name, step_number } = event.payload as {
		agent_name?: string;
		step_number?: number | null;
	};

	return (
		<div className="space-y-3">
			<div className="flex items-center gap-2">
				<StatusBadge status="suspended" />
				<span className="text-sm font-medium">Execution cancelled</span>
			</div>
			{agent_name && (
				<div className="flex items-center gap-1.5 text-xs">
					<span className="text-muted-foreground">Agent</span>
					<span className="font-mono">{agent_name}</span>
				</div>
			)}
			<div className="flex items-center gap-1.5 text-xs">
				<span className="text-muted-foreground">Step</span>
				<span className="font-mono tabular-nums">{step_number != null ? step_number : "before execution"}</span>
			</div>
			<PayloadViewer payload={event.payload} />
		</div>
	);
}

// --- Summary helpers ---

function llmRequestSummary(event: TraceEvent): string {
	const model = event.payload.model_name as string | undefined;
	return model ? `→ ${model}` : "LLM Request";
}

function llmResponseSummary(event: TraceEvent): string {
	const usage = event.payload.usage as { input_tokens: number; output_tokens: number } | undefined;
	if (usage) {
		return `${usage.input_tokens}+${usage.output_tokens} tokens`;
	}
	return "LLM Response";
}

function toolInvokeSummary(event: TraceEvent): string {
	const name = event.payload.tool_name as string | undefined;
	if (!name) return "Tool Invoke";
	const params = event.payload.parameters as Record<string, unknown> | undefined;
	if (params) {
		const keys = Object.keys(params);
		if (keys.length > 0) {
			const summary = keys.slice(0, 2).join(", ");
			return `→ ${name}(${summary}${keys.length > 2 ? ", …" : ""})`;
		}
	}
	return `→ ${name}()`;
}

function toolResultSummary(event: TraceEvent): string {
	const name = event.payload.tool_name as string | undefined;
	const success = event.payload.success as boolean | undefined;
	if (!name) return "Tool Result";
	if (success === false) {
		const error = event.payload.error as string | undefined;
		return `✗ ${name}${error ? `: ${error}` : ""}`;
	}
	return `✓ ${name}`;
}

function agentStartSummary(event: TraceEvent): string {
	const name = event.payload.agent_name as string | undefined;
	const type = event.payload.agent_type as string | undefined;
	if (!name) return "Agent Start";
	return type ? `${name} (${type})` : name;
}

function agentStepSummary(event: TraceEvent): string {
	const step = event.payload.step as number | undefined;
	const action = event.payload.action as string | undefined;
	if (step != null && action) return `Step ${step}: ${action}`;
	if (step != null) return `Step ${step}`;
	return "Agent Step";
}

function agentCompleteSummary(event: TraceEvent): string {
	const reason = event.payload.termination_reason as string | undefined;
	return reason ? `Completed: ${reason}` : "Agent Complete";
}

function spanSummary(event: TraceEvent): string {
	const name = event.payload.name as string | undefined;
	const duration = event.payload.duration_ms as number | undefined;
	if (name && duration != null) {
		const formatted = duration < 1000 ? `${Math.round(duration)}ms` : `${(duration / 1000).toFixed(1)}s`;
		return `${name} (${formatted})`;
	}
	if (name) return name;
	return event.event_type;
}

// --- Tree Search summaries ---

function treeSearchNodeCreatedSummary(event: TraceEvent): string {
	const preview = event.payload.content as string | undefined;
	const depth = event.payload.depth as number | undefined;
	const truncated = preview && preview.length > 60 ? `${preview.slice(0, 60)}…` : preview;
	if (truncated && depth != null) return `Node created: ${truncated} (depth ${depth})`;
	if (truncated) return `Node created: ${truncated}`;
	return "Node created";
}

function treeSearchNodeEvaluatedSummary(event: TraceEvent): string {
	const score = event.payload.score as number | undefined;
	const isTerminal = event.payload.is_terminal as boolean | undefined;
	const parts: string[] = [];
	if (score != null) parts.push(score.toFixed(2));
	if (isTerminal) parts.push("terminal");
	if (parts.length > 0) return `Evaluated: ${parts.join(", ")}`;
	return "Evaluated";
}

function treeSearchNodePrunedSummary(event: TraceEvent): string {
	const reason = event.payload.reason as string | undefined;
	return reason ? `Pruned: ${reason}` : "Pruned";
}

function treeSearchCompleteSummary(event: TraceEvent): string {
	const reason = event.payload.termination_reason as string | undefined;
	const total = event.payload.total_nodes as number | undefined;
	if (reason && total != null) return `Search complete: ${reason} (${total} nodes)`;
	if (reason) return `Search complete: ${reason}`;
	return "Search complete";
}

function mctsIterationSummary(event: TraceEvent): string {
	const num = event.payload.iteration_number as number | undefined;
	const expanded = event.payload.expanded_count as number | undefined;
	const best = event.payload.best_value_so_far as number | undefined;
	if (num != null && expanded != null && best != null) {
		return `MCTS #${num}: expanded ${expanded}, best=${best.toFixed(2)}`;
	}
	if (num != null) return `MCTS #${num}`;
	return "MCTS iteration";
}

function mctsBackpropagationSummary(event: TraceEvent): string {
	const value = event.payload.propagated_value as number | undefined;
	const pathLen = event.payload.path_length as number | undefined;
	if (value != null && pathLen != null) {
		return `Backprop: ${value.toFixed(2)} through ${pathLen} nodes`;
	}
	return "Backpropagation";
}

// --- Memory summaries ---

function memoryWorkingReadSummary(event: TraceEvent): string {
	const tokens = event.payload.token_count as number | undefined;
	return tokens != null ? `Working memory read (${tokens} tokens)` : "Working memory read";
}

function memoryWorkingUpdateSummary(event: TraceEvent): string {
	const source = event.payload.source as string | undefined;
	return source ? `Working memory updated (source: ${source})` : "Working memory updated";
}

function memorySemanticStoreSummary(): string {
	return "Stored in semantic memory";
}

function memorySemanticSearchSummary(event: TraceEvent): string {
	const count = event.payload.results_count as number | undefined;
	const top = event.payload.top_score as number | undefined;
	if (count != null && top != null) return `Semantic search: ${count} results (top: ${top.toFixed(3)})`;
	if (count != null) return `Semantic search: ${count} results`;
	return "Semantic search";
}

function memorySemanticDeleteSummary(): string {
	return "Deleted from semantic memory";
}

function memoryEpisodeRecordSummary(): string {
	return "Episode recorded";
}

function memoryEpisodeRecallSummary(event: TraceEvent): string {
	const count = event.payload.results_count as number | undefined;
	const top = event.payload.top_score as number | undefined;
	if (count != null && top != null) return `Episode recall: ${count} results (top: ${top.toFixed(3)})`;
	if (count != null) return `Episode recall: ${count} results`;
	return "Episode recall";
}

function memoryEpisodeForgetSummary(): string {
	return "Episode forgotten";
}

function memoryLongtermStoreSummary(event: TraceEvent): string {
	const key = event.payload.key as string | undefined;
	return key ? `Stored: ${key}` : "Stored in long-term memory";
}

function memoryLongtermRetrieveSummary(event: TraceEvent): string {
	const key = event.payload.key as string | undefined;
	const found = event.payload.found as boolean | undefined;
	if (key && found != null) return `Retrieved: ${key} (${found ? "found" : "not found"})`;
	if (key) return `Retrieved: ${key}`;
	return "Retrieved from long-term memory";
}

function memoryLongtermDeleteSummary(event: TraceEvent): string {
	const key = event.payload.key as string | undefined;
	return key ? `Deleted: ${key}` : "Deleted from long-term memory";
}

function memoryLongtermListSummary(event: TraceEvent): string {
	const keys = event.payload.keys as string[] | undefined;
	return keys ? `Listed keys (${keys.length})` : "Listed keys";
}

function memorySharedWriteSummary(event: TraceEvent): string {
	const author = event.payload.author as string | undefined;
	return author ? `Shared write by ${author}` : "Shared write";
}

function memorySharedReadSummary(event: TraceEvent): string {
	const count = event.payload.entries_returned as number | undefined;
	return count != null ? `Shared read (${count} entries)` : "Shared read";
}

function memorySharedSupersedeSummary(event: TraceEvent): string {
	const author = event.payload.author as string | undefined;
	return author ? `Shared supersede by ${author}` : "Shared supersede";
}

function memorySharedRetractSummary(event: TraceEvent): string {
	const author = event.payload.author as string | undefined;
	return author ? `Shared retract by ${author}` : "Shared retract";
}

// --- Planning summaries ---

function planCreatedSummary(event: TraceEvent): string {
	const name = event.payload.plan_name as string | undefined;
	const count = event.payload.step_count as number | undefined;
	if (name && count != null) return `Plan created: ${name} (${count} steps)`;
	if (name) return `Plan created: ${name}`;
	return "Plan created";
}

function planStepUpdatedSummary(event: TraceEvent): string {
	const desc = event.payload.step_description as string | undefined;
	const prev = event.payload.previous_status as string | undefined;
	const next = event.payload.new_status as string | undefined;
	if (desc && prev && next) return `Step '${desc}': ${prev} → ${next}`;
	if (desc) return `Step '${desc}'`;
	return "Step updated";
}

function planRevisedSummary(event: TraceEvent): string {
	const before = event.payload.steps_before as number | undefined;
	const after = event.payload.steps_after as number | undefined;
	if (before != null && after != null) return `Plan revised (${before} → ${after} steps)`;
	return "Plan revised";
}

function goalStatusChangedSummary(event: TraceEvent): string {
	const desc = event.payload.goal_description as string | undefined;
	const prev = event.payload.previous_status as string | undefined;
	const next = event.payload.new_status as string | undefined;
	if (desc && prev && next) return `Goal '${desc}': ${prev} → ${next}`;
	if (desc) return `Goal '${desc}'`;
	return "Goal status changed";
}

// --- Evaluation & Reflection summaries ---

function evaluationResultSummary(event: TraceEvent): string {
	const verdict = event.payload.verdict as string | undefined;
	const score = event.payload.score as number | undefined;
	if (verdict && score != null) return `Evaluation: ${verdict} (score: ${score.toFixed(2)})`;
	if (verdict) return `Evaluation: ${verdict}`;
	return "Evaluation result";
}

function evaluationRevisionSummary(event: TraceEvent): string {
	const attempt = event.payload.revision_attempt as number | undefined;
	const max = event.payload.max_revisions as number | undefined;
	if (attempt != null && max != null) return `Revision requested (attempt ${attempt}/${max})`;
	return "Revision requested";
}

function reflectionGeneratedSummary(event: TraceEvent): string {
	const attempt = event.payload.attempt_number as number | undefined;
	const max = event.payload.max_attempts as number | undefined;
	if (attempt != null && max != null) return `Reflection generated (attempt ${attempt}/${max})`;
	return "Reflection generated";
}

// --- Revision summaries ---

function revisionStartSummary(event: TraceEvent): string {
	const stepName = event.payload.step_name as string | undefined;
	const workerCount = event.payload.worker_count as number | undefined;
	const maxRevisions = event.payload.max_revisions as number | undefined;
	if (stepName && workerCount != null && maxRevisions != null)
		return `Revision: ${stepName} (${workerCount} workers, max ${maxRevisions})`;
	if (stepName) return `Revision: ${stepName}`;
	return "Revision workflow started";
}

function revisionAttemptSummary(event: TraceEvent): string {
	const attempt = event.payload.attempt_number as number | undefined;
	const feedback = event.payload.feedback as string | undefined;
	if (attempt != null && feedback) {
		const truncated = feedback.length > 50 ? `${feedback.slice(0, 50)}\u2026` : feedback;
		return `Revision attempt ${attempt}: ${truncated}`;
	}
	if (attempt != null) return `Revision attempt ${attempt}`;
	return "Revision attempt";
}

function revisionCompleteSummary(event: TraceEvent): string {
	const decision = event.payload.final_decision as string | undefined;
	const attempts = event.payload.total_attempts as number | undefined;
	if (decision && attempts != null) return `Revision complete: ${decision} (${attempts} attempts)`;
	if (decision) return `Revision complete: ${decision}`;
	return "Revision complete";
}

// --- HITL summaries ---

function hitlRequestSummary(event: TraceEvent): string {
	const type = event.payload.request_type as string | undefined;
	const prompt = event.payload.prompt as string | undefined;
	if (type && prompt) {
		const truncated = prompt.length > 50 ? `${prompt.slice(0, 50)}…` : prompt;
		return `HITL: ${type} — ${truncated}`;
	}
	if (type) return `HITL: ${type}`;
	return "HITL request";
}

function hitlResponseSummary(event: TraceEvent): string {
	const decision = event.payload.decision as string | undefined;
	const wait = event.payload.wait_duration_ms as number | undefined;
	if (decision && wait != null) return `HITL response: ${decision} (waited ${wait}ms)`;
	if (decision) return `HITL response: ${decision}`;
	return "HITL response";
}

// --- Code Execution summaries ---

function codeExecutionSummary(event: TraceEvent): string {
	const step = event.payload.step_number as number | undefined;
	return step != null ? `Step ${step}: execute code` : "Execute code";
}

function codeExecutionResultSummary(event: TraceEvent): string {
	const step = event.payload.step_number as number | undefined;
	const success = event.payload.success as boolean | undefined;
	const duration = event.payload.duration_ms as number | undefined;
	const status = success === true ? "success" : success === false ? "failed" : "unknown";
	const prefix = step != null ? `Step ${step}: ` : "";
	if (duration != null) return `${prefix}${status} (${duration}ms)`;
	return `${prefix}${status}`;
}

// --- Safety summaries ---

function errorRetrySummary(event: TraceEvent): string {
	const attempt = event.payload.attempt as number | undefined;
	const max = event.payload.max_attempts as number | undefined;
	const errorType = event.payload.error_type as string | undefined;
	const prefix = attempt != null && max != null ? `Retry attempt ${attempt}/${max}` : "Retry";
	if (errorType) return `${prefix} — ${errorType}`;
	return prefix;
}

function errorCorrectionSummary(event: TraceEvent): string {
	const attempt = event.payload.attempt as number | undefined;
	const max = event.payload.max_attempts as number | undefined;
	const errorType = event.payload.error_type as string | undefined;
	const prefix = attempt != null && max != null ? `Correction attempt ${attempt}/${max}` : "Correction";
	if (errorType) return `${prefix} — ${errorType}`;
	return prefix;
}

function errorDegradationSummary(event: TraceEvent): string {
	const errorType = event.payload.error_type as string | undefined;
	if (errorType) return `Degraded — ${errorType}`;
	return "Degraded";
}

function safetyIterationLimitSummary(event: TraceEvent): string {
	const current = event.payload.current_iteration as number | undefined;
	const max = event.payload.max_iterations as number | undefined;
	if (current != null && max != null) return `Iteration limit: ${current}/${max}`;
	return "Iteration limit reached";
}

function safetyCancellationSummary(event: TraceEvent): string {
	const step = event.payload.step_number as number | null | undefined;
	if (step != null) return `Cancelled at step ${step}`;
	return "Cancelled before execution";
}

// --- Registration ---

/** Creates all default renderer registrations. */
export function createDefaultRegistrations(): EventRendererRegistration[] {
	return [
		// Fallback — lowest priority
		{
			matches: () => true,
			priority: -1,
			component: GenericPayloadRenderer,
		},
		// LLM
		{
			matches: (t) => t === "llm.request",
			priority: 0,
			component: LLMRequestRenderer,
			summary: llmRequestSummary,
		},
		{
			matches: (t) => t === "llm.response",
			priority: 0,
			component: LLMResponseRenderer,
			summary: llmResponseSummary,
		},
		// Tool
		{
			matches: (t) => t === "tool.invoke",
			priority: 0,
			component: ToolInvokeRenderer,
			summary: toolInvokeSummary,
		},
		{
			matches: (t) => t === "tool.result",
			priority: 0,
			component: ToolResultRenderer,
			summary: toolResultSummary,
		},
		// Agent
		{
			matches: (t) => t === "agent.start",
			priority: 0,
			component: AgentStartRenderer,
			summary: agentStartSummary,
		},
		{
			matches: (t) => t === "agent.step",
			priority: 0,
			component: AgentStepRenderer,
			summary: agentStepSummary,
		},
		{
			matches: (t) => t === "agent.complete",
			priority: 0,
			component: AgentCompleteRenderer,
			summary: agentCompleteSummary,
		},
		// Span
		{
			matches: (t) => t === "span.start" || t === "span.end",
			priority: 0,
			component: SpanRenderer,
			summary: spanSummary,
		},
		// Memory — Working
		{
			matches: (t) => t === "memory.working.read",
			priority: 0,
			component: MemoryWorkingReadRenderer,
			summary: memoryWorkingReadSummary,
		},
		{
			matches: (t) => t === "memory.working.update",
			priority: 0,
			component: MemoryWorkingUpdateRenderer,
			summary: memoryWorkingUpdateSummary,
		},
		// Memory — Semantic
		{
			matches: (t) => t === "memory.semantic.store",
			priority: 0,
			component: MemorySemanticStoreRenderer,
			summary: memorySemanticStoreSummary,
		},
		{
			matches: (t) => t === "memory.semantic.search",
			priority: 0,
			component: MemorySemanticSearchRenderer,
			summary: memorySemanticSearchSummary,
		},
		{
			matches: (t) => t === "memory.semantic.delete",
			priority: 0,
			component: MemorySemanticDeleteRenderer,
			summary: memorySemanticDeleteSummary,
		},
		// Memory — Episodic
		{
			matches: (t) => t === "memory.episode.record",
			priority: 0,
			component: MemoryEpisodeRecordRenderer,
			summary: memoryEpisodeRecordSummary,
		},
		{
			matches: (t) => t === "memory.episode.recall",
			priority: 0,
			component: MemoryEpisodeRecallRenderer,
			summary: memoryEpisodeRecallSummary,
		},
		{
			matches: (t) => t === "memory.episode.forget",
			priority: 0,
			component: MemoryEpisodeForgetRenderer,
			summary: memoryEpisodeForgetSummary,
		},
		// Memory — Long-Term
		{
			matches: (t) => t === "memory.longterm.store",
			priority: 0,
			component: MemoryLongtermStoreRenderer,
			summary: memoryLongtermStoreSummary,
		},
		{
			matches: (t) => t === "memory.longterm.retrieve",
			priority: 0,
			component: MemoryLongtermRetrieveRenderer,
			summary: memoryLongtermRetrieveSummary,
		},
		{
			matches: (t) => t === "memory.longterm.delete",
			priority: 0,
			component: MemoryLongtermDeleteRenderer,
			summary: memoryLongtermDeleteSummary,
		},
		{
			matches: (t) => t === "memory.longterm.list",
			priority: 0,
			component: MemoryLongtermListRenderer,
			summary: memoryLongtermListSummary,
		},
		// Memory — Shared
		{
			matches: (t) => t === "memory.shared.write",
			priority: 0,
			component: MemorySharedWriteRenderer,
			summary: memorySharedWriteSummary,
		},
		{
			matches: (t) => t === "memory.shared.read",
			priority: 0,
			component: MemorySharedReadRenderer,
			summary: memorySharedReadSummary,
		},
		{
			matches: (t) => t === "memory.shared.supersede",
			priority: 0,
			component: MemorySharedSupersedeRenderer,
			summary: memorySharedSupersedeSummary,
		},
		{
			matches: (t) => t === "memory.shared.retract",
			priority: 0,
			component: MemorySharedRetractRenderer,
			summary: memorySharedRetractSummary,
		},
		// Tree Search
		{
			matches: (t) => t === "tree_search.node.created",
			priority: 0,
			component: TreeSearchNodeCreatedRenderer,
			summary: treeSearchNodeCreatedSummary,
		},
		{
			matches: (t) => t === "tree_search.node.evaluated",
			priority: 0,
			component: TreeSearchNodeEvaluatedRenderer,
			summary: treeSearchNodeEvaluatedSummary,
		},
		{
			matches: (t) => t === "tree_search.node.pruned",
			priority: 0,
			component: TreeSearchNodePrunedRenderer,
			summary: treeSearchNodePrunedSummary,
		},
		{
			matches: (t) => t === "tree_search.complete",
			priority: 0,
			component: TreeSearchCompleteRenderer,
			summary: treeSearchCompleteSummary,
		},
		// MCTS
		{
			matches: (t) => t === "mcts.iteration",
			priority: 0,
			component: MCTSIterationRenderer,
			summary: mctsIterationSummary,
		},
		{
			matches: (t) => t === "mcts.backpropagation",
			priority: 0,
			component: MCTSBackpropagationRenderer,
			summary: mctsBackpropagationSummary,
		},
		// Planning
		{
			matches: (t) => t === "planning.plan.created",
			priority: 0,
			component: PlanCreatedRenderer,
			summary: planCreatedSummary,
		},
		{
			matches: (t) => t === "planning.step.updated",
			priority: 0,
			component: PlanStepUpdatedRenderer,
			summary: planStepUpdatedSummary,
		},
		{
			matches: (t) => t === "planning.plan.revised",
			priority: 0,
			component: PlanRevisedRenderer,
			summary: planRevisedSummary,
		},
		{
			matches: (t) => t === "planning.goal.status_changed",
			priority: 0,
			component: GoalStatusChangedRenderer,
			summary: goalStatusChangedSummary,
		},
		// Evaluation & Reflection
		{
			matches: (t) => t === "evaluation.result",
			priority: 0,
			component: EvaluationResultRenderer,
			summary: evaluationResultSummary,
		},
		{
			matches: (t) => t === "evaluation.revision",
			priority: 0,
			component: EvaluationRevisionRenderer,
			summary: evaluationRevisionSummary,
		},
		{
			matches: (t) => t === "reflection.generated",
			priority: 0,
			component: ReflectionGeneratedRenderer,
			summary: reflectionGeneratedSummary,
		},
		// Revision
		{
			matches: (t) => t === "revision.start",
			priority: 0,
			component: RevisionStartRenderer,
			summary: revisionStartSummary,
		},
		{
			matches: (t) => t === "revision.attempt",
			priority: 0,
			component: RevisionAttemptRenderer,
			summary: revisionAttemptSummary,
		},
		{
			matches: (t) => t === "revision.complete",
			priority: 0,
			component: RevisionCompleteRenderer,
			summary: revisionCompleteSummary,
		},
		// HITL
		{
			matches: (t) => t === "hitl.request",
			priority: 0,
			component: HITLRequestRenderer,
			summary: hitlRequestSummary,
		},
		{
			matches: (t) => t === "hitl.response",
			priority: 0,
			component: HITLResponseRenderer,
			summary: hitlResponseSummary,
		},
		// Code Execution
		{
			matches: (t) => t === "code.execution",
			priority: 0,
			component: CodeExecutionRenderer,
			summary: codeExecutionSummary,
		},
		{
			matches: (t) => t === "code.execution.result",
			priority: 0,
			component: CodeExecutionResultRenderer,
			summary: codeExecutionResultSummary,
		},
		// Safety
		{
			matches: (t) => t === "safety.iteration_limit",
			priority: 0,
			component: SafetyIterationLimitRenderer,
			summary: safetyIterationLimitSummary,
		},
		{
			matches: (t) => t === "safety.cancellation",
			priority: 0,
			component: SafetyCancellationRenderer,
			summary: safetyCancellationSummary,
		},
		// Error Recovery
		{
			matches: (t) => t === "error.retry",
			priority: 0,
			component: ErrorRetryRenderer,
			summary: errorRetrySummary,
		},
		{
			matches: (t) => t === "error.correction",
			priority: 0,
			component: ErrorCorrectionRenderer,
			summary: errorCorrectionSummary,
		},
		{
			matches: (t) => t === "error.degradation",
			priority: 0,
			component: ErrorDegradationRenderer,
			summary: errorDegradationSummary,
		},
		// Workflow + Multi-Agent
		...createWorkflowRegistrations(),
		// Run Lifecycle
		...createRunRegistrations(),
	];
}

/** Creates a registry pre-loaded with all default renderers. */
export function createDefaultRegistry(): EventRendererRegistry {
	const registry = new EventRendererRegistry();
	for (const reg of createDefaultRegistrations()) {
		registry.register(reg);
	}
	return registry;
}

/** Creates all three registries with default registrations. */
export function createDefaultRegistries(): {
	registry: EventRendererRegistry;
	agentViewRegistry: AgentViewRegistry;
	panelRegistry: CapabilityPanelRegistry;
} {
	return {
		registry: createDefaultRegistry(),
		agentViewRegistry: createDefaultAgentViewRegistry(),
		panelRegistry: createDefaultPanelRegistry(),
	};
}
