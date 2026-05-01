import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";

interface ToolCall {
	invoke: TraceEvent;
	result: TraceEvent | null;
}

interface ToolStats {
	toolName: string;
	calls: number;
	successes: number;
	errors: number;
	totalDuration: number;
	durations: number[];
}

function collectToolCalls(events: TraceEvent[]): ToolCall[] {
	const invokes = events.filter((e) => e.event_type === "tool.invoke");
	const results = events.filter((e) => e.event_type === "tool.result");

	return invokes.map((invoke) => {
		const invokePayload = invoke.payload as { tool_call_id?: string; tool_name?: string };
		const result =
			results.find((r) => {
				const rp = r.payload as { tool_call_id?: string; tool_name?: string };
				// Match by tool_call_id if available, otherwise by tool_name and temporal proximity
				if (invokePayload.tool_call_id && rp.tool_call_id) {
					return invokePayload.tool_call_id === rp.tool_call_id;
				}
				return (
					rp.tool_name === invokePayload.tool_name &&
					new Date(r.timestamp).getTime() >= new Date(invoke.timestamp).getTime()
				);
			}) ?? null;

		return { invoke, result };
	});
}

function aggregateStats(calls: ToolCall[]): ToolStats[] {
	const byTool = new Map<string, ToolStats>();

	for (const call of calls) {
		const toolName = (call.invoke.payload as { tool_name?: string }).tool_name ?? "unknown";
		let stats = byTool.get(toolName);
		if (!stats) {
			stats = { toolName, calls: 0, successes: 0, errors: 0, totalDuration: 0, durations: [] };
			byTool.set(toolName, stats);
		}
		stats.calls++;

		if (call.result) {
			const rp = call.result.payload as { success?: boolean; duration_ms?: number };
			if (rp.success) {
				stats.successes++;
			} else {
				stats.errors++;
			}
			if (rp.duration_ms != null) {
				stats.totalDuration += rp.duration_ms;
				stats.durations.push(rp.duration_ms);
			}
		}
	}

	return Array.from(byTool.values()).sort((a, b) => b.calls - a.calls);
}

/** Tool Analytics panel — summary statistics and chronological call details. */
export function ToolAnalyticsPanel({ events }: CapabilityPanelProps) {
	const calls = collectToolCalls(events);
	const stats = aggregateStats(calls);

	if (calls.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No tool calls recorded for this agent.</div>;
	}

	const totalCalls = calls.length;
	const totalSuccesses = stats.reduce((s, t) => s + t.successes, 0);
	const totalErrors = stats.reduce((s, t) => s + t.errors, 0);
	const successRate = totalCalls > 0 ? Math.round((totalSuccesses / totalCalls) * 100) : 0;

	return (
		<div className="p-4 space-y-4">
			{/* Summary bar */}
			<div className="flex items-center gap-4 text-xs flex-wrap">
				<SummaryPill label="Total calls" value={totalCalls} />
				<SummaryPill label="Success rate" value={`${successRate}%`} />
				<SummaryPill label="Errors" value={totalErrors} variant={totalErrors > 0 ? "error" : "default"} />
			</div>

			{/* Per-tool breakdown table */}
			<div className="overflow-x-auto">
				<table className="w-full text-xs" data-testid="tool-stats-table">
					<thead>
						<tr className="border-b text-left text-muted-foreground">
							<th className="pb-1.5 pr-4 font-medium">Tool</th>
							<th className="pb-1.5 pr-4 font-medium tabular-nums text-right">Calls</th>
							<th className="pb-1.5 pr-4 font-medium text-right">Success</th>
							<th className="pb-1.5 pr-4 font-medium tabular-nums text-right">Avg Duration</th>
							<th className="pb-1.5 font-medium tabular-nums text-right">Errors</th>
						</tr>
					</thead>
					<tbody>
						{stats.map((s) => {
							const avgDuration = s.durations.length > 0 ? Math.round(s.totalDuration / s.durations.length) : null;
							const rate = s.calls > 0 ? Math.round((s.successes / s.calls) * 100) : 0;

							return (
								<tr key={s.toolName} className="border-b last:border-0">
									<td className="py-1.5 pr-4 font-mono">{s.toolName}</td>
									<td className="py-1.5 pr-4 tabular-nums text-right">{s.calls}</td>
									<td className="py-1.5 pr-4 text-right">{rate}%</td>
									<td className="py-1.5 pr-4 tabular-nums text-right">
										{avgDuration != null ? formatDuration(avgDuration) : "—"}
									</td>
									<td className={`py-1.5 tabular-nums text-right ${s.errors > 0 ? "text-destructive" : ""}`}>
										{s.errors}
									</td>
								</tr>
							);
						})}
					</tbody>
				</table>
			</div>

			{/* Call detail list */}
			<div className="space-y-1">
				<div className="text-xs text-muted-foreground mb-2">Call Detail</div>
				{calls.map((call, i) => (
					<ToolCallRow key={call.invoke.id} call={call} index={i + 1} />
				))}
			</div>
		</div>
	);
}

function ToolCallRow({ call, index }: { call: ToolCall; index: number }) {
	const [isExpanded, setIsExpanded] = useState(false);

	const invokePayload = call.invoke.payload as {
		tool_name?: string;
		parameters?: Record<string, unknown>;
	};
	const resultPayload = call.result
		? (call.result.payload as {
				success?: boolean;
				result?: unknown;
				error?: string;
				duration_ms?: number;
			})
		: null;

	const toolName = invokePayload.tool_name ?? "unknown";
	const success = resultPayload?.success;
	const duration = resultPayload?.duration_ms;

	return (
		<div
			className={`rounded-md border overflow-hidden ${
				success === false ? "border-destructive-border" : "border-transparent hover:border-border"
			}`}
		>
			<button
				type="button"
				className="w-full flex items-center gap-1.5 px-2 py-1.5 text-sm text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="text-[10px] text-muted-foreground tabular-nums">#{index}</span>
				{success === true && <span className="text-success text-xs">✓</span>}
				{success === false && <span className="text-destructive text-xs">✗</span>}
				<span className="text-xs font-mono">{toolName}</span>
				{duration != null && (
					<span className="text-[10px] text-muted-foreground tabular-nums ml-auto">{formatDuration(duration)}</span>
				)}
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2 text-xs">
					{/* Parameters */}
					{invokePayload.parameters && (
						<div>
							<span className="text-muted-foreground font-medium">Parameters:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 text-xs overflow-x-auto max-h-[200px] overflow-y-auto whitespace-pre-wrap">
								{JSON.stringify(invokePayload.parameters, null, 2)}
							</pre>
						</div>
					)}

					{/* Result */}
					{resultPayload && success && resultPayload.result != null && (
						<div>
							<span className="text-muted-foreground font-medium">Result:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 text-xs overflow-x-auto max-h-[200px] overflow-y-auto whitespace-pre-wrap">
								{typeof resultPayload.result === "string"
									? resultPayload.result
									: JSON.stringify(resultPayload.result, null, 2)}
							</pre>
						</div>
					)}

					{/* Error */}
					{resultPayload && success === false && resultPayload.error && (
						<div>
							<span className="text-destructive font-medium">Error:</span>
							<pre className="mt-1 bg-destructive-muted rounded p-2 text-xs text-destructive-muted-foreground overflow-x-auto whitespace-pre-wrap">
								{resultPayload.error}
							</pre>
						</div>
					)}

					{!call.result && <div className="text-muted-foreground italic">No result recorded</div>}
				</div>
			)}
		</div>
	);
}

function SummaryPill({
	label,
	value,
	variant = "default",
}: {
	label: string;
	value: string | number;
	variant?: "default" | "error";
}) {
	const colorClass = variant === "error" ? "text-destructive font-medium" : "text-foreground";
	return (
		<div className="flex items-center gap-1.5">
			<span className="text-muted-foreground">{label}</span>
			<span className={`font-mono tabular-nums ${colorClass}`}>{value}</span>
		</div>
	);
}

function formatDuration(ms: number): string {
	return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}
