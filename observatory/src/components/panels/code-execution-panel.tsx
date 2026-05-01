import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";
import { CodeBlock } from "../primitives/code-block";

interface CodeExecution {
	stepNumber: number;
	executionEvent: TraceEvent;
	resultEvent: TraceEvent | null;
}

function pairExecutions(events: TraceEvent[]): CodeExecution[] {
	const executions = events.filter((e) => e.event_type === "code.execution");
	const results = events.filter((e) => e.event_type === "code.execution.result");

	// Pair by step_number when available, otherwise by temporal order
	return executions.map((exec) => {
		const execPayload = exec.payload as { step_number?: number };
		const stepNumber = execPayload.step_number ?? 0;

		const result =
			results.find((r) => {
				const rp = r.payload as { step_number?: number };
				return rp.step_number === stepNumber;
			}) ?? null;

		return { stepNumber, executionEvent: exec, resultEvent: result };
	});
}

/** Code Execution panel — overview stats and chronological execution timeline. */
export function CodeExecutionPanel({ events }: CapabilityPanelProps) {
	const paired = pairExecutions(events);

	if (paired.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No code executions recorded for this agent.</div>;
	}

	const totalExecutions = paired.length;
	const successes = paired.filter((p) => {
		const rp = p.resultEvent?.payload as { success?: boolean } | null;
		return rp?.success === true;
	}).length;
	const failures = paired.filter((p) => {
		const rp = p.resultEvent?.payload as { success?: boolean } | null;
		return rp?.success === false;
	}).length;

	const durations = paired
		.map((p) => (p.resultEvent?.payload as { duration_ms?: number } | null)?.duration_ms)
		.filter((d): d is number => d != null);
	const totalDuration = durations.reduce((s, d) => s + d, 0);
	const avgDuration = durations.length > 0 ? Math.round(totalDuration / durations.length) : null;

	return (
		<div className="p-4 space-y-4">
			{/* Overview bar */}
			<div className="flex items-center gap-4 text-xs flex-wrap">
				<SummaryPill label="Executions" value={totalExecutions} />
				<SummaryPill label="Success" value={successes} />
				<SummaryPill label="Failed" value={failures} variant={failures > 0 ? "error" : "default"} />
				{avgDuration != null && <SummaryPill label="Avg duration" value={formatDuration(avgDuration)} />}
				{durations.length > 0 && <SummaryPill label="Total duration" value={formatDuration(totalDuration)} />}
			</div>

			{/* Execution timeline */}
			<div className="space-y-1">
				<div className="text-xs text-muted-foreground mb-2">Execution Timeline</div>
				{paired.map((exec) => (
					<ExecutionRow key={exec.executionEvent.id} execution={exec} />
				))}
			</div>
		</div>
	);
}

function ExecutionRow({ execution }: { execution: CodeExecution }) {
	const [isExpanded, setIsExpanded] = useState(false);

	const execPayload = execution.executionEvent.payload as {
		code?: string;
	};
	const resultPayload = execution.resultEvent
		? (execution.resultEvent.payload as {
				success?: boolean;
				stdout?: string;
				stderr?: string;
				return_value?: string | null;
				error?: string | null;
				duration_ms?: number;
			})
		: null;

	const code = execPayload.code ?? "";
	const firstLine = code.split("\n")[0] ?? "";
	const truncatedCode = firstLine.length > 60 ? `${firstLine.slice(0, 60)}…` : firstLine;
	const success = resultPayload?.success;
	const duration = resultPayload?.duration_ms;
	const isFailed = success === false;

	return (
		<div
			className={`rounded-md border overflow-hidden ${
				isFailed ? "border-destructive-border" : "border-transparent hover:border-border"
			}`}
		>
			<button
				type="button"
				className="w-full flex items-center gap-1.5 px-2 py-1.5 text-sm text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-mono text-[10px] text-muted-foreground tabular-nums">#{execution.stepNumber}</span>
				{success === true && <span className="text-success text-xs">✓</span>}
				{success === false && <span className="text-destructive text-xs">✗</span>}
				<span className="text-xs font-mono truncate">{truncatedCode || "code"}</span>
				{duration != null && (
					<span className="text-[10px] text-muted-foreground tabular-nums ml-auto">{formatDuration(duration)}</span>
				)}
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2">
					{/* Full code block */}
					{code && <CodeBlock code={code} language="python" maxHeight={300} />}

					{/* Output sections */}
					{resultPayload && <ExecutionOutput result={resultPayload} />}

					{!execution.resultEvent && <div className="text-xs text-muted-foreground italic">No result recorded</div>}
				</div>
			)}
		</div>
	);
}

function ExecutionOutput({
	result,
}: {
	result: {
		success?: boolean;
		stdout?: string;
		stderr?: string;
		return_value?: string | null;
		error?: string | null;
		duration_ms?: number;
	};
}) {
	const [stdoutExpanded, setStdoutExpanded] = useState(false);
	const hasStdout = !!result.stdout;
	const hasStderr = !!result.stderr;
	const hasReturnValue = result.return_value != null;
	const hasError = !!result.error;
	const isLongStdout = (result.stdout?.length ?? 0) > 200;

	if (!hasStdout && !hasStderr && !hasReturnValue && !hasError) {
		return null;
	}

	return (
		<div className="space-y-1 text-xs">
			{/* stdout */}
			{hasStdout && (
				<div className="space-y-0.5">
					<div className="flex items-center gap-1">
						<span className="text-muted-foreground font-mono">stdout:</span>
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
						className={`font-mono bg-muted/50 rounded p-2 whitespace-pre-wrap ${
							!stdoutExpanded && isLongStdout ? "max-h-20 overflow-hidden" : ""
						}`}
					>
						{result.stdout}
					</pre>
				</div>
			)}

			{/* stderr */}
			{hasStderr && (
				<div className="space-y-0.5">
					<span className="text-destructive font-mono">stderr:</span>
					<pre className="font-mono bg-destructive-muted text-destructive-muted-foreground rounded p-2 whitespace-pre-wrap">
						{result.stderr}
					</pre>
				</div>
			)}

			{/* return_value */}
			{hasReturnValue && (
				<div className="space-y-0.5">
					<span className="text-muted-foreground font-mono">return_value:</span>
					<pre className="font-mono bg-muted/50 rounded p-2 whitespace-pre-wrap">{result.return_value}</pre>
				</div>
			)}

			{/* error */}
			{hasError && (
				<div className="space-y-0.5">
					<span className="text-destructive font-mono">error:</span>
					<pre className="font-mono bg-destructive-muted text-destructive-muted-foreground rounded p-2 whitespace-pre-wrap">
						{result.error}
					</pre>
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
