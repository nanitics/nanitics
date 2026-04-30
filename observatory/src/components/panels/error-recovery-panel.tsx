import { XCircle } from "lucide-react";
import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";
import { statusVariant } from "../../utils/status-variants";
import { OutcomeIcon, RecoveryIcon } from "../primitives/event-icon";

type OutcomeKind = "corrected" | "degraded" | "retried" | "unresolved";
type RecoveryKind = "retry" | "correction" | "degradation" | "unknown";

const ERROR_TYPES = new Set(["error.retry", "error.correction", "error.degradation", "agent.error"]);

interface ErrorChain {
	index: number;
	initialError: TraceEvent;
	recoveryEvents: TraceEvent[];
	outcome: "corrected" | "degraded" | "retried" | "unresolved";
	outcomeEvent: TraceEvent | null;
}

function buildErrorChains(events: TraceEvent[]): ErrorChain[] {
	const errorEvents = events.filter(
		(e) =>
			ERROR_TYPES.has(e.event_type) ||
			(e.event_type === "tool.result" && (e.payload as { success?: boolean })?.success === false),
	);

	if (errorEvents.length === 0) return [];

	const chains: ErrorChain[] = [];
	let current: ErrorChain | null = null;
	let chainIndex = 0;

	for (const event of errorEvents) {
		const isInitialError =
			event.event_type === "agent.error" ||
			(event.event_type === "tool.result" && (event.payload as { success?: boolean })?.success === false);

		if (isInitialError) {
			// Finalize previous chain
			if (current) {
				finalizeChain(current, events);
				chains.push(current);
			}
			chainIndex++;
			current = {
				index: chainIndex,
				initialError: event,
				recoveryEvents: [],
				outcome: "unresolved",
				outcomeEvent: null,
			};
		} else if (event.event_type === "error.retry") {
			if (!current) {
				chainIndex++;
				current = {
					index: chainIndex,
					initialError: event,
					recoveryEvents: [],
					outcome: "unresolved",
					outcomeEvent: null,
				};
			} else {
				current.recoveryEvents.push(event);
			}
		} else if (event.event_type === "error.correction") {
			if (!current) {
				chainIndex++;
				current = {
					index: chainIndex,
					initialError: event,
					recoveryEvents: [],
					outcome: "unresolved",
					outcomeEvent: null,
				};
			} else {
				current.recoveryEvents.push(event);
			}
		} else if (event.event_type === "error.degradation") {
			if (current) {
				current.recoveryEvents.push(event);
				current.outcome = "degraded";
				current.outcomeEvent = event;
				chains.push(current);
				current = null;
			} else {
				chainIndex++;
				chains.push({
					index: chainIndex,
					initialError: event,
					recoveryEvents: [],
					outcome: "degraded",
					outcomeEvent: event,
				});
			}
		}
	}

	// Finalize remaining chain
	if (current) {
		finalizeChain(current, events);
		chains.push(current);
	}

	return chains;
}

function finalizeChain(chain: ErrorChain, allEvents: TraceEvent[]): void {
	if (chain.outcome !== "unresolved") return;

	// Check if there's a successful tool result with the same tool name after the error
	const errorTime = new Date(chain.initialError.timestamp).getTime();
	const toolName = (chain.initialError.payload as { tool_name?: string })?.tool_name;

	if (toolName) {
		const subsequentSuccess = allEvents.find(
			(e) =>
				e.event_type === "tool.result" &&
				(e.payload as { tool_name?: string })?.tool_name === toolName &&
				(e.payload as { success?: boolean })?.success === true &&
				new Date(e.timestamp).getTime() > errorTime,
		);

		if (subsequentSuccess) {
			chain.outcome = "corrected";
			chain.outcomeEvent = subsequentSuccess;
			return;
		}
	}

	// Check if recovery events include corrections
	const hasCorrection = chain.recoveryEvents.some((e) => e.event_type === "error.correction");
	const hasRetry = chain.recoveryEvents.some((e) => e.event_type === "error.retry");

	if (hasCorrection) {
		chain.outcome = "corrected";
	} else if (hasRetry) {
		chain.outcome = "retried";
	}
}

/** Error Recovery panel — error chain visualization with recovery outcomes. */
export function ErrorRecoveryPanel({ events }: CapabilityPanelProps) {
	const chains = buildErrorChains(events);

	if (chains.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No errors recorded for this agent.</div>;
	}

	const totalErrors = chains.length;
	const corrections = chains.filter((c) => c.outcome === "corrected").length;
	const degradations = chains.filter((c) => c.outcome === "degraded").length;
	const recoveryRate = totalErrors > 0 ? Math.round(((corrections + degradations) / totalErrors) * 100) : 0;

	return (
		<div className="p-4 space-y-4">
			{/* Summary header */}
			<div className="flex items-center gap-4 text-xs flex-wrap">
				<SummaryPill label="Total errors" value={totalErrors} variant="error" />
				<SummaryPill label="Corrections" value={corrections} />
				<SummaryPill label="Degradations" value={degradations} />
				<SummaryPill label="Recovery rate" value={`${recoveryRate}%`} />
			</div>

			{/* Error chains */}
			<div className="space-y-3">
				{chains.map((chain) => (
					<ErrorChainCard key={chain.index} chain={chain} />
				))}
			</div>
		</div>
	);
}

function ErrorChainCard({ chain }: { chain: ErrorChain }) {
	const [isExpanded, setIsExpanded] = useState(true);

	const outcomeStyles: Record<OutcomeKind, { kind: OutcomeKind; label: string; className: string }> = {
		corrected: {
			kind: "corrected",
			label: "Corrected",
			className: "text-success",
		},
		degraded: {
			kind: "degraded",
			label: "Degraded",
			className: "text-warning",
		},
		retried: {
			kind: "retried",
			label: "Retried",
			className: "text-info",
		},
		unresolved: {
			kind: "unresolved",
			label: "Unresolved",
			className: "text-destructive",
		},
	};
	const outcomeStyle = outcomeStyles[chain.outcome];

	return (
		<div className="border rounded-lg overflow-hidden">
			{/* Chain header */}
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">Error Chain {chain.index}</span>
				<span className={`text-xs font-medium inline-flex items-center gap-1 ${outcomeStyle.className}`}>
					<OutcomeIcon kind={outcomeStyle.kind} className="h-4 w-4" /> {outcomeStyle.label}
				</span>
				{chain.recoveryEvents.length > 0 && (
					<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5 ml-auto">
						{chain.recoveryEvents.length} recovery step{chain.recoveryEvents.length !== 1 ? "s" : ""}
					</span>
				)}
			</button>

			{/* Chain detail */}
			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2">
					{/* Initial error */}
					<ErrorNode event={chain.initialError} />

					{/* Recovery events */}
					{chain.recoveryEvents.map((event) => (
						<RecoveryNode key={event.id} event={event} />
					))}

					{/* Outcome */}
					{chain.outcomeEvent &&
						chain.outcomeEvent !== chain.initialError &&
						!chain.recoveryEvents.includes(chain.outcomeEvent) && (
							<OutcomeNode event={chain.outcomeEvent} outcome={chain.outcome} />
						)}
				</div>
			)}
		</div>
	);
}

function ErrorNode({ event }: { event: TraceEvent }) {
	const payload = event.payload as Record<string, unknown>;

	let errorType: string;
	let errorMessage: string;

	if (event.event_type === "tool.result") {
		errorType = `${payload.tool_name ?? "Tool"} failed`;
		errorMessage = String(payload.error ?? "Unknown error");
	} else if (event.event_type === "agent.error") {
		errorType = String(payload.error_type ?? "Error");
		errorMessage = String(payload.error_message ?? "");
	} else {
		errorType = String(payload.error_type ?? event.event_type);
		errorMessage = String(payload.error_message ?? "");
	}

	return (
		<div className={`flex items-start gap-2 p-2 rounded-md border ${statusVariant("error")}`}>
			<span className="flex-shrink-0 mt-0.5">
				<XCircle aria-hidden="true" className="h-4 w-4" />
			</span>
			<div className="min-w-0">
				<div className="text-xs font-medium">
					{errorType}
					{event.event_type === "agent.error" && payload.step_number != null && (
						<span className="ml-1.5 font-mono text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
							Step {String(payload.step_number)}
						</span>
					)}
				</div>
				{errorMessage && <div className="text-xs opacity-80 mt-0.5">{errorMessage}</div>}
			</div>
		</div>
	);
}

function RecoveryNode({ event }: { event: TraceEvent }) {
	const payload = event.payload as Record<string, unknown>;

	let kind: RecoveryKind;
	let label: string;
	let detail: string;
	let bgClass: string;

	switch (event.event_type) {
		case "error.retry": {
			kind = "retry";
			label = `Retry (attempt ${payload.attempt}/${payload.max_attempts})`;
			detail = payload.delay_ms ? `delay: ${payload.delay_ms}ms` : "";
			bgClass = statusVariant("warning");
			break;
		}
		case "error.correction": {
			kind = "correction";
			label = `Correction (attempt ${payload.attempt ?? 1}/${payload.max_attempts ?? "?"})`;
			detail = String(payload.correction_prompt ?? "");
			bgClass = statusVariant("warning");
			break;
		}
		case "error.degradation": {
			kind = "degradation";
			label = "Degradation";
			detail = String(payload.degradation_message ?? "");
			bgClass = statusVariant("warning");
			break;
		}
		default: {
			kind = "unknown";
			label = event.event_type;
			detail = "";
			bgClass = "bg-muted/50 border-border text-foreground";
		}
	}

	const errorType = payload.error_type ? String(payload.error_type) : "";
	const errorMessage = payload.error_message ? String(payload.error_message) : "";
	const category = event.event_type === "error.retry" && payload.category ? String(payload.category) : "";

	return (
		<div className={`flex items-start gap-2 p-2 rounded-md border ${bgClass}`}>
			<span className="flex-shrink-0 mt-0.5">
				<RecoveryIcon kind={kind} />
			</span>
			<div className="min-w-0">
				<div className="text-xs font-medium">
					{label}
					{category && (
						<span className="ml-1.5 font-mono text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
							{category}
						</span>
					)}
				</div>
				{detail && <div className="text-xs text-muted-foreground mt-0.5 break-words">{detail}</div>}
				{(errorType || errorMessage) && (
					<div className="text-xs text-muted-foreground mt-0.5">
						{errorType && <span className="font-mono">{errorType}</span>}
						{errorType && errorMessage && <span>: </span>}
						{errorMessage && <span>{errorMessage}</span>}
					</div>
				)}
			</div>
		</div>
	);
}

function OutcomeNode({ event, outcome }: { event: TraceEvent; outcome: string }) {
	const payload = event.payload as Record<string, unknown>;

	const isSuccess = outcome === "corrected";
	const bgClass = isSuccess ? statusVariant("success") : statusVariant("warning");
	const label = isSuccess ? "Corrected successfully" : "Degraded";

	const toolName = event.event_type === "tool.result" ? String(payload.tool_name ?? "") : "";

	return (
		<div className={`flex items-start gap-2 p-2 rounded-md border ${bgClass}`}>
			<span className="flex-shrink-0 mt-0.5">
				<OutcomeIcon kind={isSuccess ? "success" : "warning"} className="h-4 w-4" />
			</span>
			<div className="min-w-0">
				<div className="text-xs font-medium">{label}</div>
				{toolName && <div className="text-xs text-muted-foreground mt-0.5">Next call to {toolName} succeeded</div>}
			</div>
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
		<div className="flex items-center gap-1.5 text-xs">
			<span className="text-muted-foreground">{label}</span>
			<span className={`font-mono tabular-nums ${colorClass}`}>{value}</span>
		</div>
	);
}
