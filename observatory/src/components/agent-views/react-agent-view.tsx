import { CheckCircle2, Eye, Lightbulb, Zap } from "lucide-react";
import { useState } from "react";
import { useObservatory } from "../../context/observatory-context";
import type { AgentViewProps } from "../../registry/agent-view-registry";
import type { TraceEvent } from "../../types";
import { statusVariant } from "../../utils/status-variants";
import { RecoveryIcon } from "../primitives/event-icon";

type RecoveryKind = "retry" | "correction" | "degradation" | "error" | "unknown";

interface ReActStep {
	stepNumber: number;
	stepEvent: TraceEvent;
	thought?: string;
	action?: string;
	observation?: string;
	/** Child events: LLM calls, tool invocations, etc. */
	childEvents: TraceEvent[];
	/** Error-related events within this step. */
	errorEvents: TraceEvent[];
}

const ERROR_EVENT_TYPES = new Set(["error.retry", "error.correction", "error.degradation", "agent.error"]);

function isErrorEvent(event: TraceEvent): boolean {
	return (
		ERROR_EVENT_TYPES.has(event.event_type) ||
		(event.event_type === "tool.result" && (event.payload as { success?: boolean })?.success === false)
	);
}

function groupReActSteps(events: TraceEvent[]): {
	steps: ReActStep[];
	headerEvents: TraceEvent[];
	completeEvent: TraceEvent | null;
} {
	const stepEvents = events.filter((e) => e.event_type === "agent.step");
	const nonStepEvents = events.filter((e) => e.event_type !== "agent.step");

	const completeEvent = nonStepEvents.find((e) => e.event_type === "agent.complete") ?? null;

	const headerEvents = nonStepEvents.filter((e) => e.event_type === "agent.start");

	if (stepEvents.length === 0) {
		return { steps: [], headerEvents: nonStepEvents, completeEvent: null };
	}

	const sortedSteps = [...stepEvents].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

	const steps: ReActStep[] = sortedSteps.map((stepEvent, idx) => {
		const payload = stepEvent.payload as {
			step?: number;
			thought?: string;
			action?: string;
			observation?: string;
		};

		const stepStart = new Date(stepEvent.timestamp).getTime();
		const stepEnd = idx < sortedSteps.length - 1 ? new Date(sortedSteps[idx + 1].timestamp).getTime() : Infinity;

		const childCandidates = nonStepEvents.filter((e) => {
			if (e.event_type === "agent.start" || e.event_type === "agent.complete") {
				return false;
			}
			const t = new Date(e.timestamp).getTime();
			return t >= stepStart && t < stepEnd;
		});

		const errorEvents = childCandidates.filter(isErrorEvent);
		const childEvents = childCandidates.filter((e) => !isErrorEvent(e));

		return {
			stepNumber: payload.step ?? idx + 1,
			stepEvent,
			thought: payload.thought,
			action: payload.action,
			observation: payload.observation,
			childEvents,
			errorEvents,
		};
	});

	return { steps, headerEvents, completeEvent };
}

/** ReAct agent timeline — step-by-step with thought/action/observation labeling. */
export function ReActAgentView({ events }: AgentViewProps) {
	const { steps, headerEvents, completeEvent } = groupReActSteps(events);

	if (events.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No events recorded for this agent.</div>;
	}

	return (
		<div className="p-4 space-y-2">
			{/* Header events */}
			{headerEvents.length > 0 && (
				<div className="space-y-1">
					{headerEvents.map((event) => (
						<EventRow key={event.id} event={event} />
					))}
				</div>
			)}

			{/* ReAct steps */}
			{steps.map((step) => (
				<ReActStepSection key={step.stepNumber} step={step} />
			))}

			{/* If no steps, show all events flat */}
			{steps.length === 0 && headerEvents.length === 0 && (
				<div className="space-y-1">
					{events.map((event) => (
						<EventRow key={event.id} event={event} />
					))}
				</div>
			)}

			{/* Final output */}
			{completeEvent && <CompletionSection event={completeEvent} />}
		</div>
	);
}

function ReActStepSection({ step }: { step: ReActStep }) {
	const [isExpanded, setIsExpanded] = useState(true);
	const [showChildEvents, setShowChildEvents] = useState(false);

	const hasChildEvents = step.childEvents.length > 0;
	const hasErrorEvents = step.errorEvents.length > 0;

	return (
		<div className="border rounded-lg overflow-hidden">
			{/* Step header */}
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">Step {step.stepNumber}</span>
				{step.action && <span className="text-muted-foreground truncate">{step.action}</span>}
				{hasErrorEvents && (
					<span className="text-[10px] text-warning bg-warning-muted rounded-full px-1.5 py-0.5 flex-shrink-0">
						{step.errorEvents.length} error{step.errorEvents.length !== 1 ? "s" : ""}
					</span>
				)}
				{hasChildEvents && (
					<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5 ml-auto flex-shrink-0">
						{step.childEvents.length}
					</span>
				)}
			</button>

			{/* Step content */}
			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2">
					{/* Thought */}
					{step.thought && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground flex items-center gap-1">
								<Lightbulb aria-hidden="true" className="h-3.5 w-3.5" /> Thought
							</span>
							<div className="text-sm bg-muted/50 rounded-md p-2">{step.thought}</div>
						</div>
					)}

					{/* Action */}
					{step.action && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground flex items-center gap-1">
								<Zap aria-hidden="true" className="h-3.5 w-3.5" /> Action
							</span>
							<div className="text-sm font-mono bg-muted/50 rounded-md p-2">{step.action}</div>
						</div>
					)}

					{/* Child events (LLM calls, tool invocations) — expandable */}
					{hasChildEvents && (
						<div>
							<button
								type="button"
								className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 py-1"
								onClick={() => setShowChildEvents(!showChildEvents)}
							>
								<span className="w-3">{showChildEvents ? "▾" : "▸"}</span>
								{step.childEvents.length} event{step.childEvents.length !== 1 ? "s" : ""} (LLM calls, tool invocations)
							</button>
							{showChildEvents && (
								<div className="space-y-1 ml-4 border-l pl-2">
									{step.childEvents.map((event) => (
										<EventRow key={event.id} event={event} />
									))}
								</div>
							)}
						</div>
					)}

					{/* Error recovery events — always visible inline */}
					{hasErrorEvents && (
						<div className="space-y-1">
							{step.errorEvents.map((event) => (
								<ErrorEventRow key={event.id} event={event} />
							))}
						</div>
					)}

					{/* Observation */}
					{step.observation && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground flex items-center gap-1">
								<Eye aria-hidden="true" className="h-3.5 w-3.5" /> Observation
							</span>
							<div className="text-sm bg-muted/50 rounded-md p-2">{step.observation}</div>
						</div>
					)}

					{!step.thought && !step.action && !step.observation && !hasChildEvents && !hasErrorEvents && (
						<div className="text-xs text-muted-foreground">No additional events in this step.</div>
					)}
				</div>
			)}
		</div>
	);
}

function ErrorEventRow({ event }: { event: TraceEvent }) {
	const { registry } = useObservatory();
	const [isExpanded, setIsExpanded] = useState(false);

	const Renderer = registry.getRenderer(event.event_type);
	const payload = event.payload as Record<string, unknown>;

	let kind: RecoveryKind;
	let label: string;
	let detail: string;
	let colorClass: string;

	switch (event.event_type) {
		case "error.retry": {
			kind = "retry";
			label = "Retry";
			detail = `Attempt ${payload.attempt}/${payload.max_attempts}`;
			if (payload.error) detail += ` — ${payload.error}`;
			colorClass = statusVariant("warning");
			break;
		}
		case "error.correction": {
			kind = "correction";
			label = "Correction";
			detail = `Attempt ${payload.attempt ?? 1}`;
			if (payload.correction_prompt) detail += ` — ${payload.correction_prompt}`;
			colorClass = statusVariant("warning");
			break;
		}
		case "error.degradation": {
			kind = "degradation";
			label = "Degradation";
			detail = String(payload.reason ?? payload.fallback ?? "");
			colorClass = statusVariant("error");
			break;
		}
		case "agent.error": {
			kind = "error";
			label = String(payload.error_type ?? "Error");
			detail = String(payload.message ?? "");
			colorClass = statusVariant("error");
			break;
		}
		default: {
			// tool.result with success=false
			kind = "error";
			label = `${payload.tool_name ?? "Tool"} failed`;
			detail = String(payload.error ?? "");
			colorClass = statusVariant("error");
			break;
		}
	}

	return (
		<div className={`rounded-md border ${colorClass}`}>
			<button
				type="button"
				className="w-full flex items-center gap-1.5 px-2 py-1.5 text-sm text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="flex-shrink-0">
					<RecoveryIcon kind={kind} />
				</span>
				<span className="text-xs font-medium flex-shrink-0">{label}</span>
				<span className="text-xs text-muted-foreground truncate flex-1 min-w-0">{detail}</span>
				<span className="text-xs text-muted-foreground w-3 flex-shrink-0">{isExpanded ? "▾" : "▸"}</span>
			</button>

			{isExpanded && Renderer && (
				<div className="px-2 pb-2 pt-1 border-t ml-6">
					<Renderer event={event} />
				</div>
			)}
		</div>
	);
}

function CompletionSection({ event }: { event: TraceEvent }) {
	const payload = event.payload as Record<string, unknown>;
	const terminationReason = payload.termination_reason as string | undefined;
	const totalSteps = payload.total_steps as number | undefined;

	return (
		<div className="border rounded-lg overflow-hidden border-success-border">
			<div className="px-3 py-2 bg-success-muted">
				<div className="flex items-center gap-2 text-sm">
					<CheckCircle2 aria-hidden="true" className="h-4 w-4 text-success" />
					<span className="font-medium">Completed</span>
					{terminationReason && <span className="text-xs text-muted-foreground">({terminationReason})</span>}
					{totalSteps != null && (
						<span className="text-xs text-muted-foreground ml-auto">
							{totalSteps} step{totalSteps !== 1 ? "s" : ""}
						</span>
					)}
				</div>
			</div>
		</div>
	);
}

function EventRow({ event }: { event: TraceEvent }) {
	const { registry } = useObservatory();
	const [isExpanded, setIsExpanded] = useState(false);

	const Renderer = registry.getRenderer(event.event_type);
	const summary = registry.getSummary(event);

	return (
		<div className="rounded-md border border-transparent hover:border-border">
			<button
				type="button"
				className="w-full flex items-center gap-1.5 px-2 py-1 text-sm text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4 flex-shrink-0">{isExpanded ? "▾" : "▸"}</span>
				<span className="text-xs font-mono text-muted-foreground flex-shrink-0">{event.event_type}</span>
				<span className="text-xs text-muted-foreground truncate flex-1 min-w-0">{summary}</span>
				<span className="text-[10px] text-muted-foreground tabular-nums flex-shrink-0">
					{new Date(event.timestamp).toLocaleTimeString()}
				</span>
			</button>

			{isExpanded && Renderer && (
				<div className="px-2 pb-2 pt-1 border-t ml-6">
					<Renderer event={event} />
				</div>
			)}
		</div>
	);
}
