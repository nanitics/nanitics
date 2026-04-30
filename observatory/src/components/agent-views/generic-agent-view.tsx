import { useState } from "react";
import { useObservatory } from "../../context/observatory-context";
import type { AgentViewProps } from "../../registry/agent-view-registry";
import type { TraceEvent } from "../../types";

/**
 * Represents a step boundary with its associated events.
 * Events between this step's timestamp and the next step's timestamp belong here.
 */
interface StepGroup {
	stepNumber: number;
	stepEvent: TraceEvent;
	thought?: string;
	action?: string;
	observation?: string;
	events: TraceEvent[];
}

/** Events not associated with any step (e.g., agent.start, agent.complete). */
interface UngroupedEvents {
	header: TraceEvent[];
	footer: TraceEvent[];
}

function groupEventsByStep(events: TraceEvent[]): {
	steps: StepGroup[];
	ungrouped: UngroupedEvents;
} {
	const stepEvents = events.filter((e) => e.event_type === "agent.step");
	const nonStepEvents = events.filter((e) => e.event_type !== "agent.step");

	if (stepEvents.length === 0) {
		// No steps — everything goes to header
		return {
			steps: [],
			ungrouped: { header: nonStepEvents, footer: [] },
		};
	}

	// Sort step events by timestamp
	const sortedSteps = [...stepEvents].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

	const steps: StepGroup[] = sortedSteps.map((stepEvent, idx) => {
		const payload = stepEvent.payload as {
			step?: number;
			thought?: string;
			action?: string;
			observation?: string;
		};

		const stepStart = new Date(stepEvent.timestamp).getTime();
		const stepEnd = idx < sortedSteps.length - 1 ? new Date(sortedSteps[idx + 1].timestamp).getTime() : Infinity;

		// Collect events that fall within this step's time window
		const stepChildEvents = nonStepEvents.filter((e) => {
			// Skip agent lifecycle events — they go to header/footer
			if (e.event_type === "agent.start" || e.event_type === "agent.complete") {
				return false;
			}
			const t = new Date(e.timestamp).getTime();
			return t >= stepStart && t < stepEnd;
		});

		return {
			stepNumber: payload.step ?? idx + 1,
			stepEvent,
			thought: payload.thought,
			action: payload.action,
			observation: payload.observation,
			events: stepChildEvents,
		};
	});

	// Header: events before first step + agent.start
	const firstStepTime = new Date(sortedSteps[0].timestamp).getTime();
	const header = nonStepEvents.filter((e) => {
		if (e.event_type === "agent.start") return true;
		const t = new Date(e.timestamp).getTime();
		return t < firstStepTime && e.event_type !== "agent.complete";
	});

	// Footer: agent.complete events
	const footer = nonStepEvents.filter((e) => e.event_type === "agent.complete");

	return { steps, ungrouped: { header, footer } };
}

/** Fallback timeline view — groups events by step number. */
export function GenericAgentView({ events }: AgentViewProps) {
	const { steps, ungrouped } = groupEventsByStep(events);

	if (events.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No events recorded for this agent.</div>;
	}

	return (
		<div className="p-4 space-y-2">
			{/* Header events (agent.start, pre-step events) */}
			{ungrouped.header.length > 0 && (
				<div className="space-y-1">
					{ungrouped.header.map((event) => (
						<EventRow key={event.id} event={event} />
					))}
				</div>
			)}

			{/* Step groups */}
			{steps.map((step) => (
				<StepSection key={step.stepNumber} step={step} />
			))}

			{/* If no steps, show all non-lifecycle events */}
			{steps.length === 0 && ungrouped.header.length === 0 && (
				<div className="space-y-1">
					{events.map((event) => (
						<EventRow key={event.id} event={event} />
					))}
				</div>
			)}

			{/* Footer events (agent.complete) */}
			{ungrouped.footer.length > 0 && (
				<div className="space-y-1 border-t pt-2">
					{ungrouped.footer.map((event) => (
						<EventRow key={event.id} event={event} />
					))}
				</div>
			)}
		</div>
	);
}

function StepSection({ step }: { step: StepGroup }) {
	const [isExpanded, setIsExpanded] = useState(true);

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
				{step.events.length > 0 && (
					<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5 ml-auto flex-shrink-0">
						{step.events.length}
					</span>
				)}
			</button>

			{/* Step content */}
			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2">
					{step.thought && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground">Thought</span>
							<div className="text-sm bg-muted/50 rounded-md p-2">{step.thought}</div>
						</div>
					)}

					{step.observation && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground">Observation</span>
							<div className="text-sm bg-muted/50 rounded-md p-2">{step.observation}</div>
						</div>
					)}

					{/* Child events */}
					{step.events.length > 0 && (
						<div className="space-y-1 pt-1">
							{step.events.map((event) => (
								<EventRow key={event.id} event={event} />
							))}
						</div>
					)}

					{step.events.length === 0 && !step.thought && !step.observation && (
						<div className="text-xs text-muted-foreground">No additional events in this step.</div>
					)}
				</div>
			)}
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
