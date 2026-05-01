import { CheckCircle2, Lightbulb, NotebookPen, StopCircle, XCircle } from "lucide-react";
import { useState } from "react";
import { useObservatory } from "../../context/observatory-context";
import type { AgentViewProps } from "../../registry/agent-view-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Attempt {
	attemptNumber: number;
	stepEvent: TraceEvent | null;
	innerEvents: TraceEvent[];
	evaluation: EvaluationInfo | null;
	reflection: ReflectionInfo | null;
	episodeRecorded: EpisodeInfo | null;
}

interface EvaluationInfo {
	event: TraceEvent;
	verdict: string;
	score: number | null;
	feedback: string | null;
	evaluatorName: string | null;
}

interface ReflectionInfo {
	event: TraceEvent;
	reflectionText: string;
	evaluationFeedback: string | null;
	episodeId: string | null;
}

interface EpisodeInfo {
	event: TraceEvent;
	episodeId: string;
	situation: string;
	outcome: string;
	hasReflection: boolean;
}

interface ReflexionParsed {
	attempts: Attempt[];
	headerEvents: TraceEvent[];
	completeEvent: TraceEvent | null;
	finalVerdict: string | null;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

function parseReflexionEvents(events: TraceEvent[]): ReflexionParsed {
	const headerEvents: TraceEvent[] = [];
	const completeEvent = events.find((e) => e.event_type === "agent.complete") ?? null;

	// Collect header events
	for (const event of events) {
		if (event.event_type === "agent.start") {
			headerEvents.push(event);
		}
	}

	// Collect evaluation results, reflections, and episodes
	const evaluations = events
		.filter((e) => e.event_type === "evaluation.result")
		.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

	const reflections = events.filter((e) => e.event_type === "reflection.generated");

	const episodes = events.filter((e) => e.event_type === "memory.episode.record");

	// Collect agent.step events — each represents one attempt
	const stepEvents = events
		.filter((e) => e.event_type === "agent.step")
		.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

	// Non-lifecycle, non-step events for grouping as inner events
	const lifecycleTypes = new Set([
		"agent.start",
		"agent.complete",
		"agent.step",
		"evaluation.result",
		"evaluation.revision",
		"reflection.generated",
		"memory.episode.record",
	]);

	const innerCandidates = events.filter((e) => !lifecycleTypes.has(e.event_type));

	// Build attempts
	const attempts: Attempt[] = [];

	if (evaluations.length > 0) {
		// Build attempts from evaluation results
		for (let i = 0; i < evaluations.length; i++) {
			const evalEvent = evaluations[i];
			const evalPayload = evalEvent.payload as {
				verdict?: string;
				score?: number;
				feedback?: string;
				evaluator_name?: string;
				revision_attempt?: number;
			};

			const attemptNumber = evalPayload.revision_attempt ?? i + 1;

			// Find matching step
			const matchedStep = stepEvents[i] ?? null;

			// Find inner events for this attempt (between this step and the next step)
			const stepStart = matchedStep
				? new Date(matchedStep.timestamp).getTime()
				: i === 0
					? 0
					: new Date(evaluations[i - 1].timestamp).getTime();
			const stepEnd =
				i < stepEvents.length - 1
					? new Date(stepEvents[i + 1].timestamp).getTime()
					: stepEvents.length > 0
						? new Date(evalEvent.timestamp).getTime()
						: Infinity;

			const attemptInnerEvents = innerCandidates.filter((e) => {
				const t = new Date(e.timestamp).getTime();
				return t >= stepStart && t <= stepEnd;
			});

			// Find reflection for this attempt
			const matchedReflection = reflections.find((r) => {
				const rp = r.payload as { attempt_number?: number };
				return rp.attempt_number === attemptNumber;
			});

			// Find episode recorded after this evaluation
			const evalTime = new Date(evalEvent.timestamp).getTime();
			const nextEvalTime = i < evaluations.length - 1 ? new Date(evaluations[i + 1].timestamp).getTime() : Infinity;
			const matchedEpisode = episodes.find((ep) => {
				const t = new Date(ep.timestamp).getTime();
				return t >= evalTime && t < nextEvalTime;
			});

			attempts.push({
				attemptNumber,
				stepEvent: matchedStep,
				innerEvents: attemptInnerEvents,
				evaluation: {
					event: evalEvent,
					verdict: evalPayload.verdict ?? "unknown",
					score: evalPayload.score ?? null,
					feedback: evalPayload.feedback ?? null,
					evaluatorName: evalPayload.evaluator_name ?? null,
				},
				reflection: matchedReflection
					? {
							event: matchedReflection,
							reflectionText: (matchedReflection.payload as { reflection_text?: string }).reflection_text ?? "",
							evaluationFeedback:
								(
									matchedReflection.payload as {
										evaluation_feedback?: string;
									}
								).evaluation_feedback ?? null,
							episodeId: (matchedReflection.payload as { episode_id?: string }).episode_id ?? null,
						}
					: null,
				episodeRecorded: matchedEpisode
					? {
							event: matchedEpisode,
							episodeId: (matchedEpisode.payload as { episode_id?: string }).episode_id ?? "",
							situation: (matchedEpisode.payload as { situation?: string }).situation ?? "",
							outcome: (matchedEpisode.payload as { outcome?: string }).outcome ?? "",
							hasReflection: (matchedEpisode.payload as { has_reflection?: boolean }).has_reflection ?? false,
						}
					: null,
			});
		}
	} else if (stepEvents.length > 0) {
		// Fallback: build attempts from step events without evaluation info
		for (let i = 0; i < stepEvents.length; i++) {
			const stepStart = new Date(stepEvents[i].timestamp).getTime();
			const stepEnd = i < stepEvents.length - 1 ? new Date(stepEvents[i + 1].timestamp).getTime() : Infinity;

			const attemptInnerEvents = innerCandidates.filter((e) => {
				const t = new Date(e.timestamp).getTime();
				return t >= stepStart && t < stepEnd;
			});

			attempts.push({
				attemptNumber: i + 1,
				stepEvent: stepEvents[i],
				innerEvents: attemptInnerEvents,
				evaluation: null,
				reflection: null,
				episodeRecorded: null,
			});
		}
	}

	// Determine final verdict
	const lastEval = evaluations[evaluations.length - 1];
	const finalVerdict = lastEval
		? ((lastEval.payload as { verdict?: string }).verdict ?? null)
		: completeEvent
			? ((completeEvent.payload as { termination_reason?: string }).termination_reason ?? null)
			: null;

	return { attempts, headerEvents, completeEvent, finalVerdict };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** Reflexion agent view — score progression and attempt timeline. */
export function ReflexionAgentView({ events }: AgentViewProps) {
	const { attempts, headerEvents, completeEvent, finalVerdict } = parseReflexionEvents(events);

	if (events.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No events recorded for this agent.</div>;
	}

	return (
		<div className="p-4 space-y-3">
			{/* Header events */}
			{headerEvents.length > 0 && (
				<div className="space-y-1">
					{headerEvents.map((event) => (
						<EventRow key={event.id} event={event} />
					))}
				</div>
			)}

			{/* Score progression header */}
			{attempts.length > 0 && attempts.some((a) => a.evaluation) && <ScoreProgression attempts={attempts} />}

			{/* Attempt sections */}
			{attempts.map((attempt) => (
				<AttemptSection key={attempt.attemptNumber} attempt={attempt} />
			))}

			{/* Completion / Final outcome */}
			{completeEvent && (
				<FinalOutcome event={completeEvent} finalVerdict={finalVerdict} totalAttempts={attempts.length} />
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Score Progression
// ---------------------------------------------------------------------------

function ScoreProgression({ attempts }: { attempts: Attempt[] }) {
	const evaluatedAttempts = attempts.filter((a) => a.evaluation);

	return (
		<div className="border rounded-lg px-3 py-2">
			<div className="flex items-center gap-1.5 flex-wrap">
				{evaluatedAttempts.map((attempt, i) => {
					const eval_ = attempt.evaluation!;
					const isAccept = eval_.verdict === "accept";
					const isLast = i === evaluatedAttempts.length - 1;

					return (
						<div key={attempt.attemptNumber} className="flex items-center gap-1.5">
							{i > 0 && <span className="text-muted-foreground text-xs">→</span>}
							<span
								className={`inline-flex items-center gap-1 text-xs font-mono rounded-full px-2 py-0.5 ${
									isAccept
										? "text-success-muted-foreground bg-success-muted"
										: "text-warning-muted-foreground bg-warning-muted"
								}`}
							>
								{eval_.score != null ? eval_.score.toFixed(1) : eval_.verdict}
								{isAccept && isLast && " ✓"}
								{!isAccept && " ×"}
							</span>
						</div>
					);
				})}
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Attempt Section
// ---------------------------------------------------------------------------

function AttemptSection({ attempt }: { attempt: Attempt }) {
	const [isExpanded, setIsExpanded] = useState(true);
	const [showInnerEvents, setShowInnerEvents] = useState(false);

	const eval_ = attempt.evaluation;
	const isAccept = eval_?.verdict === "accept";

	return (
		<div className="border rounded-lg overflow-hidden">
			{/* Attempt header */}
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">Attempt {attempt.attemptNumber}</span>
				{eval_ && (
					<>
						<span
							className={`text-[10px] rounded-full px-1.5 py-0.5 ${
								isAccept
									? "text-success-muted-foreground bg-success-muted"
									: "text-warning-muted-foreground bg-warning-muted"
							}`}
						>
							{eval_.verdict}
						</span>
						{eval_.score != null && <span className="text-xs text-muted-foreground">score: {eval_.score}</span>}
					</>
				)}
				{attempt.innerEvents.length > 0 && (
					<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5 ml-auto flex-shrink-0">
						{attempt.innerEvents.length} event
						{attempt.innerEvents.length !== 1 ? "s" : ""}
					</span>
				)}
			</button>

			{/* Attempt content */}
			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2">
					{/* Inner agent events — collapsible */}
					{attempt.innerEvents.length > 0 && (
						<div>
							<button
								type="button"
								className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 py-1"
								onClick={() => setShowInnerEvents(!showInnerEvents)}
							>
								<span className="w-3">{showInnerEvents ? "▾" : "▸"}</span>
								{attempt.innerEvents.length} inner agent event
								{attempt.innerEvents.length !== 1 ? "s" : ""}
							</button>
							{showInnerEvents && (
								<div className="space-y-1 ml-4 border-l pl-2">
									{attempt.innerEvents.map((event) => (
										<EventRow key={event.id} event={event} />
									))}
								</div>
							)}
						</div>
					)}

					{/* Evaluation result */}
					{eval_ && <EvaluationCard evaluation={eval_} />}

					{/* Reflection */}
					{attempt.reflection && <ReflectionCard reflection={attempt.reflection} />}

					{/* Episode recorded */}
					{attempt.episodeRecorded && <EpisodeBadge episode={attempt.episodeRecorded} isSuccess={isAccept ?? false} />}

					{!eval_ && !attempt.reflection && attempt.innerEvents.length === 0 && (
						<div className="text-xs text-muted-foreground">No events in this attempt.</div>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Evaluation Card
// ---------------------------------------------------------------------------

function EvaluationCard({ evaluation }: { evaluation: EvaluationInfo }) {
	const isAccept = evaluation.verdict === "accept";

	return (
		<div
			className={`border rounded-md px-2.5 py-2 text-xs space-y-1 ${
				isAccept ? "border-success-border bg-success-muted" : "border-warning-border bg-warning-muted"
			}`}
		>
			<div className="flex items-center gap-2">
				<span className="font-medium">Evaluation</span>
				{evaluation.evaluatorName && <span className="text-muted-foreground">{evaluation.evaluatorName}</span>}
				<span
					className={`rounded-full px-1.5 py-0.5 ${
						isAccept
							? "text-success-muted-foreground bg-success-muted"
							: "text-warning-muted-foreground bg-warning-muted"
					}`}
				>
					{evaluation.verdict}
				</span>
				{evaluation.score != null && <span className="text-muted-foreground">score: {evaluation.score}</span>}
			</div>
			{evaluation.feedback && <div className="text-muted-foreground">{evaluation.feedback}</div>}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Reflection Card
// ---------------------------------------------------------------------------

function ReflectionCard({ reflection }: { reflection: ReflectionInfo }) {
	return (
		<div className="border border-accent-status-border bg-accent-status-muted rounded-md px-2.5 py-2 text-xs space-y-1">
			<div className="flex items-center gap-2">
				<span className="font-medium inline-flex items-center gap-1">
					<Lightbulb aria-hidden="true" className="h-3.5 w-3.5" /> Reflection
				</span>
				{reflection.episodeId && (
					<span className="text-[10px] text-muted-foreground font-mono">
						episode: {reflection.episodeId.slice(0, 8)}…
					</span>
				)}
			</div>
			<div className="text-sm whitespace-pre-wrap">{reflection.reflectionText}</div>
			{reflection.evaluationFeedback && (
				<div className="text-muted-foreground border-t border-accent-status-border pt-1 mt-1">
					<span className="font-medium">Triggered by: </span>
					{reflection.evaluationFeedback}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Episode Badge
// ---------------------------------------------------------------------------

function EpisodeBadge({ episode, isSuccess }: { episode: EpisodeInfo; isSuccess: boolean }) {
	return (
		<div className="flex items-center gap-2 text-xs text-muted-foreground">
			<span className="text-[10px] bg-muted rounded-full px-1.5 py-0.5 inline-flex items-center gap-1">
				<NotebookPen aria-hidden="true" className="h-3 w-3" />
				{isSuccess ? "Success episode recorded" : "Episode recorded"}
			</span>
			<span className="font-mono text-[10px]">{episode.episodeId.slice(0, 8)}…</span>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Final Outcome
// ---------------------------------------------------------------------------

function FinalOutcome({
	event,
	finalVerdict,
	totalAttempts,
}: {
	event: TraceEvent;
	finalVerdict: string | null;
	totalAttempts: number;
}) {
	const payload = event.payload as Record<string, unknown>;
	const terminationReason = (payload.termination_reason as string) ?? finalVerdict;
	const isSuccess = terminationReason === "complete" || finalVerdict === "accept";
	const isFailed = terminationReason === "evaluation_failed";

	return (
		<div
			className={`border rounded-lg overflow-hidden ${
				isSuccess ? "border-success-border" : isFailed ? "border-destructive-border" : "border-border"
			}`}
		>
			<div className={`px-3 py-2 ${isSuccess ? "bg-success-muted" : isFailed ? "bg-destructive-muted" : ""}`}>
				<div className="flex items-center gap-2 text-sm">
					{isSuccess ? (
						<CheckCircle2 aria-hidden="true" className="h-4 w-4 text-success" />
					) : isFailed ? (
						<XCircle aria-hidden="true" className="h-4 w-4 text-destructive" />
					) : (
						<StopCircle aria-hidden="true" className="h-4 w-4" />
					)}
					<span className="font-medium">{isSuccess ? "Accepted" : isFailed ? "Failed" : "Completed"}</span>
					{terminationReason && <span className="text-xs text-muted-foreground">({terminationReason})</span>}
					<span className="text-xs text-muted-foreground ml-auto">
						{totalAttempts} attempt{totalAttempts !== 1 ? "s" : ""}
					</span>
				</div>
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Event Row (shared utility)
// ---------------------------------------------------------------------------

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
