import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Data extraction
// ---------------------------------------------------------------------------

interface EvaluationResult {
	event: TraceEvent;
	evaluatorName: string;
	verdict: string;
	score: number | null;
	feedback: string;
	revisionAttempt: number;
}

interface RevisionRequest {
	event: TraceEvent;
	feedback: string;
	revisionAttempt: number;
	maxRevisions: number;
}

interface Reflection {
	event: TraceEvent;
	reflectionText: string;
	evaluationFeedback: string;
	episodeId: string | null;
	attemptNumber: number;
	maxAttempts: number;
}

function extractEvaluations(events: TraceEvent[]): EvaluationResult[] {
	return events
		.filter((e) => e.event_type === "evaluation.result")
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				event: e,
				evaluatorName: String(p.evaluator_name ?? "Evaluator"),
				verdict: String(p.verdict ?? "unknown"),
				score: typeof p.score === "number" ? p.score : null,
				feedback: String(p.feedback ?? ""),
				revisionAttempt: (p.revision_attempt as number) ?? 0,
			};
		});
}

function extractRevisions(events: TraceEvent[]): RevisionRequest[] {
	return events
		.filter((e) => e.event_type === "evaluation.revision")
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				event: e,
				feedback: String(p.feedback ?? ""),
				revisionAttempt: (p.revision_attempt as number) ?? 0,
				maxRevisions: (p.max_revisions as number) ?? 0,
			};
		});
}

function extractReflections(events: TraceEvent[]): Reflection[] {
	return events
		.filter((e) => e.event_type === "reflection.generated")
		.map((e) => {
			const p = e.payload as Record<string, unknown>;
			return {
				event: e,
				reflectionText: String(p.reflection_text ?? ""),
				evaluationFeedback: String(p.evaluation_feedback ?? ""),
				episodeId: (p.episode_id as string) ?? null,
				attemptNumber: (p.attempt_number as number) ?? 0,
				maxAttempts: (p.max_attempts as number) ?? 0,
			};
		});
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

/** Evaluation & Revision panel — score progression, evaluations, reflections. */
export function EvaluationPanel({ events }: CapabilityPanelProps) {
	const evaluations = extractEvaluations(events);
	const revisions = extractRevisions(events);
	const reflections = extractReflections(events);

	if (evaluations.length === 0 && revisions.length === 0 && reflections.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No evaluation events recorded for this agent.</div>;
	}

	const totalAttempts = evaluations.length;
	const finalEval = evaluations[evaluations.length - 1];
	const finalVerdict = finalEval?.verdict ?? "unknown";

	return (
		<div className="p-4 space-y-4">
			{/* Outcome summary */}
			<div className="flex items-center gap-3 text-xs">
				<span className="text-muted-foreground">
					{totalAttempts} evaluation{totalAttempts !== 1 ? "s" : ""}
				</span>
				{finalVerdict !== "unknown" && <VerdictBadge verdict={finalVerdict} />}
				{finalEval?.score != null && (
					<span className="text-muted-foreground">Final score: {finalEval.score.toFixed(2)}</span>
				)}
			</div>

			{/* Score progression */}
			{evaluations.length > 1 && <ScoreProgression evaluations={evaluations} />}

			{/* Per-attempt detail */}
			{evaluations.map((ev, i) => {
				const revision = revisions.find((r) => r.revisionAttempt === ev.revisionAttempt);
				const reflection = reflections.find((r) => r.attemptNumber === ev.revisionAttempt);
				return (
					<EvaluationCard
						key={ev.event.id}
						evaluation={ev}
						index={i}
						total={evaluations.length}
						revision={revision}
						reflection={reflection}
					/>
				);
			})}

			{/* Reflections without matching evaluations (standalone) */}
			{reflections
				.filter((r) => !evaluations.some((ev) => ev.revisionAttempt === r.attemptNumber))
				.map((r) => (
					<ReflectionCard key={r.event.id} reflection={r} />
				))}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Score Progression
// ---------------------------------------------------------------------------

function ScoreProgression({ evaluations }: { evaluations: EvaluationResult[] }) {
	const maxScore = Math.max(...evaluations.map((e) => e.score ?? 0), 1);

	return (
		<div className="space-y-1.5">
			<div className="text-xs font-medium text-muted-foreground">Score Progression</div>
			<div className="flex items-end gap-1 h-16">
				{evaluations.map((ev, i) => {
					const height = ev.score != null ? (ev.score / maxScore) * 100 : 10;
					const isAccepted = ev.verdict.toLowerCase() === "accept" || ev.verdict.toLowerCase() === "accepted";
					return (
						<div key={ev.event.id} className="flex flex-col items-center gap-0.5 flex-1">
							<span className="text-[9px] text-muted-foreground">{ev.score != null ? ev.score.toFixed(1) : "?"}</span>
							<div
								className={`w-full max-w-[40px] rounded-t transition-all ${isAccepted ? "bg-success" : "bg-warning"}`}
								style={{ height: `${height}%` }}
							/>
							<span className="text-[9px] text-muted-foreground">{i + 1}</span>
						</div>
					);
				})}
			</div>
		</div>
	);
}

// ---------------------------------------------------------------------------
// Evaluation Card
// ---------------------------------------------------------------------------

function EvaluationCard({
	evaluation,
	index,
	total,
	revision,
	reflection,
}: {
	evaluation: EvaluationResult;
	index: number;
	total: number;
	revision?: RevisionRequest;
	reflection?: Reflection;
}) {
	const [isExpanded, setIsExpanded] = useState(false);

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">
					{index + 1} of {total}
				</span>
				<VerdictBadge verdict={evaluation.verdict} />
				{evaluation.score != null && (
					<span className="text-xs text-muted-foreground">Score: {evaluation.score.toFixed(2)}</span>
				)}
				<span className="ml-auto text-[10px] text-muted-foreground">{evaluation.evaluatorName}</span>
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-3 text-xs">
					{/* Evaluation details */}
					<div className="space-y-1">
						<DetailRow label="Evaluator" value={evaluation.evaluatorName} />
						<DetailRow label="Verdict" value={evaluation.verdict} />
						{evaluation.score != null && <DetailRow label="Score" value={evaluation.score.toFixed(3)} />}
						<DetailRow label="Attempt" value={String(evaluation.revisionAttempt)} />
					</div>

					{/* Feedback */}
					{evaluation.feedback && (
						<div>
							<span className="text-muted-foreground">Feedback:</span>
							<div className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[200px] overflow-y-auto">
								{evaluation.feedback}
							</div>
						</div>
					)}

					{/* Revision request */}
					{revision && (
						<div className="border-l-2 border-warning pl-2">
							<span className="text-muted-foreground font-medium">Revision requested</span>
							<span className="text-[10px] text-muted-foreground ml-2">
								(attempt {revision.revisionAttempt}/{revision.maxRevisions})
							</span>
							{revision.feedback && <div className="mt-1 text-muted-foreground">{revision.feedback}</div>}
						</div>
					)}

					{/* Reflection */}
					{reflection && <ReflectionCard reflection={reflection} />}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Reflection Card
// ---------------------------------------------------------------------------

function ReflectionCard({ reflection }: { reflection: Reflection }) {
	return (
		<div className="border rounded-lg px-3 py-2 border-accent-status-border bg-accent-status-muted/50">
			<div className="flex items-center gap-2 text-xs">
				<span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-accent-status-muted text-accent-status-muted-foreground">
					reflection
				</span>
				<span className="text-muted-foreground">
					Attempt {reflection.attemptNumber}
					{reflection.maxAttempts > 0 && `/${reflection.maxAttempts}`}
				</span>
				{reflection.episodeId && (
					<span className="text-[10px] text-muted-foreground ml-auto">Episode: {reflection.episodeId}</span>
				)}
			</div>
			{reflection.reflectionText && (
				<div className="mt-1.5 text-xs whitespace-pre-wrap bg-muted/50 rounded p-2 max-h-[200px] overflow-y-auto">
					{reflection.reflectionText}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

function VerdictBadge({ verdict }: { verdict: string }) {
	const lower = verdict.toLowerCase();
	const isAccepted = lower === "accept" || lower === "accepted";
	const colors = isAccepted
		? "bg-success-muted text-success-muted-foreground"
		: "bg-warning-muted text-warning-muted-foreground";

	return <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${colors}`}>{verdict}</span>;
}

function DetailRow({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-start gap-2">
			<span className="text-muted-foreground shrink-0">{label}:</span>
			<span className="font-mono break-all">{value}</span>
		</div>
	);
}
