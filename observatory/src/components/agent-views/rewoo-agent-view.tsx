import { Brain, CheckCircle2, ClipboardList, Zap } from "lucide-react";
import { useState } from "react";
import { useObservatory } from "../../context/observatory-context";
import type { AgentViewProps } from "../../registry/agent-view-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PlanStep {
	stepNumber: number;
	stepId: string;
	description: string;
	toolName: string;
	args: Record<string, string>;
	variable: string;
	dependsOn: number[];
	executionLevel: number;
}

interface PlanInfo {
	planId: string;
	planName: string;
	steps: PlanStep[];
}

interface StepExecution {
	stepNumber: number;
	stepId: string;
	description: string;
	previousStatus: string;
	newStatus: string;
	hasResult: boolean;
	toolInvoke: TraceEvent | null;
	toolResult: TraceEvent | null;
}

interface SolverInfo {
	llmRequest: TraceEvent | null;
	llmResponse: TraceEvent | null;
	evaluationResult: TraceEvent | null;
	stepEvent: TraceEvent | null;
}

interface ReWOOParsed {
	plan: PlanInfo | null;
	executionLevels: Map<number, StepExecution[]>;
	solver: SolverInfo;
	headerEvents: TraceEvent[];
	completeEvent: TraceEvent | null;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

function parseReWOOEvents(events: TraceEvent[]): ReWOOParsed {
	let plan: PlanInfo | null = null;
	const stepUpdates: StepExecution[] = [];
	const solver: SolverInfo = {
		llmRequest: null,
		llmResponse: null,
		evaluationResult: null,
		stepEvent: null,
	};

	const headerEvents: TraceEvent[] = [];
	const completeEvent = events.find((e) => e.event_type === "agent.complete") ?? null;

	// Tool events for matching to steps
	const toolInvokes = events.filter((e) => e.event_type === "tool.invoke");
	const toolResults = events.filter((e) => e.event_type === "tool.result");

	// All agent.step events sorted chronologically
	const stepEvents = events
		.filter((e) => e.event_type === "agent.step")
		.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

	// Extract plan from planning.plan.created
	const planEvent = events.find((e) => e.event_type === "planning.plan.created");
	if (planEvent) {
		const p = planEvent.payload as {
			plan_id?: string;
			plan_name?: string;
			steps?: Array<{
				step_id: string;
				description: string;
				metadata?: {
					tool?: string;
					args?: Record<string, string>;
					variable?: string;
					depends_on?: number[];
					execution_level?: number;
				};
			}>;
		};
		plan = {
			planId: p.plan_id ?? "",
			planName: p.plan_name ?? "Plan",
			steps: (p.steps ?? []).map((s, i) => ({
				stepNumber: i + 1,
				stepId: s.step_id,
				description: s.description,
				toolName: s.metadata?.tool ?? "",
				args: s.metadata?.args ?? {},
				variable: s.metadata?.variable ?? `#${i + 1}`,
				dependsOn: s.metadata?.depends_on ?? [],
				executionLevel: s.metadata?.execution_level ?? 0,
			})),
		};
	}

	// Extract step updates
	for (const event of events) {
		if (event.event_type === "planning.step.updated") {
			const p = event.payload as {
				step_id?: string;
				step_description?: string;
				previous_status?: string;
				new_status?: string;
				has_result?: boolean;
			};

			// Try to match tool events to this step by step_id or temporal proximity
			const stepIndex = plan?.steps.findIndex((s) => s.stepId === p.step_id);
			const matchedStep = stepIndex != null && stepIndex >= 0 ? plan?.steps[stepIndex] : null;

			// Find tool invoke/result for this step by tool name match
			let matchedInvoke: TraceEvent | null = null;
			let matchedResult: TraceEvent | null = null;
			if (matchedStep) {
				matchedInvoke =
					toolInvokes.find((t) => (t.payload as { tool_name?: string }).tool_name === matchedStep.toolName) ?? null;
				if (matchedInvoke) {
					const invokeTime = new Date(matchedInvoke.timestamp).getTime();
					matchedResult =
						toolResults.find(
							(t) =>
								(t.payload as { tool_name?: string }).tool_name === matchedStep.toolName &&
								new Date(t.timestamp).getTime() >= invokeTime,
						) ?? null;
				}
			}

			stepUpdates.push({
				stepNumber: matchedStep?.stepNumber ?? stepUpdates.length + 1,
				stepId: p.step_id ?? "",
				description: p.step_description ?? "",
				previousStatus: p.previous_status ?? "not_started",
				newStatus: p.new_status ?? "completed",
				hasResult: p.has_result ?? false,
				toolInvoke: matchedInvoke,
				toolResult: matchedResult,
			});
		}
	}

	// Group step executions by execution level
	const executionLevels = new Map<number, StepExecution[]>();
	for (const update of stepUpdates) {
		const matchedPlanStep = plan?.steps.find((s) => s.stepId === update.stepId);
		const level = matchedPlanStep?.executionLevel ?? 0;
		if (!executionLevels.has(level)) {
			executionLevels.set(level, []);
		}
		executionLevels.get(level)?.push(update);
	}

	// Solver: last agent.step after all step updates, plus last LLM events
	const lastStepUpdateTime =
		stepUpdates.length > 0
			? Math.max(
					...events.filter((e) => e.event_type === "planning.step.updated").map((e) => new Date(e.timestamp).getTime()),
				)
			: 0;

	const llmRequests = events.filter(
		(e) => e.event_type === "llm.request" && new Date(e.timestamp).getTime() > lastStepUpdateTime,
	);
	const llmResponses = events.filter(
		(e) => e.event_type === "llm.response" && new Date(e.timestamp).getTime() > lastStepUpdateTime,
	);

	if (llmRequests.length > 0) {
		solver.llmRequest = llmRequests[llmRequests.length - 1];
	}
	if (llmResponses.length > 0) {
		solver.llmResponse = llmResponses[llmResponses.length - 1];
	}

	solver.evaluationResult = events.find((e) => e.event_type === "evaluation.result") ?? null;

	// Last agent.step is the solver step
	if (stepEvents.length > 0) {
		solver.stepEvent = stepEvents[stepEvents.length - 1];
	}

	// Header events
	for (const event of events) {
		if (event.event_type === "agent.start") {
			headerEvents.push(event);
		}
	}

	return { plan, executionLevels, solver, headerEvents, completeEvent };
}

// ---------------------------------------------------------------------------
// Highlight #N variable references in text
// ---------------------------------------------------------------------------

function highlightVariableRefs(text: string): React.ReactNode {
	const parts = text.split(/(#\d+)/g);
	return parts.map((part, i) =>
		/^#\d+$/.test(part) ? (
			// biome-ignore lint/suspicious/noArrayIndexKey: text split fragments have no stable ID
			<span key={i} className="font-mono text-info-muted-foreground bg-info-muted rounded px-0.5">
				{part}
			</span>
		) : (
			// biome-ignore lint/suspicious/noArrayIndexKey: text split fragments have no stable ID
			<span key={i}>{part}</span>
		),
	);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/** ReWOO agent view — plan → execute → solve timeline. */
export function ReWOOAgentView({ events }: AgentViewProps) {
	const { plan, executionLevels, solver, headerEvents, completeEvent } = parseReWOOEvents(events);

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

			{/* Plan section */}
			{plan && <PlanSection plan={plan} />}

			{/* Execute section */}
			{executionLevels.size > 0 && <ExecuteSection executionLevels={executionLevels} planSteps={plan?.steps ?? []} />}

			{/* Solve section */}
			<SolveSection solver={solver} planSteps={plan?.steps ?? []} />

			{/* Completion */}
			{completeEvent && <CompletionSection event={completeEvent} />}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Plan Section
// ---------------------------------------------------------------------------

function PlanSection({ plan }: { plan: PlanInfo }) {
	const [isExpanded, setIsExpanded] = useState(true);

	// Group steps by execution level for the dependency graph
	const levelGroups = new Map<number, PlanStep[]>();
	for (const step of plan.steps) {
		if (!levelGroups.has(step.executionLevel)) {
			levelGroups.set(step.executionLevel, []);
		}
		levelGroups.get(step.executionLevel)?.push(step);
	}
	const sortedLevels = [...levelGroups.keys()].sort((a, b) => a - b);

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-medium inline-flex items-center gap-1">
					<ClipboardList aria-hidden="true" className="h-4 w-4" /> Plan
				</span>
				<span className="text-[10px] text-info-muted-foreground bg-info-muted rounded-full px-1.5 py-0.5">
					{plan.steps.length} step{plan.steps.length !== 1 ? "s" : ""}
				</span>
				<span className="text-xs text-muted-foreground ml-1 truncate">{plan.planName}</span>
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-3 space-y-3">
					{/* Dependency graph — steps grouped by execution level */}
					{sortedLevels.map((level) => (
						<div key={level} className="space-y-1">
							<span className="text-[10px] text-muted-foreground uppercase tracking-wider">Level {level}</span>
							<div className="flex gap-2 flex-wrap">
								{levelGroups.get(level)?.map((step) => (
									<PlanStepNode key={step.stepId} step={step} />
								))}
							</div>
							{/* Dependency arrows (represented textually) */}
							{level < sortedLevels[sortedLevels.length - 1] && (
								<div className="flex justify-center text-muted-foreground text-xs py-0.5">↓</div>
							)}
						</div>
					))}
				</div>
			)}
		</div>
	);
}

function PlanStepNode({ step }: { step: PlanStep }) {
	return (
		<div className="border rounded-md px-2.5 py-1.5 text-xs bg-muted/30 flex-1 min-w-[140px] max-w-[300px]">
			<div className="flex items-center gap-1.5 mb-1">
				<span className="font-mono text-info-muted-foreground font-medium">{step.variable}</span>
				<span className="font-mono text-muted-foreground">{step.toolName}</span>
			</div>
			<div className="text-muted-foreground">{highlightVariableRefs(step.description)}</div>
			{step.dependsOn.length > 0 && (
				<div className="text-[10px] text-muted-foreground mt-1">
					depends on:{" "}
					{step.dependsOn
						.map((d) => (
							<span key={d} className="font-mono text-info-muted-foreground">
								#{d}
							</span>
						))
						.reduce<React.ReactNode[]>((acc, el, i) => {
							if (i > 0) acc.push(", ");
							acc.push(el);
							return acc;
						}, [])}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Execute Section
// ---------------------------------------------------------------------------

function ExecuteSection({
	executionLevels,
	planSteps,
}: {
	executionLevels: Map<number, StepExecution[]>;
	planSteps: PlanStep[];
}) {
	const [isExpanded, setIsExpanded] = useState(true);
	const sortedLevels = [...executionLevels.keys()].sort((a, b) => a - b);
	const totalSteps = [...executionLevels.values()].reduce((sum, steps) => sum + steps.length, 0);

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-medium inline-flex items-center gap-1">
					<Zap aria-hidden="true" className="h-4 w-4" /> Execution
				</span>
				<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5">
					{sortedLevels.length} level{sortedLevels.length !== 1 ? "s" : ""}, {totalSteps} step
					{totalSteps !== 1 ? "s" : ""}
				</span>
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-3 space-y-3">
					{sortedLevels.map((level) => {
						const steps = executionLevels.get(level)!;
						const isParallel = steps.length > 1;
						return (
							<div key={level} className="space-y-1.5">
								<div className="flex items-center gap-2">
									<span className="text-[10px] text-muted-foreground uppercase tracking-wider">Level {level}</span>
									{isParallel && (
										<span className="text-[10px] text-accent-status-muted-foreground bg-accent-status-muted rounded-full px-1.5 py-0.5">
											parallel
										</span>
									)}
								</div>
								<div className={isParallel ? "grid grid-cols-2 gap-2" : "space-y-2"}>
									{steps.map((step) => (
										<ExecutionStepCard
											key={step.stepId}
											step={step}
											planStep={planSteps.find((ps) => ps.stepId === step.stepId)}
										/>
									))}
								</div>
							</div>
						);
					})}
				</div>
			)}
		</div>
	);
}

function ExecutionStepCard({ step, planStep }: { step: StepExecution; planStep?: PlanStep }) {
	const [showDetail, setShowDetail] = useState(false);
	const { registry } = useObservatory();
	const isCompleted = step.newStatus === "completed";
	const isFailed = step.newStatus === "failed";

	const statusColor = isCompleted
		? "text-success-muted-foreground bg-success-muted"
		: isFailed
			? "text-destructive-muted-foreground bg-destructive-muted"
			: "text-muted-foreground bg-muted";

	return (
		<div className="border rounded-md overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-2.5 py-1.5 text-xs text-left hover:bg-accent/30 transition-colors"
				onClick={() => setShowDetail(!showDetail)}
			>
				<span className="text-muted-foreground w-3">{showDetail ? "▾" : "▸"}</span>
				{planStep && <span className="font-mono text-info-muted-foreground">{planStep.variable}</span>}
				<span className="truncate flex-1">{step.description}</span>
				<span className={`text-[10px] rounded-full px-1.5 py-0.5 flex-shrink-0 ${statusColor}`}>{step.newStatus}</span>
			</button>

			{showDetail && (
				<div className="border-t px-2.5 py-2 space-y-2 text-xs">
					{/* Tool invocation detail */}
					{step.toolInvoke && (
						<div>
							<span className="text-muted-foreground">Tool invocation:</span>
							{(() => {
								const Renderer = registry.getRenderer("tool.invoke");
								return Renderer ? (
									<div className="mt-1">
										<Renderer event={step.toolInvoke} />
									</div>
								) : (
									<pre className="mt-1 bg-muted/50 rounded p-1.5 overflow-auto text-[11px]">
										{JSON.stringify(step.toolInvoke.payload, null, 2)}
									</pre>
								);
							})()}
						</div>
					)}

					{/* Tool result detail */}
					{step.toolResult && (
						<div>
							<span className="text-muted-foreground">Result:</span>
							{(() => {
								const Renderer = registry.getRenderer("tool.result");
								return Renderer ? (
									<div className="mt-1">
										<Renderer event={step.toolResult} />
									</div>
								) : (
									<pre className="mt-1 bg-muted/50 rounded p-1.5 overflow-auto text-[11px]">
										{JSON.stringify(step.toolResult.payload, null, 2)}
									</pre>
								);
							})()}
						</div>
					)}

					{!step.toolInvoke && !step.toolResult && (
						<span className="text-muted-foreground">No tool details available.</span>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Solve Section
// ---------------------------------------------------------------------------

function SolveSection({ solver, planSteps }: { solver: SolverInfo; planSteps: PlanStep[] }) {
	const [isExpanded, setIsExpanded] = useState(true);
	const { registry } = useObservatory();

	const hasSolverContent = solver.llmRequest || solver.llmResponse || solver.stepEvent;

	if (!hasSolverContent && !solver.evaluationResult) {
		return null;
	}

	const solverThought = solver.stepEvent ? (solver.stepEvent.payload as { thought?: string }).thought : null;

	// Variable references used — all plan step variables
	const allVariables = planSteps.map((s) => s.variable);

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<span className="font-medium inline-flex items-center gap-1">
					<Brain aria-hidden="true" className="h-4 w-4" /> Synthesis
				</span>
				{solver.evaluationResult &&
					(() => {
						const p = solver.evaluationResult.payload as {
							verdict?: string;
							score?: number;
						};
						const isAccept = p.verdict === "accept";
						return (
							<span
								className={`text-[10px] rounded-full px-1.5 py-0.5 ${
									isAccept
										? "text-success-muted-foreground bg-success-muted"
										: "text-warning-muted-foreground bg-warning-muted"
								}`}
							>
								{p.verdict}
								{p.score != null && ` (${p.score})`}
							</span>
						);
					})()}
			</button>

			{isExpanded && (
				<div className="border-t px-3 py-3 space-y-2">
					{/* Variables resolved */}
					{allVariables.length > 0 && (
						<div className="text-xs text-muted-foreground">
							Variables resolved:{" "}
							{allVariables.map((v, i) => (
								<span key={v}>
									{i > 0 && ", "}
									<span className="font-mono text-info-muted-foreground">{v}</span>
								</span>
							))}
						</div>
					)}

					{/* Solver output */}
					{solverThought && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground">Synthesized answer:</span>
							<div className="text-sm bg-muted/50 rounded-md p-2 whitespace-pre-wrap">
								{highlightVariableRefs(solverThought)}
							</div>
						</div>
					)}

					{/* LLM call details */}
					{solver.llmResponse && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground">LLM call:</span>
							{(() => {
								const Renderer = registry.getRenderer("llm.response");
								return Renderer ? (
									<Renderer event={solver.llmResponse} />
								) : (
									<div className="text-xs text-muted-foreground">
										{(solver.llmResponse.payload as { model_name?: string }).model_name ?? "LLM call"}
									</div>
								);
							})()}
						</div>
					)}

					{/* Evaluation result */}
					{solver.evaluationResult && (
						<div className="space-y-1">
							<span className="text-xs text-muted-foreground">Evaluation:</span>
							{(() => {
								const Renderer = registry.getRenderer("evaluation.result");
								return Renderer ? (
									<Renderer event={solver.evaluationResult} />
								) : (
									<EvaluationResultInline event={solver.evaluationResult} />
								);
							})()}
						</div>
					)}
				</div>
			)}
		</div>
	);
}

function EvaluationResultInline({ event }: { event: TraceEvent }) {
	const p = event.payload as {
		verdict?: string;
		score?: number;
		feedback?: string;
		evaluator_name?: string;
	};

	const isAccept = p.verdict === "accept";

	return (
		<div className="text-xs space-y-1">
			<div className="flex items-center gap-2">
				{p.evaluator_name && <span className="text-muted-foreground">{p.evaluator_name}</span>}
				<span
					className={`rounded-full px-1.5 py-0.5 ${
						isAccept
							? "text-success-muted-foreground bg-success-muted"
							: "text-warning-muted-foreground bg-warning-muted"
					}`}
				>
					{p.verdict}
				</span>
				{p.score != null && <span className="text-muted-foreground">score: {p.score}</span>}
			</div>
			{p.feedback && <div className="bg-muted/50 rounded p-1.5 text-muted-foreground">{p.feedback}</div>}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Completion Section
// ---------------------------------------------------------------------------

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
