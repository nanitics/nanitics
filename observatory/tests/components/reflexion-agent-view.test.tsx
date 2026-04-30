import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ReflexionAgentView } from "../../src/components/agent-views/reflexion-agent-view";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "reflexion-agent",
	agent_type: "reflexion",
	span_id: "agent-1",
	capabilities: ["evaluation", "memory"],
	stats: {
		llm_calls: 6,
		tool_calls: 2,
		input_tokens: 4000,
		output_tokens: 2000,
		duration_ms: 15000,
		errors: 0,
		iterations: 3,
	},
};

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "reflexion-agent",
	summary: {
		event_count: 20,
		duration_ms: 15000,
		has_errors: false,
		agent_name: "reflexion-agent",
		agent_type: "reflexion",
	},
	events: [],
	children: [],
};

function renderView(events: TraceEvent[]) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<ReflexionAgentView agent={mockAgent} events={events} spanTree={mockSpanTree} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Multi-attempt scenario: attempt 1 (revise), attempt 2 (revise), attempt 3 (accept). */
function makeMultiAttemptEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "agent.start",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:00Z",
			payload: { agent_name: "reflexion-agent" },
		}),
		// Attempt 1
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:01Z",
			payload: { step: 1, thought: "First try at the task" },
		}),
		makeEvent({
			event_type: "llm.request",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:02Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "llm.response",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:03Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "evaluation.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:04Z",
			payload: {
				evaluator_name: "quality-evaluator",
				verdict: "revise",
				score: 0.4,
				feedback: "Response lacks depth and specificity",
				revision_attempt: 1,
			},
		}),
		makeEvent({
			event_type: "reflection.generated",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:05Z",
			payload: {
				attempt_number: 1,
				max_attempts: 3,
				reflection_text: "I need to provide more specific examples and data points.",
				evaluation_feedback: "Response lacks depth and specificity",
				episode_id: "ep-1",
			},
		}),
		makeEvent({
			event_type: "memory.episode.record",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:05.5Z",
			payload: {
				episode_id: "ep-1",
				situation: "Write about climate change",
				outcome: "failure",
				has_reflection: true,
			},
		}),
		// Attempt 2
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:06Z",
			payload: { step: 2, thought: "Second try with more detail" },
		}),
		makeEvent({
			event_type: "llm.request",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:07Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "llm.response",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:08Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "evaluation.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:09Z",
			payload: {
				evaluator_name: "quality-evaluator",
				verdict: "revise",
				score: 0.6,
				feedback: "Better but still needs more concrete data",
				revision_attempt: 2,
			},
		}),
		makeEvent({
			event_type: "reflection.generated",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:10Z",
			payload: {
				attempt_number: 2,
				max_attempts: 3,
				reflection_text: "I should include specific statistics and cite sources.",
				evaluation_feedback: "Better but still needs more concrete data",
				episode_id: "ep-2",
			},
		}),
		makeEvent({
			event_type: "memory.episode.record",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:10.5Z",
			payload: {
				episode_id: "ep-2",
				situation: "Write about climate change",
				outcome: "failure",
				has_reflection: true,
			},
		}),
		// Attempt 3 (accepted)
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:11Z",
			payload: { step: 3, thought: "Third attempt with statistics" },
		}),
		makeEvent({
			event_type: "llm.request",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:12Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "llm.response",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:13Z",
			payload: { model_name: "claude-haiku-4-5-20251001" },
		}),
		makeEvent({
			event_type: "evaluation.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:14Z",
			payload: {
				evaluator_name: "quality-evaluator",
				verdict: "accept",
				score: 0.9,
				feedback: "Excellent response with concrete data",
				revision_attempt: 3,
			},
		}),
		makeEvent({
			event_type: "memory.episode.record",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:14.5Z",
			payload: {
				episode_id: "ep-3",
				situation: "Write about climate change",
				outcome: "success",
				has_reflection: false,
			},
		}),
		makeEvent({
			event_type: "agent.complete",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:15Z",
			payload: { termination_reason: "complete", total_steps: 3 },
		}),
	];
}

/** Single attempt (immediate accept) scenario. */
function makeSingleAttemptEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "agent.start",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:00Z",
			payload: { agent_name: "reflexion-agent" },
		}),
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:01Z",
			payload: { step: 1, thought: "Single attempt" },
		}),
		makeEvent({
			event_type: "evaluation.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:03Z",
			payload: {
				evaluator_name: "quality-evaluator",
				verdict: "accept",
				score: 0.95,
				feedback: "Excellent work",
				revision_attempt: 1,
			},
		}),
		makeEvent({
			event_type: "memory.episode.record",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:03.5Z",
			payload: {
				episode_id: "ep-1",
				situation: "Simple task",
				outcome: "success",
				has_reflection: false,
			},
		}),
		makeEvent({
			event_type: "agent.complete",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:04Z",
			payload: { termination_reason: "complete", total_steps: 1 },
		}),
	];
}

/** Max-attempts-exhausted scenario. */
function makeMaxAttemptsEvents(): TraceEvent[] {
	return [
		makeEvent({
			event_type: "agent.start",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:00Z",
			payload: { agent_name: "reflexion-agent" },
		}),
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:01Z",
			payload: { step: 1 },
		}),
		makeEvent({
			event_type: "evaluation.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:02Z",
			payload: {
				evaluator_name: "strict-eval",
				verdict: "revise",
				score: 0.3,
				feedback: "Not good enough",
				revision_attempt: 1,
			},
		}),
		makeEvent({
			event_type: "reflection.generated",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:03Z",
			payload: {
				attempt_number: 1,
				max_attempts: 2,
				reflection_text: "Need to improve approach",
				evaluation_feedback: "Not good enough",
				episode_id: "ep-fail-1",
			},
		}),
		makeEvent({
			event_type: "agent.step",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:04Z",
			payload: { step: 2 },
		}),
		makeEvent({
			event_type: "evaluation.result",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:05Z",
			payload: {
				evaluator_name: "strict-eval",
				verdict: "revise",
				score: 0.4,
				feedback: "Still not sufficient",
				revision_attempt: 2,
			},
		}),
		makeEvent({
			event_type: "agent.complete",
			span_id: "agent-1",
			timestamp: "2026-03-05T10:00:06Z",
			payload: { termination_reason: "evaluation_failed" },
		}),
	];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ReflexionAgentView", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows empty state when no events", () => {
		renderView([]);
		expect(screen.getByText("No events recorded for this agent.")).toBeInTheDocument();
	});

	it("renders score progression header with correct scores", () => {
		renderView(makeMultiAttemptEvents());

		// Scores appear in both progression and attempt cards
		expect(screen.getAllByText(/0\.4/).length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText(/0\.6/).length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText(/0\.9/).length).toBeGreaterThanOrEqual(1);
	});

	it("renders attempt sections with evaluation results", () => {
		renderView(makeMultiAttemptEvents());

		// Three attempt headers
		expect(screen.getByText("Attempt 1")).toBeInTheDocument();
		expect(screen.getByText("Attempt 2")).toBeInTheDocument();
		expect(screen.getByText("Attempt 3")).toBeInTheDocument();

		// Evaluation cards present — "Evaluation" label in cards
		expect(screen.getAllByText("Evaluation").length).toBe(3);

		// Verdicts
		expect(screen.getAllByText("revise").length).toBeGreaterThanOrEqual(2);
		expect(screen.getAllByText("accept").length).toBeGreaterThanOrEqual(1);
	});

	it("shows reflection cards between attempts", () => {
		renderView(makeMultiAttemptEvents());

		// Reflection cards (label text only; the icon is aria-hidden)
		expect(screen.getAllByText("Reflection").length).toBe(2);
		expect(screen.getByText("I need to provide more specific examples and data points.")).toBeInTheDocument();
		expect(screen.getByText("I should include specific statistics and cite sources.")).toBeInTheDocument();
	});

	it("shows final outcome", () => {
		renderView(makeMultiAttemptEvents());

		expect(screen.getByText("Accepted")).toBeInTheDocument();
		expect(screen.getByText("3 attempts")).toBeInTheDocument();
	});

	it("handles single-attempt (immediate accept) scenario", () => {
		renderView(makeSingleAttemptEvents());

		expect(screen.getByText("Attempt 1")).toBeInTheDocument();
		expect(screen.getByText("Accepted")).toBeInTheDocument();
		expect(screen.getByText("1 attempt")).toBeInTheDocument();

		// No reflection cards
		expect(screen.queryByText("Reflection")).not.toBeInTheDocument();
	});

	it("handles max-attempts-exhausted scenario", () => {
		renderView(makeMaxAttemptsEvents());

		expect(screen.getByText("Attempt 1")).toBeInTheDocument();
		expect(screen.getByText("Attempt 2")).toBeInTheDocument();

		// Shows failed outcome
		expect(screen.getByText("Failed")).toBeInTheDocument();
		expect(screen.getByText("(evaluation_failed)")).toBeInTheDocument();
		expect(screen.getByText("2 attempts")).toBeInTheDocument();
	});

	it("shows evaluator name and feedback in evaluation cards", () => {
		renderView(makeMultiAttemptEvents());

		// Evaluator name appears
		expect(screen.getAllByText("quality-evaluator").length).toBeGreaterThanOrEqual(1);

		// Feedback text appears (may appear in both evaluation card and reflection "triggered by")
		expect(screen.getAllByText("Response lacks depth and specificity").length).toBeGreaterThanOrEqual(1);
	});

	it("shows episode recorded badges", () => {
		renderView(makeMultiAttemptEvents());

		// Success episode for accepted attempt (text includes emoji so use regex)
		expect(screen.getByText(/Success episode recorded/)).toBeInTheDocument();

		// Failed attempt episodes (2 failure episodes + 1 success = 3 total matches of /Episode recorded/)
		// But getAllByText matches text content of elements, and the success badge
		// text is "Success episode recorded" which also contains "Episode recorded"
		const allEpisodeBadges = screen.getAllByText(/episode recorded/i);
		expect(allEpisodeBadges.length).toBeGreaterThanOrEqual(2);
	});

	it("shows reflection triggered-by feedback", () => {
		renderView(makeMultiAttemptEvents());

		// Reflection cards show "Triggered by:" with linked feedback
		expect(screen.getAllByText("Triggered by:").length).toBe(2);
	});

	it("shows inner agent events as expandable", () => {
		renderView(makeMultiAttemptEvents());

		// Each attempt has inner events (LLM calls)
		const innerEventToggles = screen.getAllByText(/inner agent event/);
		expect(innerEventToggles.length).toBeGreaterThanOrEqual(1);

		// Click to expand first one
		fireEvent.click(innerEventToggles[0]);

		// LLM events should be visible
		expect(screen.getAllByText("llm.request").length).toBeGreaterThanOrEqual(1);
	});

	it("renders score progression with checkmark on accepted attempt", () => {
		renderView(makeMultiAttemptEvents());

		// The accepted score (0.9) should have a checkmark
		expect(screen.getByText(/0\.9 ✓/)).toBeInTheDocument();
	});

	it("renders score progression with × on revised attempts", () => {
		renderView(makeMultiAttemptEvents());

		// Revised attempts should have ×
		expect(screen.getByText(/0\.4 ×/)).toBeInTheDocument();
		expect(screen.getByText(/0\.6 ×/)).toBeInTheDocument();
	});
});
