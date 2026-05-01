import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { EventDetailPanel } from "../../src/components/event-detail/event-detail-panel";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { createDefaultRegistrations, createDefaultRegistry } from "../../src/registry/default-renderers";
import type { TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderEvent(event: TraceEvent) {
	const client = new ObservatoryClient("/test");
	const registry = createDefaultRegistry();

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<EventDetailPanel event={event} />
		</ObservatoryProvider>,
	);
}

function getSummary(event: TraceEvent): string {
	const registrations = createDefaultRegistrations();
	for (const reg of registrations) {
		if (reg.matches(event.event_type) && reg.summary) {
			return reg.summary(event);
		}
	}
	return event.event_type;
}

// ---------------------------------------------------------------------------
// Memory Event Renderers
// ---------------------------------------------------------------------------

describe("Memory event renderers", () => {
	describe("memory.working.read", () => {
		it("renders token count", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.working.read",
					payload: { content: "Some working memory content", token_count: 450 },
				}),
			);
			expect(screen.getAllByText("450").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("Tokens")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.working.read",
					payload: { token_count: 450 },
				}),
			);
			expect(summary).toBe("Working memory read (450 tokens)");
		});

		it("handles missing optional fields", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.working.read",
					payload: {},
				}),
			);
			expect(screen.getByText("memory.working.read")).toBeInTheDocument();
		});
	});

	describe("memory.working.update", () => {
		it("renders source and content diff", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.working.update",
					payload: {
						source: "tool_result",
						previous_content: "Old content",
						new_content: "New content",
					},
				}),
			);
			expect(screen.getByText("tool_result")).toBeInTheDocument();
			expect(screen.getByText("Old content")).toBeInTheDocument();
			expect(screen.getByText("New content")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.working.update",
					payload: { source: "llm_response" },
				}),
			);
			expect(summary).toBe("Working memory updated (source: llm_response)");
		});
	});

	describe("memory.semantic.store", () => {
		it("renders entry and namespace", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.semantic.store",
					payload: { content: "Important fact", entry_id: "entry-1", namespace: "knowledge" },
				}),
			);
			expect(screen.getByText("entry-1")).toBeInTheDocument();
			expect(screen.getByText("knowledge")).toBeInTheDocument();
			expect(screen.getByText("Important fact")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(makeEvent({ event_type: "memory.semantic.store", payload: {} }));
			expect(summary).toBe("Stored in semantic memory");
		});
	});

	describe("memory.semantic.search", () => {
		it("renders search results info", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.semantic.search",
					payload: { query: "find facts", results_count: 3, top_score: 0.95, namespace: "kb" },
				}),
			);
			expect(screen.getAllByText("3").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("0.950")).toBeInTheDocument();
			expect(screen.getByText("Query")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.semantic.search",
					payload: { results_count: 5, top_score: 0.87 },
				}),
			);
			expect(summary).toBe("Semantic search: 5 results (top: 0.870)");
		});
	});

	describe("memory.semantic.delete", () => {
		it("renders entry id", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.semantic.delete",
					payload: { entry_id: "entry-42" },
				}),
			);
			expect(screen.getByText("entry-42")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(makeEvent({ event_type: "memory.semantic.delete", payload: {} }));
			expect(summary).toBe("Deleted from semantic memory");
		});
	});

	describe("memory.episode.record", () => {
		it("renders episode details", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.episode.record",
					payload: {
						episode_id: "ep-1",
						situation: "Agent tried approach A",
						outcome: "Succeeded with modifications",
						has_reflection: true,
					},
				}),
			);
			expect(screen.getByText("ep-1")).toBeInTheDocument();
			expect(screen.getByText("Agent tried approach A")).toBeInTheDocument();
			expect(screen.getByText("Succeeded with modifications")).toBeInTheDocument();
			expect(screen.getByText("has reflection")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(makeEvent({ event_type: "memory.episode.record", payload: {} }));
			expect(summary).toBe("Episode recorded");
		});
	});

	describe("memory.episode.recall", () => {
		it("renders recall results", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.episode.recall",
					payload: { query: "similar situations", results_count: 2, top_score: 0.82 },
				}),
			);
			expect(screen.getAllByText("2").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("0.820")).toBeInTheDocument();
			expect(screen.getByText("Query")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.episode.recall",
					payload: { results_count: 2, top_score: 0.82 },
				}),
			);
			expect(summary).toBe("Episode recall: 2 results (top: 0.820)");
		});
	});

	describe("memory.episode.forget", () => {
		it("renders episode id", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.episode.forget",
					payload: { episode_id: "ep-old" },
				}),
			);
			expect(screen.getByText("ep-old")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(makeEvent({ event_type: "memory.episode.forget", payload: {} }));
			expect(summary).toBe("Episode forgotten");
		});
	});

	describe("memory.longterm.store", () => {
		it("renders key and value", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.longterm.store",
					payload: { key: "user_preference", value: "dark mode", namespace: "settings" },
				}),
			);
			expect(screen.getByText("user_preference")).toBeInTheDocument();
			expect(screen.getByText("dark mode")).toBeInTheDocument();
			expect(screen.getByText("settings")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.longterm.store",
					payload: { key: "user_preference" },
				}),
			);
			expect(summary).toBe("Stored: user_preference");
		});
	});

	describe("memory.longterm.retrieve", () => {
		it("renders found status", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.longterm.retrieve",
					payload: { key: "api_key", found: true, value: "sk-123" },
				}),
			);
			expect(screen.getByText("api_key")).toBeInTheDocument();
			expect(screen.getByText("found")).toBeInTheDocument();
		});

		it("renders not found status", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.longterm.retrieve",
					payload: { key: "missing_key", found: false },
				}),
			);
			expect(screen.getByText("not found")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.longterm.retrieve",
					payload: { key: "test_key", found: true },
				}),
			);
			expect(summary).toBe("Retrieved: test_key (found)");
		});
	});

	describe("memory.longterm.delete", () => {
		it("renders key", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.longterm.delete",
					payload: { key: "old_key", namespace: "cache" },
				}),
			);
			expect(screen.getByText("old_key")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.longterm.delete",
					payload: { key: "old_key" },
				}),
			);
			expect(summary).toBe("Deleted: old_key");
		});
	});

	describe("memory.longterm.list", () => {
		it("renders key list", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.longterm.list",
					payload: { keys: ["key1", "key2", "key3"], namespace: "data" },
				}),
			);
			expect(screen.getByText("key1")).toBeInTheDocument();
			expect(screen.getByText("key2")).toBeInTheDocument();
			expect(screen.getByText("key3")).toBeInTheDocument();
			expect(screen.getByText("Keys (3)")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.longterm.list",
					payload: { keys: ["a", "b"] },
				}),
			);
			expect(summary).toBe("Listed keys (2)");
		});
	});

	describe("memory.shared.write", () => {
		it("renders author and content", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.shared.write",
					payload: {
						author: "agent-1",
						content: "Shared observation",
						scope: "team",
						entry_id: "shared-1",
					},
				}),
			);
			expect(screen.getByText("agent-1")).toBeInTheDocument();
			expect(screen.getByText("Shared observation")).toBeInTheDocument();
			expect(screen.getByText("team")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.shared.write",
					payload: { author: "agent-1" },
				}),
			);
			expect(summary).toBe("Shared write by agent-1");
		});
	});

	describe("memory.shared.read", () => {
		it("renders entry count", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.shared.read",
					payload: { entries_returned: 5, scope: "global" },
				}),
			);
			expect(screen.getByText("Entries")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.shared.read",
					payload: { entries_returned: 3 },
				}),
			);
			expect(summary).toBe("Shared read (3 entries)");
		});
	});

	describe("memory.shared.supersede", () => {
		it("renders original and new entries", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.shared.supersede",
					payload: {
						author: "agent-2",
						original_entry_id: "old-entry",
						new_entry_id: "new-entry",
						content: "Updated content",
					},
				}),
			);
			expect(screen.getByText("agent-2")).toBeInTheDocument();
			expect(screen.getByText("old-entry")).toBeInTheDocument();
			expect(screen.getByText("new-entry")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.shared.supersede",
					payload: { author: "agent-2" },
				}),
			);
			expect(summary).toBe("Shared supersede by agent-2");
		});
	});

	describe("memory.shared.retract", () => {
		it("renders retract reason", () => {
			renderEvent(
				makeEvent({
					event_type: "memory.shared.retract",
					payload: {
						author: "agent-1",
						entry_id: "retracted-1",
						reason: "Information was incorrect",
					},
				}),
			);
			expect(screen.getByText("agent-1")).toBeInTheDocument();
			expect(screen.getByText("Information was incorrect")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "memory.shared.retract",
					payload: { author: "agent-1" },
				}),
			);
			expect(summary).toBe("Shared retract by agent-1");
		});
	});
});

// ---------------------------------------------------------------------------
// Planning Event Renderers
// ---------------------------------------------------------------------------

describe("Planning event renderers", () => {
	describe("planning.plan.created", () => {
		it("renders plan name and step count", () => {
			renderEvent(
				makeEvent({
					event_type: "planning.plan.created",
					payload: {
						plan_name: "Research Plan",
						step_count: 5,
						goal_count: 2,
						steps: [
							{ step_id: "s1", description: "Gather data" },
							{ step_id: "s2", description: "Analyze results" },
						],
					},
				}),
			);
			expect(screen.getByText("Research Plan")).toBeInTheDocument();
			expect(screen.getByText("5 steps")).toBeInTheDocument();
			expect(screen.getByText("2 goals")).toBeInTheDocument();
			expect(screen.getByText("Gather data")).toBeInTheDocument();
			expect(screen.getByText("Analyze results")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "planning.plan.created",
					payload: { plan_name: "Build App", step_count: 3 },
				}),
			);
			expect(summary).toBe("Plan created: Build App (3 steps)");
		});
	});

	describe("planning.step.updated", () => {
		it("renders status transition", () => {
			renderEvent(
				makeEvent({
					event_type: "planning.step.updated",
					payload: {
						step_description: "Fetch API data",
						previous_status: "not_started",
						new_status: "in_progress",
						has_result: false,
					},
				}),
			);
			expect(screen.getByText("Fetch API data")).toBeInTheDocument();
			expect(screen.getByText("not_started")).toBeInTheDocument();
			expect(screen.getByText("in_progress")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "planning.step.updated",
					payload: {
						step_description: "Fetch API data",
						previous_status: "not_started",
						new_status: "completed",
					},
				}),
			);
			expect(summary).toBe("Step 'Fetch API data': not_started → completed");
		});
	});

	describe("planning.plan.revised", () => {
		it("renders revision details", () => {
			renderEvent(
				makeEvent({
					event_type: "planning.plan.revised",
					payload: {
						steps_before: 3,
						steps_after: 5,
						steps_preserved: 2,
						revision_reason: "New requirements discovered",
					},
				}),
			);
			expect(screen.getByText("New requirements discovered")).toBeInTheDocument();
			expect(screen.getByText("Preserved")).toBeInTheDocument();
			expect(screen.getByText("steps")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "planning.plan.revised",
					payload: { steps_before: 3, steps_after: 5 },
				}),
			);
			expect(summary).toBe("Plan revised (3 → 5 steps)");
		});
	});

	describe("planning.goal.status_changed", () => {
		it("renders goal status transition", () => {
			renderEvent(
				makeEvent({
					event_type: "planning.goal.status_changed",
					payload: {
						goal_description: "Complete data analysis",
						previous_status: "in_progress",
						new_status: "achieved",
					},
				}),
			);
			expect(screen.getByText("Complete data analysis")).toBeInTheDocument();
			expect(screen.getByText("in_progress")).toBeInTheDocument();
			expect(screen.getByText("achieved")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "planning.goal.status_changed",
					payload: {
						goal_description: "Build feature",
						previous_status: "in_progress",
						new_status: "achieved",
					},
				}),
			);
			expect(summary).toBe("Goal 'Build feature': in_progress → achieved");
		});
	});
});

// ---------------------------------------------------------------------------
// Evaluation & Reflection Event Renderers
// ---------------------------------------------------------------------------

describe("Evaluation & Reflection event renderers", () => {
	describe("evaluation.result", () => {
		it("renders verdict and score", () => {
			renderEvent(
				makeEvent({
					event_type: "evaluation.result",
					payload: {
						evaluator_name: "QualityEvaluator",
						verdict: "ACCEPT",
						score: 0.92,
						feedback: "Well-structured response",
						revision_attempt: 2,
					},
				}),
			);
			expect(screen.getByText("QualityEvaluator")).toBeInTheDocument();
			expect(screen.getByText("ACCEPT")).toBeInTheDocument();
			expect(screen.getAllByText("0.92").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("Well-structured response")).toBeInTheDocument();
			expect(screen.getByText("attempt 2")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "evaluation.result",
					payload: { verdict: "REVISE", score: 0.4 },
				}),
			);
			expect(summary).toBe("Evaluation: REVISE (score: 0.40)");
		});
	});

	describe("evaluation.revision", () => {
		it("renders attempt info", () => {
			renderEvent(
				makeEvent({
					event_type: "evaluation.revision",
					payload: {
						feedback: "Please improve formatting",
						revision_attempt: 1,
						max_revisions: 3,
					},
				}),
			);
			expect(screen.getByText("Attempt 1 of 3")).toBeInTheDocument();
			expect(screen.getByText("Please improve formatting")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "evaluation.revision",
					payload: { revision_attempt: 2, max_revisions: 5 },
				}),
			);
			expect(summary).toBe("Revision requested (attempt 2/5)");
		});
	});

	describe("reflection.generated", () => {
		it("renders reflection text and feedback", () => {
			renderEvent(
				makeEvent({
					event_type: "reflection.generated",
					payload: {
						attempt_number: 1,
						max_attempts: 3,
						reflection_text: "I should focus more on conciseness",
						evaluation_feedback: "Response was too verbose",
						episode_id: "ep-reflection-1",
					},
				}),
			);
			expect(screen.getByText("Attempt 1 of 3")).toBeInTheDocument();
			expect(screen.getByText("I should focus more on conciseness")).toBeInTheDocument();
			expect(screen.getByText("Response was too verbose")).toBeInTheDocument();
			expect(screen.getByText("ep-reflection-1")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "reflection.generated",
					payload: { attempt_number: 2, max_attempts: 4 },
				}),
			);
			expect(summary).toBe("Reflection generated (attempt 2/4)");
		});
	});
});

// ---------------------------------------------------------------------------
// HITL Event Renderers
// ---------------------------------------------------------------------------

describe("HITL event renderers", () => {
	describe("hitl.request", () => {
		it("renders request type and prompt", () => {
			renderEvent(
				makeEvent({
					event_type: "hitl.request",
					payload: {
						request_id: "req-1",
						request_type: "approval",
						prompt: "Should I proceed with deletion?",
						agent_name: "cleanup-agent",
						tool_name: "delete_file",
					},
				}),
			);
			expect(screen.getByText("approval")).toBeInTheDocument();
			expect(screen.getByText("Should I proceed with deletion?")).toBeInTheDocument();
			expect(screen.getByText("cleanup-agent")).toBeInTheDocument();
			expect(screen.getByText("delete_file")).toBeInTheDocument();
		});

		it("produces correct summary with truncation", () => {
			const longPrompt = "A".repeat(60);
			const summary = getSummary(
				makeEvent({
					event_type: "hitl.request",
					payload: { request_type: "approval", prompt: longPrompt },
				}),
			);
			expect(summary).toContain("HITL: approval");
			expect(summary).toContain("…");
			expect(summary.length).toBeLessThan(80);
		});

		it("handles missing optional fields", () => {
			renderEvent(
				makeEvent({
					event_type: "hitl.request",
					payload: { request_id: "req-2", request_type: "question" },
				}),
			);
			expect(screen.getByText("question")).toBeInTheDocument();
		});
	});

	describe("hitl.response", () => {
		it("renders decision and wait duration", () => {
			renderEvent(
				makeEvent({
					event_type: "hitl.response",
					payload: {
						request_id: "req-1",
						decision: "approve",
						has_content: false,
						wait_duration_ms: 5200,
					},
				}),
			);
			expect(screen.getByText("approve")).toBeInTheDocument();
			expect(screen.getByText("5.2s")).toBeInTheDocument();
		});

		it("renders reject decision with red styling", () => {
			renderEvent(
				makeEvent({
					event_type: "hitl.response",
					payload: {
						request_id: "req-2",
						decision: "reject",
						has_content: false,
						wait_duration_ms: 800,
					},
				}),
			);
			expect(screen.getByText("reject")).toBeInTheDocument();
			expect(screen.getByText("800ms")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "hitl.response",
					payload: { decision: "approve", wait_duration_ms: 3000 },
				}),
			);
			expect(summary).toBe("HITL response: approve (waited 3000ms)");
		});
	});
});

// ---------------------------------------------------------------------------
// Renderer registration verification
// ---------------------------------------------------------------------------

describe("Default registrations include memory and planning renderers", () => {
	const registrations = createDefaultRegistrations();

	const expectedEventTypes = [
		"memory.working.read",
		"memory.working.update",
		"memory.semantic.store",
		"memory.semantic.search",
		"memory.semantic.delete",
		"memory.episode.record",
		"memory.episode.recall",
		"memory.episode.forget",
		"memory.longterm.store",
		"memory.longterm.retrieve",
		"memory.longterm.delete",
		"memory.longterm.list",
		"memory.shared.write",
		"memory.shared.read",
		"memory.shared.supersede",
		"memory.shared.retract",
		"planning.plan.created",
		"planning.step.updated",
		"planning.plan.revised",
		"planning.goal.status_changed",
		"evaluation.result",
		"evaluation.revision",
		"reflection.generated",
		"hitl.request",
		"hitl.response",
	];

	for (const eventType of expectedEventTypes) {
		it(`has a renderer for ${eventType}`, () => {
			const match = registrations.find((r) => r.matches(eventType) && r.priority >= 0);
			expect(match).toBeDefined();
			expect(match?.summary).toBeDefined();
		});
	}
});
