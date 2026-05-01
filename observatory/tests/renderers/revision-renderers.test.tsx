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
// revision.start
// ---------------------------------------------------------------------------

describe("revision.start renderer", () => {
	it("renders step name, worker count, and max revisions", () => {
		renderEvent(
			makeEvent({
				event_type: "revision.start",
				payload: {
					step_name: "review_step",
					worker_count: 2,
					max_revisions: 10,
				},
			}),
		);
		expect(screen.getByText("Revision workflow")).toBeInTheDocument();
		expect(screen.getByText("review_step")).toBeInTheDocument();
		expect(screen.getByText("Workers")).toBeInTheDocument();
		expect(screen.getByText("Max revisions")).toBeInTheDocument();
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "revision.start",
				payload: {
					step_name: "review_step",
					worker_count: 2,
					max_revisions: 10,
				},
			}),
		);
		expect(summary).toBe("Revision: review_step (2 workers, max 10)");
	});

	it("produces fallback summary without details", () => {
		const summary = getSummary(makeEvent({ event_type: "revision.start", payload: {} }));
		expect(summary).toBe("Revision workflow started");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "revision.start", payload: {} }));
		expect(screen.getByText("revision.start")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// revision.attempt
// ---------------------------------------------------------------------------

describe("revision.attempt renderer", () => {
	it("renders attempt number and feedback", () => {
		renderEvent(
			makeEvent({
				event_type: "revision.attempt",
				payload: {
					step_name: "review_step",
					attempt_number: 1,
					feedback: "Please add more detail to the analysis",
				},
			}),
		);
		expect(screen.getByText("Attempt 1")).toBeInTheDocument();
		expect(screen.getByText("Please add more detail to the analysis")).toBeInTheDocument();
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "revision.attempt",
				payload: {
					attempt_number: 1,
					feedback: "needs more detail",
				},
			}),
		);
		expect(summary).toBe("Revision attempt 1: needs more detail");
	});

	it("truncates long feedback in summary", () => {
		const longFeedback = "A".repeat(60);
		const summary = getSummary(
			makeEvent({
				event_type: "revision.attempt",
				payload: { attempt_number: 2, feedback: longFeedback },
			}),
		);
		expect(summary).toBe(`Revision attempt 2: ${"A".repeat(50)}\u2026`);
	});

	it("produces fallback summary without details", () => {
		const summary = getSummary(makeEvent({ event_type: "revision.attempt", payload: {} }));
		expect(summary).toBe("Revision attempt");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "revision.attempt", payload: {} }));
		expect(screen.getByText("revision.attempt")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// revision.complete
// ---------------------------------------------------------------------------

describe("revision.complete renderer", () => {
	it("renders final decision and total attempts", () => {
		renderEvent(
			makeEvent({
				event_type: "revision.complete",
				payload: {
					step_name: "review_step",
					final_decision: "approve",
					total_attempts: 2,
				},
			}),
		);
		expect(screen.getByText("approve")).toBeInTheDocument();
		expect(screen.getByText("Total attempts")).toBeInTheDocument();
		expect(screen.getAllByText("2").length).toBeGreaterThanOrEqual(1);
	});

	it("renders reject decision with correct styling", () => {
		renderEvent(
			makeEvent({
				event_type: "revision.complete",
				payload: {
					step_name: "review_step",
					final_decision: "reject",
					total_attempts: 1,
				},
			}),
		);
		expect(screen.getByText("reject")).toBeInTheDocument();
	});

	it("renders max_revisions_exceeded decision", () => {
		renderEvent(
			makeEvent({
				event_type: "revision.complete",
				payload: {
					step_name: "review_step",
					final_decision: "max_revisions_exceeded",
					total_attempts: 10,
				},
			}),
		);
		expect(screen.getByText("max_revisions_exceeded")).toBeInTheDocument();
		expect(screen.getAllByText("10").length).toBeGreaterThanOrEqual(1);
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "revision.complete",
				payload: { final_decision: "approve", total_attempts: 2 },
			}),
		);
		expect(summary).toBe("Revision complete: approve (2 attempts)");
	});

	it("produces fallback summary without details", () => {
		const summary = getSummary(makeEvent({ event_type: "revision.complete", payload: {} }));
		expect(summary).toBe("Revision complete");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "revision.complete", payload: {} }));
		expect(screen.getByText("revision.complete")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

describe("revision renderer registrations", () => {
	it("registers renderers for all three revision event types", () => {
		const registrations = createDefaultRegistrations();
		const revisionTypes = ["revision.start", "revision.attempt", "revision.complete"];

		for (const eventType of revisionTypes) {
			const match = registrations.find((r) => r.matches(eventType) && r.priority === 0 && r.summary);
			expect(match).toBeDefined();
		}
	});
});
