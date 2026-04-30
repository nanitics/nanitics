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
// CheckpointSaved
// ---------------------------------------------------------------------------

describe("checkpoint.saved renderer", () => {
	it("renders checkpoint type and IDs", () => {
		renderEvent(
			makeEvent({
				event_type: "checkpoint.saved",
				payload: {
					checkpoint_id: "cp-001",
					checkpoint_type: "orchestration",
					run_id: "run-abc",
				},
			}),
		);
		expect(screen.getByText("orchestration")).toBeInTheDocument();
		expect(screen.getByText("cp-001")).toBeInTheDocument();
		expect(screen.getByText("run-abc")).toBeInTheDocument();
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "checkpoint.saved",
				payload: { checkpoint_type: "agent" },
			}),
		);
		expect(summary).toBe("Checkpoint saved (agent)");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "checkpoint.saved", payload: {} }));
		expect(screen.getByText("checkpoint.saved")).toBeInTheDocument();
	});

	it("summary falls back for missing checkpoint_type", () => {
		const summary = getSummary(makeEvent({ event_type: "checkpoint.saved", payload: {} }));
		expect(summary).toBe("Checkpoint saved (unknown)");
	});
});

// ---------------------------------------------------------------------------
// ExecutionSuspended
// ---------------------------------------------------------------------------

describe("execution.suspended renderer", () => {
	it("renders suspension type, step, and agent", () => {
		renderEvent(
			makeEvent({
				event_type: "execution.suspended",
				payload: {
					suspension_id: "sus-001",
					suspension_type: "hitl",
					checkpoint_id: "cp-001",
					step_name: "review_step",
					agent_name: "reviewer",
				},
			}),
		);
		expect(screen.getByText("hitl")).toBeInTheDocument();
		expect(screen.getByText("review_step")).toBeInTheDocument();
		expect(screen.getByText("reviewer")).toBeInTheDocument();
		expect(screen.getByText("sus-001")).toBeInTheDocument();
		expect(screen.getByText("cp-001")).toBeInTheDocument();
	});

	it("produces summary with step name", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "execution.suspended",
				payload: { step_name: "review_step", suspension_type: "hitl" },
			}),
		);
		expect(summary).toBe("Suspended at 'review_step' (hitl)");
	});

	it("produces summary without step name", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "execution.suspended",
				payload: { suspension_type: "hitl" },
			}),
		);
		expect(summary).toBe("Execution suspended (hitl)");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "execution.suspended", payload: {} }));
		expect(screen.getByText("execution.suspended")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// ExecutionResumed
// ---------------------------------------------------------------------------

describe("execution.resumed renderer", () => {
	it("renders Resumed badge and IDs", () => {
		renderEvent(
			makeEvent({
				event_type: "execution.resumed",
				payload: {
					checkpoint_id: "cp-001",
					suspension_id: "sus-001",
					resumed_from_step: "review_step",
				},
			}),
		);
		expect(screen.getByText("Resumed")).toBeInTheDocument();
		expect(screen.getByText("review_step")).toBeInTheDocument();
		expect(screen.getByText("cp-001")).toBeInTheDocument();
		expect(screen.getByText("sus-001")).toBeInTheDocument();
	});

	it("produces summary with step name", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "execution.resumed",
				payload: { resumed_from_step: "review_step" },
			}),
		);
		expect(summary).toBe("Resumed from 'review_step'");
	});

	it("produces summary without step name", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "execution.resumed",
				payload: {},
			}),
		);
		expect(summary).toBe("Execution resumed");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "execution.resumed", payload: {} }));
		expect(screen.getByText("Resumed")).toBeInTheDocument();
	});
});
