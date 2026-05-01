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
// Run Lifecycle Event Renderers
// ---------------------------------------------------------------------------

describe("Run lifecycle event renderers", () => {
	describe("run.start", () => {
		it("renders run_id and workflow name", () => {
			renderEvent(
				makeEvent({
					event_type: "run.start",
					payload: { run_id: "run-abc-123", workflow_name: "research", metadata: {} },
				}),
			);
			expect(screen.getByText("run-abc-123")).toBeInTheDocument();
			expect(screen.getByText("research")).toBeInTheDocument();
			expect(screen.getByText("running")).toBeInTheDocument();
		});

		it("produces correct summary with workflow name", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.start",
					payload: { run_id: "run-abc-123", workflow_name: "research" },
				}),
			);
			expect(summary).toBe("Run started: 'research'");
		});

		it("produces summary with run_id when no workflow name", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.start",
					payload: { run_id: "run-abc-123" },
				}),
			);
			expect(summary).toBe("Run started: run-abc-123");
		});

		it("handles missing optional fields", () => {
			renderEvent(makeEvent({ event_type: "run.start", payload: {} }));
			expect(screen.getByText("run.start")).toBeInTheDocument();
		});
	});

	describe("run.complete", () => {
		it("renders run_id and duration", () => {
			renderEvent(
				makeEvent({
					event_type: "run.complete",
					payload: { run_id: "run-abc-123", duration_ms: 4500 },
				}),
			);
			expect(screen.getByText("run-abc-123")).toBeInTheDocument();
			expect(screen.getAllByText("4.5s").length).toBeGreaterThanOrEqual(1);
			expect(screen.getByText("completed")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.complete",
					payload: { run_id: "run-abc-123", duration_ms: 800 },
				}),
			);
			expect(summary).toBe("Run completed (800ms)");
		});

		it("handles missing duration", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.complete",
					payload: { run_id: "run-abc-123" },
				}),
			);
			expect(summary).toBe("Run completed (—)");
		});

		it("handles missing optional fields", () => {
			renderEvent(makeEvent({ event_type: "run.complete", payload: {} }));
			expect(screen.getByText("run.complete")).toBeInTheDocument();
		});
	});

	describe("run.failed", () => {
		it("renders error details", () => {
			renderEvent(
				makeEvent({
					event_type: "run.failed",
					payload: {
						run_id: "run-abc-123",
						error_type: "TimeoutError",
						error_message: "Agent exceeded time limit",
					},
				}),
			);
			expect(screen.getByText("run-abc-123")).toBeInTheDocument();
			expect(screen.getByText("TimeoutError")).toBeInTheDocument();
			expect(screen.getByText("Agent exceeded time limit")).toBeInTheDocument();
			expect(screen.getByText("failed")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.failed",
					payload: { run_id: "run-abc-123", error_type: "TimeoutError", error_message: "timeout" },
				}),
			);
			expect(summary).toBe("Run failed: TimeoutError");
		});

		it("produces summary with unknown error when missing error_type", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.failed",
					payload: { run_id: "run-abc-123" },
				}),
			);
			expect(summary).toBe("Run failed: unknown error");
		});

		it("handles missing optional fields", () => {
			renderEvent(makeEvent({ event_type: "run.failed", payload: {} }));
			expect(screen.getByText("run.failed")).toBeInTheDocument();
		});
	});

	describe("run.suspended", () => {
		it("renders suspension details", () => {
			renderEvent(
				makeEvent({
					event_type: "run.suspended",
					payload: { run_id: "run-abc-123", suspension_id: "susp-456" },
				}),
			);
			expect(screen.getByText("run-abc-123")).toBeInTheDocument();
			expect(screen.getByText("susp-456")).toBeInTheDocument();
			expect(screen.getByText("suspended")).toBeInTheDocument();
		});

		it("produces correct summary", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.suspended",
					payload: { run_id: "run-abc-123", suspension_id: "susp-456" },
				}),
			);
			expect(summary).toBe("Run suspended (suspension: susp-456)");
		});

		it("produces summary without suspension_id", () => {
			const summary = getSummary(
				makeEvent({
					event_type: "run.suspended",
					payload: { run_id: "run-abc-123" },
				}),
			);
			expect(summary).toBe("Run suspended");
		});

		it("handles missing optional fields", () => {
			renderEvent(makeEvent({ event_type: "run.suspended", payload: {} }));
			expect(screen.getByText("run.suspended")).toBeInTheDocument();
		});
	});
});
