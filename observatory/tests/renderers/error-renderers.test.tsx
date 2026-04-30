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
// ErrorRetryRenderer
// ---------------------------------------------------------------------------

describe("error.retry renderer", () => {
	it("displays all fields", () => {
		renderEvent(
			makeEvent({
				event_type: "error.retry",
				payload: {
					error_type: "ConnectionError",
					error_message: "Connection refused",
					attempt: 2,
					max_attempts: 5,
					delay_ms: 1500,
					category: "transient",
				},
			}),
		);

		expect(screen.getByText("ConnectionError")).toBeInTheDocument();
		expect(screen.getByText("Connection refused")).toBeInTheDocument();
		expect(screen.getByText("attempt 2/5")).toBeInTheDocument();
		expect(screen.getByText("1500ms")).toBeInTheDocument();
		expect(screen.getByText("transient")).toBeInTheDocument();
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "error.retry",
				payload: {
					error_type: "Timeout",
					attempt: 1,
					max_attempts: 3,
				},
			}),
		);
		expect(summary).toBe("Retry attempt 1/3 — Timeout");
	});

	it("produces fallback summary without fields", () => {
		const summary = getSummary(makeEvent({ event_type: "error.retry", payload: {} }));
		expect(summary).toBe("Retry");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "error.retry", payload: {} }));
		// Should render without crashing — the PayloadViewer fallback shows event type
		expect(screen.getByText("error.retry")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// ErrorCorrectionRenderer
// ---------------------------------------------------------------------------

describe("error.correction renderer", () => {
	it("displays all fields including correction_prompt", () => {
		renderEvent(
			makeEvent({
				event_type: "error.correction",
				payload: {
					error_type: "ParseError",
					error_message: "Invalid JSON output",
					correction_prompt: "Please respond with valid JSON only.",
					attempt: 1,
					max_attempts: 3,
				},
			}),
		);

		expect(screen.getByText("ParseError")).toBeInTheDocument();
		expect(screen.getByText("Invalid JSON output")).toBeInTheDocument();
		expect(screen.getByText("attempt 1/3")).toBeInTheDocument();
		expect(screen.getByText("Please respond with valid JSON only.")).toBeInTheDocument();
		expect(screen.getByText("Correction prompt")).toBeInTheDocument();
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "error.correction",
				payload: {
					error_type: "FormatError",
					attempt: 2,
					max_attempts: 3,
				},
			}),
		);
		expect(summary).toBe("Correction attempt 2/3 — FormatError");
	});

	it("produces fallback summary without fields", () => {
		const summary = getSummary(makeEvent({ event_type: "error.correction", payload: {} }));
		expect(summary).toBe("Correction");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "error.correction", payload: {} }));
		expect(screen.getByText("error.correction")).toBeInTheDocument();
	});
});

// ---------------------------------------------------------------------------
// ErrorDegradationRenderer
// ---------------------------------------------------------------------------

describe("error.degradation renderer", () => {
	it("displays degradation_message (not reason/fallback)", () => {
		renderEvent(
			makeEvent({
				event_type: "error.degradation",
				payload: {
					error_type: "ServiceUnavailable",
					error_message: "External API down",
					degradation_message: "Using cached results from previous run",
				},
			}),
		);

		expect(screen.getByText("ServiceUnavailable")).toBeInTheDocument();
		expect(screen.getByText("External API down")).toBeInTheDocument();
		expect(screen.getByText("Using cached results from previous run")).toBeInTheDocument();
		expect(screen.getByText("Degradation")).toBeInTheDocument();
	});

	it("produces correct summary", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "error.degradation",
				payload: { error_type: "TimeoutError" },
			}),
		);
		expect(summary).toBe("Degraded — TimeoutError");
	});

	it("produces fallback summary without fields", () => {
		const summary = getSummary(makeEvent({ event_type: "error.degradation", payload: {} }));
		expect(summary).toBe("Degraded");
	});

	it("handles missing optional fields", () => {
		renderEvent(makeEvent({ event_type: "error.degradation", payload: {} }));
		expect(screen.getByText("error.degradation")).toBeInTheDocument();
	});
});
