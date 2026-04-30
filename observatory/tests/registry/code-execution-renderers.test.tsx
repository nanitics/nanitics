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

function getRenderer(eventType: string) {
	const registry = createDefaultRegistry();
	return registry.getRenderer(eventType);
}

// ---------------------------------------------------------------------------
// code.execution
// ---------------------------------------------------------------------------

describe("code.execution renderer", () => {
	it("resolves to a renderer", () => {
		expect(getRenderer("code.execution")).not.toBeNull();
	});

	it("renders step number and agent name", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution",
				payload: {
					agent_name: "code-agent",
					code: "print('hello')",
					step_number: 3,
				},
			}),
		);

		expect(screen.getByText("Step 3")).toBeInTheDocument();
		expect(screen.getByText("code-agent")).toBeInTheDocument();
	});

	it("renders code block", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution",
				payload: {
					code: "x = 42\nprint(x)",
					step_number: 1,
				},
			}),
		);

		expect(screen.getByText("x = 42")).toBeInTheDocument();
		expect(screen.getByText("print(x)")).toBeInTheDocument();
	});

	it("produces correct summary with step number", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "code.execution",
				payload: { step_number: 2 },
			}),
		);
		expect(summary).toBe("Step 2: execute code");
	});

	it("produces fallback summary without step number", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "code.execution",
				payload: {},
			}),
		);
		expect(summary).toBe("Execute code");
	});
});

// ---------------------------------------------------------------------------
// code.execution.result
// ---------------------------------------------------------------------------

describe("code.execution.result renderer", () => {
	it("resolves to a renderer", () => {
		expect(getRenderer("code.execution.result")).not.toBeNull();
	});

	it("renders step badge and success status", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: true,
					duration_ms: 150,
					stdout: "",
					stderr: "",
					return_value: null,
					error: null,
				},
			}),
		);

		expect(screen.getByText("Step 1")).toBeInTheDocument();
		expect(screen.getByText("success")).toBeInTheDocument();
		expect(screen.getByText("150ms")).toBeInTheDocument();
	});

	it("renders failure status and error", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 2,
					success: false,
					duration_ms: 50,
					error: "NameError: name 'foo' is not defined",
					stdout: "",
					stderr: "",
					return_value: null,
				},
			}),
		);

		expect(screen.getByText("failed")).toBeInTheDocument();
		expect(screen.getByText("NameError: name 'foo' is not defined")).toBeInTheDocument();
	});

	it("renders stdout", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: true,
					stdout: "Hello, world!",
					stderr: "",
					return_value: null,
					error: null,
					duration_ms: 10,
				},
			}),
		);

		expect(screen.getAllByText("stdout:").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("Hello, world!")).toBeInTheDocument();
	});

	it("renders stderr with destructive styling", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: false,
					stdout: "",
					stderr: "Warning: deprecated",
					return_value: null,
					error: null,
					duration_ms: 10,
				},
			}),
		);

		expect(screen.getAllByText("stderr:").length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("Warning: deprecated")).toBeInTheDocument();
	});

	it("renders return_value when present", () => {
		renderEvent(
			makeEvent({
				event_type: "code.execution.result",
				payload: {
					step_number: 1,
					success: true,
					stdout: "",
					stderr: "",
					return_value: "42",
					error: null,
					duration_ms: 5,
				},
			}),
		);

		expect(screen.getAllByText("return_value:").length).toBeGreaterThanOrEqual(1);
		expect(screen.getAllByText("42").length).toBeGreaterThanOrEqual(1);
	});

	it("produces correct summary for success", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "code.execution.result",
				payload: { step_number: 3, success: true, duration_ms: 200 },
			}),
		);
		expect(summary).toBe("Step 3: success (200ms)");
	});

	it("produces correct summary for failure", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "code.execution.result",
				payload: { step_number: 1, success: false, duration_ms: 50 },
			}),
		);
		expect(summary).toBe("Step 1: failed (50ms)");
	});

	it("produces summary without duration", () => {
		const summary = getSummary(
			makeEvent({
				event_type: "code.execution.result",
				payload: { step_number: 2, success: true },
			}),
		);
		expect(summary).toBe("Step 2: success");
	});
});
