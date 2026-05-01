import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type ConnectionState, StreamingClient } from "../../src/client/streaming-client";
import type { TraceEvent } from "../../src/types";

// -----------------------------------------------------------------------
// Mock EventSource
// -----------------------------------------------------------------------

type EventSourceListener = (e: MessageEvent | Event) => void;

class MockEventSource {
	static CONNECTING = 0;
	static OPEN = 1;
	static CLOSED = 2;

	static instances: MockEventSource[] = [];

	url: string;
	readyState: number = MockEventSource.CONNECTING;
	private listeners: Record<string, EventSourceListener[]> = {};

	constructor(url: string) {
		this.url = url;
		MockEventSource.instances.push(this);
	}

	addEventListener(type: string, listener: EventSourceListener) {
		if (!this.listeners[type]) this.listeners[type] = [];
		this.listeners[type].push(listener);
	}

	close() {
		this.readyState = MockEventSource.CLOSED;
	}

	// Test helpers

	simulateOpen() {
		this.readyState = MockEventSource.OPEN;
		for (const fn of this.listeners.open ?? []) {
			fn(new Event("open"));
		}
	}

	simulateEvent(type: string, data: unknown) {
		for (const fn of this.listeners[type] ?? []) {
			fn(new MessageEvent(type, { data: JSON.stringify(data) }));
		}
	}

	simulateError() {
		for (const fn of this.listeners.error ?? []) {
			fn(new Event("error"));
		}
	}
}

// Install mock globally
beforeEach(() => {
	MockEventSource.instances = [];
	vi.stubGlobal("EventSource", MockEventSource);
});

afterEach(() => {
	vi.restoreAllMocks();
});

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

function makeTraceEvent(overrides: Partial<TraceEvent> = {}): TraceEvent {
	return {
		id: 1,
		event_type: "test.event",
		level: "info",
		trace_id: "trace-1",
		span_id: "span-1",
		parent_span_id: null,
		timestamp: "2026-03-05T10:00:00Z",
		payload: {},
		...overrides,
	};
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

describe("StreamingClient", () => {
	it("constructs the correct SSE URL", () => {
		const client = new StreamingClient("/api/observatory");
		client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
		});

		expect(MockEventSource.instances).toHaveLength(1);
		expect(MockEventSource.instances[0].url).toBe("/api/observatory/runs/run-1/stream");
	});

	it("includes min_level query param when provided", () => {
		const client = new StreamingClient("/api/observatory");
		client.connect({
			runId: "run-1",
			minLevel: "debug",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
		});

		expect(MockEventSource.instances[0].url).toBe("/api/observatory/runs/run-1/stream?min_level=debug");
	});

	it("encodes runId in the URL", () => {
		const client = new StreamingClient("/api/observatory");
		client.connect({
			runId: "run with spaces",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
		});

		expect(MockEventSource.instances[0].url).toBe("/api/observatory/runs/run%20with%20spaces/stream");
	});

	it("starts in connecting state and transitions to connected on open", () => {
		const client = new StreamingClient("/api/observatory");
		const states: ConnectionState[] = [];

		const conn = client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
			onStateChange: (s) => states.push(s),
		});

		expect(conn.state).toBe("connecting");

		MockEventSource.instances[0].simulateOpen();

		expect(conn.state).toBe("connected");
		expect(states).toEqual(["connected"]);
	});

	it("calls onEvent with parsed TraceEvent for trace events", () => {
		const client = new StreamingClient("/api/observatory");
		const onEvent = vi.fn();

		client.connect({
			runId: "run-1",
			onEvent,
			onRunComplete: vi.fn(),
		});

		const event = makeTraceEvent({ id: 42, event_type: "llm.request" });
		MockEventSource.instances[0].simulateOpen();
		MockEventSource.instances[0].simulateEvent("trace", event);

		expect(onEvent).toHaveBeenCalledWith(event);
	});

	it("calls onRunComplete and closes connection on run_complete event", () => {
		const client = new StreamingClient("/api/observatory");
		const onRunComplete = vi.fn();
		const states: ConnectionState[] = [];

		const conn = client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete,
			onStateChange: (s) => states.push(s),
		});

		const source = MockEventSource.instances[0];
		source.simulateOpen();
		source.simulateEvent("run_complete", { status: "completed" });

		expect(onRunComplete).toHaveBeenCalledWith("completed");
		expect(conn.state).toBe("closed");
		expect(source.readyState).toBe(MockEventSource.CLOSED);
		expect(states).toContain("closed");
	});

	it("transitions to reconnecting on error when previously connected", () => {
		const client = new StreamingClient("/api/observatory");
		const states: ConnectionState[] = [];

		client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
			onStateChange: (s) => states.push(s),
		});

		const source = MockEventSource.instances[0];
		source.simulateOpen();

		// Simulate transient error (EventSource auto-reconnects, readyState stays CONNECTING)
		source.readyState = MockEventSource.CONNECTING;
		source.simulateError();

		expect(states).toEqual(["connected", "reconnecting"]);
	});

	it("transitions to closed when error occurs with CLOSED readyState", () => {
		const client = new StreamingClient("/api/observatory");
		const states: ConnectionState[] = [];

		client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
			onStateChange: (s) => states.push(s),
		});

		const source = MockEventSource.instances[0];
		source.readyState = MockEventSource.CLOSED;
		source.simulateError();

		expect(states).toEqual(["closed"]);
	});

	it("close() closes the EventSource and sets state to closed", () => {
		const client = new StreamingClient("/api/observatory");
		const states: ConnectionState[] = [];

		const conn = client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
			onStateChange: (s) => states.push(s),
		});

		MockEventSource.instances[0].simulateOpen();
		conn.close();

		expect(conn.state).toBe("closed");
		expect(MockEventSource.instances[0].readyState).toBe(MockEventSource.CLOSED);
	});

	it("calls onError callback when an error occurs", () => {
		const client = new StreamingClient("/api/observatory");
		const onError = vi.fn();

		client.connect({
			runId: "run-1",
			onEvent: vi.fn(),
			onRunComplete: vi.fn(),
			onError,
		});

		MockEventSource.instances[0].readyState = MockEventSource.CLOSED;
		MockEventSource.instances[0].simulateError();

		expect(onError).toHaveBeenCalled();
	});

	it("handles multiple trace events in sequence", () => {
		const client = new StreamingClient("/api/observatory");
		const events: TraceEvent[] = [];

		client.connect({
			runId: "run-1",
			onEvent: (e) => events.push(e),
			onRunComplete: vi.fn(),
		});

		const source = MockEventSource.instances[0];
		source.simulateOpen();
		source.simulateEvent("trace", makeTraceEvent({ id: 1 }));
		source.simulateEvent("trace", makeTraceEvent({ id: 2 }));
		source.simulateEvent("trace", makeTraceEvent({ id: 3 }));

		expect(events).toHaveLength(3);
		expect(events.map((e) => e.id)).toEqual([1, 2, 3]);
	});
});
