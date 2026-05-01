import { describe, expect, it } from "vitest";
import { type EventDetailProps, EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { TraceEvent } from "../../src/types";

function makeEvent(overrides: Partial<TraceEvent> = {}): TraceEvent {
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

function DummyRenderer(_props: EventDetailProps) {
	return null;
}

function AnotherRenderer(_props: EventDetailProps) {
	return null;
}

describe("EventRendererRegistry", () => {
	it("returns null when no renderers are registered", () => {
		const registry = new EventRendererRegistry();
		expect(registry.getRenderer("llm.request")).toBeNull();
	});

	it("returns the event type as summary when no renderers match", () => {
		const registry = new EventRendererRegistry();
		const event = makeEvent({ event_type: "custom.event" });
		expect(registry.getSummary(event)).toBe("custom.event");
	});

	it("registers and looks up a renderer by event type", () => {
		const registry = new EventRendererRegistry();
		registry.register({
			matches: (t) => t === "llm.request",
			priority: 0,
			component: DummyRenderer,
		});

		expect(registry.getRenderer("llm.request")).toBe(DummyRenderer);
		expect(registry.getRenderer("llm.response")).toBeNull();
	});

	it("returns higher-priority renderer when multiple match", () => {
		const registry = new EventRendererRegistry();
		registry.register({
			matches: () => true,
			priority: -1,
			component: DummyRenderer,
		});
		registry.register({
			matches: (t) => t === "llm.request",
			priority: 5,
			component: AnotherRenderer,
		});

		expect(registry.getRenderer("llm.request")).toBe(AnotherRenderer);
		// Fallback for non-matching specific
		expect(registry.getRenderer("tool.invoke")).toBe(DummyRenderer);
	});

	it("maintains priority order regardless of registration order", () => {
		const registry = new EventRendererRegistry();

		// Register high priority first
		registry.register({
			matches: (t) => t === "llm.request",
			priority: 10,
			component: AnotherRenderer,
		});
		// Then register low priority
		registry.register({
			matches: (t) => t === "llm.request",
			priority: 0,
			component: DummyRenderer,
		});

		expect(registry.getRenderer("llm.request")).toBe(AnotherRenderer);
	});

	it("uses custom summary when provided", () => {
		const registry = new EventRendererRegistry();
		registry.register({
			matches: (t) => t === "llm.response",
			priority: 0,
			component: DummyRenderer,
			summary: (event) => {
				const usage = event.payload.usage as { input_tokens: number; output_tokens: number } | undefined;
				return usage ? `${usage.input_tokens}+${usage.output_tokens} tokens` : "LLM Response";
			},
		});

		const event = makeEvent({
			event_type: "llm.response",
			payload: { usage: { input_tokens: 100, output_tokens: 50 } },
		});
		expect(registry.getSummary(event)).toBe("100+50 tokens");
	});

	it("falls back to event type when renderer has no summary", () => {
		const registry = new EventRendererRegistry();
		registry.register({
			matches: (t) => t === "tool.invoke",
			priority: 0,
			component: DummyRenderer,
			// no summary function
		});

		const event = makeEvent({ event_type: "tool.invoke" });
		expect(registry.getSummary(event)).toBe("tool.invoke");
	});

	it("uses highest-priority summary when multiple match", () => {
		const registry = new EventRendererRegistry();
		registry.register({
			matches: () => true,
			priority: -1,
			component: DummyRenderer,
			summary: () => "fallback",
		});
		registry.register({
			matches: (t) => t === "llm.request",
			priority: 5,
			component: AnotherRenderer,
			summary: () => "specific",
		});

		const event = makeEvent({ event_type: "llm.request" });
		expect(registry.getSummary(event)).toBe("specific");

		const otherEvent = makeEvent({ event_type: "unknown.type" });
		expect(registry.getSummary(otherEvent)).toBe("fallback");
	});

	it("supports pattern-based matching", () => {
		const registry = new EventRendererRegistry();
		registry.register({
			matches: (t) => t.startsWith("agent."),
			priority: 0,
			component: DummyRenderer,
			summary: () => "agent event",
		});

		expect(registry.getRenderer("agent.start")).toBe(DummyRenderer);
		expect(registry.getRenderer("agent.step")).toBe(DummyRenderer);
		expect(registry.getRenderer("agent.complete")).toBe(DummyRenderer);
		expect(registry.getRenderer("tool.invoke")).toBeNull();
	});
});
