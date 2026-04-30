import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { EventDetailPanel } from "../../src/components/event-detail/event-detail-panel";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { createDefaultRegistrations } from "../../src/registry/default-renderers";
import type { EventDetailProps } from "../../src/registry/renderer-registry";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderPanel(
	event: Parameters<typeof EventDetailPanel>[0]["event"],
	registrations: Array<{
		matches: (t: string) => boolean;
		priority: number;
		component: React.ComponentType<EventDetailProps>;
	}> = [],
	onNavigateToAgent?: (spanId: string) => void,
) {
	const client = new ObservatoryClient("/test");
	const registry = new EventRendererRegistry();
	for (const reg of registrations) {
		registry.register(reg);
	}

	return render(
		<ObservatoryProvider client={client} registry={registry}>
			<EventDetailPanel event={event} onNavigateToAgent={onNavigateToAgent} />
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("EventDetailPanel", () => {
	it("shows event type in header", () => {
		const event = makeEvent({ event_type: "llm.request" });
		renderPanel(event);
		expect(screen.getByText("llm.request")).toBeInTheDocument();
	});

	it("shows level badge", () => {
		const event = makeEvent({ level: "debug" });
		renderPanel(event);
		expect(screen.getByText("debug")).toBeInTheDocument();
	});

	it("shows span ID", () => {
		const event = makeEvent({ span_id: "span-abc-123" });
		renderPanel(event);
		expect(screen.getByText("span-abc-123")).toBeInTheDocument();
	});

	it("falls back to PayloadViewer when no renderer matches", () => {
		const event = makeEvent({
			event_type: "unknown.custom.event",
			payload: { myField: "hello" },
		});
		renderPanel(event);
		// PayloadViewer renders object keys as "key:" text
		expect(screen.getByText("myField:")).toBeInTheDocument();
	});

	it("uses custom renderer when one matches", () => {
		function CustomRenderer({ event }: EventDetailProps) {
			return <div data-testid="custom">Custom: {event.event_type}</div>;
		}

		const event = makeEvent({ event_type: "llm.request" });
		renderPanel(event, [
			{
				matches: (t) => t === "llm.request",
				priority: 0,
				component: CustomRenderer,
			},
		]);

		expect(screen.getByTestId("custom")).toBeInTheDocument();
		expect(screen.getByText("Custom: llm.request")).toBeInTheDocument();
	});

	it("prefers higher-priority renderer", () => {
		function LowPriority() {
			return <div>low</div>;
		}
		function HighPriority() {
			return <div>high</div>;
		}

		const event = makeEvent({ event_type: "llm.request" });
		renderPanel(event, [
			{ matches: () => true, priority: -1, component: LowPriority },
			{
				matches: (t) => t === "llm.request",
				priority: 5,
				component: HighPriority,
			},
		]);

		expect(screen.getByText("high")).toBeInTheDocument();
		expect(screen.queryByText("low")).not.toBeInTheDocument();
	});

	it("shows 'View agent details' link for agent.start when onNavigateToAgent is provided", () => {
		const onNavigateToAgent = vi.fn();
		const event = makeEvent({
			event_type: "agent.start",
			span_id: "agent-span-1",
			payload: { agent_name: "researcher", agent_type: "react" },
		});

		renderPanel(event, createDefaultRegistrations(), onNavigateToAgent);

		const link = screen.getByText("View agent details →");
		expect(link).toBeInTheDocument();
		fireEvent.click(link);
		expect(onNavigateToAgent).toHaveBeenCalledWith("agent-span-1");
	});

	it("does not show 'View agent details' link when onNavigateToAgent is not provided", () => {
		const event = makeEvent({
			event_type: "agent.start",
			span_id: "agent-span-1",
			payload: { agent_name: "researcher", agent_type: "react" },
		});

		renderPanel(event, createDefaultRegistrations());

		expect(screen.queryByText("View agent details →")).not.toBeInTheDocument();
	});
});
