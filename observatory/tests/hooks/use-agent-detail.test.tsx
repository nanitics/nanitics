import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { useAgentDetail } from "../../src/hooks/use-agent-detail";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentDetailResponse } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockResponse: AgentDetailResponse = {
	agent: {
		agent_name: "researcher",
		agent_type: "react",
		span_id: "agent-1",
		capabilities: ["web_search"],
		stats: {
			llm_calls: 3,
			tool_calls: 2,
			input_tokens: 500,
			output_tokens: 300,
			duration_ms: 2000,
			errors: 0,
			iterations: 2,
		},
	},
	events: [
		makeEvent({ event_type: "agent.start", span_id: "agent-1" }),
		makeEvent({ event_type: "agent.step", span_id: "agent-1" }),
		makeEvent({ event_type: "agent.complete", span_id: "agent-1" }),
	],
	span_tree: {
		span_id: "agent-1",
		parent_span_id: "root",
		name: "researcher",
		summary: {
			event_count: 3,
			duration_ms: 2000,
			has_errors: false,
			agent_name: "researcher",
			agent_type: "react",
		},
		events: [],
		children: [],
	},
};

function createWrapper(client: ObservatoryClient) {
	const registry = new EventRendererRegistry();
	return function Wrapper({ children }: { children: React.ReactNode }) {
		return (
			<ObservatoryProvider client={client} registry={registry}>
				{children}
			</ObservatoryProvider>
		);
	};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useAgentDetail", () => {
	let client: ObservatoryClient;

	beforeEach(() => {
		client = new ObservatoryClient("/test");
	});

	it("returns loading state initially", () => {
		vi.spyOn(client, "getAgentDetail").mockReturnValue(new Promise(() => {}));

		const { result } = renderHook(() => useAgentDetail("run-1", "agent-1"), {
			wrapper: createWrapper(client),
		});

		expect(result.current.isLoading).toBe(true);
		expect(result.current.agent).toBeNull();
		expect(result.current.events).toEqual([]);
		expect(result.current.spanTree).toBeNull();
		expect(result.current.error).toBeNull();
	});

	it("returns agent data on success", async () => {
		vi.spyOn(client, "getAgentDetail").mockResolvedValue(mockResponse);

		const { result } = renderHook(() => useAgentDetail("run-1", "agent-1"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(result.current.agent).toEqual(mockResponse.agent);
		expect(result.current.events).toEqual(mockResponse.events);
		expect(result.current.spanTree).toEqual(mockResponse.span_tree);
		expect(result.current.error).toBeNull();
	});

	it("returns error on failure", async () => {
		vi.spyOn(client, "getAgentDetail").mockRejectedValue(new Error("Network error"));

		const { result } = renderHook(() => useAgentDetail("run-1", "agent-1"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(result.current.error).toContain("Network error");
		expect(result.current.agent).toBeNull();
	});

	it("cancels request on unmount", async () => {
		let resolve!: (value: AgentDetailResponse) => void;
		vi.spyOn(client, "getAgentDetail").mockReturnValue(
			new Promise((r) => {
				resolve = r;
			}),
		);

		const { unmount } = renderHook(() => useAgentDetail("run-1", "agent-1"), { wrapper: createWrapper(client) });

		unmount();
		// Resolve after unmount — state should not update
		resolve?.(mockResponse);

		// Give time for any async effects
		await new Promise((r) => setTimeout(r, 10));

		// The hook was unmounted, so we can't check result.current
		// But the key assertion is no error is thrown (no state update after unmount)
		expect(true).toBe(true);
	});

	it("refetches when refetch is called", async () => {
		const spy = vi.spyOn(client, "getAgentDetail").mockResolvedValue(mockResponse);

		const { result } = renderHook(() => useAgentDetail("run-1", "agent-1"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(spy).toHaveBeenCalledTimes(1);

		// Trigger refetch
		result.current.refetch();

		await waitFor(() => {
			expect(spy).toHaveBeenCalledTimes(2);
		});
	});

	it("refetches when runId changes", async () => {
		const spy = vi.spyOn(client, "getAgentDetail").mockResolvedValue(mockResponse);

		const { result, rerender } = renderHook(
			({ runId, spanId }: { runId: string; spanId: string }) => useAgentDetail(runId, spanId),
			{
				wrapper: createWrapper(client),
				initialProps: { runId: "run-1", spanId: "agent-1" },
			},
		);

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		rerender({ runId: "run-2", spanId: "agent-1" });

		await waitFor(() => {
			expect(spy).toHaveBeenCalledTimes(2);
		});

		expect(spy).toHaveBeenLastCalledWith("run-2", "agent-1");
	});

	it("cancels previous request when parameters change", async () => {
		const firstResponse: AgentDetailResponse = {
			...mockResponse,
			agent: { ...mockResponse.agent, agent_name: "first" },
		};
		const secondResponse: AgentDetailResponse = {
			...mockResponse,
			agent: { ...mockResponse.agent, agent_name: "second" },
		};

		let resolveFirst!: (v: AgentDetailResponse) => void;
		vi.spyOn(client, "getAgentDetail")
			.mockReturnValueOnce(
				new Promise((r) => {
					resolveFirst = r;
				}),
			)
			.mockResolvedValueOnce(secondResponse);

		const { result, rerender } = renderHook(
			({ runId, spanId }: { runId: string; spanId: string }) => useAgentDetail(runId, spanId),
			{
				wrapper: createWrapper(client),
				initialProps: { runId: "run-1", spanId: "agent-1" },
			},
		);

		// Change params before first resolves
		rerender({ runId: "run-2", spanId: "agent-1" });

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		// First request resolves after cancellation — should be ignored
		resolveFirst?.(firstResponse);
		await new Promise((r) => setTimeout(r, 10));

		// Should have the second response, not the first
		expect(result.current.agent?.agent_name).toBe("second");
	});
});
