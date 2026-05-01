import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { useWorkflowDAG } from "../../src/hooks/use-workflow-dag";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { WorkflowDAGResponse } from "../../src/types";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const mockWorkflow: WorkflowDAGResponse = {
	workflow_name: "analysis-pipeline",
	workflow_type: "sequential",
	steps: [
		{
			name: "collect",
			step_type: "agent",
			index: 0,
			depends_on: [],
			parallel_group: null,
			status: "completed",
			duration_ms: 1200,
			agent_span_id: "span-collect",
			metadata: {},
		},
		{
			name: "analyze",
			step_type: "agent",
			index: 1,
			depends_on: ["collect"],
			parallel_group: null,
			status: "completed",
			duration_ms: 3400,
			agent_span_id: "span-analyze",
			metadata: {},
		},
	],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

describe("useWorkflowDAG", () => {
	let client: ObservatoryClient;

	beforeEach(() => {
		client = new ObservatoryClient("/test");
	});

	it("returns loading state initially", () => {
		vi.spyOn(client, "getWorkflow").mockReturnValue(new Promise(() => {}));

		const { result } = renderHook(() => useWorkflowDAG("run-1"), {
			wrapper: createWrapper(client),
		});

		expect(result.current.isLoading).toBe(true);
		expect(result.current.workflow).toBeNull();
		expect(result.current.layout).toBeNull();
		expect(result.current.error).toBeNull();
	});

	it("returns workflow data and computed layout on success", async () => {
		vi.spyOn(client, "getWorkflow").mockResolvedValue(mockWorkflow);

		const { result } = renderHook(() => useWorkflowDAG("run-1"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(result.current.workflow).toEqual(mockWorkflow);
		expect(result.current.layout).not.toBeNull();
		expect(result.current.layout?.nodes).toHaveLength(2);
		expect(result.current.layout?.edges).toHaveLength(1);
		expect(result.current.error).toBeNull();
	});

	it("returns error on failure", async () => {
		vi.spyOn(client, "getWorkflow").mockRejectedValue(new Error("Network error"));

		const { result } = renderHook(() => useWorkflowDAG("run-1"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(result.current.error).toContain("Network error");
		expect(result.current.workflow).toBeNull();
		expect(result.current.layout).toBeNull();
	});

	it("does not update state after unmount", async () => {
		let resolve!: (value: WorkflowDAGResponse) => void;
		vi.spyOn(client, "getWorkflow").mockReturnValue(
			new Promise((r) => {
				resolve = r;
			}),
		);

		const { unmount } = renderHook(() => useWorkflowDAG("run-1"), {
			wrapper: createWrapper(client),
		});

		unmount();
		resolve?.(mockWorkflow);

		// Give time for any async effects — no error should be thrown
		await new Promise((r) => setTimeout(r, 10));
	});

	it("refetches when refetch is called", async () => {
		const spy = vi.spyOn(client, "getWorkflow").mockResolvedValue(mockWorkflow);

		const { result } = renderHook(() => useWorkflowDAG("run-1"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(spy).toHaveBeenCalledTimes(1);

		result.current.refetch();

		await waitFor(() => {
			expect(spy).toHaveBeenCalledTimes(2);
		});
	});

	it("refetches when runId changes", async () => {
		const spy = vi.spyOn(client, "getWorkflow").mockResolvedValue(mockWorkflow);

		const { result, rerender } = renderHook(({ runId }: { runId: string }) => useWorkflowDAG(runId), {
			wrapper: createWrapper(client),
			initialProps: { runId: "run-1" },
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		rerender({ runId: "run-2" });

		await waitFor(() => {
			expect(spy).toHaveBeenCalledTimes(2);
		});

		expect(spy).toHaveBeenLastCalledWith("run-2");
	});

	it("cancels previous request when runId changes", async () => {
		const secondWorkflow: WorkflowDAGResponse = {
			...mockWorkflow,
			workflow_name: "second-pipeline",
		};

		let resolveFirst!: (v: WorkflowDAGResponse) => void;
		vi.spyOn(client, "getWorkflow")
			.mockReturnValueOnce(
				new Promise((r) => {
					resolveFirst = r;
				}),
			)
			.mockResolvedValueOnce(secondWorkflow);

		const { result, rerender } = renderHook(({ runId }: { runId: string }) => useWorkflowDAG(runId), {
			wrapper: createWrapper(client),
			initialProps: { runId: "run-1" },
		});

		// Change runId before first resolves
		rerender({ runId: "run-2" });

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		// First request resolves after cancellation — should be ignored
		resolveFirst?.(mockWorkflow);
		await new Promise((r) => setTimeout(r, 10));

		expect(result.current.workflow?.workflow_name).toBe("second-pipeline");
	});

	it("passes runId to client.getWorkflow", async () => {
		const spy = vi.spyOn(client, "getWorkflow").mockResolvedValue(mockWorkflow);

		const { result } = renderHook(() => useWorkflowDAG("my-run-id"), {
			wrapper: createWrapper(client),
		});

		await waitFor(() => {
			expect(result.current.isLoading).toBe(false);
		});

		expect(spy).toHaveBeenCalledWith("my-run-id");
	});
});
