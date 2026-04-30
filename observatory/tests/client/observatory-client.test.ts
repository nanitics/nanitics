import { beforeEach, describe, expect, it, vi } from "vitest";
import { ObservatoryClient } from "../../src/client/observatory-client";

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);

function mockResponse(body: unknown, status = 200) {
	fetchMock.mockResolvedValueOnce({
		ok: status >= 200 && status < 300,
		status,
		statusText: status === 200 ? "OK" : "Not Found",
		json: () => Promise.resolve(body),
	});
}

beforeEach(() => {
	fetchMock.mockReset();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("ObservatoryClient", () => {
	const client = new ObservatoryClient("/api/observatory");

	describe("listRuns", () => {
		it("constructs URL without query params when no options given", async () => {
			mockResponse({ runs: [], total: 0 });
			await client.listRuns();
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs", { signal: undefined });
		});

		it("appends status, limit, and offset as query params", async () => {
			mockResponse({ runs: [], total: 0 });
			await client.listRuns({ status: "running", limit: 10, offset: 5 });
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs?status=running&limit=10&offset=5", {
				signal: undefined,
			});
		});

		it("parses response correctly", async () => {
			const body = {
				runs: [
					{
						id: "r1",
						trace_id: "t1",
						status: "completed",
						started_at: "2026-01-01T00:00:00Z",
						completed_at: null,
						metadata: {},
						error: null,
					},
				],
				total: 1,
			};
			mockResponse(body);
			const result = await client.listRuns();
			expect(result).toEqual(body);
		});

		it("throws on non-2xx response", async () => {
			mockResponse({}, 500);
			await expect(client.listRuns()).rejects.toThrow("Failed to list runs");
		});

		it("threads AbortSignal through to fetch", async () => {
			mockResponse({ runs: [], total: 0 });
			const controller = new AbortController();
			await client.listRuns({ signal: controller.signal });
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs", { signal: controller.signal });
		});
	});

	describe("getRun", () => {
		it("constructs correct URL with encoded run ID", async () => {
			mockResponse({ run: {}, summary: {} });
			await client.getRun("run with spaces");
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/run%20with%20spaces");
		});

		it("throws on 404", async () => {
			mockResponse({}, 404);
			await expect(client.getRun("missing")).rejects.toThrow("Failed to get run missing");
		});
	});

	describe("getSpanTree", () => {
		it("constructs URL without params when no options", async () => {
			mockResponse({ trace_id: "t1", root: {} });
			await client.getSpanTree("run-1");
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/run-1/tree");
		});

		it("appends min_level param", async () => {
			mockResponse({ trace_id: "t1", root: {} });
			await client.getSpanTree("run-1", { minLevel: "info" });
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/run-1/tree?min_level=info");
		});
	});

	describe("queryEvents", () => {
		it("constructs URL with all options", async () => {
			mockResponse({ events: [], has_more: false });
			await client.queryEvents("r1", {
				level: "debug",
				eventTypes: ["llm.request", "tool.invoke"],
				limit: 50,
				after: 10,
			});
			expect(fetchMock).toHaveBeenCalledWith(
				"/api/observatory/runs/r1/events?level=debug&event_types=llm.request%2Ctool.invoke&limit=50&after=10",
			);
		});
	});

	describe("getEvent", () => {
		it("constructs correct URL with event ID", async () => {
			mockResponse({ id: 42, event_type: "llm.request" });
			await client.getEvent(42);
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/events/42");
		});

		it("throws on non-2xx", async () => {
			mockResponse({}, 404);
			await expect(client.getEvent(999)).rejects.toThrow("Failed to get event 999");
		});
	});

	describe("getSummary", () => {
		it("constructs correct URL", async () => {
			mockResponse({ total_events: 10 });
			await client.getSummary("r1");
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/r1/summary");
		});
	});

	describe("listAgents", () => {
		it("constructs correct URL", async () => {
			mockResponse({ agents: [] });
			await client.listAgents("r1");
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/r1/agents");
		});
	});

	describe("getAgentDetail", () => {
		it("encodes both runId and spanId", async () => {
			mockResponse({ agent: {}, events: [], span_tree: {} });
			await client.getAgentDetail("r 1", "s 2");
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/r%201/agents/s%202");
		});
	});

	describe("getWorkflow", () => {
		it("constructs correct URL", async () => {
			mockResponse({ workflow_name: "test", steps: [] });
			await client.getWorkflow("r1");
			expect(fetchMock).toHaveBeenCalledWith("/api/observatory/runs/r1/workflow");
		});

		it("throws on error", async () => {
			mockResponse({}, 500);
			await expect(client.getWorkflow("r1")).rejects.toThrow("Failed to get workflow for run r1");
		});
	});

	describe("getBaseUrl", () => {
		it("returns the base URL", () => {
			expect(client.getBaseUrl()).toBe("/api/observatory");
		});
	});
});
