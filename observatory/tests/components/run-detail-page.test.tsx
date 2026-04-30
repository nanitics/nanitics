import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { RunDetailPage } from "../../src/pages/run-detail-page";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentListResponse, RunDetailResponse, SpanTreeResponse } from "../../src/types";
import { makeRun, makeSummary } from "../fixtures/scenarios";

// Stub EventSource for the streaming hook (jsdom has no native EventSource).
class MockEventSource {
	static readonly CONNECTING = 0;
	static readonly OPEN = 1;
	static readonly CLOSED = 2;
	readyState = MockEventSource.CONNECTING;
	url: string;
	constructor(url: string) {
		this.url = url;
	}
	addEventListener() {}
	close() {
		this.readyState = MockEventSource.CLOSED;
	}
}
vi.stubGlobal("EventSource", MockEventSource);

const RUN_ID = "run-header-shell-first";

const mockRunDetail: RunDetailResponse = {
	run: makeRun({ id: RUN_ID, status: "completed", metadata: { description: "Clickable Header" } }),
	summary: makeSummary(),
};

const mockAgentList: AgentListResponse = { agents: [] };

describe("RunDetailPage", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("renders the full RunDetailSkeleton while the run header is still loading", () => {
		const client = new ObservatoryClient("/test");
		// Run header never resolves — we stay in the first-paint skeleton.
		vi.spyOn(client, "getRun").mockReturnValue(new Promise(() => {}));
		vi.spyOn(client, "getSpanTree").mockReturnValue(new Promise(() => {}));
		vi.spyOn(client, "listAgents").mockResolvedValue(mockAgentList);
		vi.spyOn(client, "getBaseUrl").mockReturnValue("/test");

		const { container } = render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<RunDetailPage runId={RUN_ID} onBack={vi.fn()} />
			</ObservatoryProvider>,
		);

		// A11y contract: one role="status" + aria-busy="true" container on the page.
		expect(screen.getByRole("status")).toBeInTheDocument();
		expect(container.querySelectorAll("[aria-busy='true']").length).toBeGreaterThan(0);
	});

	it("renders the page-level <main> landmark and right-side complementary panel without axe violations", async () => {
		const client = new ObservatoryClient("/test");
		const mockSpanTree: SpanTreeResponse = {
			trace_id: "trace-1",
			root: {
				span_id: "root-span",
				parent_span_id: null,
				name: "root",
				summary: { event_count: 0, duration_ms: 0, has_errors: false, agent_name: null, agent_type: null },
				events: [],
				children: [],
			},
		};
		vi.spyOn(client, "getRun").mockResolvedValue(mockRunDetail);
		vi.spyOn(client, "getSpanTree").mockResolvedValue(mockSpanTree);
		vi.spyOn(client, "listAgents").mockResolvedValue(mockAgentList);
		vi.spyOn(client, "getBaseUrl").mockReturnValue("/test");

		const { container } = render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<RunDetailPage runId={RUN_ID} onBack={vi.fn()} />
			</ObservatoryProvider>,
		);

		// Wait for the page header to land before scanning landmarks.
		await waitFor(() => {
			expect(screen.getByText(RUN_ID)).toBeInTheDocument();
		});

		// Landmark contract: the page contributes a single <main> and a
		// complementary aside for the right-side detail panel.
		expect(container.querySelector("main")).not.toBeNull();
		expect(screen.getByRole("complementary", { name: /Selected event details/i })).toBeInTheDocument();

		const results = await axe(container);
		expect(results).toHaveNoViolations();
	});

	it("renders the real header AND the tree-row skeleton once the run resolves but the tree is still loading", async () => {
		const client = new ObservatoryClient("/test");
		// `getRun` resolves immediately; `getSpanTree` hangs so `treeLoading` stays true.
		vi.spyOn(client, "getRun").mockResolvedValue(mockRunDetail);
		vi.spyOn(client, "getSpanTree").mockReturnValue(new Promise(() => {}));
		vi.spyOn(client, "listAgents").mockResolvedValue(mockAgentList);
		vi.spyOn(client, "getBaseUrl").mockReturnValue("/test");

		const { container } = render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<RunDetailPage runId={RUN_ID} onBack={vi.fn()} />
			</ObservatoryProvider>,
		);

		// Header-shell-first: the real run id is in the DOM once `useRunDetail`
		// resolves, even while the tree is still fetching.
		await waitFor(() => {
			expect(screen.getByText(RUN_ID)).toBeInTheDocument();
		});
		// And the tree-row skeleton is present underneath.
		expect(screen.getByRole("status")).toBeInTheDocument();
		expect(container.querySelectorAll("[aria-busy='true']").length).toBeGreaterThan(0);
	});
});
