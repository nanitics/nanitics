import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { WorkflowDetailPage } from "../../src/pages/workflow-detail-page";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { WorkflowDAGResponse, WorkflowStep } from "../../src/types";

// Stub EventSource for streaming hook (not available in jsdom)
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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeStep(overrides: Partial<WorkflowStep> & { name: string }): WorkflowStep {
	return {
		step_type: "function",
		index: null,
		depends_on: [],
		parallel_group: null,
		status: "completed",
		duration_ms: null,
		agent_span_id: null,
		metadata: {},
		...overrides,
	};
}

const mockWorkflow: WorkflowDAGResponse = {
	workflow_name: "Research Pipeline",
	workflow_type: "sequential",
	steps: [
		makeStep({ name: "gather", index: 0, status: "completed", duration_ms: 1200 }),
		makeStep({
			name: "analyze",
			index: 1,
			depends_on: ["gather"],
			status: "completed",
			duration_ms: 3500,
			agent_span_id: "agent-1",
		}),
		makeStep({
			name: "report",
			index: 2,
			depends_on: ["analyze"],
			status: "pending",
		}),
	],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderPage(
	props?: Partial<React.ComponentProps<typeof WorkflowDetailPage>>,
	workflowData?: WorkflowDAGResponse,
) {
	const client = new ObservatoryClient("/test");
	vi.spyOn(client, "getWorkflow").mockResolvedValue(workflowData ?? mockWorkflow);
	// Mock getBaseUrl for streaming client
	vi.spyOn(client, "getBaseUrl").mockReturnValue("/test");

	return render(
		<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
			<WorkflowDetailPage
				runId="run-1"
				onBack={props?.onBack ?? vi.fn()}
				onBackToRuns={props?.onBackToRuns ?? vi.fn()}
				onNavigateToAgent={props?.onNavigateToAgent ?? vi.fn()}
				runLabel={props?.runLabel}
			/>
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("WorkflowDetailPage", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows loading skeleton initially", () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "getWorkflow").mockReturnValue(new Promise(() => {}));

		const { container } = render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<WorkflowDetailPage runId="run-1" onBack={vi.fn()} onBackToRuns={vi.fn()} onNavigateToAgent={vi.fn()} />
			</ObservatoryProvider>,
		);

		// `<RunDetailSkeleton>` is reused on the workflow page — verify the
		// a11y contract rather than brittle structural detail.
		expect(screen.getByRole("status")).toBeInTheDocument();
		expect(container.querySelectorAll("[aria-busy='true']").length).toBeGreaterThan(0);
	});

	it("shows error state on failure", async () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "getWorkflow").mockRejectedValue(new Error("Network error"));

		render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<WorkflowDetailPage runId="run-1" onBack={vi.fn()} onBackToRuns={vi.fn()} onNavigateToAgent={vi.fn()} />
			</ObservatoryProvider>,
		);

		// The page-level ErrorState surfaces the default title and carries
		// role="alert"; the raw error lives inside the <details> pane.
		await waitFor(() => {
			expect(screen.getByRole("alert")).toBeInTheDocument();
		});
		expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
		expect(screen.getByText(/Network error/)).toBeInTheDocument();
	});

	it("renders breadcrumbs with workflow name", async () => {
		renderPage({ runLabel: "My Run" });

		await waitFor(() => {
			expect(screen.getByText("Runs")).toBeInTheDocument();
			expect(screen.getByText("My Run")).toBeInTheDocument();
			expect(screen.getAllByText("Research Pipeline").length).toBeGreaterThan(0);
		});
	});

	it("renders workflow header with name and type badge", async () => {
		renderPage();

		await waitFor(() => {
			expect(screen.getByRole("heading", { name: "Research Pipeline" })).toBeInTheDocument();
			expect(screen.getByText("sequential")).toBeInTheDocument();
		});
	});

	it("shows step count", async () => {
		renderPage();

		await waitFor(() => {
			expect(screen.getByText("3 steps")).toBeInTheDocument();
		});
	});

	it("renders the DAG visualization with all nodes", async () => {
		renderPage();

		await waitFor(() => {
			expect(screen.getByTestId("dag-visualization")).toBeInTheDocument();
			expect(screen.getByTestId("node-gather")).toBeInTheDocument();
			expect(screen.getByTestId("node-analyze")).toBeInTheDocument();
			expect(screen.getByTestId("node-report")).toBeInTheDocument();
		});
	});

	it("shows selected node detail panel on click", async () => {
		renderPage();

		await waitFor(() => {
			expect(screen.getByTestId("node-analyze")).toBeInTheDocument();
		});

		fireEvent.click(screen.getByTestId("node-analyze"));

		await waitFor(() => {
			expect(screen.getByTestId("selected-node-detail")).toBeInTheDocument();
		});
	});

	it("navigates back via breadcrumb buttons", async () => {
		const onBack = vi.fn();
		const onBackToRuns = vi.fn();
		renderPage({ onBack, onBackToRuns });

		await waitFor(() => {
			expect(screen.getByText("Runs")).toBeInTheDocument();
		});

		fireEvent.click(screen.getByText("Runs"));
		expect(onBackToRuns).toHaveBeenCalled();

		fireEvent.click(screen.getByText("run-1"));
		expect(onBack).toHaveBeenCalled();
	});

	it("has critical path toggle", async () => {
		renderPage();

		await waitFor(() => {
			expect(screen.getByText("Critical path")).toBeInTheDocument();
		});

		const checkbox = screen.getByRole("checkbox");
		expect(checkbox).not.toBeChecked();

		fireEvent.click(checkbox);
		expect(checkbox).toBeChecked();
	});

	it("renders edges between dependent steps", async () => {
		renderPage();

		await waitFor(() => {
			expect(screen.getByTestId("edge-gather-analyze")).toBeInTheDocument();
			expect(screen.getByTestId("edge-analyze-report")).toBeInTheDocument();
		});
	});

	it("renders the page-level <main> landmark and complementary panel without axe violations", async () => {
		const { container } = renderPage();

		await waitFor(() => {
			expect(screen.getByTestId("workflow-detail-page")).toBeInTheDocument();
		});

		// Select a node to surface the bottom complementary panel.
		fireEvent.click(screen.getByTestId("node-analyze"));
		await waitFor(() => {
			expect(screen.getByTestId("selected-node-detail")).toBeInTheDocument();
		});

		// Landmark contract: the page is a single <main>, with a complementary
		// aside under it for the selected workflow step.
		expect(container.querySelector("main")).not.toBeNull();
		expect(screen.getByRole("complementary", { name: /Selected workflow step/i })).toBeInTheDocument();

		const results = await axe(container);
		expect(results).toHaveNoViolations();
	});

	it("shows View agent link for steps with agentSpanId", async () => {
		const onNavigateToAgent = vi.fn();
		renderPage({ onNavigateToAgent });

		await waitFor(() => {
			expect(screen.getByTestId("node-analyze")).toBeInTheDocument();
		});

		// Click the node to select it and see details
		fireEvent.click(screen.getByTestId("node-analyze"));

		await waitFor(() => {
			const viewAgentLinks = screen.getAllByText("View agent →");
			expect(viewAgentLinks.length).toBeGreaterThan(0);
		});
	});
});
