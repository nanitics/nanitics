import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ObservatoryClient } from "../../src/client/observatory-client";
import { ObservatoryProvider } from "../../src/context/observatory-context";
import { AgentDetailPage } from "../../src/pages/agent-detail-page";
import type { AgentViewProps } from "../../src/registry/agent-view-registry";
import { AgentViewRegistry } from "../../src/registry/agent-view-registry";
import type { CapabilityPanelProps } from "../../src/registry/capability-panel-registry";
import { CapabilityPanelRegistry } from "../../src/registry/capability-panel-registry";
import { EventRendererRegistry } from "../../src/registry/renderer-registry";
import type { AgentDetailResponse, AgentInfo, SpanTreeNode, TraceEvent } from "../../src/types";
import { makeEvent } from "../fixtures/scenarios";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const mockAgent: AgentInfo = {
	agent_name: "researcher",
	agent_type: "react",
	span_id: "agent-1",
	capabilities: ["web_search", "evaluation"],
	stats: {
		llm_calls: 5,
		tool_calls: 3,
		input_tokens: 1200,
		output_tokens: 800,
		duration_ms: 3500,
		errors: 1,
		iterations: 4,
	},
};

const mockEvents: TraceEvent[] = [
	makeEvent({ event_type: "agent.start", span_id: "agent-1" }),
	makeEvent({ event_type: "agent.step", span_id: "agent-1" }),
	makeEvent({ event_type: "agent.complete", span_id: "agent-1" }),
];

const mockSpanTree: SpanTreeNode = {
	span_id: "agent-1",
	parent_span_id: "root",
	name: "researcher",
	summary: {
		event_count: 3,
		duration_ms: 3500,
		has_errors: false,
		agent_name: "researcher",
		agent_type: "react",
	},
	events: [],
	children: [],
};

const mockResponse: AgentDetailResponse = {
	agent: mockAgent,
	events: mockEvents,
	span_tree: mockSpanTree,
};

function FakeTimelineView({ agent }: AgentViewProps) {
	return <div data-testid="timeline-view">Timeline for {agent.agent_name}</div>;
}

function FakeLLMPanel({ agent }: CapabilityPanelProps) {
	return <div data-testid="llm-panel">LLM Calls for {agent.agent_name}</div>;
}

function FakeToolsPanel({ agent }: CapabilityPanelProps) {
	return <div data-testid="tools-panel">Tools for {agent.agent_name}</div>;
}

function FakeErrorsPanel({ agent }: CapabilityPanelProps) {
	return <div data-testid="errors-panel">Errors for {agent.agent_name}</div>;
}

function createContext(overrides?: { agentViewRegistry?: AgentViewRegistry; panelRegistry?: CapabilityPanelRegistry }) {
	const client = new ObservatoryClient("/test");
	vi.spyOn(client, "getAgentDetail").mockResolvedValue(mockResponse);

	const agentViewRegistry = overrides?.agentViewRegistry ?? new AgentViewRegistry();
	if (!overrides?.agentViewRegistry) {
		agentViewRegistry.registerFallback(FakeTimelineView);
	}

	const panelRegistry = overrides?.panelRegistry ?? new CapabilityPanelRegistry();

	return { client, agentViewRegistry, panelRegistry };
}

function renderPage(
	props?: Partial<React.ComponentProps<typeof AgentDetailPage>>,
	contextOverrides?: Parameters<typeof createContext>[0],
) {
	const { client, agentViewRegistry, panelRegistry } = createContext(contextOverrides);

	return render(
		<ObservatoryProvider
			client={client}
			registry={new EventRendererRegistry()}
			agentViewRegistry={agentViewRegistry}
			panelRegistry={panelRegistry}
		>
			<AgentDetailPage
				runId="run-1"
				spanId="agent-1"
				onBack={props?.onBack ?? vi.fn()}
				onBackToRuns={props?.onBackToRuns ?? vi.fn()}
				runLabel={props?.runLabel}
			/>
		</ObservatoryProvider>,
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("AgentDetailPage", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("shows loading skeleton initially", () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "getAgentDetail").mockReturnValue(new Promise(() => {}));

		const { container } = render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<AgentDetailPage runId="run-1" spanId="agent-1" onBack={vi.fn()} onBackToRuns={vi.fn()} />
			</ObservatoryProvider>,
		);

		// AgentDetailSkeleton carries the a11y contract — verify it rather
		// than the specific placeholder shape.
		expect(screen.getByRole("status")).toBeInTheDocument();
		expect(container.querySelectorAll("[aria-busy='true']").length).toBeGreaterThan(0);
	});

	it("shows error state on failure", async () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "getAgentDetail").mockRejectedValue(new Error("Network error"));

		render(
			<ObservatoryProvider client={client} registry={new EventRendererRegistry()}>
				<AgentDetailPage runId="run-1" spanId="agent-1" onBack={vi.fn()} onBackToRuns={vi.fn()} />
			</ObservatoryProvider>,
		);

		// The page-level ErrorState surfaces the default title, the stable subtitle,
		// and the raw error inside the <details> pane.
		expect(await screen.findByRole("alert")).toBeInTheDocument();
		expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
		expect(screen.getByText(/Network error/)).toBeInTheDocument();
	});

	it("renders agent header with name, type badge, and capabilities", async () => {
		renderPage();

		// Agent name appears in both breadcrumb and header
		const names = await screen.findAllByText("researcher");
		expect(names.length).toBeGreaterThanOrEqual(1);
		expect(screen.getByText("react")).toBeInTheDocument();
		expect(screen.getByText("web_search")).toBeInTheDocument();
		expect(screen.getByText("evaluation")).toBeInTheDocument();
	});

	it("renders stats in the header", async () => {
		renderPage();

		await screen.findByText("LLM calls");

		expect(screen.getByText("5")).toBeInTheDocument();
		expect(screen.getByText("Tool calls")).toBeInTheDocument();
		expect(screen.getByText("3")).toBeInTheDocument();
		expect(screen.getByText("Iterations")).toBeInTheDocument();
		expect(screen.getByText("4")).toBeInTheDocument();
		expect(screen.getByText("Errors")).toBeInTheDocument();
		expect(screen.getByText("1")).toBeInTheDocument();
	});

	it("renders breadcrumb with Runs, run label, and agent name", async () => {
		renderPage({ runLabel: "My Research Run" });

		await screen.findByText("Runs");

		expect(screen.getByText("My Research Run")).toBeInTheDocument();
	});

	it("renders breadcrumb with runId when no label provided", async () => {
		renderPage();

		await screen.findByText("run-1");

		expect(screen.getByText("run-1")).toBeInTheDocument();
	});

	it("calls onBackToRuns when clicking Runs breadcrumb", async () => {
		const onBackToRuns = vi.fn();
		renderPage({ onBackToRuns });

		await screen.findByText("Runs");
		fireEvent.click(screen.getByText("Runs"));

		expect(onBackToRuns).toHaveBeenCalledOnce();
	});

	it("calls onBack when clicking run breadcrumb", async () => {
		const onBack = vi.fn();
		renderPage({ onBack, runLabel: "My Run" });

		await screen.findByText("My Run");
		fireEvent.click(screen.getByText("My Run"));

		expect(onBack).toHaveBeenCalledOnce();
	});

	it("renders Timeline tab and dispatches to correct view", async () => {
		renderPage();

		expect(await screen.findByText("Timeline")).toBeInTheDocument();
		expect(screen.getByTestId("timeline-view")).toHaveTextContent("Timeline for researcher");
	});

	it("renders dynamic capability panel tabs from registry", async () => {
		const panelRegistry = new CapabilityPanelRegistry();
		panelRegistry.register({
			id: "llm-calls",
			label: "LLM Calls",
			order: 10,
			isVisible: () => true,
			component: FakeLLMPanel,
		});
		panelRegistry.register({
			id: "tools",
			label: "Tools",
			order: 20,
			isVisible: (agent) => agent.stats.tool_calls > 0,
			component: FakeToolsPanel,
		});
		panelRegistry.register({
			id: "errors",
			label: "Error Recovery",
			order: 30,
			isVisible: (agent) => agent.stats.errors > 0,
			component: FakeErrorsPanel,
		});

		renderPage({}, { panelRegistry });

		await screen.findByText("LLM Calls");

		expect(screen.getByText("Tools")).toBeInTheDocument();
		expect(screen.getByText("Error Recovery")).toBeInTheDocument();
	});

	it("hides panels when visibility predicate returns false", async () => {
		const panelRegistry = new CapabilityPanelRegistry();
		panelRegistry.register({
			id: "errors",
			label: "Error Recovery",
			order: 30,
			isVisible: () => false,
			component: FakeErrorsPanel,
		});

		renderPage({}, { panelRegistry });

		await screen.findByText("Timeline");

		expect(screen.queryByText("Error Recovery")).not.toBeInTheDocument();
	});

	it("switches tab content when clicking a capability panel tab", async () => {
		const panelRegistry = new CapabilityPanelRegistry();
		panelRegistry.register({
			id: "llm-calls",
			label: "LLM Calls",
			order: 10,
			isVisible: () => true,
			component: FakeLLMPanel,
		});

		renderPage({}, { panelRegistry });

		await screen.findByText("Timeline");

		// Initially shows timeline
		expect(screen.getByTestId("timeline-view")).toBeInTheDocument();
		expect(screen.queryByTestId("llm-panel")).not.toBeInTheDocument();

		// Click LLM Calls tab
		fireEvent.click(screen.getByText("LLM Calls"));

		expect(screen.getByTestId("llm-panel")).toHaveTextContent("LLM Calls for researcher");
		expect(screen.queryByTestId("timeline-view")).not.toBeInTheDocument();
	});

	it("switches back to Timeline tab", async () => {
		const panelRegistry = new CapabilityPanelRegistry();
		panelRegistry.register({
			id: "llm-calls",
			label: "LLM Calls",
			order: 10,
			isVisible: () => true,
			component: FakeLLMPanel,
		});

		renderPage({}, { panelRegistry });

		await screen.findByText("Timeline");

		// Switch to LLM Calls
		fireEvent.click(screen.getByText("LLM Calls"));
		expect(screen.getByTestId("llm-panel")).toBeInTheDocument();

		// Switch back to Timeline
		fireEvent.click(screen.getByText("Timeline"));
		expect(screen.getByTestId("timeline-view")).toBeInTheDocument();
		expect(screen.queryByTestId("llm-panel")).not.toBeInTheDocument();
	});

	it("dispatches to correct agent view based on agent type", async () => {
		const agentViewRegistry = new AgentViewRegistry();
		function ReactView({ agent }: AgentViewProps) {
			return <div data-testid="react-view">ReAct: {agent.agent_name}</div>;
		}
		agentViewRegistry.register({
			agentType: "react",
			component: ReactView,
		});
		agentViewRegistry.registerFallback(FakeTimelineView);

		renderPage({}, { agentViewRegistry });

		expect(await screen.findByTestId("react-view")).toHaveTextContent("ReAct: researcher");
	});

	it("falls back to generic view for unknown agent type", async () => {
		const client = new ObservatoryClient("/test");
		const unknownAgent = {
			...mockResponse,
			agent: { ...mockAgent, agent_type: "custom-unknown" },
		};
		vi.spyOn(client, "getAgentDetail").mockResolvedValue(unknownAgent);

		const agentViewRegistry = new AgentViewRegistry();
		agentViewRegistry.registerFallback(FakeTimelineView);

		render(
			<ObservatoryProvider
				client={client}
				registry={new EventRendererRegistry()}
				agentViewRegistry={agentViewRegistry}
				panelRegistry={new CapabilityPanelRegistry()}
			>
				<AgentDetailPage runId="run-1" spanId="agent-1" onBack={vi.fn()} onBackToRuns={vi.fn()} />
			</ObservatoryProvider>,
		);

		expect(await screen.findByTestId("timeline-view")).toHaveTextContent("Timeline for researcher");
	});

	it("renders agent type in breadcrumb parenthetical", async () => {
		renderPage();

		await screen.findByText("(react)");

		expect(screen.getByText("(react)")).toBeInTheDocument();
	});

	it("renders a user-visible notice when related-agents fetch fails", async () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "getAgentDetail").mockResolvedValue(mockResponse);
		vi.spyOn(client, "listAgents").mockRejectedValue(new Error("Network error"));
		vi.spyOn(client, "queryEvents").mockRejectedValue(new Error("Network error"));

		const agentViewRegistry = new AgentViewRegistry();
		agentViewRegistry.registerFallback(FakeTimelineView);

		const { container } = render(
			<ObservatoryProvider
				client={client}
				registry={new EventRendererRegistry()}
				agentViewRegistry={agentViewRegistry}
				panelRegistry={new CapabilityPanelRegistry()}
			>
				<AgentDetailPage
					runId="run-1"
					spanId="agent-1"
					onBack={vi.fn()}
					onBackToRuns={vi.fn()}
					onNavigateToAgent={vi.fn()}
				/>
			</ObservatoryProvider>,
		);

		// The notice appears after the rejection flushes.
		const notice = await screen.findByText(/Couldn't load related agents/i);
		expect(notice).toBeInTheDocument();

		// Rehosted on <ErrorState variant="inline">, which carries role="alert".
		expect(screen.getByRole("alert")).toBeInTheDocument();

		// The raw error message is available inside a <details> for developers.
		expect(screen.getByText(/Network error/)).toBeInTheDocument();

		// The notice renders without a11y violations.
		await waitFor(async () => {
			expect(await axe(container)).toHaveNoViolations();
		});
	});

	it("renders the page-level <main> landmark and tab region without axe violations", async () => {
		const { container } = renderPage();

		// Wait for the page to fully render (not the loading skeleton).
		await screen.findByTestId("timeline-view");

		// Landmark contract: the page is a single <main>, the tab toolbar carries
		// an accessible name, and the tab content section is labelled by the
		// active tab button id.
		expect(container.querySelector("main")).not.toBeNull();
		expect(screen.getByRole("toolbar", { name: /Agent detail views/i })).toBeInTheDocument();
		// `<section>` with aria-labelledby has the implicit `region` role.
		expect(screen.getByRole("region")).toBeInTheDocument();

		const results = await axe(container);
		expect(results).toHaveNoViolations();
	});

	it("omits type badge and breadcrumb parenthetical when agent_type is null", async () => {
		const client = new ObservatoryClient("/test");
		vi.spyOn(client, "getAgentDetail").mockResolvedValue({
			...mockResponse,
			agent: { ...mockAgent, agent_type: null },
		});

		const agentViewRegistry = new AgentViewRegistry();
		agentViewRegistry.registerFallback(FakeTimelineView);

		render(
			<ObservatoryProvider
				client={client}
				registry={new EventRendererRegistry()}
				agentViewRegistry={agentViewRegistry}
				panelRegistry={new CapabilityPanelRegistry()}
			>
				<AgentDetailPage runId="run-1" spanId="agent-1" onBack={vi.fn()} onBackToRuns={vi.fn()} />
			</ObservatoryProvider>,
		);

		await screen.findByTestId("timeline-view");

		// No type badge or parenthetical
		expect(screen.queryByText("(react)")).not.toBeInTheDocument();
		expect(screen.queryByText("react")).not.toBeInTheDocument();
	});
});
