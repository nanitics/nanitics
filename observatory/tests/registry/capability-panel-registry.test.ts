import { describe, expect, it } from "vitest";
import { type CapabilityPanelProps, CapabilityPanelRegistry } from "../../src/registry/capability-panel-registry";
import type { AgentInfo } from "../../src/types";

function DummyPanel(_props: CapabilityPanelProps) {
	return null;
}

function AnotherPanel(_props: CapabilityPanelProps) {
	return null;
}

function makeAgent(overrides: Partial<AgentInfo> = {}): AgentInfo {
	return {
		agent_name: "test-agent",
		agent_type: "react",
		span_id: "span-1",
		capabilities: [],
		stats: {
			llm_calls: 3,
			tool_calls: 2,
			input_tokens: 500,
			output_tokens: 300,
			duration_ms: 1000,
			errors: 0,
			iterations: 2,
		},
		...overrides,
	};
}

describe("CapabilityPanelRegistry", () => {
	it("returns empty array when no panels registered", () => {
		const registry = new CapabilityPanelRegistry();
		const panels = registry.getPanels(makeAgent(), []);
		expect(panels).toEqual([]);
	});

	it("returns all visible panels sorted by order", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "tools",
			label: "Tools",
			order: 20,
			isVisible: () => true,
			component: AnotherPanel,
		});
		registry.register({
			id: "llm-calls",
			label: "LLM Calls",
			order: 10,
			isVisible: () => true,
			component: DummyPanel,
		});

		const panels = registry.getPanels(makeAgent(), []);
		expect(panels.map((p) => p.id)).toEqual(["llm-calls", "tools"]);
	});

	it("filters out panels where isVisible returns false", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "llm-calls",
			label: "LLM Calls",
			order: 10,
			isVisible: () => true,
			component: DummyPanel,
		});
		registry.register({
			id: "errors",
			label: "Errors",
			order: 30,
			isVisible: (agent) => agent.stats.errors > 0,
			component: AnotherPanel,
		});

		const agentNoErrors = makeAgent({ stats: { ...makeAgent().stats, errors: 0 } });
		const panels = registry.getPanels(agentNoErrors, []);
		expect(panels.map((p) => p.id)).toEqual(["llm-calls"]);
	});

	it("shows error panel when agent has errors", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "errors",
			label: "Errors",
			order: 30,
			isVisible: (agent) => agent.stats.errors > 0,
			component: AnotherPanel,
		});

		const agentWithErrors = makeAgent({
			stats: { ...makeAgent().stats, errors: 2 },
		});
		const panels = registry.getPanels(agentWithErrors, []);
		expect(panels).toHaveLength(1);
		expect(panels[0].id).toBe("errors");
	});

	it("shows tools panel when agent has tool calls", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "tools",
			label: "Tools",
			order: 20,
			isVisible: (agent) => agent.stats.tool_calls > 0,
			component: DummyPanel,
		});

		const agentWithTools = makeAgent({
			stats: { ...makeAgent().stats, tool_calls: 5 },
		});
		const panels = registry.getPanels(agentWithTools, []);
		expect(panels).toHaveLength(1);
		expect(panels[0].id).toBe("tools");
	});

	it("hides tools panel when agent has no tool calls", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "tools",
			label: "Tools",
			order: 20,
			isVisible: (agent) => agent.stats.tool_calls > 0,
			component: DummyPanel,
		});

		const agentNoTools = makeAgent({
			stats: { ...makeAgent().stats, tool_calls: 0 },
		});
		const panels = registry.getPanels(agentNoTools, []);
		expect(panels).toEqual([]);
	});

	it("passes events to visibility predicate", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "custom",
			label: "Custom",
			order: 10,
			isVisible: (_agent, events) => events.length > 0,
			component: DummyPanel,
		});

		expect(registry.getPanels(makeAgent(), [])).toEqual([]);
		expect(
			registry.getPanels(makeAgent(), [
				{
					id: 1,
					event_type: "test",
					level: "info",
					trace_id: "t",
					span_id: "s",
					parent_span_id: null,
					timestamp: "2026-01-01T00:00:00Z",
					payload: {},
				},
			]),
		).toHaveLength(1);
	});

	it("maintains stable sort for equal order values", () => {
		const registry = new CapabilityPanelRegistry();
		registry.register({
			id: "first",
			label: "First",
			order: 10,
			isVisible: () => true,
			component: DummyPanel,
		});
		registry.register({
			id: "second",
			label: "Second",
			order: 10,
			isVisible: () => true,
			component: AnotherPanel,
		});

		const panels = registry.getPanels(makeAgent(), []);
		// Array.sort is stable in modern JS — registration order preserved for equal order values
		expect(panels.map((p) => p.id)).toEqual(["first", "second"]);
	});
});
