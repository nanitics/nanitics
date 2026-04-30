import { CodeExecutionPanel } from "../components/panels/code-execution-panel";
import { DurabilityPanel } from "../components/panels/durability-panel";
import { ErrorRecoveryPanel } from "../components/panels/error-recovery-panel";
import { EvaluationPanel } from "../components/panels/evaluation-panel";
import { HITLPanel } from "../components/panels/hitl-panel";
import { LLMCallsPanel } from "../components/panels/llm-calls-panel";
import { MemoryInspectorPanel } from "../components/panels/memory-inspector-panel";
import { PlanningPanel } from "../components/panels/planning-panel";
import { ToolAnalyticsPanel } from "../components/panels/tool-analytics-panel";
import { CapabilityPanelRegistry } from "./capability-panel-registry";

/** Creates a CapabilityPanelRegistry with default panel registrations. */
export function createDefaultPanelRegistry(): CapabilityPanelRegistry {
	const registry = new CapabilityPanelRegistry();

	registry.register({
		id: "llm-calls",
		label: "LLM Calls",
		order: 10,
		isVisible: () => true,
		component: LLMCallsPanel,
	});

	registry.register({
		id: "tools",
		label: "Tools",
		order: 20,
		isVisible: (agent) => agent.stats.tool_calls > 0,
		component: ToolAnalyticsPanel,
	});

	registry.register({
		id: "code-execution",
		label: "Code Execution",
		order: 25,
		isVisible: (_agent, events) => events.some((e) => e.event_type === "code.execution"),
		component: CodeExecutionPanel,
	});

	registry.register({
		id: "errors",
		label: "Errors",
		order: 30,
		isVisible: (agent) => agent.stats.errors > 0,
		component: ErrorRecoveryPanel,
	});

	registry.register({
		id: "memory",
		label: "Memory",
		order: 40,
		isVisible: (_agent, events) => events.some((e) => e.event_type.startsWith("memory.")),
		component: MemoryInspectorPanel,
	});

	registry.register({
		id: "planning",
		label: "Planning",
		order: 50,
		isVisible: (agent, events) =>
			agent.capabilities.includes("planning") || events.some((e) => e.event_type.startsWith("planning.")),
		component: PlanningPanel,
	});

	registry.register({
		id: "evaluation",
		label: "Evaluation",
		order: 60,
		isVisible: (_agent, events) =>
			events.some((e) => e.event_type.startsWith("evaluation.") || e.event_type.startsWith("reflection.")),
		component: EvaluationPanel,
	});

	registry.register({
		id: "hitl",
		label: "Human-in-the-Loop",
		order: 70,
		isVisible: (_agent, events) =>
			events.some(
				(e) =>
					e.event_type === "hitl.request" || e.event_type === "hitl.response" || e.event_type.startsWith("revision."),
			),
		component: HITLPanel,
	});

	registry.register({
		id: "durability",
		label: "Durability",
		order: 75,
		isVisible: (_agent, events) =>
			events.some(
				(e) =>
					e.event_type === "execution.suspended" ||
					e.event_type === "execution.resumed" ||
					e.event_type === "checkpoint.saved",
			),
		component: DurabilityPanel,
	});

	return registry;
}
