import { describe, expect, it } from "vitest";
import { type AgentViewProps, AgentViewRegistry } from "../../src/registry/agent-view-registry";

function FallbackView(_props: AgentViewProps) {
	return null;
}

function ReactView(_props: AgentViewProps) {
	return null;
}

function CodeActView(_props: AgentViewProps) {
	return null;
}

describe("AgentViewRegistry", () => {
	it("throws when no view registered and no fallback", () => {
		const registry = new AgentViewRegistry();
		expect(() => registry.getView("react")).toThrow(
			'No agent view registered for type "react" and no fallback registered',
		);
	});

	it("returns fallback for null agent type", () => {
		const registry = new AgentViewRegistry();
		registry.registerFallback(FallbackView);
		expect(registry.getView(null)).toBe(FallbackView);
	});

	it("returns fallback for unregistered agent type", () => {
		const registry = new AgentViewRegistry();
		registry.registerFallback(FallbackView);
		expect(registry.getView("unknown")).toBe(FallbackView);
	});

	it("returns registered view for exact agent type match", () => {
		const registry = new AgentViewRegistry();
		registry.registerFallback(FallbackView);
		registry.register({ agentType: "react", component: ReactView });
		registry.register({ agentType: "codeact", component: CodeActView });

		expect(registry.getView("react")).toBe(ReactView);
		expect(registry.getView("codeact")).toBe(CodeActView);
	});

	it("does not match partial agent type strings", () => {
		const registry = new AgentViewRegistry();
		registry.registerFallback(FallbackView);
		registry.register({ agentType: "react", component: ReactView });

		expect(registry.getView("react-v2")).toBe(FallbackView);
	});

	it("last registration wins for duplicate agent types", () => {
		const registry = new AgentViewRegistry();
		registry.register({ agentType: "react", component: FallbackView });
		registry.register({ agentType: "react", component: ReactView });

		expect(registry.getView("react")).toBe(ReactView);
	});

	it("returns agentType as label when no custom label", () => {
		const registry = new AgentViewRegistry();
		registry.register({ agentType: "react", component: ReactView });
		expect(registry.getLabel("react")).toBe("react");
	});

	it("returns custom label when provided", () => {
		const registry = new AgentViewRegistry();
		registry.register({
			agentType: "react",
			component: ReactView,
			label: "ReAct Agent",
		});
		expect(registry.getLabel("react")).toBe("ReAct Agent");
	});

	it("returns agentType for unknown agent type label", () => {
		const registry = new AgentViewRegistry();
		expect(registry.getLabel("unknown")).toBe("unknown");
	});
});
