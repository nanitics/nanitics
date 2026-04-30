import type { AgentInfo, SpanTreeNode, TraceEvent } from "../types";

export interface CapabilityPanelProps {
	agent: AgentInfo;
	events: TraceEvent[];
	spanTree: SpanTreeNode;
}

export interface CapabilityPanelRegistration {
	/** Unique panel ID (e.g., "llm-calls"). */
	id: string;
	/** Tab label (e.g., "LLM Calls"). */
	label: string;
	/** Tab ordering — lower values appear further left. */
	order: number;
	/** Determines whether the panel tab is shown for a given agent. */
	isVisible: (agent: AgentInfo, events: TraceEvent[]) => boolean;
	/** React component to render the panel content. */
	component: React.ComponentType<CapabilityPanelProps>;
}

export class CapabilityPanelRegistry {
	private registrations: CapabilityPanelRegistration[] = [];

	register(registration: CapabilityPanelRegistration): void {
		this.registrations.push(registration);
	}

	getPanels(agent: AgentInfo, events: TraceEvent[]): CapabilityPanelRegistration[] {
		return this.registrations.filter((reg) => reg.isVisible(agent, events)).sort((a, b) => a.order - b.order);
	}
}
