import type { AgentInfo, SpanTreeNode, TraceEvent } from "../types";

export interface AgentViewProps {
	agent: AgentInfo;
	events: TraceEvent[];
	spanTree: SpanTreeNode;
}

export interface AgentViewRegistration {
	/** Agent type string to match (e.g., "react", "codeact"). */
	agentType: string;
	/** React component to render the agent timeline view. */
	component: React.ComponentType<AgentViewProps>;
	/** Display label (defaults to agentType). */
	label?: string;
}

export class AgentViewRegistry {
	private registrations = new Map<string, AgentViewRegistration>();
	private fallback: React.ComponentType<AgentViewProps> | null = null;

	register(registration: AgentViewRegistration): void {
		this.registrations.set(registration.agentType, registration);
	}

	registerFallback(component: React.ComponentType<AgentViewProps>): void {
		this.fallback = component;
	}

	getView(agentType: string | null): React.ComponentType<AgentViewProps> {
		if (agentType) {
			const reg = this.registrations.get(agentType);
			if (reg) return reg.component;
		}
		if (this.fallback) return this.fallback;
		throw new Error(`No agent view registered for type "${agentType}" and no fallback registered`);
	}

	getLabel(agentType: string): string {
		const reg = this.registrations.get(agentType);
		return reg?.label ?? agentType;
	}
}
