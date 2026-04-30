import { createContext, useContext, useMemo } from "react";
import type { ObservatoryClient } from "../client/observatory-client";
import { AgentViewRegistry } from "../registry/agent-view-registry";
import { CapabilityPanelRegistry } from "../registry/capability-panel-registry";
import type { EventRendererRegistry } from "../registry/renderer-registry";

export interface ObservatoryContextValue {
	client: ObservatoryClient;
	registry: EventRendererRegistry;
	agentViewRegistry: AgentViewRegistry;
	panelRegistry: CapabilityPanelRegistry;
}

const ObservatoryContext = createContext<ObservatoryContextValue | null>(null);

export function ObservatoryProvider({
	client,
	registry,
	agentViewRegistry,
	panelRegistry,
	children,
}: {
	client: ObservatoryClient;
	registry: EventRendererRegistry;
	agentViewRegistry?: AgentViewRegistry;
	panelRegistry?: CapabilityPanelRegistry;
	children: React.ReactNode;
}) {
	const resolvedAgentViewRegistry = useMemo(() => agentViewRegistry ?? new AgentViewRegistry(), [agentViewRegistry]);
	const resolvedPanelRegistry = useMemo(() => panelRegistry ?? new CapabilityPanelRegistry(), [panelRegistry]);

	return (
		<ObservatoryContext.Provider
			value={{
				client,
				registry,
				agentViewRegistry: resolvedAgentViewRegistry,
				panelRegistry: resolvedPanelRegistry,
			}}
		>
			{children}
		</ObservatoryContext.Provider>
	);
}

export function useObservatory(): ObservatoryContextValue {
	const ctx = useContext(ObservatoryContext);
	if (!ctx) {
		throw new Error("useObservatory must be used within an ObservatoryProvider");
	}
	return ctx;
}
