import { CodeActAgentView } from "../components/agent-views/codeact-agent-view";
import { GenericAgentView } from "../components/agent-views/generic-agent-view";
import { LATSAgentView } from "../components/agent-views/lats-agent-view";
import { ReActAgentView } from "../components/agent-views/react-agent-view";
import { ReflexionAgentView } from "../components/agent-views/reflexion-agent-view";
import { ReWOOAgentView } from "../components/agent-views/rewoo-agent-view";
import { TreeOfThoughtAgentView } from "../components/agent-views/tree-of-thought-agent-view";
import { AgentViewRegistry } from "./agent-view-registry";

/** Creates an AgentViewRegistry with default registrations. */
export function createDefaultAgentViewRegistry(): AgentViewRegistry {
	const registry = new AgentViewRegistry();
	registry.register({ agentType: "react", component: ReActAgentView, label: "ReAct" });
	registry.register({ agentType: "codeact", component: CodeActAgentView, label: "CodeAct" });
	registry.register({ agentType: "rewoo", component: ReWOOAgentView, label: "ReWOO" });
	registry.register({ agentType: "reflexion", component: ReflexionAgentView, label: "Reflexion" });
	registry.register({ agentType: "tree_of_thought", component: TreeOfThoughtAgentView, label: "Tree of Thought" });
	registry.register({ agentType: "lats", component: LATSAgentView, label: "LATS" });
	registry.registerFallback(GenericAgentView);
	return registry;
}
