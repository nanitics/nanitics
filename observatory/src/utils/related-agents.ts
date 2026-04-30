import type { AgentInfo, TraceEvent } from "../types";

export interface RelatedAgent {
	agentName: string;
	spanId: string;
	relationship: string;
	eventType: string;
}

/**
 * Scans trace events for multi-agent pattern events and finds agents
 * related to the current agent via delegation, handoff, coordination, etc.
 */
export function findRelatedAgents(currentAgentName: string, events: TraceEvent[], agents: AgentInfo[]): RelatedAgent[] {
	const agentMap = new Map(agents.map((a) => [a.agent_name, a.span_id]));
	const seen = new Map<string, RelatedAgent>();

	function add(agentName: string, relationship: string, eventType: string): void {
		if (agentName === currentAgentName) return;
		if (!agentMap.has(agentName)) return;
		// Keep first relationship found per agent
		if (seen.has(agentName)) return;
		seen.set(agentName, {
			agentName,
			spanId: agentMap.get(agentName)!,
			relationship,
			eventType,
		});
	}

	for (const event of events) {
		const p = event.payload as Record<string, unknown>;
		switch (event.event_type) {
			case "multi_agent.delegation": {
				const caller = p.caller_agent as string | undefined;
				const delegate = p.delegate_agent as string | undefined;
				if (caller === currentAgentName && delegate) {
					add(delegate, "delegated to", event.event_type);
				}
				if (delegate === currentAgentName && caller) {
					add(caller, "delegated from", event.event_type);
				}
				break;
			}

			case "multi_agent.handoff": {
				const from = p.from_agent as string | undefined;
				const to = p.to_agent as string | undefined;
				if (from === currentAgentName && to) {
					add(to, "handed off to", event.event_type);
				}
				if (to === currentAgentName && from) {
					add(from, "received handoff from", event.event_type);
				}
				break;
			}

			case "multi_agent.supervision": {
				const supervised = p.supervised_agent as string | undefined;
				const reassigned = p.reassigned_to as string | undefined;
				if (supervised === currentAgentName && reassigned) {
					add(reassigned, "reassigned to", event.event_type);
				}
				if (reassigned === currentAgentName && supervised) {
					add(supervised, "supervises", event.event_type);
				}
				break;
			}

			case "multi_agent.bidding.start": {
				const names = p.participant_names as string[] | undefined;
				if (names?.includes(currentAgentName)) {
					for (const name of names) {
						add(name, "bidding participant", event.event_type);
					}
				}
				break;
			}

			case "multi_agent.debate.start": {
				const names = p.debater_names as string[] | undefined;
				if (names?.includes(currentAgentName)) {
					for (const name of names) {
						add(name, "debate participant", event.event_type);
					}
				}
				break;
			}

			case "multi_agent.consensus.start": {
				const names = p.agent_names as string[] | undefined;
				if (names?.includes(currentAgentName)) {
					for (const name of names) {
						add(name, "consensus participant", event.event_type);
					}
				}
				break;
			}

			case "multi_agent.broadcast.start": {
				const names = p.agent_names as string[] | undefined;
				if (names?.includes(currentAgentName)) {
					for (const name of names) {
						add(name, "broadcast participant", event.event_type);
					}
				}
				break;
			}

			case "blackboard.start": {
				const names = p.agent_names as string[] | undefined;
				if (names?.includes(currentAgentName)) {
					for (const name of names) {
						add(name, "blackboard participant", event.event_type);
					}
				}
				break;
			}

			case "multi_agent.peer.start": {
				const entry = p.entry_agent as string | undefined;
				const peers = p.peer_names as string[] | undefined;
				if (entry === currentAgentName || peers?.includes(currentAgentName)) {
					if (entry && entry !== currentAgentName) {
						add(entry, "peer network entry", event.event_type);
					}
					if (peers) {
						for (const name of peers) {
							add(name, "peer", event.event_type);
						}
					}
				}
				break;
			}

			case "multi_agent.peer.consultation": {
				const from = p.from_agent as string | undefined;
				const to = p.to_agent as string | undefined;
				if (from === currentAgentName && to) {
					add(to, "consulted", event.event_type);
				}
				if (to === currentAgentName && from) {
					add(from, "consulted by", event.event_type);
				}
				break;
			}

			case "multi_agent.bus.start": {
				const subscriptions = p.subscriptions as Record<string, string[]> | undefined;
				if (subscriptions) {
					// Check if current agent is a subscriber
					const allSubscribers = new Set(Object.values(subscriptions).flat());
					if (allSubscribers.has(currentAgentName)) {
						for (const name of allSubscribers) {
							add(name, "message bus participant", event.event_type);
						}
					}
				}
				break;
			}
		}
	}

	return Array.from(seen.values());
}
