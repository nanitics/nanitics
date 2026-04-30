import type { SpanTreeNode, TraceEvent } from "../types";

export type PatternType =
	| "delegation"
	| "broadcast"
	| "bidding"
	| "supervision"
	| "handoff"
	| "debate"
	| "consensus"
	| "blackboard"
	| "peer_network"
	| "message_bus";

export interface DetectedPattern {
	type: PatternType;
	events: TraceEvent[];
	/** The span where the pattern's root event lives */
	spanId: string;
	/** Human-readable label derived from events */
	label: string;
}

/**
 * Walk the span tree and detect multi-agent coordination patterns.
 * Returns array of DetectedPattern instances, one per pattern occurrence.
 */
export function detectPatterns(tree: SpanTreeNode): DetectedPattern[] {
	const patterns: DetectedPattern[] = [];
	const allHandoffs: TraceEvent[] = [];

	walkTree(tree, patterns, allHandoffs);

	// Handoffs are grouped across spans by timestamp order into chains
	if (allHandoffs.length > 0) {
		const chains = buildHandoffChains(allHandoffs);
		for (const chain of chains) {
			const first = chain[0].payload as Record<string, unknown>;
			const names: string[] = [first.from_agent as string];
			for (const ev of chain) {
				const p = ev.payload as Record<string, unknown>;
				names.push(p.to_agent as string);
			}
			patterns.push({
				type: "handoff",
				events: chain,
				spanId: chain[0].span_id,
				label: names.join(" → "),
			});
		}
	}

	return patterns;
}

function walkTree(node: SpanTreeNode, patterns: DetectedPattern[], allHandoffs: TraceEvent[]): void {
	// Collect events by prefix within this span
	const delegations: TraceEvent[] = [];
	const broadcastEvents: TraceEvent[] = [];
	const biddingEvents: TraceEvent[] = [];
	const supervisionEvents: TraceEvent[] = [];
	const debateEvents: TraceEvent[] = [];
	const consensusEvents: TraceEvent[] = [];
	const blackboardEvents: TraceEvent[] = [];
	const peerNetworkEvents: TraceEvent[] = [];
	const messageBusEvents: TraceEvent[] = [];

	for (const event of node.events) {
		switch (event.event_type) {
			case "multi_agent.delegation":
				delegations.push(event);
				break;
			case "multi_agent.broadcast.start":
			case "multi_agent.broadcast.response":
			case "multi_agent.broadcast.complete":
				broadcastEvents.push(event);
				break;
			case "multi_agent.bidding.start":
			case "multi_agent.bidding.bid":
			case "multi_agent.bidding.allocated":
			case "multi_agent.bidding.complete":
				biddingEvents.push(event);
				break;
			case "multi_agent.supervision":
				supervisionEvents.push(event);
				break;
			case "multi_agent.handoff":
				allHandoffs.push(event);
				break;
			case "multi_agent.debate.start":
			case "multi_agent.debate.argument":
			case "multi_agent.debate.resolution":
			case "multi_agent.debate.complete":
				debateEvents.push(event);
				break;
			case "multi_agent.consensus.start":
			case "multi_agent.consensus.vote":
			case "multi_agent.consensus.agreement":
			case "multi_agent.consensus.complete":
				consensusEvents.push(event);
				break;
			case "blackboard.start":
			case "blackboard.round":
			case "blackboard.complete":
				blackboardEvents.push(event);
				break;
			case "multi_agent.peer.start":
			case "multi_agent.peer.consultation":
			case "multi_agent.peer.complete":
				peerNetworkEvents.push(event);
				break;
			case "multi_agent.bus.start":
			case "multi_agent.bus.published":
			case "multi_agent.bus.delivered":
			case "multi_agent.bus.complete":
				messageBusEvents.push(event);
				break;
		}
	}

	// Each delegation event is its own pattern instance
	for (const event of delegations) {
		const p = event.payload as Record<string, unknown>;
		const caller = (p.caller_agent as string) ?? "?";
		const delegate = (p.delegate_agent as string) ?? "?";
		patterns.push({
			type: "delegation",
			events: [event],
			spanId: node.span_id,
			label: `${caller} → ${delegate}`,
		});
	}

	// Broadcast: group all broadcast.* events in same span
	if (broadcastEvents.some((e) => e.event_type === "multi_agent.broadcast.start")) {
		const startEvent = broadcastEvents.find((e) => e.event_type === "multi_agent.broadcast.start")!;
		const p = startEvent.payload as Record<string, unknown>;
		const agentNames = (p.agent_names as string[]) ?? [];
		patterns.push({
			type: "broadcast",
			events: broadcastEvents,
			spanId: node.span_id,
			label: `Broadcast to ${agentNames.length} agents`,
		});
	}

	// Bidding: group all bidding.* events in same span
	if (biddingEvents.some((e) => e.event_type === "multi_agent.bidding.start")) {
		const startEvent = biddingEvents.find((e) => e.event_type === "multi_agent.bidding.start")!;
		const p = startEvent.payload as Record<string, unknown>;
		const participants = (p.participant_names as string[]) ?? [];
		patterns.push({
			type: "bidding",
			events: biddingEvents,
			spanId: node.span_id,
			label: `Bidding (${participants.length} participants)`,
		});
	}

	// Supervision: group by supervised_agent within same span
	if (supervisionEvents.length > 0) {
		const byAgent = new Map<string, TraceEvent[]>();
		for (const event of supervisionEvents) {
			const p = event.payload as Record<string, unknown>;
			const agent = (p.supervised_agent as string) ?? "unknown";
			if (!byAgent.has(agent)) byAgent.set(agent, []);
			byAgent.get(agent)?.push(event);
		}
		for (const [agentName, events] of byAgent) {
			patterns.push({
				type: "supervision",
				events,
				spanId: node.span_id,
				label: `Supervision: ${agentName} (${events.length} intervention${events.length > 1 ? "s" : ""})`,
			});
		}
	}

	// Debate: group all debate.* events in same span
	if (debateEvents.some((e) => e.event_type === "multi_agent.debate.start")) {
		const startEvent = debateEvents.find((e) => e.event_type === "multi_agent.debate.start")!;
		const completeEvent = debateEvents.find((e) => e.event_type === "multi_agent.debate.complete");
		const sp = startEvent.payload as Record<string, unknown>;
		const debaterNames = (sp.debater_names as string[]) ?? [];
		const cp = completeEvent?.payload as Record<string, unknown> | undefined;
		const roundsCompleted = (cp?.rounds_completed as number) ?? undefined;
		const roundsLabel = roundsCompleted != null ? `${roundsCompleted} round${roundsCompleted !== 1 ? "s" : ""}` : "";
		patterns.push({
			type: "debate",
			events: debateEvents,
			spanId: node.span_id,
			label: `Debate (${debaterNames.length} debaters${roundsLabel ? `, ${roundsLabel}` : ""})`,
		});
	}

	// Consensus: group all consensus.* events in same span
	if (consensusEvents.some((e) => e.event_type === "multi_agent.consensus.start")) {
		const startEvent = consensusEvents.find((e) => e.event_type === "multi_agent.consensus.start")!;
		const sp = startEvent.payload as Record<string, unknown>;
		const agentNames = (sp.agent_names as string[]) ?? [];
		const strategy = (sp.strategy as string) ?? "unknown";
		patterns.push({
			type: "consensus",
			events: consensusEvents,
			spanId: node.span_id,
			label: `Consensus (${agentNames.length} agents, ${strategy})`,
		});
	}

	// Blackboard: group all blackboard.* events in same span
	if (blackboardEvents.some((e) => e.event_type === "blackboard.start")) {
		const startEvent = blackboardEvents.find((e) => e.event_type === "blackboard.start")!;
		const sp = startEvent.payload as Record<string, unknown>;
		const agentNames = (sp.agent_names as string[]) ?? [];
		const control = (sp.control_strategy as string) ?? "unknown";
		patterns.push({
			type: "blackboard",
			events: blackboardEvents,
			spanId: node.span_id,
			label: `Blackboard (${agentNames.length} agents, ${control})`,
		});
	}

	// Peer Network: group all peer.* events in same span
	if (peerNetworkEvents.some((e) => e.event_type === "multi_agent.peer.start")) {
		const startEvent = peerNetworkEvents.find((e) => e.event_type === "multi_agent.peer.start")!;
		const sp = startEvent.payload as Record<string, unknown>;
		const entry = (sp.entry_agent as string) ?? "unknown";
		const peers = (sp.peer_names as string[]) ?? [];
		patterns.push({
			type: "peer_network",
			events: peerNetworkEvents,
			spanId: node.span_id,
			label: `Peer network from ${entry} (${peers.length} peers)`,
		});
	}

	// Message Bus: group all bus.* events in same span
	if (messageBusEvents.some((e) => e.event_type === "multi_agent.bus.start")) {
		const completeEvent = messageBusEvents.find((e) => e.event_type === "multi_agent.bus.complete");
		const publishedCount = messageBusEvents.filter((e) => e.event_type === "multi_agent.bus.published").length;
		const startEvent = messageBusEvents.find((e) => e.event_type === "multi_agent.bus.start")!;
		const sp = startEvent.payload as Record<string, unknown>;
		const seedTopics = (sp.seed_topics as string[]) ?? [];
		const cp = completeEvent?.payload as Record<string, unknown> | undefined;
		const totalMessages = (cp?.total_messages as number) ?? publishedCount;
		patterns.push({
			type: "message_bus",
			events: messageBusEvents,
			spanId: node.span_id,
			label: `Message bus (${seedTopics.length} topics, ${totalMessages} messages)`,
		});
	}

	// Recurse into children
	for (const child of node.children) {
		walkTree(child, patterns, allHandoffs);
	}
}

/**
 * Group handoff events into chains by matching to_agent → from_agent
 * across sequential events (ordered by timestamp).
 */
function buildHandoffChains(events: TraceEvent[]): TraceEvent[][] {
	// Sort by timestamp
	const sorted = [...events].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

	const chains: TraceEvent[][] = [];
	const used = new Set<number>();

	for (let i = 0; i < sorted.length; i++) {
		if (used.has(i)) continue;

		const chain: TraceEvent[] = [sorted[i]];
		used.add(i);

		// Extend chain by finding the next event where from_agent matches current to_agent
		let current = sorted[i];
		let currentTo = (current.payload as Record<string, unknown>).to_agent as string;

		for (let j = i + 1; j < sorted.length; j++) {
			if (used.has(j)) continue;
			const candidate = sorted[j];
			const candidateFrom = (candidate.payload as Record<string, unknown>).from_agent as string;
			if (candidateFrom === currentTo) {
				chain.push(candidate);
				used.add(j);
				current = candidate;
				currentTo = (current.payload as Record<string, unknown>).to_agent as string;
			}
		}

		chains.push(chain);
	}

	return chains;
}
