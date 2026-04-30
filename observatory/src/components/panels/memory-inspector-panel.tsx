import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";

// ---------------------------------------------------------------------------
// Sub-tab definitions
// ---------------------------------------------------------------------------

interface SubTabDef {
	id: string;
	label: string;
	prefix: string;
	render: (events: TraceEvent[]) => React.ReactNode;
}

const SUB_TABS: SubTabDef[] = [
	{
		id: "working",
		label: "Working Memory",
		prefix: "memory.working.",
		render: (events) => <WorkingMemoryTab events={events} />,
	},
	{
		id: "semantic",
		label: "Semantic",
		prefix: "memory.semantic.",
		render: (events) => <SemanticTab events={events} />,
	},
	{
		id: "episodic",
		label: "Episodic",
		prefix: "memory.episode.",
		render: (events) => <EpisodicTab events={events} />,
	},
	{
		id: "longterm",
		label: "Long-Term",
		prefix: "memory.longterm.",
		render: (events) => <LongTermTab events={events} />,
	},
	{
		id: "shared",
		label: "Shared",
		prefix: "memory.shared.",
		render: (events) => <SharedTab events={events} />,
	},
];

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

/** Memory Inspector panel — sub-tabbed view of all memory operations. */
export function MemoryInspectorPanel({ events }: CapabilityPanelProps) {
	const memoryEvents = events.filter((e) => e.event_type.startsWith("memory."));

	if (memoryEvents.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No memory events recorded for this agent.</div>;
	}

	// Determine which sub-tabs have data
	const availableTabs = SUB_TABS.filter((tab) => memoryEvents.some((e) => e.event_type.startsWith(tab.prefix)));

	// Always add Timeline tab
	const allTabs = [
		...availableTabs,
		{
			id: "timeline",
			label: "Timeline",
			prefix: "memory.",
			render: (evts: TraceEvent[]) => <TimelineTab events={evts} />,
		},
	];

	return <SubTabContainer tabs={allTabs} events={memoryEvents} />;
}

// ---------------------------------------------------------------------------
// Sub-tab container
// ---------------------------------------------------------------------------

function SubTabContainer({ tabs, events }: { tabs: SubTabDef[]; events: TraceEvent[] }) {
	const [activeTab, setActiveTab] = useState(tabs[0]?.id ?? "timeline");
	const active = tabs.find((t) => t.id === activeTab) ?? tabs[0];

	const filtered = active.id === "timeline" ? events : events.filter((e) => e.event_type.startsWith(active.prefix));

	return (
		<div className="p-4 space-y-3">
			<div className="text-xs text-muted-foreground">
				{events.length} memory event{events.length !== 1 ? "s" : ""}
			</div>

			{/* Sub-tab bar */}
			<div className="flex gap-1 border-b">
				{tabs.map((tab) => (
					<button
						type="button"
						key={tab.id}
						className={`px-3 py-1.5 text-xs transition-colors ${
							activeTab === tab.id
								? "border-b-2 border-foreground font-medium"
								: "text-muted-foreground hover:text-foreground"
						}`}
						onClick={() => setActiveTab(tab.id)}
					>
						{tab.label}
					</button>
				))}
			</div>

			{/* Tab content */}
			{active.render(filtered)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Working Memory sub-tab
// ---------------------------------------------------------------------------

function WorkingMemoryTab({ events }: { events: TraceEvent[] }) {
	const updates = events.filter((e) => e.event_type === "memory.working.update");
	const reads = events.filter((e) => e.event_type === "memory.working.read");
	const latestRead = reads[reads.length - 1];
	const tokenCount = latestRead
		? ((latestRead.payload as Record<string, unknown>).token_count as number | undefined)
		: undefined;

	if (updates.length === 0 && reads.length === 0) {
		return <EmptySubTab label="working memory" />;
	}

	return (
		<div className="space-y-2 mt-2">
			{tokenCount != null && (
				<div className="text-xs text-muted-foreground">Current size: {tokenCount.toLocaleString()} tokens</div>
			)}
			{updates.map((event) => (
				<WorkingMemoryUpdateCard key={event.id} event={event} />
			))}
			{updates.length === 0 && reads.length > 0 && (
				<div className="text-xs text-muted-foreground">
					{reads.length} read{reads.length !== 1 ? "s" : ""}, no updates recorded.
				</div>
			)}
		</div>
	);
}

function WorkingMemoryUpdateCard({ event }: { event: TraceEvent }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const payload = event.payload as Record<string, unknown>;
	const source = (payload.source ?? "unknown") as string;
	const previousContent = payload.previous_content as string | undefined;
	const newContent = payload.new_content as string | undefined;

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<TypeBadge type="update" />
				<span className="text-xs truncate">Source: {source}</span>
				<span className="ml-auto text-[10px] text-muted-foreground">{formatTime(event.timestamp)}</span>
			</button>
			{isExpanded && (
				<div className="border-t px-3 py-2 space-y-2 text-xs">
					{previousContent != null && (
						<div>
							<span className="text-muted-foreground">Previous:</span>
							<pre className="mt-1 bg-destructive-muted rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{truncate(previousContent, 500)}
							</pre>
						</div>
					)}
					{newContent != null && (
						<div>
							<span className="text-muted-foreground">Updated:</span>
							<pre className="mt-1 bg-success-muted rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{truncate(newContent, 500)}
							</pre>
						</div>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Semantic sub-tab
// ---------------------------------------------------------------------------

function SemanticTab({ events }: { events: TraceEvent[] }) {
	if (events.length === 0) return <EmptySubTab label="semantic memory" />;

	return (
		<div className="space-y-2 mt-2">
			{events.map((event) => (
				<SemanticEventCard key={event.id} event={event} />
			))}
		</div>
	);
}

function SemanticEventCard({ event }: { event: TraceEvent }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const payload = event.payload as Record<string, unknown>;

	let summary: string;
	switch (event.event_type) {
		case "memory.semantic.search": {
			const count = payload.results_count as number | undefined;
			const topScore = payload.top_score as number | undefined;
			summary = `Search: ${count ?? 0} results`;
			if (topScore != null) summary += ` (top: ${topScore.toFixed(2)})`;
			break;
		}
		case "memory.semantic.store":
			summary = `Stored entry${payload.entry_id ? ` (${payload.entry_id})` : ""}`;
			break;
		case "memory.semantic.delete":
			summary = `Deleted entry${payload.entry_id ? ` (${payload.entry_id})` : ""}`;
			break;
		default:
			summary = event.event_type;
	}

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<TypeBadge type={event.event_type.split(".").pop() ?? "event"} />
				<span className="text-xs truncate">{summary}</span>
				<span className="ml-auto text-[10px] text-muted-foreground">{formatTime(event.timestamp)}</span>
			</button>
			{isExpanded && (
				<div className="border-t px-3 py-2 text-xs space-y-1">
					{payload.query != null && <DetailRow label="Query" value={String(payload.query)} />}
					{payload.namespace != null && <DetailRow label="Namespace" value={String(payload.namespace)} />}
					{payload.entry_id != null && <DetailRow label="Entry ID" value={String(payload.entry_id)} />}
					{payload.content != null && (
						<div>
							<span className="text-muted-foreground">Content:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{truncate(String(payload.content), 500)}
							</pre>
						</div>
					)}
					{payload.results_count != null && <DetailRow label="Results" value={String(payload.results_count)} />}
					{payload.top_score != null && (
						<DetailRow label="Top score" value={(payload.top_score as number).toFixed(3)} />
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Episodic sub-tab
// ---------------------------------------------------------------------------

function EpisodicTab({ events }: { events: TraceEvent[] }) {
	if (events.length === 0) return <EmptySubTab label="episodic memory" />;

	return (
		<div className="space-y-2 mt-2">
			{events.map((event) => (
				<EpisodicEventCard key={event.id} event={event} />
			))}
		</div>
	);
}

function EpisodicEventCard({ event }: { event: TraceEvent }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const payload = event.payload as Record<string, unknown>;

	let summary: string;
	switch (event.event_type) {
		case "memory.episode.record":
			summary = `Episode recorded${payload.episode_id ? ` (${payload.episode_id})` : ""}`;
			break;
		case "memory.episode.recall": {
			const count = payload.results_count as number | undefined;
			const topScore = payload.top_score as number | undefined;
			summary = `Recall: ${count ?? 0} results`;
			if (topScore != null) summary += ` (top: ${topScore.toFixed(2)})`;
			break;
		}
		case "memory.episode.forget":
			summary = `Episode forgotten${payload.episode_id ? ` (${payload.episode_id})` : ""}`;
			break;
		default:
			summary = event.event_type;
	}

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<TypeBadge type={event.event_type.split(".").pop() ?? "event"} />
				<span className="text-xs truncate">{summary}</span>
				<span className="ml-auto text-[10px] text-muted-foreground">{formatTime(event.timestamp)}</span>
			</button>
			{isExpanded && (
				<div className="border-t px-3 py-2 text-xs space-y-1">
					{payload.episode_id != null && <DetailRow label="Episode ID" value={String(payload.episode_id)} />}
					{payload.query != null && <DetailRow label="Query" value={String(payload.query)} />}
					{payload.situation != null && (
						<div>
							<span className="text-muted-foreground">Situation:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[100px] overflow-y-auto">
								{truncate(String(payload.situation), 300)}
							</pre>
						</div>
					)}
					{payload.outcome != null && (
						<div>
							<span className="text-muted-foreground">Outcome:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[100px] overflow-y-auto">
								{truncate(String(payload.outcome), 300)}
							</pre>
						</div>
					)}
					{payload.has_reflection != null && (
						<DetailRow label="Reflection" value={payload.has_reflection ? "Yes" : "No"} />
					)}
					{payload.results_count != null && <DetailRow label="Results" value={String(payload.results_count)} />}
					{payload.top_score != null && (
						<DetailRow label="Top score" value={(payload.top_score as number).toFixed(3)} />
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Long-Term sub-tab
// ---------------------------------------------------------------------------

function LongTermTab({ events }: { events: TraceEvent[] }) {
	if (events.length === 0) return <EmptySubTab label="long-term memory" />;

	return (
		<div className="space-y-2 mt-2">
			{events.map((event) => (
				<LongTermEventCard key={event.id} event={event} />
			))}
		</div>
	);
}

function LongTermEventCard({ event }: { event: TraceEvent }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const payload = event.payload as Record<string, unknown>;
	const key = payload.key as string | undefined;
	const op = event.event_type.split(".").pop() ?? "event";

	let summary: string;
	switch (event.event_type) {
		case "memory.longterm.store":
			summary = `Stored: ${key ?? "?"}`;
			break;
		case "memory.longterm.retrieve":
			summary = `Retrieved: ${key ?? "?"}${payload.found != null ? (payload.found ? " (found)" : " (not found)") : ""}`;
			break;
		case "memory.longterm.delete":
			summary = `Deleted: ${key ?? "?"}`;
			break;
		case "memory.longterm.list": {
			const keys = payload.keys as string[] | undefined;
			summary = `Listed keys (${keys?.length ?? "?"})`;
			break;
		}
		default:
			summary = event.event_type;
	}

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<TypeBadge type={op} />
				<span className="text-xs truncate">{summary}</span>
				<span className="ml-auto text-[10px] text-muted-foreground">{formatTime(event.timestamp)}</span>
			</button>
			{isExpanded && (
				<div className="border-t px-3 py-2 text-xs space-y-1">
					{key != null && <DetailRow label="Key" value={key} />}
					{payload.namespace != null && <DetailRow label="Namespace" value={String(payload.namespace)} />}
					{payload.value != null && (
						<div>
							<span className="text-muted-foreground">Value:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{truncate(String(payload.value), 500)}
							</pre>
						</div>
					)}
					{payload.found != null && <DetailRow label="Found" value={payload.found ? "Yes" : "No"} />}
					{event.event_type === "memory.longterm.list" && Array.isArray(payload.keys) && (
						<div>
							<span className="text-muted-foreground">Keys:</span>
							<div className="mt-1 flex flex-wrap gap-1">
								{(payload.keys as string[]).map((k) => (
									<span key={k} className="font-mono text-[10px] px-1.5 py-0.5 bg-muted rounded">
										{k}
									</span>
								))}
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Shared sub-tab
// ---------------------------------------------------------------------------

function SharedTab({ events }: { events: TraceEvent[] }) {
	if (events.length === 0) return <EmptySubTab label="shared memory" />;

	return (
		<div className="space-y-2 mt-2">
			{events.map((event) => (
				<SharedEventCard key={event.id} event={event} />
			))}
		</div>
	);
}

function SharedEventCard({ event }: { event: TraceEvent }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const payload = event.payload as Record<string, unknown>;
	const author = payload.author as string | undefined;
	const op = event.event_type.split(".").pop() ?? "event";

	let summary: string;
	switch (event.event_type) {
		case "memory.shared.write":
			summary = `Write by ${author ?? "unknown"}`;
			break;
		case "memory.shared.read": {
			const count = payload.entries_returned as number | undefined;
			summary = `Read (${count ?? 0} entries)`;
			break;
		}
		case "memory.shared.supersede":
			summary = `Supersede by ${author ?? "unknown"}`;
			break;
		case "memory.shared.retract":
			summary = `Retract by ${author ?? "unknown"}`;
			break;
		default:
			summary = event.event_type;
	}

	return (
		<div className="border rounded-lg overflow-hidden">
			<button
				type="button"
				className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
				<TypeBadge type={op} />
				<span className="text-xs truncate">{summary}</span>
				<span className="ml-auto text-[10px] text-muted-foreground">{formatTime(event.timestamp)}</span>
			</button>
			{isExpanded && (
				<div className="border-t px-3 py-2 text-xs space-y-1">
					{author != null && <DetailRow label="Author" value={author} />}
					{payload.scope != null && <DetailRow label="Scope" value={String(payload.scope)} />}
					{payload.entry_id != null && <DetailRow label="Entry ID" value={String(payload.entry_id)} />}
					{payload.content != null && (
						<div>
							<span className="text-muted-foreground">Content:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{truncate(String(payload.content), 500)}
							</pre>
						</div>
					)}
					{payload.original_entry_id != null && (
						<DetailRow label="Original entry" value={String(payload.original_entry_id)} />
					)}
					{payload.new_content != null && (
						<div>
							<span className="text-muted-foreground">New content:</span>
							<pre className="mt-1 bg-muted/50 rounded p-2 whitespace-pre-wrap max-h-[150px] overflow-y-auto">
								{truncate(String(payload.new_content), 500)}
							</pre>
						</div>
					)}
					{payload.reason != null && <DetailRow label="Reason" value={String(payload.reason)} />}
					{payload.entries_returned != null && (
						<DetailRow label="Entries returned" value={String(payload.entries_returned)} />
					)}
				</div>
			)}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Timeline sub-tab
// ---------------------------------------------------------------------------

function TimelineTab({ events }: { events: TraceEvent[] }) {
	if (events.length === 0) return <EmptySubTab label="memory" />;

	return (
		<div className="space-y-1.5 mt-2">
			{events.map((event) => {
				const typeSegments = event.event_type.split(".");
				const category = typeSegments[1] ?? "memory";
				const operation = typeSegments[2] ?? "event";
				const payload = event.payload as Record<string, unknown>;

				let info = "";
				if (payload.source) info = `source: ${payload.source}`;
				else if (payload.key) info = `key: ${payload.key}`;
				else if (payload.query) info = `query: ${truncate(String(payload.query), 60)}`;
				else if (payload.author) info = `author: ${payload.author}`;
				else if (payload.episode_id) info = `episode: ${payload.episode_id}`;

				return (
					<div key={event.id} className="flex items-center gap-2 px-2 py-1 text-xs rounded hover:bg-accent/30">
						<span className="text-[10px] text-muted-foreground tabular-nums w-16 shrink-0">
							{formatTime(event.timestamp)}
						</span>
						<TypeBadge type={category} />
						<span className="font-medium">{operation}</span>
						{info && <span className="text-muted-foreground truncate">{info}</span>}
					</div>
				);
			})}
		</div>
	);
}

// ---------------------------------------------------------------------------
// Shared UI primitives
// ---------------------------------------------------------------------------

function EmptySubTab({ label }: { label: string }) {
	return <div className="py-6 text-center text-xs text-muted-foreground">No {label} events recorded.</div>;
}

const TYPE_COLORS: Record<string, string> = {
	update: "bg-info-muted text-info-muted-foreground",
	read: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
	store: "bg-success-muted text-success-muted-foreground",
	search: "bg-accent-status-muted text-accent-status-muted-foreground",
	delete: "bg-destructive-muted text-destructive-muted-foreground",
	record: "bg-success-muted text-success-muted-foreground",
	recall: "bg-accent-status-muted text-accent-status-muted-foreground",
	forget: "bg-destructive-muted text-destructive-muted-foreground",
	retrieve: "bg-accent-status-muted text-accent-status-muted-foreground",
	list: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
	write: "bg-success-muted text-success-muted-foreground",
	supersede: "bg-warning-muted text-warning-muted-foreground",
	retract: "bg-destructive-muted text-destructive-muted-foreground",
	// Category-level colors for timeline
	working: "bg-info-muted text-info-muted-foreground",
	semantic: "bg-accent-status-muted text-accent-status-muted-foreground",
	episode: "bg-teal-100 text-teal-700 dark:bg-teal-950 dark:text-teal-300",
	longterm: "bg-warning-muted text-warning-muted-foreground",
	shared: "bg-info-muted text-info-muted-foreground",
};

function TypeBadge({ type }: { type: string }) {
	const colors = TYPE_COLORS[type] ?? "bg-muted text-muted-foreground";
	return <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${colors}`}>{type}</span>;
}

function DetailRow({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-start gap-2">
			<span className="text-muted-foreground shrink-0">{label}:</span>
			<span className="font-mono break-all">{value}</span>
		</div>
	);
}

function truncate(text: string, max: number): string {
	return text.length > max ? `${text.slice(0, max)}…` : text;
}

function formatTime(timestamp: string): string {
	try {
		const d = new Date(timestamp);
		return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
	} catch {
		return timestamp;
	}
}
