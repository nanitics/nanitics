import { ArrowDown, ArrowUp, Info } from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";
import type { CapabilityPanelProps } from "../../registry/capability-panel-registry";
import type { TraceEvent } from "../../types";
import { TokenUsage } from "../primitives/token-usage";

const LLM_EVENT_TYPES = new Set(["llm.request", "llm.response"]);
const CONTEXT_EVENT_TYPES = new Set(["context.assembly", "context.truncation", "context.summarization"]);

interface LLMCall {
	index: number;
	request: TraceEvent;
	response: TraceEvent | null;
	contextEvents: TraceEvent[];
}

function pairLLMCalls(events: TraceEvent[]): LLMCall[] {
	const relevant = events.filter((e) => LLM_EVENT_TYPES.has(e.event_type) || CONTEXT_EVENT_TYPES.has(e.event_type));

	const calls: LLMCall[] = [];
	let pendingContextEvents: TraceEvent[] = [];
	let callIndex = 0;

	for (const event of relevant) {
		if (CONTEXT_EVENT_TYPES.has(event.event_type)) {
			pendingContextEvents.push(event);
		} else if (event.event_type === "llm.request") {
			callIndex++;
			calls.push({
				index: callIndex,
				request: event,
				response: null,
				contextEvents: pendingContextEvents,
			});
			pendingContextEvents = [];
		} else if (event.event_type === "llm.response" && calls.length > 0) {
			// Pair with the most recent unpaired request
			const last = calls[calls.length - 1];
			if (!last.response) {
				last.response = event;
			}
		}
	}

	return calls;
}

/** LLM Calls panel — chronological LLM calls with messages, tokens, and context events. */
export function LLMCallsPanel({ events }: CapabilityPanelProps) {
	const calls = pairLLMCalls(events);

	if (calls.length === 0) {
		return <div className="p-4 text-sm text-muted-foreground">No LLM calls recorded for this agent.</div>;
	}

	return (
		<div className="p-4 space-y-3">
			<div className="text-xs text-muted-foreground">
				{calls.length} LLM call{calls.length !== 1 ? "s" : ""}
			</div>
			{calls.map((call) => (
				<LLMCallCard key={call.index} call={call} total={calls.length} />
			))}
		</div>
	);
}

function LLMCallCard({ call, total }: { call: LLMCall; total: number }) {
	const [isExpanded, setIsExpanded] = useState(false);

	const reqPayload = call.request.payload as Record<string, unknown>;
	const resPayload = (call.response?.payload ?? {}) as Record<string, unknown>;
	const usage = (resPayload.usage ?? {}) as {
		input_tokens?: number;
		output_tokens?: number;
	};

	const modelName = (reqPayload.model_name ?? "unknown") as string;
	const inputTokens = usage.input_tokens ?? (reqPayload.input_tokens as number | undefined) ?? 0;
	const outputTokens = usage.output_tokens ?? 0;
	const stopReason = resPayload.stop_reason as string | undefined;
	const messagesCount = reqPayload.messages_count as number | undefined;
	const durationMs = computeDuration(call.request, call.response);
	const messages = Array.isArray(reqPayload.messages) ? (reqPayload.messages as Array<Record<string, unknown>>) : null;
	const responseContent = resPayload.content != null ? String(resPayload.content) : null;
	const toolCalls = resPayload.tool_calls != null ? JSON.stringify(resPayload.tool_calls) : null;

	return (
		<div>
			{/* Context events before this LLM call */}
			{call.contextEvents.map((ce) => (
				<ContextBanner key={ce.id} event={ce} />
			))}

			<div className="border rounded-lg overflow-hidden">
				{/* Header */}
				<button
					type="button"
					className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-accent/50 transition-colors text-left"
					onClick={() => setIsExpanded(!isExpanded)}
				>
					<span className="text-xs text-muted-foreground w-4">{isExpanded ? "▾" : "▸"}</span>
					<span className="font-mono text-xs px-1.5 py-0.5 rounded bg-muted">
						Call {call.index} of {total}
					</span>
					<span className="text-xs text-muted-foreground truncate">{modelName}</span>
					{messagesCount != null && (
						<span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5 py-0.5">
							{messagesCount} msg{messagesCount !== 1 ? "s" : ""}
						</span>
					)}
					<span className="ml-auto flex items-center gap-3">
						<TokenUsage inputTokens={inputTokens} outputTokens={outputTokens} />
						{durationMs != null && (
							<span className="text-[10px] text-muted-foreground tabular-nums">{formatDuration(durationMs)}</span>
						)}
					</span>
				</button>

				{/* Expanded detail */}
				{isExpanded && (
					<div className="border-t px-3 py-2 space-y-3">
						{/* Request details */}
						<div className="space-y-1">
							<SectionLabel icon={<ArrowUp aria-hidden="true" className="h-3.5 w-3.5" />} label="Request" />
							<div className="text-xs space-y-1">
								<DetailRow label="Model" value={modelName} />
								{messagesCount != null && <DetailRow label="Messages" value={String(messagesCount)} />}
								{messages && <MessageList messages={messages} />}
							</div>
						</div>

						{/* Response details */}
						{call.response && (
							<div className="space-y-1">
								<SectionLabel icon={<ArrowDown aria-hidden="true" className="h-3.5 w-3.5" />} label="Response" />
								<div className="text-xs space-y-1">
									{stopReason && <DetailRow label="Stop reason" value={stopReason} />}
									{responseContent && <CollapsibleText label="Content" text={responseContent} />}
									{toolCalls && <DetailRow label="Tool calls" value={toolCalls} />}
								</div>
							</div>
						)}

						{!call.response && <div className="text-xs text-muted-foreground italic">No response recorded</div>}
					</div>
				)}
			</div>
		</div>
	);
}

function ContextBanner({ event }: { event: TraceEvent }) {
	const payload = event.payload as Record<string, unknown>;
	let text: string;

	switch (event.event_type) {
		case "context.assembly": {
			const contributions = payload.contributions as unknown[] | undefined;
			const tokens = payload.total_injected as number | undefined;
			text = `Context assembled from ${contributions?.length ?? "?"} providers`;
			if (tokens != null) text += ` (${tokens.toLocaleString()} tokens injected)`;
			break;
		}
		case "context.truncation": {
			const beforeTokens = payload.tokens_before as number | undefined;
			const afterTokens = payload.tokens_after as number | undefined;
			const beforeMessages = payload.messages_before as number | undefined;
			const afterMessages = payload.messages_after as number | undefined;
			text = "Context truncated";
			if (beforeMessages != null && afterMessages != null) {
				text += `: ${beforeMessages}→${afterMessages} messages`;
			}
			if (beforeTokens != null && afterTokens != null) {
				text += `, ${beforeTokens.toLocaleString()}→${afterTokens.toLocaleString()} tokens`;
			}
			break;
		}
		case "context.summarization": {
			const count = payload.messages_summarized as number | undefined;
			const beforeTk = payload.original_tokens as number | undefined;
			const afterTk = payload.summary_tokens as number | undefined;
			text = `${count ?? "?"} messages summarized`;
			if (beforeTk != null && afterTk != null) {
				text += ` (${beforeTk.toLocaleString()}→${afterTk.toLocaleString()} tokens)`;
			}
			break;
		}
		default:
			text = event.event_type;
	}

	return (
		<div className="flex items-center gap-2 px-3 py-1.5 mb-1 text-xs rounded-md bg-info-muted text-info-muted-foreground border border-info-border">
			<Info aria-hidden="true" className="h-3.5 w-3.5" />
			<span>{text}</span>
		</div>
	);
}

function MessageList({ messages }: { messages: Array<Record<string, unknown>> }) {
	const [isExpanded, setIsExpanded] = useState(false);

	return (
		<div>
			<button
				type="button"
				onClick={() => setIsExpanded(!isExpanded)}
				className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
			>
				<span className="w-3">{isExpanded ? "▾" : "▸"}</span>
				{messages.length} message{messages.length !== 1 ? "s" : ""}
			</button>
			{isExpanded && (
				<div className="ml-4 mt-1 space-y-1.5 border-l pl-2">
					{messages.map((msg, i) => {
						const role = String(msg.role ?? "unknown");
						const content = String(msg.content ?? "");
						return (
							// biome-ignore lint/suspicious/noArrayIndexKey: LLM messages have no unique ID
							<div key={i} className="text-xs">
								<span className="font-medium text-muted-foreground capitalize">{role}:</span>{" "}
								<span className="text-foreground">{content.length > 200 ? `${content.slice(0, 200)}…` : content}</span>
							</div>
						);
					})}
				</div>
			)}
		</div>
	);
}

function CollapsibleText({ label, text }: { label: string; text: string }) {
	const [isExpanded, setIsExpanded] = useState(false);
	const isLong = text.length > 200;

	return (
		<div>
			<button
				type="button"
				className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
				onClick={() => setIsExpanded(!isExpanded)}
			>
				<span className="w-3">{isExpanded ? "▾" : "▸"}</span>
				{label}
			</button>
			{isExpanded && (
				<div className="ml-4 mt-1 text-xs whitespace-pre-wrap bg-muted/50 rounded p-2 max-h-[300px] overflow-y-auto">
					{text}
				</div>
			)}
			{!isExpanded && isLong && <div className="ml-4 text-xs text-muted-foreground">{text.slice(0, 100)}…</div>}
		</div>
	);
}

function SectionLabel({ icon, label }: { icon: ReactNode; label: string }) {
	return (
		<div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
			<span className="inline-flex items-center">{icon}</span>
			<span>{label}</span>
		</div>
	);
}

function DetailRow({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex items-center gap-2">
			<span className="text-muted-foreground">{label}:</span>
			<span className="font-mono">{value}</span>
		</div>
	);
}

function computeDuration(request: TraceEvent, response: TraceEvent | null): number | null {
	if (!response) return null;
	const start = new Date(request.timestamp).getTime();
	const end = new Date(response.timestamp).getTime();
	const diff = end - start;
	return diff >= 0 ? diff : null;
}

function formatDuration(ms: number): string {
	return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}
