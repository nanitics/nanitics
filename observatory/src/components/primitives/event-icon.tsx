import {
	AlertTriangle,
	Bot,
	Brain,
	CheckCircle2,
	Circle,
	ClipboardList,
	HelpCircle,
	MessageSquare,
	Paperclip,
	RotateCw,
	User,
	Workflow,
	Wrench,
	XCircle,
} from "lucide-react";
import type { ComponentType, SVGProps } from "react";

type LucideIcon = ComponentType<SVGProps<SVGSVGElement>>;

interface EventIconProps {
	eventType: string;
	className?: string;
}

interface OutcomeIconProps {
	kind: "corrected" | "degraded" | "retried" | "unresolved" | "success" | "warning";
	className?: string;
}

interface RecoveryIconProps {
	kind: "retry" | "correction" | "degradation" | "error" | "unknown";
	className?: string;
}

/** Map an event-type prefix to its canonical lucide glyph. */
function eventIconFor(eventType: string): LucideIcon {
	if (eventType.startsWith("agent.")) return Bot;
	if (eventType.startsWith("llm.")) return MessageSquare;
	if (eventType.startsWith("tool.")) return Wrench;
	if (eventType.startsWith("memory.")) return Brain;
	if (eventType.startsWith("planning.")) return ClipboardList;
	if (eventType.startsWith("error.") || eventType.startsWith("correction.")) return AlertTriangle;
	if (eventType.startsWith("span.")) return Paperclip;
	if (eventType.startsWith("workflow.")) return Workflow;
	if (eventType.startsWith("hitl.")) return User;
	return Circle;
}

/** Map an outcome kind to its canonical lucide glyph. */
function outcomeIconFor(kind: OutcomeIconProps["kind"]): LucideIcon {
	switch (kind) {
		case "corrected":
		case "success":
			return CheckCircle2;
		case "degraded":
		case "warning":
			return AlertTriangle;
		case "retried":
			return RotateCw;
		case "unresolved":
			return HelpCircle;
	}
}

/** Map a recovery kind to its canonical lucide glyph. */
function recoveryIconFor(kind: RecoveryIconProps["kind"]): LucideIcon {
	switch (kind) {
		case "retry":
			return RotateCw;
		case "correction":
			return Wrench;
		case "degradation":
			return AlertTriangle;
		case "error":
			return XCircle;
		case "unknown":
			return HelpCircle;
	}
}

/**
 * Renders a lucide glyph for a trace event based on its event-type prefix.
 * Decorative: `aria-hidden="true"`. Semantics carried by neighboring text.
 */
export function EventIcon({ eventType, className = "h-4 w-4" }: EventIconProps) {
	const Icon = eventIconFor(eventType);
	return <Icon aria-hidden="true" className={className} data-testid="event-icon" />;
}

/**
 * Renders a lucide glyph for an outcome marker (success / warning / retried / unresolved).
 * Decorative: `aria-hidden="true"`.
 */
export function OutcomeIcon({ kind, className = "h-5 w-5" }: OutcomeIconProps) {
	const Icon = outcomeIconFor(kind);
	return <Icon aria-hidden="true" className={className} data-testid="outcome-icon" />;
}

/**
 * Renders a lucide glyph for a recovery-row marker (retry / correction / degradation / error / unknown).
 * Decorative: `aria-hidden="true"`.
 */
export function RecoveryIcon({ kind, className = "h-4 w-4" }: RecoveryIconProps) {
	const Icon = recoveryIconFor(kind);
	return <Icon aria-hidden="true" className={className} data-testid="recovery-icon" />;
}
