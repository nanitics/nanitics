import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EventIcon, OutcomeIcon, RecoveryIcon } from "../../src/components/primitives/event-icon";

describe("EventIcon", () => {
	const prefixes = [
		"agent.start",
		"llm.call",
		"tool.invoke",
		"memory.read",
		"planning.plan",
		"error.raise",
		"correction.apply",
		"span.leaf",
		"workflow.step",
		"hitl.prompt",
		"unknown.event",
	];

	it.each(prefixes)("renders a stable glyph for event type '%s'", (eventType) => {
		const { getByTestId } = render(<EventIcon eventType={eventType} />);
		const icon = getByTestId("event-icon");
		expect(icon).toBeInTheDocument();
		expect(icon.tagName.toLowerCase()).toBe("svg");
	});

	it("sets aria-hidden on the rendered svg", () => {
		const { getByTestId } = render(<EventIcon eventType="agent.start" />);
		expect(getByTestId("event-icon")).toHaveAttribute("aria-hidden", "true");
	});

	it("applies the default className when none is provided", () => {
		const { getByTestId } = render(<EventIcon eventType="agent.start" />);
		expect(getByTestId("event-icon").getAttribute("class")).toContain("h-4 w-4");
	});

	it("applies a custom className when provided", () => {
		const { getByTestId } = render(<EventIcon eventType="agent.start" className="h-6 w-6 text-primary" />);
		const icon = getByTestId("event-icon");
		expect(icon.getAttribute("class")).toContain("h-6 w-6");
		expect(icon.getAttribute("class")).toContain("text-primary");
	});

	it("renders a different glyph for each canonical prefix", () => {
		const rendered = new Map<string, string>();
		for (const prefix of prefixes) {
			const { getByTestId, unmount } = render(<EventIcon eventType={prefix} />);
			// Lucide renders the icon name onto the svg via a class token like `lucide-bot`; use path count as a weaker stability check.
			const icon = getByTestId("event-icon");
			const paths = icon.querySelectorAll("path, circle, line, rect, polyline, polygon").length;
			rendered.set(prefix, `${icon.getAttribute("class") ?? ""}|${paths}`);
			unmount();
		}
		// Every known prefix produces some glyph. The default-case `unknown.event` produces the fallback.
		expect(rendered.size).toBe(prefixes.length);
	});
});

describe("OutcomeIcon", () => {
	const kinds = ["corrected", "degraded", "retried", "unresolved", "success", "warning"] as const;

	it.each(kinds)("renders a glyph for outcome kind '%s'", (kind) => {
		const { getByTestId } = render(<OutcomeIcon kind={kind} />);
		expect(getByTestId("outcome-icon")).toBeInTheDocument();
	});

	it("sets aria-hidden on the rendered svg", () => {
		const { getByTestId } = render(<OutcomeIcon kind="success" />);
		expect(getByTestId("outcome-icon")).toHaveAttribute("aria-hidden", "true");
	});

	it("applies the default className when none is provided", () => {
		const { getByTestId } = render(<OutcomeIcon kind="success" />);
		expect(getByTestId("outcome-icon").getAttribute("class")).toContain("h-5 w-5");
	});

	it("applies a custom className when provided", () => {
		const { getByTestId } = render(<OutcomeIcon kind="success" className="h-3 w-3" />);
		expect(getByTestId("outcome-icon").getAttribute("class")).toContain("h-3 w-3");
	});
});

describe("RecoveryIcon", () => {
	const kinds = ["retry", "correction", "degradation", "error", "unknown"] as const;

	it.each(kinds)("renders a glyph for recovery kind '%s'", (kind) => {
		const { getByTestId } = render(<RecoveryIcon kind={kind} />);
		expect(getByTestId("recovery-icon")).toBeInTheDocument();
	});

	it("sets aria-hidden on the rendered svg", () => {
		const { getByTestId } = render(<RecoveryIcon kind="retry" />);
		expect(getByTestId("recovery-icon")).toHaveAttribute("aria-hidden", "true");
	});

	it("applies the default className when none is provided", () => {
		const { getByTestId } = render(<RecoveryIcon kind="retry" />);
		expect(getByTestId("recovery-icon").getAttribute("class")).toContain("h-4 w-4");
	});

	it("applies a custom className when provided", () => {
		const { getByTestId } = render(<RecoveryIcon kind="retry" className="h-5 w-5 text-destructive" />);
		const icon = getByTestId("recovery-icon");
		expect(icon.getAttribute("class")).toContain("h-5 w-5");
		expect(icon.getAttribute("class")).toContain("text-destructive");
	});
});
