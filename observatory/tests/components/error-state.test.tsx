import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ErrorState } from "../../src/components/feedback/error-state";

describe("ErrorState — matrix of {title} × {onRetry} × {variant}", () => {
	const titles: Array<string | undefined> = [undefined, "Custom title"];
	const retries: Array<"set" | "unset"> = ["set", "unset"];
	const variants = ["page", "inline"] as const;

	for (const title of titles) {
		for (const retry of retries) {
			for (const variant of variants) {
				const label = `title=${title ?? "undefined"} onRetry=${retry} variant=${variant}`;
				it(`mounts without error: ${label}`, () => {
					const onRetry = retry === "set" ? vi.fn() : undefined;
					const { container } = render(
						<ErrorState title={title} error={new Error("boom")} onRetry={onRetry} variant={variant} />,
					);
					expect(container).toBeTruthy();
				});
			}
		}
	}
});

describe("ErrorState — structural contract", () => {
	it("outer container has role='alert' in variant='page'", () => {
		render(<ErrorState error={new Error("boom")} />);
		expect(screen.getByRole("alert")).toBeInTheDocument();
	});

	it("outer container has role='alert' in variant='inline'", () => {
		render(<ErrorState error={new Error("boom")} variant="inline" />);
		expect(screen.getByRole("alert")).toBeInTheDocument();
	});

	it("uses the default title 'Something went wrong' when no title prop is passed", () => {
		render(<ErrorState error={new Error("boom")} />);
		expect(screen.getByText("Something went wrong")).toBeInTheDocument();
	});

	it("uses the caller-provided title when passed", () => {
		render(<ErrorState title="My custom title" error={new Error("boom")} />);
		expect(screen.getByText("My custom title")).toBeInTheDocument();
	});

	it("applies the caller-provided className on variant='page'", () => {
		const { container } = render(<ErrorState error={new Error("boom")} className="extra-page-class" />);
		const alert = container.querySelector("[role='alert']");
		expect(alert?.className).toContain("extra-page-class");
	});

	it("applies the caller-provided className on variant='inline'", () => {
		const { container } = render(
			<ErrorState error={new Error("boom")} variant="inline" className="extra-inline-class" />,
		);
		const alert = container.querySelector("[role='alert']");
		expect(alert?.className).toContain("extra-inline-class");
	});

	it("marks the AlertCircle glyph as aria-hidden in variant='page'", () => {
		const { container } = render(<ErrorState error={new Error("boom")} />);
		const svg = container.querySelector("svg");
		expect(svg).not.toBeNull();
		expect(svg).toHaveAttribute("aria-hidden", "true");
	});
});

describe("ErrorState — axe accessibility", () => {
	it("passes axe on variant='page' with no retry", async () => {
		const { container } = render(<ErrorState error={new Error("boom")} />);
		expect(await axe(container)).toHaveNoViolations();
	});

	it("passes axe on variant='page' with a retry button", async () => {
		const { container } = render(<ErrorState error={new Error("boom")} onRetry={vi.fn()} />);
		expect(await axe(container)).toHaveNoViolations();
	});

	it("passes axe on variant='inline'", async () => {
		const { container } = render(<ErrorState error={new Error("boom")} variant="inline" />);
		expect(await axe(container)).toHaveNoViolations();
	});
});

describe("ErrorState — <details> pane reveals stringified error", () => {
	it("keeps the <details> pane collapsed by default", () => {
		const { container } = render(<ErrorState error={new Error("boom")} />);
		const details = container.querySelector("details");
		expect(details).not.toBeNull();
		expect(details?.hasAttribute("open")).toBe(false);
	});

	it("reveals the stringified error when the summary is clicked", () => {
		const err = new Error("specific failure");
		const { container } = render(<ErrorState error={err} />);
		const summary = container.querySelector("summary");
		expect(summary).not.toBeNull();
		// Before click: <details> closed
		const details = container.querySelector("details");
		expect(details?.hasAttribute("open")).toBe(false);
		// After click: <details> open and the error message is rendered
		fireEvent.click(summary as Element);
		expect(screen.getByText(/specific failure/)).toBeInTheDocument();
	});

	it("renders a stringified error in variant='inline' as well", () => {
		const { container } = render(<ErrorState error="inline-only-error" variant="inline" />);
		const summary = container.querySelector("summary");
		fireEvent.click(summary as Element);
		expect(screen.getByText("inline-only-error")).toBeInTheDocument();
	});
});

describe("ErrorState — retry button semantics", () => {
	it("renders the retry button on variant='page' when onRetry is set", () => {
		render(<ErrorState error={new Error("boom")} onRetry={vi.fn()} />);
		expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
	});

	it("does not render a retry button when onRetry is undefined", () => {
		render(<ErrorState error={new Error("boom")} />);
		expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
	});

	it("invokes onRetry exactly once when the retry button is clicked", () => {
		const onRetry = vi.fn();
		render(<ErrorState error={new Error("boom")} onRetry={onRetry} />);
		fireEvent.click(screen.getByRole("button", { name: /retry/i }));
		expect(onRetry).toHaveBeenCalledTimes(1);
	});

	it("does NOT render a retry button on variant='inline' even when onRetry is passed", () => {
		render(<ErrorState error={new Error("boom")} onRetry={vi.fn()} variant="inline" />);
		expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
	});
});

describe("ErrorState — stringifyError behavior through the component", () => {
	function openDetails(container: HTMLElement) {
		const summary = container.querySelector("summary");
		fireEvent.click(summary as Element);
	}

	it("renders `Error.message` for an Error instance", () => {
		const { container } = render(<ErrorState error={new Error("specific-message")} />);
		openDetails(container);
		const pre = container.querySelector("pre");
		expect(pre?.textContent).toContain("specific-message");
	});

	it("renders a string error verbatim", () => {
		const { container } = render(<ErrorState error="raw-string-error" />);
		openDetails(container);
		const pre = container.querySelector("pre");
		expect(pre?.textContent).toBe("raw-string-error");
	});

	it("renders a plain object as JSON", () => {
		const { container } = render(<ErrorState error={{ code: 500, msg: "server" }} />);
		openDetails(container);
		const pre = container.querySelector("pre");
		expect(pre?.textContent).toContain('"code": 500');
		expect(pre?.textContent).toContain('"msg": "server"');
	});

	it("falls back to String(error) on a circular/un-serializable object without throwing", () => {
		// Build a circular reference — JSON.stringify will throw, triggering the fallback branch.
		const circular: Record<string, unknown> = { self: null };
		circular.self = circular;

		expect(() => render(<ErrorState error={circular} />)).not.toThrow();
		const { container } = render(<ErrorState error={circular} />);
		openDetails(container);
		const pre = container.querySelector("pre");
		// `String({})` yields "[object Object]"; confirm the fallback activated.
		expect(pre?.textContent).toBe("[object Object]");
	});
});
