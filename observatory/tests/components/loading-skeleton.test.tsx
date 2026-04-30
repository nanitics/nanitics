import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import {
	AgentDetailSkeleton,
	RunDetailSkeleton,
	RunListSkeleton,
	TreeSkeleton,
} from "../../src/components/feedback/loading-skeleton";

const ALL_FULL_PAGE_VARIANTS: Array<[string, React.ComponentType<{ className?: string }>]> = [
	["RunListSkeleton", RunListSkeleton],
	["RunDetailSkeleton", RunDetailSkeleton],
	["AgentDetailSkeleton", AgentDetailSkeleton],
];

describe("Loading skeletons — shared a11y contract", () => {
	it.each(ALL_FULL_PAGE_VARIANTS)("<%s> outer container has role='status' and aria-busy='true'", (_name, Cmp) => {
		const { container } = render(<Cmp />);
		const status = container.querySelector("[role='status']");
		expect(status).not.toBeNull();
		expect(status).toHaveAttribute("aria-busy", "true");
	});

	it.each(ALL_FULL_PAGE_VARIANTS)("<%s> renders the visually-hidden 'Loading…' string", (_name, Cmp) => {
		const { container } = render(<Cmp />);
		const sr = container.querySelector(".sr-only");
		expect(sr).not.toBeNull();
		expect(sr?.textContent).toMatch(/Loading/);
	});

	it.each(ALL_FULL_PAGE_VARIANTS)("<%s> uses animate-pulse on at least one placeholder", (_name, Cmp) => {
		const { container } = render(<Cmp />);
		expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);
	});

	it.each(ALL_FULL_PAGE_VARIANTS)("<%s> appends a caller-provided className to the outer container", (_name, Cmp) => {
		const { container } = render(<Cmp className="extra-layout-class" />);
		const status = container.querySelector("[role='status']");
		expect(status?.className).toContain("extra-layout-class");
	});

	it.each(ALL_FULL_PAGE_VARIANTS)("<%s> passes axe with no violations", async (_name, Cmp) => {
		const { container } = render(<Cmp />);
		expect(await axe(container)).toHaveNoViolations();
	});
});

describe("RunListSkeleton", () => {
	it("renders exactly 5 card shells by default (count=5)", () => {
		const { container } = render(<RunListSkeleton />);
		expect(container.querySelectorAll(".rounded-lg").length).toBe(5);
	});

	it("renders exactly N card shells when count={N} is provided", () => {
		for (const n of [1, 3, 8]) {
			const { container, unmount } = render(<RunListSkeleton count={n} />);
			expect(container.querySelectorAll(".rounded-lg").length).toBe(n);
			unmount();
		}
	});

	it("renders zero card shells when count={0}", () => {
		const { container } = render(<RunListSkeleton count={0} />);
		expect(container.querySelectorAll(".rounded-lg").length).toBe(0);
	});
});

describe("RunDetailSkeleton", () => {
	it("renders eight tree-row shells", () => {
		const { container } = render(<RunDetailSkeleton />);
		// h-6 my-1 rows are only used by the tree-row shells in this variant.
		const treeRows = container.querySelectorAll(".h-6.bg-muted.rounded.my-1");
		expect(treeRows.length).toBe(8);
	});

	it("renders two filter-toolbar placeholders", () => {
		const { container } = render(<RunDetailSkeleton />);
		const toolbarShells = container.querySelectorAll(".h-8.w-32.bg-muted");
		expect(toolbarShells.length).toBe(2);
	});
});

describe("AgentDetailSkeleton", () => {
	it("renders six timeline-row shells", () => {
		const { container } = render(<AgentDetailSkeleton />);
		const timelineRows = container.querySelectorAll(".h-6.bg-muted.rounded.my-1");
		expect(timelineRows.length).toBe(6);
	});

	it("renders six stats-row placeholders", () => {
		const { container } = render(<AgentDetailSkeleton />);
		const statPlaceholders = container.querySelectorAll(".h-3.w-20.bg-muted");
		expect(statPlaceholders.length).toBe(6);
	});
});

describe("TreeSkeleton (module-local helper)", () => {
	it("has role='status' and aria-busy='true'", () => {
		const { container } = render(<TreeSkeleton />);
		const status = container.querySelector("[role='status']");
		expect(status).not.toBeNull();
		expect(status).toHaveAttribute("aria-busy", "true");
	});

	it("renders eight tree-row shells", () => {
		const { container } = render(<TreeSkeleton />);
		expect(container.querySelectorAll(".h-6.bg-muted.rounded.my-1").length).toBe(8);
	});

	it("passes axe with no violations", async () => {
		const { container } = render(<TreeSkeleton />);
		expect(await axe(container)).toHaveNoViolations();
	});

	it("appends a caller-provided className to the outer container", () => {
		const { container } = render(<TreeSkeleton className="tree-only" />);
		const status = container.querySelector("[role='status']");
		expect(status?.className).toContain("tree-only");
	});
});
