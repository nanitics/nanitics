import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CodeBlock } from "../../src/components/primitives/code-block";

describe("CodeBlock", () => {
	beforeEach(() => {
		vi.restoreAllMocks();
	});

	it("renders code content", () => {
		render(<CodeBlock code="print('hello')" />);
		expect(screen.getByText("print('hello')")).toBeInTheDocument();
	});

	it("shows line numbers by default", () => {
		render(<CodeBlock code={"line 1\nline 2\nline 3"} />);
		expect(screen.getByText("1")).toBeInTheDocument();
		expect(screen.getByText("2")).toBeInTheDocument();
		expect(screen.getByText("3")).toBeInTheDocument();
	});

	it("hides line numbers when showLineNumbers is false", () => {
		render(<CodeBlock code={"line 1\nline 2"} showLineNumbers={false} />);
		expect(screen.getByText("line 1")).toBeInTheDocument();
		// No line number cells should be present
		expect(screen.queryByText("1")).not.toBeInTheDocument();
	});

	it("renders copy button", () => {
		render(<CodeBlock code="x = 1" />);
		expect(screen.getByRole("button", { name: "Copy code" })).toBeInTheDocument();
	});

	it("copies code to clipboard on copy button click", async () => {
		const writeText = vi.fn().mockResolvedValue(undefined);
		Object.assign(navigator, { clipboard: { writeText } });

		render(<CodeBlock code="x = 42" />);
		fireEvent.click(screen.getByRole("button", { name: "Copy code" }));

		expect(writeText).toHaveBeenCalledWith("x = 42");
	});

	it("uses custom highlighter when provided", () => {
		const highlighter = vi.fn((code: string, _lang: string) => <span data-testid="highlighted">{code}</span>);

		render(<CodeBlock code="x = 1" highlighter={highlighter} />);

		expect(highlighter).toHaveBeenCalledWith("x = 1", "python");
		expect(screen.getByTestId("highlighted")).toBeInTheDocument();
	});

	it("applies maxHeight style when provided", () => {
		const { container } = render(<CodeBlock code="x = 1" maxHeight={200} />);
		const scrollContainer = container.querySelector(".overflow-auto");
		expect(scrollContainer).toHaveStyle({ maxHeight: "200px" });
	});

	it("handles empty code", () => {
		render(<CodeBlock code="" />);
		// Should render without error — single empty line
		expect(screen.getByText("1")).toBeInTheDocument();
	});

	it("handles multiline code correctly", () => {
		const code = "import pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.head())";
		render(<CodeBlock code={code} />);
		expect(screen.getByText("import pandas as pd")).toBeInTheDocument();
		expect(screen.getByText("print(df.head())")).toBeInTheDocument();
		expect(screen.getByText("4")).toBeInTheDocument();
	});
});
