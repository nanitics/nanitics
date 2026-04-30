import { useCallback, useState } from "react";

export interface CodeBlockProps {
	code: string;
	language?: string;
	highlighter?: (code: string, language: string) => React.ReactNode;
	showLineNumbers?: boolean;
	maxHeight?: number;
}

/** Monospace code display with line numbers and copy button. */
export function CodeBlock({
	code,
	language = "python",
	highlighter,
	showLineNumbers = true,
	maxHeight,
}: CodeBlockProps) {
	const [copied, setCopied] = useState(false);

	const handleCopy = useCallback(() => {
		navigator.clipboard.writeText(code).then(() => {
			setCopied(true);
			setTimeout(() => setCopied(false), 2000);
		});
	}, [code]);

	const lines = code.split("\n");

	return (
		<div className="relative group rounded-md overflow-hidden border border-zinc-700 bg-zinc-900">
			{/* Copy button */}
			<button
				type="button"
				onClick={handleCopy}
				className="absolute top-2 right-2 text-xs px-2 py-1 rounded bg-zinc-700 text-zinc-300 hover:bg-zinc-600 opacity-0 group-hover:opacity-100 transition-opacity z-10"
				aria-label="Copy code"
			>
				{copied ? "Copied!" : "Copy"}
			</button>

			{/* Code content */}
			<div className="overflow-auto" style={maxHeight ? { maxHeight: `${maxHeight}px` } : undefined}>
				{highlighter ? (
					<div className="p-3 text-sm font-mono">{highlighter(code, language)}</div>
				) : (
					<table className="w-full text-sm font-mono">
						<tbody>
							{lines.map((line, i) => (
								// biome-ignore lint/suspicious/noArrayIndexKey: code lines keyed by line number
								<tr key={i} className="leading-relaxed">
									{showLineNumbers && (
										<td className="select-none text-right pr-4 pl-3 text-zinc-500 w-[1%] whitespace-nowrap align-top">
											{i + 1}
										</td>
									)}
									<td className="pr-4 text-zinc-100 whitespace-pre">{line || " "}</td>
								</tr>
							))}
						</tbody>
					</table>
				)}
			</div>
		</div>
	);
}
