interface TreeControlsProps {
	onExpandAll: () => void;
	onCollapseAll: () => void;
}

export function TreeControls({ onExpandAll, onCollapseAll }: TreeControlsProps) {
	return (
		<div className="flex items-center gap-2 px-3 py-2 border-b">
			<button
				type="button"
				onClick={onExpandAll}
				className="text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded hover:bg-accent"
			>
				Expand all
			</button>
			<button
				type="button"
				onClick={onCollapseAll}
				className="text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded hover:bg-accent"
			>
				Collapse all
			</button>
		</div>
	);
}
