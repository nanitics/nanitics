export type StatusVariant = "error" | "success" | "warning" | "info" | "accent";

const variants: Record<StatusVariant, string> = {
	error: "bg-destructive-muted border-destructive-border text-destructive-muted-foreground",
	success: "bg-success-muted border-success-border text-success-muted-foreground",
	warning: "bg-warning-muted border-warning-border text-warning-muted-foreground",
	info: "bg-info-muted border-info-border text-info-muted-foreground",
	accent: "bg-accent-status-muted border-accent-status-border text-accent-status-muted-foreground",
};

/** Returns composed className for a status color variant (background + border + text). */
export function statusVariant(variant: StatusVariant): string {
	return variants[variant];
}
