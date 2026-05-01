import { useCallback, useEffect, useRef, useState } from "react";

interface ViewBox {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface ContentBounds {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

interface UseSVGViewportOptions {
	/** Padding around content for fit-to-content (default: 60) */
	padding?: number;
	/** Minimum zoom level as fraction of initial size (default: 0.1) */
	minZoom?: number;
	/** Maximum zoom level as fraction of initial size (default: 10) */
	maxZoom?: number;
}

interface UseSVGViewportResult {
	/** Ref to attach to the SVG element */
	svgRef: React.RefObject<SVGSVGElement | null>;
	/** Current viewBox string for the SVG element */
	viewBox: string;
	/** Whether the user is currently dragging/panning */
	isDragging: boolean;
	/** Zoom in by one step */
	zoomIn: () => void;
	/** Zoom out by one step */
	zoomOut: () => void;
	/** Fit all content in view */
	fitToContent: () => void;
	/** Event handlers to attach to the SVG element */
	svgHandlers: {
		onMouseDown: React.MouseEventHandler;
		onMouseMove: React.MouseEventHandler;
		onMouseUp: React.MouseEventHandler;
		onMouseLeave: React.MouseEventHandler;
		onWheel: React.WheelEventHandler;
	};
}

function viewBoxString(vb: ViewBox): string {
	return `${vb.x} ${vb.y} ${vb.width} ${vb.height}`;
}

export function useSVGViewport(
	contentBounds: ContentBounds | null,
	options?: UseSVGViewportOptions,
): UseSVGViewportResult {
	const { padding = 60 } = options ?? {};

	const svgRef = useRef<SVGSVGElement | null>(null);
	const [viewBox, setViewBox] = useState<ViewBox>({ x: 0, y: 0, width: 800, height: 600 });
	const [dragging, setDragging] = useState(false);
	const dragStart = useRef<{ x: number; y: number; vb: ViewBox } | null>(null);

	const fitToContent = useCallback(() => {
		if (!contentBounds) return;
		setViewBox({
			x: contentBounds.minX - padding,
			y: contentBounds.minY - padding,
			width: contentBounds.maxX - contentBounds.minX + padding * 2,
			height: contentBounds.maxY - contentBounds.minY + padding * 2,
		});
	}, [contentBounds, padding]);

	// Auto-fit when content bounds change
	useEffect(() => {
		fitToContent();
	}, [fitToContent]);

	const handleWheel = useCallback((e: React.WheelEvent) => {
		e.preventDefault();
		const factor = e.deltaY > 0 ? 1.1 : 0.9;
		setViewBox((prev) => {
			const cx = prev.x + prev.width / 2;
			const cy = prev.y + prev.height / 2;
			const newW = prev.width * factor;
			const newH = prev.height * factor;
			return { x: cx - newW / 2, y: cy - newH / 2, width: newW, height: newH };
		});
	}, []);

	const handleMouseDown = useCallback(
		(e: React.MouseEvent) => {
			if (e.button !== 0) return;
			if ((e.target as Element).closest("[data-tree-node], [data-dag-node]")) return;
			setDragging(true);
			dragStart.current = { x: e.clientX, y: e.clientY, vb: { ...viewBox } };
		},
		[viewBox],
	);

	const handleMouseMove = useCallback(
		(e: React.MouseEvent) => {
			if (!dragging || !dragStart.current || !svgRef.current) return;
			const rect = svgRef.current.getBoundingClientRect();
			const scaleX = dragStart.current.vb.width / rect.width;
			const scaleY = dragStart.current.vb.height / rect.height;
			const dx = (e.clientX - dragStart.current.x) * scaleX;
			const dy = (e.clientY - dragStart.current.y) * scaleY;
			setViewBox({
				...dragStart.current.vb,
				x: dragStart.current.vb.x - dx,
				y: dragStart.current.vb.y - dy,
			});
		},
		[dragging],
	);

	const handleMouseUp = useCallback(() => {
		setDragging(false);
		dragStart.current = null;
	}, []);

	const zoomIn = useCallback(() => {
		setViewBox((vb) => {
			const cx = vb.x + vb.width / 2;
			const cy = vb.y + vb.height / 2;
			const w = vb.width * 0.8;
			const h = vb.height * 0.8;
			return { x: cx - w / 2, y: cy - h / 2, width: w, height: h };
		});
	}, []);

	const zoomOut = useCallback(() => {
		setViewBox((vb) => {
			const cx = vb.x + vb.width / 2;
			const cy = vb.y + vb.height / 2;
			const w = vb.width * 1.25;
			const h = vb.height * 1.25;
			return { x: cx - w / 2, y: cy - h / 2, width: w, height: h };
		});
	}, []);

	return {
		svgRef,
		viewBox: viewBoxString(viewBox),
		isDragging: dragging,
		zoomIn,
		zoomOut,
		fitToContent,
		svgHandlers: {
			onMouseDown: handleMouseDown,
			onMouseMove: handleMouseMove,
			onMouseUp: handleMouseUp,
			onMouseLeave: handleMouseUp,
			onWheel: handleWheel,
		},
	};
}
