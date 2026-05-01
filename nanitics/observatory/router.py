"""FastAPI observatory router factory.

Creates an :class:`~fastapi.APIRouter` with all observatory endpoints
backed by a :class:`PersistentTraceStore`.  Applications mount the router
at whatever prefix makes sense for their domain::

    from nanitics.observatory import create_observatory_router

    router = create_observatory_router(store)
    app.include_router(router, prefix="/api/observatory")

When ``static_dir`` points to built observatory assets, visiting the
router's root (``/api/observatory/``) serves the observatory UI.
Build with ``just observatory-build``.
"""

from __future__ import annotations

# FastAPI route handlers are registered dynamically via decorators, so the
# closure-scoped handlers below intentionally omit return annotations. The
# blanket suppression below lets mypy remain strict for the rest of the file.
# mypy: disable-error-code="no-untyped-def"
import os.path
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_args

from starlette.requests import Request

from nanitics.infrastructure.observability.levels import TraceLevel
from nanitics.observatory.models import (
    RunCreateRequest,
    RunStatusUpdateRequest,
)

if TYPE_CHECKING:
    from nanitics.infrastructure.observability.storage import PersistentTraceStore

_VALID_TRACE_LEVELS: frozenset[TraceLevel] = frozenset(get_args(TraceLevel))


def create_observatory_router(
    store: PersistentTraceStore,
    *,
    static_dir: Path | None = None,
):
    """Create a FastAPI :class:`APIRouter` with all observatory endpoints.

    Args:
        store: A :class:`PersistentTraceStore` implementation.
        static_dir: Path to a directory containing built observatory UI
            assets (``index.html`` + ``assets/``).  When *None*, the
            embedded UI endpoints return a helpful fallback message.

    Returns:
        An :class:`APIRouter` with run, trace hierarchy, agent, workflow,
        event, and streaming endpoints.
    """
    from fastapi import APIRouter, HTTPException, Query
    from fastapi.responses import Response
    from starlette.responses import StreamingResponse

    from nanitics.infrastructure.observability.levels import (
        LEVEL_ORDER,
        is_level_included,
    )
    from nanitics.infrastructure.observability.storage import (
        DEFAULT_EVENTS_LIMIT,
        DEFAULT_RUNS_LIMIT,
        MAX_EVENTS_LIMIT,
        MAX_RUNS_LIMIT,
        RunStatus,
    )
    from nanitics.observatory.service import ObservatoryService, SortOption
    from nanitics.observatory.streaming import stream_run_events

    service = ObservatoryService(store)
    router = APIRouter()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _get_run_or_404(run_id: str):
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    def _expand_levels(min_level: TraceLevel) -> list[TraceLevel]:
        """Expand a minimum level into an inclusive list of levels."""
        return [lvl for lvl in LEVEL_ORDER if is_level_included(lvl, min_level)]

    def _validate_trace_level(value: str) -> TraceLevel:
        """Narrow a query-string level to a valid ``TraceLevel`` or 422."""
        if value not in _VALID_TRACE_LEVELS:
            raise HTTPException(status_code=422, detail=f"Invalid trace level: {value}")
        return value

    # ------------------------------------------------------------------
    # Run endpoints
    # ------------------------------------------------------------------

    @router.get("/runs")
    async def list_runs(
        status: str | None = Query(None, pattern="^(running|completed|failed|suspended|rejected)$"),
        started_after: str | None = Query(None),
        started_before: str | None = Query(None),
        sort: str = Query("started_at_desc", pattern="^(started_at_desc|started_at_asc|duration_desc|duration_asc)$"),
        search: str | None = Query(None),
        limit: int = Query(DEFAULT_RUNS_LIMIT, ge=1, le=MAX_RUNS_LIMIT),
        offset: int = Query(0, ge=0),
    ):
        parsed_after = datetime.fromisoformat(started_after).astimezone(UTC) if started_after else None
        parsed_before = datetime.fromisoformat(started_before).astimezone(UTC) if started_before else None
        return await service.list_runs(
            status=cast(RunStatus | None, status),
            started_after=parsed_after,
            started_before=parsed_before,
            sort=cast(SortOption, sort),
            search=search,
            limit=limit,
            offset=offset,
        )

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str):
        result = await service.get_run(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return result

    @router.post("/runs", status_code=201)
    async def create_run(body: RunCreateRequest):
        return await service.register_run(
            run_id=body.run_id,
            trace_id=body.trace_id,
            metadata=body.metadata,
        )

    @router.patch("/runs/{run_id}/status", status_code=204)
    async def update_run_status(run_id: str, body: RunStatusUpdateRequest):
        await _get_run_or_404(run_id)
        await service.update_run_status(run_id, body.status, error=body.error)
        return Response(status_code=204)

    @router.delete("/runs/{run_id}", status_code=204)
    async def delete_run(run_id: str):
        deleted = await service.delete_run(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Run not found")
        return Response(status_code=204)

    # ------------------------------------------------------------------
    # Trace hierarchy endpoints
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/tree")
    async def get_span_tree(
        run_id: str,
        min_level: str = Query("info", pattern="^(info|debug|verbose)$"),
    ):
        run = await _get_run_or_404(run_id)
        level = _validate_trace_level(min_level)
        return await service.get_span_tree(run.trace_id, min_level=level)

    @router.get("/runs/{run_id}/spans/{span_id}/events")
    async def get_span_events(
        run_id: str,
        span_id: str,
        levels: str | None = Query(None),
        event_types: str | None = Query(None),
    ):
        run = await _get_run_or_404(run_id)
        levels_list: list[TraceLevel] | None = None
        if levels:
            levels_list = [_validate_trace_level(lv.strip()) for lv in levels.split(",") if lv.strip()]
        types_list: list[str] | None = None
        if event_types:
            types_list = [t.strip() for t in event_types.split(",") if t.strip()]
        return await service.get_events_for_span(run.trace_id, span_id, levels=levels_list, event_types=types_list)

    # ------------------------------------------------------------------
    # Agent endpoints
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/agents")
    async def list_agents(run_id: str):
        run = await _get_run_or_404(run_id)
        return await service.list_agents(run.trace_id)

    @router.get("/runs/{run_id}/agents/{span_id}")
    async def get_agent_detail(run_id: str, span_id: str):
        run = await _get_run_or_404(run_id)
        result = await service.get_agent_detail(run.trace_id, span_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return result

    # ------------------------------------------------------------------
    # Workflow endpoint
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/workflow")
    async def get_workflow(run_id: str):
        run = await _get_run_or_404(run_id)
        result = await service.get_workflow_structure(run.trace_id)
        if result is None:
            raise HTTPException(status_code=404, detail="No workflow structure found")
        return result

    # ------------------------------------------------------------------
    # Flat event endpoints
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/events")
    async def list_events(
        run_id: str,
        level: str | None = Query(None, pattern="^(info|debug|verbose)$"),
        event_types: str | None = Query(None),
        limit: int = Query(DEFAULT_EVENTS_LIMIT, ge=1, le=MAX_EVENTS_LIMIT),
        after: int | None = Query(None, ge=1),
    ):
        await _get_run_or_404(run_id)
        levels_list: list[TraceLevel] | None = None
        if level:
            levels_list = _expand_levels(_validate_trace_level(level))
        types_list: list[str] | None = None
        if event_types:
            types_list = [t.strip() for t in event_types.split(",") if t.strip()]
        return await service.query_events(
            run_id,
            levels=levels_list,
            event_types=types_list,
            after_id=after,
            limit=limit,
        )

    @router.get("/events/{event_id}")
    async def get_event(event_id: int):
        result = await service.get_event(event_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Event not found")
        return result

    # ------------------------------------------------------------------
    # Summary endpoint
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/summary")
    async def get_summary(run_id: str):
        await _get_run_or_404(run_id)
        return await service.get_run_summary(run_id)

    # ------------------------------------------------------------------
    # SSE streaming endpoint
    # ------------------------------------------------------------------

    @router.get("/runs/{run_id}/stream")
    async def stream_events(
        run_id: str,
        request: Request,
        min_level: str = Query("info", pattern="^(info|debug|verbose)$"),
    ):
        await _get_run_or_404(run_id)
        last_event_id: int | None = None
        raw = request.headers.get("last-event-id")
        if raw is not None:
            last_event_id = int(raw)
        level = _validate_trace_level(min_level)

        generator = stream_run_events(
            store,
            run_id,
            request,
            min_level=level,
            last_event_id=last_event_id,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Embedded UI (static files)
    # ------------------------------------------------------------------

    _MIME_TYPES = {
        ".html": "text/html",
        ".js": "application/javascript",
        ".css": "text/css",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".ico": "image/x-icon",
        ".json": "application/json",
    }
    _CACHE_MAX_AGE = 31_536_000  # 1 year in seconds

    _ui_dir = Path(static_dir) if static_dir is not None else None

    @router.get("/")
    async def observatory_ui():
        if _ui_dir is None or not (_ui_dir / "index.html").is_file():
            from fastapi.responses import HTMLResponse

            return HTMLResponse(
                "<h1>Observatory UI not built</h1>"
                "<p>Run <code>just observatory-build</code> to build the embedded UI, "
                "then restart the server.</p>",
                status_code=200,
            )
        return Response(
            content=(_ui_dir / "index.html").read_bytes(),
            media_type="text/html",
        )

    @router.get("/assets/{path:path}")
    async def observatory_assets(path: str):
        if _ui_dir is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        assets_dir = _ui_dir / "assets"
        resolved = (assets_dir / path).resolve()
        if not resolved.is_file() or not str(resolved).startswith(str(assets_dir.resolve()) + os.sep):
            raise HTTPException(status_code=404, detail="Asset not found")
        suffix = resolved.suffix.lower()
        media_type = _MIME_TYPES.get(suffix, "application/octet-stream")
        headers = {}
        if suffix in (".js", ".css"):
            headers["Cache-Control"] = f"public, max-age={_CACHE_MAX_AGE}, immutable"
        return Response(
            content=resolved.read_bytes(),
            media_type=media_type,
            headers=headers,
        )

    return router
