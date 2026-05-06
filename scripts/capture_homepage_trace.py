"""Capture the homepage example's trace into a live Observatory UI.

Runs ``examples/homepage.main()`` once through a :class:`TracedExecutor`
backed by an :class:`InMemoryPersistentTraceStore`, then serves the
embedded Observatory UI against that same store so the run can be
inspected in a browser and screenshotted for the website's proof
section (Phase 2.3 of ``temp/website/program.md``).

Prerequisite: the embed bundle at ``observatory/dist-embed/`` must
exist. Run ``just observatory-build`` first if it is missing or stale.

Usage::

    uv run python scripts/capture_homepage_trace.py

Then open http://localhost:8002/api/observatory/ and navigate to the
printed run id. Stop with Ctrl-C. The store is in-memory, so the run
is gone after the process exits — re-run to recapture.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from examples import homepage
from nanitics import InMemoryPersistentTraceStore, TracedExecutor
from nanitics.observatory import create_observatory_router

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "observatory" / "dist-embed"
HOST = "127.0.0.1"
PORT = 8002


async def _populate_store(store: InMemoryPersistentTraceStore) -> str:
    executor = TracedExecutor(store)

    async def _work(emitter, run_id):  # type: ignore[no-untyped-def]
        del run_id  # unused; the executor still records it
        result, _ = await homepage.main(emitter=emitter)
        return result.output

    run_id, _ = await executor.execute(
        _work,
        metadata={"source": "homepage-snapshot", "example": "examples/homepage.py"},
    )
    return run_id


def _build_app(store: InMemoryPersistentTraceStore) -> FastAPI:
    app = FastAPI(title="Nanitics Observatory — homepage trace capture")
    app.include_router(
        create_observatory_router(store, static_dir=UI_DIR),
        prefix="/api/observatory",
    )
    return app


def main() -> None:
    if not UI_DIR.is_dir():
        raise SystemExit(f"Embed bundle not found at {UI_DIR}. Run `just observatory-build` first.")

    store = InMemoryPersistentTraceStore()
    run_id = asyncio.run(_populate_store(store))

    print()
    print(f"  Captured run id: {run_id}")
    print(f"  Observatory UI:  http://{HOST}:{PORT}/api/observatory/")
    print(f"  Direct run URL:  http://{HOST}:{PORT}/api/observatory/#/runs/{run_id}")
    print("  Stop with Ctrl-C.")
    print()

    uvicorn.run(_build_app(store), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
