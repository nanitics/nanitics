"""Local-dev Nanitics app serving an embedded Observatory.

Not a production example. See docs/guides/observatory-integration.md for
the adopter-facing guide — this file is the minimum glue the local-dev
compose uses to produce visible traces.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from nanitics.infrastructure import (
    LLMClient,
    LLMResponse,
    MockLLMClient,
)
from nanitics.observatory import create_observatory_router
from nanitics.strategies import (
    ReActAgent,
    tool,
)
from nanitics.tracing import (
    InMemoryPersistentTraceStore,
    ToolCall,
    TracedExecutor,
    Usage,
)

UI_DIR = Path("/srv/observatory-ui")


@tool("greet", "Greet someone by name.")
async def greet(name: str) -> str:
    """Return a friendly greeting for ``name``."""
    return f"Hello, {name}!"


def _scripted_mock_client() -> MockLLMClient:
    """Two-turn ReAct script: call ``greet(name='world')``, then finish."""
    usage = Usage(input_tokens=0, output_tokens=0)
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call-1", name="greet", arguments={"name": "world"})],
                usage=usage,
                model="mock",
                stop_reason="tool_use",
            ),
            LLMResponse(
                content="Hello, world!",
                usage=usage,
                model="mock",
                stop_reason="end_turn",
            ),
        ]
    )


def _make_llm_client() -> LLMClient:
    """Return a real Anthropic client when an API key is set, else a scripted mock."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        from nanitics.infrastructure import AnthropicLLMClient

        return AnthropicLLMClient(api_key=key, model="claude-haiku-4-5-20251001")
    return _scripted_mock_client()


store = InMemoryPersistentTraceStore()
executor = TracedExecutor(store)
app = FastAPI(title="Nanitics Observatory — local dev")
app.include_router(
    create_observatory_router(store, static_dir=UI_DIR),
    prefix="/api/observatory",
)


class RunRequest(BaseModel):
    task: str = "Say hello to the world."


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(body: RunRequest) -> dict[str, str]:
    """Execute a demo ReAct agent and return the run id and final output."""

    async def _work(emitter, run_id):  # type: ignore[no-untyped-def]
        del run_id  # unused in this factory
        agent = ReActAgent(
            name="demo",
            llm_client=_make_llm_client(),
            emitter=emitter,
            tools=[greet],
            system_prompt="You are a helpful assistant.",
        )
        return (await agent.run(body.task)).output

    run_id, result = await executor.execute(_work, metadata={"source": "observatory-dev-compose"})
    return {"run_id": run_id, "result": str(result)}
