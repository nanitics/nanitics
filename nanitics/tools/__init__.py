"""Reference tools — curated :class:`~nanitics.strategies.Tool` implementations.

Four factories ship with the SDK so a developer can build a useful
multi-agent system without writing tool plumbing:

- :func:`create_web_search_tool` — search the web via Tavily or Brave.
- :func:`create_http_tool` — call allow-listed HTTP endpoints.
- :func:`create_file_read_tool` — read files under an allow-listed set of
  paths.
- :func:`create_code_execution_tool` — execute code in a sandbox satisfying
  the :class:`~nanitics.safety.Sandbox` protocol.

``web_search`` and ``http_request`` require the ``http-tools`` extra
(``pip install nanitics[http-tools]`` or the umbrella ``nanitics[tools]``).
When ``httpx`` is not installed the factory names resolve to stubs that
raise :class:`ImportError` with an install hint on first call, so a plain
``import nanitics.tools`` still succeeds regardless of extras.
"""

from __future__ import annotations

from typing import Any, NoReturn

from nanitics.tools._result_models import (
    CodeExecutionResult,
    FileReadResult,
    HttpResponse,
    WebSearchResult,
    WebSearchResultItem,
)
from nanitics.tools.code_execution import create_code_execution_tool
from nanitics.tools.file_read import create_file_read_tool

try:
    from nanitics.tools.http import create_http_tool
    from nanitics.tools.web_search import create_web_search_tool
except ImportError:
    # httpx not installed; provide stubs that fail loudly on call rather than
    # silently resolve to None (which would surface as a cryptic TypeError).
    def _missing_http_extra(_factory: str) -> NoReturn:
        raise ImportError(f"{_factory} requires the 'http-tools' extra: pip install nanitics[http-tools]")

    def create_http_tool(*_args: Any, **_kwargs: Any) -> NoReturn:  # type: ignore[misc]
        _missing_http_extra("create_http_tool")

    def create_web_search_tool(*_args: Any, **_kwargs: Any) -> NoReturn:  # type: ignore[misc]
        _missing_http_extra("create_web_search_tool")


__all__ = [
    "CodeExecutionResult",
    "FileReadResult",
    "HttpResponse",
    "WebSearchResult",
    "WebSearchResultItem",
    "create_code_execution_tool",
    "create_file_read_tool",
    "create_http_tool",
    "create_web_search_tool",
]
