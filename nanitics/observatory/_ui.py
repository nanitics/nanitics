"""Static-asset helpers for the embedded Observatory UI.

The wheel ships a prebuilt single-page application under
``nanitics/observatory/ui_assets/`` (populated by ``just observatory-build``
before ``uv build`` runs). :func:`default_ui_dir` returns that path via
:mod:`importlib.resources` so the UI works both from an installed wheel and
from an editable install.

The SPA needs to know its own base URL at runtime because consumers can
mount the router at any prefix (``/observatory``, ``/admin/runs``, …).
:func:`render_index_html` rewrites the ``id="nanitics-observatory-base"``
script tag in ``index.html`` to set ``window.__NANITICS_OBSERVATORY_BASE__``
to the live mount prefix; the React client reads it on boot. The base URL
is JSON-encoded so a hostile prefix cannot break out of the script literal.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

_BUNDLED_DIR_NAME = "ui_assets"

# The SPA template ships a script tag with this id that primes the base
# URL for `npm run dev` (Vite proxy). When the Python router serves the
# bundle it rewrites the script body to the live mount prefix at request
# time, so the same artifact serves every prefix.
_BASE_URL_SCRIPT_RE = re.compile(
    r'<script id="nanitics-observatory-base">.*?</script>',
    re.DOTALL,
)


def default_ui_dir() -> Path | None:
    """Return the wheel-bundled UI directory if it exists, else ``None``.

    The directory is populated by ``just observatory-build`` and shipped
    inside the wheel. Returns ``None`` for SDK contributors who have not
    built the UI yet — the router falls back to a "UI not built" page in
    that case.
    """
    try:
        anchor = resources.files("nanitics.observatory") / _BUNDLED_DIR_NAME
    except (ModuleNotFoundError, TypeError):  # pragma: no cover - defensive
        return None
    # ``files()`` returns a ``Traversable``; concrete on-disk packages give
    # back a ``MultiplexedPath`` or ``PosixPath``. Coerce to ``Path`` so
    # callers can ``read_bytes`` / iterate / resolve consistently.
    candidate = Path(str(anchor))
    if not candidate.is_dir() or not (candidate / "index.html").is_file():
        return None
    return candidate


def render_index_html(html: bytes, base_url: str) -> bytes:
    """Rewrite the SPA's base-URL bootstrap with the live mount prefix.

    Args:
        html: Raw ``index.html`` bytes as built by Vite.
        base_url: The path prefix the router is mounted at (no trailing
            slash), e.g. ``"/observatory"`` or ``"/admin/runs"``.

    Returns:
        ``index.html`` with the ``id="nanitics-observatory-base"`` script
        tag rewritten so ``window.__NANITICS_OBSERVATORY_BASE__`` carries
        the live mount prefix. The base URL is JSON-encoded so the value
        cannot break out of the script literal.

    Raises:
        ValueError: if the bootstrap script tag is missing — the SPA
            template always carries it, so a missing tag signals a
            malformed bundle.
    """
    text = html.decode("utf-8")
    replacement = (
        f'<script id="nanitics-observatory-base">window.__NANITICS_OBSERVATORY_BASE__={json.dumps(base_url)};</script>'
    )
    new_text, count = _BASE_URL_SCRIPT_RE.subn(replacement, text, count=1)
    if count == 0:
        raise ValueError(
            'Observatory index.html is missing the <script id="nanitics-observatory-base"> '
            "bootstrap tag; bundle is malformed."
        )
    return new_text.encode("utf-8")


def compute_base_url(root_path: str, mount_path: str) -> str:
    """Compute the SPA base URL from the ASGI request scope.

    FastAPI exposes the application's mount prefix as
    ``request.scope["root_path"]`` and the matched route path as
    ``request.url.path``. The UI root handler matches ``"/"``, so we
    combine ``root_path`` with the request path minus the trailing slash
    to recover the full prefix the SPA was reached at.

    Args:
        root_path: The ASGI ``root_path`` (empty in tests, ``/api`` behind
            a reverse proxy that mounts the app at ``/api``, etc.).
        mount_path: The URL path of the request to the UI root, e.g.
            ``"/observatory/"`` or ``"/admin/runs/"``.

    Returns:
        The base URL with no trailing slash; an empty string when the SPA
        is mounted at the application root.
    """
    combined = (root_path or "") + (mount_path or "")
    return combined.rstrip("/")
