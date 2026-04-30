from __future__ import annotations

import contextlib
import os

try:
    import httpx
except ImportError as _err:  # pragma: no cover
    raise ImportError("VoyageEmbeddingClient requires the 'voyage' extra: pip install nanitics[voyage]") from _err

from nanitics.infrastructure.errors import EmbeddingProviderError, EmbeddingRateLimitError

_VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddingClient:
    """Production embedding client using the Voyage AI API.

    Converts text into semantic embedding vectors via Voyage AI's
    embedding models. Requires the ``voyage`` extra:
    ``pip install nanitics[voyage]``.

    Args:
        api_key: Voyage AI API key. If not provided, reads from the
            ``VOYAGE_API_KEY`` environment variable.
        model: Voyage model name (default: ``"voyage-3-lite"``).

    Raises:
        ValueError: If no API key is provided or found in environment.
        EmbeddingRateLimitError: On HTTP 429 responses.
        EmbeddingProviderError: On other HTTP errors or connection failures.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-3-lite",
    ) -> None:
        self._model = model
        resolved_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not resolved_key:
            raise ValueError("Voyage API key required. Pass api_key or set VOYAGE_API_KEY.")
        self._api_key = resolved_key
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                _VOYAGE_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": self._model},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                retry_after = None
                raw = e.response.headers.get("retry-after")
                if raw is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        retry_after = float(raw)
                raise EmbeddingRateLimitError(str(e), retry_after=retry_after) from e
            raise EmbeddingProviderError(str(e), status_code=status, provider="voyage") from e
        except httpx.ConnectError as e:
            raise EmbeddingProviderError(str(e), provider="voyage") from e
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self) -> None:
        await self._client.aclose()
