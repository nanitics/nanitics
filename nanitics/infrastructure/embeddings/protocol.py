from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for converting text into embedding vectors.

    Embedding clients are used by ``SemanticStore`` and ``EpisodeStore``
    to enable similarity-based retrieval.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more texts into vector representations.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...
