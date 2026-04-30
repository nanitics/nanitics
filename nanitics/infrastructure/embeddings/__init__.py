from nanitics.infrastructure.embeddings.mock import MockEmbeddingClient
from nanitics.infrastructure.embeddings.protocol import EmbeddingClient

try:
    from nanitics.infrastructure.embeddings.voyage import VoyageEmbeddingClient
except ImportError:
    VoyageEmbeddingClient = None  # type: ignore[assignment,misc]

__all__ = [
    "EmbeddingClient",
    "MockEmbeddingClient",
    "VoyageEmbeddingClient",
]
