from __future__ import annotations

import hashlib
import struct


class MockEmbeddingClient:
    """Deterministic embedding client for testing.

    Generates reproducible vectors from text content using SHA-256 hashing
    and a linear congruential generator. Vectors are normalized to unit
    length. Same input always produces the same output.

    Does not capture semantic meaning — similar texts will **not** produce
    similar vectors. Use ``VoyageEmbeddingClient`` for real similarity.

    Args:
        dimension: Vector dimension (default: 1024).
    """

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._deterministic_vector(text) for text in texts]

    def _deterministic_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        rng_seed = struct.unpack("<I", digest[:4])[0]
        # Use a simple LCG to generate deterministic floats from the seed
        state = rng_seed
        vector: list[float] = []
        for _ in range(self.dimension):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            vector.append((state / 0x7FFFFFFF) * 2.0 - 1.0)
        # Normalize to unit vector
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        return vector
