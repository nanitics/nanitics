from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanitics.composition.durability.models import RunCheckpoint


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for persisting and retrieving execution checkpoints.

    Implementations store checkpoints so workflows can resume after
    suspension, potentially in a different process.
    """

    async def save(self, checkpoint: RunCheckpoint) -> None:
        """Persist a checkpoint."""
        ...

    async def load(self, run_id: str) -> RunCheckpoint | None:
        """Load the most recent checkpoint for a run, or None if none exists."""
        ...

    async def delete(self, checkpoint_id: str) -> None:
        """Delete a specific checkpoint."""
        ...

    async def delete_for_run(self, run_id: str) -> None:
        """Delete all checkpoints for a run."""
        ...


class InMemoryCheckpointStore:
    """In-memory implementation of CheckpointStore for testing.

    Stores checkpoints in a dictionary keyed by checkpoint ID.
    ``load()`` returns the most recent checkpoint for a run by ``created_at``.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, RunCheckpoint] = {}

    async def save(self, checkpoint: RunCheckpoint) -> None:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    async def load(self, run_id: str) -> RunCheckpoint | None:
        matches = [cp for cp in self._checkpoints.values() if cp.run_id == run_id]
        if not matches:
            return None
        return max(matches, key=lambda cp: cp.created_at)

    async def delete(self, checkpoint_id: str) -> None:
        self._checkpoints.pop(checkpoint_id, None)

    async def delete_for_run(self, run_id: str) -> None:
        to_delete = [cp_id for cp_id, cp in self._checkpoints.items() if cp.run_id == run_id]
        for cp_id in to_delete:
            del self._checkpoints[cp_id]
