from __future__ import annotations

from typing import Protocol, runtime_checkable

from nanitics.composition.durability.models import RunCheckpoint, StepRecord


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for persisting and retrieving execution checkpoints.

    Implementations store checkpoints so workflows can resume after
    suspension, potentially in a different process. Two distinct pieces of
    durable state are kept: a cursor snapshot (``save`` / ``load``, the
    ``RunCheckpoint`` recording loop position) and an append-only step-result
    journal (``append_step`` / ``load_journal``, the ``StepRecord`` entries
    recording completed-step results that protect side effects on replay).
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

    async def append_step(self, record: StepRecord) -> None:
        """Append a completed-step result to the run's journal.

        Idempotent on ``(run_id, step_path)``: re-appending the same step key
        is a no-op / last-write-wins, since a given step key has one canonical
        result. This is the write site that makes a completed step replayable
        rather than re-dispatched on resume.
        """
        ...

    async def load_journal(self, run_id: str) -> list[StepRecord]:
        """Return all step records for a run, in append order.

        Returns an empty list when the run has no journal entries.
        """
        ...


class InMemoryCheckpointStore:
    """In-memory implementation of CheckpointStore for testing.

    Stores checkpoints in a dictionary keyed by checkpoint ID.
    ``load()`` returns the most recent checkpoint for a run by ``created_at``.

    The step-result journal is stored keyed by ``(run_id, step_path)``; the
    insertion-ordered dict gives both idempotency on the step key
    (last-write-wins, original position retained) and append-order iteration.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, RunCheckpoint] = {}
        self._journal: dict[tuple[str, str], StepRecord] = {}

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
        journal_keys = [key for key in self._journal if key[0] == run_id]
        for key in journal_keys:
            del self._journal[key]

    async def append_step(self, record: StepRecord) -> None:
        # Keying by (run_id, step_path) makes re-appending the same step key
        # last-write-wins while retaining the entry's original append position.
        self._journal[(record.run_id, record.step_path)] = record

    async def load_journal(self, run_id: str) -> list[StepRecord]:
        return [record for key, record in self._journal.items() if key[0] == run_id]
