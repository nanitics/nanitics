import pytest

from nanitics.composition.durability.models import RunCheckpoint, SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore


def _make_checkpoint(
    run_id: str = "run-1",
    checkpoint_id: str | None = None,
    checkpoint_type: str = "orchestration",
) -> RunCheckpoint:
    kwargs: dict = {
        "run_id": run_id,
        "checkpoint_type": checkpoint_type,
        "state": {"step": 0},
        "suspension_info": SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        ),
    }
    if checkpoint_id:
        kwargs["checkpoint_id"] = checkpoint_id
    return RunCheckpoint(**kwargs)


class TestInMemoryCheckpointStore:
    @pytest.fixture
    def store(self) -> InMemoryCheckpointStore:
        return InMemoryCheckpointStore()

    async def test_save_and_load(self, store: InMemoryCheckpointStore) -> None:
        cp = _make_checkpoint(checkpoint_id="cp-1")
        await store.save(cp)
        loaded = await store.load("run-1")
        assert loaded is not None
        assert loaded.checkpoint_id == "cp-1"

    async def test_load_returns_none_for_missing(self, store: InMemoryCheckpointStore) -> None:
        result = await store.load("nonexistent")
        assert result is None

    async def test_load_returns_most_recent(self, store: InMemoryCheckpointStore) -> None:
        from datetime import UTC, datetime

        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )
        older = RunCheckpoint(
            checkpoint_id="cp-old",
            run_id="run-1",
            checkpoint_type="orchestration",
            state={"step": 0},
            suspension_info=info,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        newer = RunCheckpoint(
            checkpoint_id="cp-new",
            run_id="run-1",
            checkpoint_type="orchestration",
            state={"step": 1},
            suspension_info=info,
            created_at=datetime(2025, 1, 2, tzinfo=UTC),
        )
        await store.save(older)
        await store.save(newer)
        loaded = await store.load("run-1")
        assert loaded is not None
        assert loaded.checkpoint_id == "cp-new"

    async def test_delete(self, store: InMemoryCheckpointStore) -> None:
        cp = _make_checkpoint(checkpoint_id="cp-1")
        await store.save(cp)
        await store.delete("cp-1")
        loaded = await store.load("run-1")
        assert loaded is None

    async def test_delete_nonexistent_is_noop(self, store: InMemoryCheckpointStore) -> None:
        await store.delete("nonexistent")  # Should not raise

    async def test_delete_for_run(self, store: InMemoryCheckpointStore) -> None:
        cp1 = _make_checkpoint(run_id="run-1", checkpoint_id="cp-1")
        cp2 = _make_checkpoint(run_id="run-1", checkpoint_id="cp-2")
        cp3 = _make_checkpoint(run_id="run-2", checkpoint_id="cp-3")
        await store.save(cp1)
        await store.save(cp2)
        await store.save(cp3)

        await store.delete_for_run("run-1")

        assert await store.load("run-1") is None
        assert await store.load("run-2") is not None

    async def test_delete_for_run_nonexistent_is_noop(self, store: InMemoryCheckpointStore) -> None:
        await store.delete_for_run("nonexistent")  # Should not raise
