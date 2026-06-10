from datetime import UTC, datetime

import pytest

from nanitics.composition.durability.models import RunCheckpoint, StepRecord, SuspensionInfo
from nanitics.composition.durability.store import InMemoryCheckpointStore


def _make_step(
    *,
    run_id: str = "run-1",
    step_path: str,
    result: dict | None = None,
) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        step_path=step_path,
        step_kind="tool_call",
        result=result if result is not None else {"content": "ok"},
    )


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


class TestInMemoryJournal:
    @pytest.fixture
    def store(self) -> InMemoryCheckpointStore:
        return InMemoryCheckpointStore()

    async def test_load_journal_empty_when_none(self, store: InMemoryCheckpointStore) -> None:
        assert await store.load_journal("run-1") == []

    async def test_append_and_load_round_trip(self, store: InMemoryCheckpointStore) -> None:
        rec = _make_step(step_path="seq#0/tool#0:send_email")
        await store.append_step(rec)
        loaded = await store.load_journal("run-1")
        assert loaded == [rec]

    async def test_load_journal_returns_append_order(self, store: InMemoryCheckpointStore) -> None:
        r0 = _make_step(step_path="seq#0")
        r1 = _make_step(step_path="seq#1")
        r2 = _make_step(step_path="seq#2")
        await store.append_step(r1)
        await store.append_step(r0)
        await store.append_step(r2)
        loaded = await store.load_journal("run-1")
        assert [r.step_path for r in loaded] == ["seq#1", "seq#0", "seq#2"]

    async def test_append_is_idempotent_last_write_wins(self, store: InMemoryCheckpointStore) -> None:
        """Re-appending the same (run_id, step_path) is a no-op for ordering.

        The newer value wins (last-write-wins) and the entry keeps its
        original append position.
        """
        first = _make_step(step_path="seq#0", result={"content": "v1"})
        middle = _make_step(step_path="seq#1", result={"content": "mid"})
        replacement = _make_step(step_path="seq#0", result={"content": "v2"})
        await store.append_step(first)
        await store.append_step(middle)
        await store.append_step(replacement)
        loaded = await store.load_journal("run-1")
        assert [r.step_path for r in loaded] == ["seq#0", "seq#1"]
        assert loaded[0].result == {"content": "v2"}

    async def test_journal_is_scoped_per_run(self, store: InMemoryCheckpointStore) -> None:
        await store.append_step(_make_step(run_id="run-1", step_path="seq#0"))
        await store.append_step(_make_step(run_id="run-2", step_path="seq#0"))
        assert len(await store.load_journal("run-1")) == 1
        assert len(await store.load_journal("run-2")) == 1

    async def test_delete_for_run_clears_journal(self, store: InMemoryCheckpointStore) -> None:
        await store.append_step(_make_step(run_id="run-1", step_path="seq#0"))
        await store.append_step(_make_step(run_id="run-2", step_path="seq#0"))
        await store.delete_for_run("run-1")
        assert await store.load_journal("run-1") == []
        assert len(await store.load_journal("run-2")) == 1

    async def test_save_and_journal_are_independent(self, store: InMemoryCheckpointStore) -> None:
        """The cursor snapshot and the journal are separate durable artifacts."""
        info = SuspensionInfo(
            suspension_id="sus-1",
            request_id="req-1",
            request_type="approval",
            prompt="Approve?",
        )
        cp = RunCheckpoint(
            checkpoint_id="cp-1",
            run_id="run-1",
            checkpoint_type="orchestration",
            state={"cursor": 1},
            suspension_info=info,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await store.save(cp)
        await store.append_step(_make_step(step_path="seq#0"))
        assert (await store.load("run-1")) is not None
        assert len(await store.load_journal("run-1")) == 1
