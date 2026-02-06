"""Tests for checkpoint management."""

from __future__ import annotations

import tempfile
from pathlib import Path

from az_search_to_milvus.utils.checkpoint import CheckpointManager, MigrationCheckpoint


class TestMigrationCheckpoint:
    def test_initial_state(self) -> None:
        cp = MigrationCheckpoint(
            index_name="test",
            collection_name="test_col",
            total_documents=1000,
        )
        assert cp.status == "pending"
        assert cp.migrated_documents == 0
        assert cp.progress_pct == 0.0
        assert cp.created_at > 0

    def test_mark_in_progress(self) -> None:
        cp = MigrationCheckpoint(index_name="test", collection_name="test_col")
        cp.mark_in_progress()
        assert cp.status == "in_progress"
        assert cp.updated_at > 0

    def test_advance(self) -> None:
        cp = MigrationCheckpoint(
            index_name="test",
            collection_name="test_col",
            total_documents=100,
        )
        cp.advance(25, "doc-025")
        assert cp.migrated_documents == 25
        assert cp.last_document_key == "doc-025"
        assert cp.batch_number == 1
        assert cp.progress_pct == 25.0

        cp.advance(25, "doc-050")
        assert cp.migrated_documents == 50
        assert cp.batch_number == 2
        assert cp.progress_pct == 50.0

    def test_mark_completed(self) -> None:
        cp = MigrationCheckpoint(index_name="test", collection_name="test_col")
        cp.mark_completed()
        assert cp.status == "completed"

    def test_mark_failed(self) -> None:
        cp = MigrationCheckpoint(index_name="test", collection_name="test_col")
        cp.mark_failed("connection error")
        assert cp.status == "failed"
        assert cp.error_message == "connection error"


class TestCheckpointManager:
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            cp = MigrationCheckpoint(
                index_name="test-index",
                collection_name="test_col",
                total_documents=500,
                migrated_documents=100,
                last_document_key="doc-100",
                status="in_progress",
            )
            mgr.save(cp)

            loaded = mgr.load("test-index")
            assert loaded is not None
            assert loaded.index_name == "test-index"
            assert loaded.migrated_documents == 100
            assert loaded.status == "in_progress"

    def test_load_nonexistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            result = mgr.load("nonexistent")
            assert result is None

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            cp = MigrationCheckpoint(index_name="to-delete", collection_name="col")
            mgr.save(cp)
            assert mgr.load("to-delete") is not None

            mgr.delete("to-delete")
            assert mgr.load("to-delete") is None

    def test_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir)
            cp = MigrationCheckpoint(
                index_name="overwrite-test",
                collection_name="col",
                migrated_documents=10,
            )
            mgr.save(cp)

            cp.advance(20, "doc-030")
            mgr.save(cp)

            loaded = mgr.load("overwrite-test")
            assert loaded is not None
            assert loaded.migrated_documents == 30
