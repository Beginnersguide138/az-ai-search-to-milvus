"""再開可能なデータ移行のためのチェックポイント管理。

チェックポイントは JSON ファイルとして保存され、障害発生後に
移行済みドキュメントを再処理することなく移行を再開できます。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MigrationCheckpoint:
    """実行中の移行の状態。"""

    index_name: str
    collection_name: str
    total_documents: int = 0
    migrated_documents: int = 0
    last_document_key: str = ""
    failed_document_keys: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | in_progress | completed | failed
    created_at: float = 0.0
    updated_at: float = 0.0
    batch_number: int = 0
    error_message: str = ""

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()

    def mark_in_progress(self) -> None:
        self.status = "in_progress"
        self.updated_at = time.time()

    def advance(self, count: int, last_key: str) -> None:
        self.migrated_documents += count
        self.last_document_key = last_key
        self.batch_number += 1
        self.updated_at = time.time()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.updated_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error_message = error
        self.updated_at = time.time()

    @property
    def progress_pct(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return (self.migrated_documents / self.total_documents) * 100


class CheckpointManager:
    """:class:`MigrationCheckpoint` をファイルシステムに永続化する。"""

    def __init__(self, checkpoint_dir: str | Path = ".checkpoints") -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, index_name: str) -> Path:
        safe = index_name.replace("/", "_").replace("\\", "_")
        return self.dir / f"{safe}.checkpoint.json"

    def save(self, cp: MigrationCheckpoint) -> Path:
        path = self._path(cp.index_name)
        path.write_text(json.dumps(asdict(cp), indent=2, ensure_ascii=False))
        return path

    def load(self, index_name: str) -> MigrationCheckpoint | None:
        path = self._path(index_name)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return MigrationCheckpoint(**data)

    def delete(self, index_name: str) -> None:
        path = self._path(index_name)
        if path.exists():
            path.unlink()
