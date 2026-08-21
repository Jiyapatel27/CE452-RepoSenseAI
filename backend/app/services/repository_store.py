"""In-memory registry of analysed repositories.

Deliberately simple for this milestone: a dict guarded by a lock. PostgreSQL is
explicitly out of scope, so state resets when the server restarts (the cloned
files on disk survive).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.services.chunker_service import ChunkSummary
from app.services.github_service import DiscoveredFile
from app.services.parser_service import ParseSummary


@dataclass
class RepositoryRecord:
    repository_id: str
    repository: str
    owner: str
    branch: str | None
    html_url: str
    local_path: str
    status: str
    total_files_scanned: int
    files: list[DiscoveredFile]
    skipped_files: dict[str, int]
    language_counts: dict[str, int]
    truncated: bool
    # True when a GitHub token was supplied for the clone. The token itself is
    # deliberately never stored.
    authenticated: bool = False
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Step 2: parsing statistics. File contents are deliberately NOT stored -
    # they are re-read from `local_path` on demand.
    parse_summary: ParseSummary = field(default_factory=ParseSummary)

    # Step 3: chunk statistics. Like file contents, chunk *text* is not stored -
    # it is regenerated on demand from the files on disk.
    chunk_summary: ChunkSummary = field(default_factory=ChunkSummary)
    chunk_size: int = 0
    chunk_overlap: int = 0

    # Populated by Step 5 (Qdrant indexing).
    indexed: bool = False

    @property
    def total_chunks(self) -> int:
        return self.chunk_summary.total_chunks

    @property
    def supported_files(self) -> int:
        return len(self.files)

    @property
    def repo_root(self) -> Path:
        return Path(self.local_path)


class RepositoryStore:
    def __init__(self) -> None:
        self._records: dict[str, RepositoryRecord] = {}
        self._lock = threading.Lock()

    def save(self, record: RepositoryRecord) -> RepositoryRecord:
        with self._lock:
            self._records[record.repository_id] = record
        return record

    def get(self, repository_id: str) -> RepositoryRecord | None:
        with self._lock:
            return self._records.get(repository_id)

    def list_all(self) -> list[RepositoryRecord]:
        with self._lock:
            return sorted(
                self._records.values(), key=lambda r: r.analyzed_at, reverse=True
            )


# Single shared instance used by the routes.
repository_store = RepositoryStore()


def restore_from_vector_store() -> int:
    """Rebuild repository records from Qdrant. Returns how many were restored.

    Vectors persist across restarts but this store does not, so without this a
    freshly started server would report "no analysed repository" for code it can
    already search - forcing a pointless re-clone.

    The restored record is deliberately *partial*: Qdrant holds chunk payloads,
    not the ingestion statistics. `total_files_scanned`, `skipped_files` and the
    parse counts are unknown and left at zero, and `status` says `restored` so
    the difference is visible rather than silently faked. Re-running /analyze
    fills them in properly.
    """
    # Imported here rather than at module scope: vector_store imports the
    # embedding model, and importing that at startup for a store that may never
    # be used would pull PyTorch into every process that touches this module.
    from app.services.vector_store import list_repository_summaries

    restored = 0
    for summary in list_repository_summaries():
        if repository_store.get(summary.repository_id) is not None:
            continue  # a live record is always better than a rebuilt one

        repo_root = get_settings().repo_storage_dir / summary.repository_id
        files = [
            DiscoveredFile(
                file_path=path,
                file_name=path.rsplit("/", 1)[-1],
                extension=("." + path.rsplit(".", 1)[-1]) if "." in path else "",
                language=language,
                # Read from disk when the clone survived; 0 when it did not.
                size_bytes=_file_size(repo_root / path),
            )
            for path, language in sorted(summary.files.items())
        ]

        language_counts: dict[str, int] = {}
        for file in files:
            language_counts[file.language] = language_counts.get(file.language, 0) + 1

        chunk_summary = ChunkSummary(total_chunks=summary.chunk_count)

        owner, _, name = summary.repository_id.partition("__")
        repository_store.save(
            RepositoryRecord(
                repository_id=summary.repository_id,
                repository=summary.repository,
                owner=owner,
                branch=None,
                html_url=f"https://github.com/{owner}/{name or summary.repository}",
                local_path=str(repo_root),
                status="restored",
                total_files_scanned=0,
                files=files,
                skipped_files={},
                language_counts=dict(
                    sorted(language_counts.items(), key=lambda kv: -kv[1])
                ),
                truncated=False,
                chunk_summary=chunk_summary,
                indexed=True,
            )
        )
        restored += 1

    return restored


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
