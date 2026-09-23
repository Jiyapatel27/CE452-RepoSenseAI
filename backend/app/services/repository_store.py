"""In-memory registry of analysed repositories.

Deliberately simple for this milestone: a dict guarded by a lock. PostgreSQL is
explicitly out of scope, so state resets when the server restarts (the cloned
files on disk survive).
"""

from __future__ import annotations

import logging
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


def restore_repositories() -> dict[str, int]:
    """Rebuild the in-memory store on startup, best source first.

    SQLite has the complete picture (including ingestion statistics that exist
    nowhere else), so it wins. Qdrant is then used only for repositories that
    have vectors but no database row - which happens for anything indexed
    before Step 10 added persistence.
    """
    from_database = restore_from_database()
    from_vectors = restore_from_vector_store()
    return {"from_database": from_database, "from_vectors": from_vectors}


def restore_from_database() -> int:
    """Load repository records from SQLite. Returns how many were restored."""
    from app.db.repositories import list_repositories

    restored = 0
    for stored in list_repositories():
        if repository_store.get(stored.repository_id) is not None:
            continue

        repo_root = Path(stored.local_path)
        # Rebuild the file list from disk so /chunks and /file keep working.
        # The clone may be gone (user cleared .reposense-data), in which case
        # those endpoints correctly report 410 rather than crashing.
        files = _scan_files_if_present(repo_root)

        parse_summary = ParseSummary(
            files_parsed=stored.parse_stats.get("files_parsed", 0),
            files_failed=stored.parse_stats.get("files_failed", 0),
            total_lines=stored.parse_stats.get("total_lines", 0),
            total_characters=stored.parse_stats.get("total_characters", 0),
        )
        chunk_summary = ChunkSummary(
            total_chunks=stored.total_chunks,
            files_chunked=stored.chunk_stats.get("files_chunked", 0),
            files_without_chunks=stored.chunk_stats.get("files_without_chunks", 0),
            total_characters=stored.chunk_stats.get("total_characters", 0),
            max_chunk_chars=stored.chunk_stats.get("max_chunk_chars", 0),
            min_chunk_chars=stored.chunk_stats.get("min_chunk_chars", 0),
            total_tokens=stored.chunk_stats.get("total_tokens", 0),
            max_chunk_tokens=stored.chunk_stats.get("max_chunk_tokens", 0),
            chunks_per_language=stored.chunk_stats.get("chunks_per_language", {}),
        )

        repository_store.save(
            RepositoryRecord(
                repository_id=stored.repository_id,
                repository=stored.repository,
                owner=stored.owner,
                branch=stored.branch,
                html_url=stored.html_url,
                local_path=stored.local_path,
                status=stored.status,
                total_files_scanned=stored.total_files_scanned,
                files=files,
                skipped_files=stored.skipped_files,
                language_counts=stored.language_counts,
                truncated=False,
                authenticated=stored.authenticated,
                analyzed_at=stored.analyzed_at,
                parse_summary=parse_summary,
                chunk_summary=chunk_summary,
                indexed=stored.indexed,
            )
        )
        restored += 1

    return restored


def _scan_files_if_present(repo_root: Path) -> list[DiscoveredFile]:
    """Re-walk a clone to rebuild its file list, or [] if it is gone."""
    if not repo_root.exists():
        return []
    try:
        from app.config import get_settings as _get_settings
        from app.services.github_service import scan_repository_files

        settings = _get_settings()
        return scan_repository_files(
            repo_root,
            max_file_size_bytes=settings.max_file_size_bytes,
            max_files=settings.max_files_per_repo,
        ).files
    except Exception as exc:  # noqa: BLE001 - restore must not fail on this
        logger = __import__("logging").getLogger(__name__)
        logger.warning("Could not re-scan %s: %s", repo_root, exc)
        return []


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

        # Backfill a metadata row so this repository stops being invisible to
        # anything that reads SQLite (the dashboard's per-repository panel).
        # The statistics Qdrant cannot provide stay zero and the status says
        # "restored", so the record never pretends to be complete.
        _backfill_metadata(
            repository_id=summary.repository_id,
            repository=summary.repository,
            owner=owner,
            name=name,
            local_path=str(repo_root),
            files=files,
            language_counts=language_counts,
            total_chunks=summary.chunk_count,
        )
        restored += 1

    return restored


def _backfill_metadata(
    *,
    repository_id: str,
    repository: str,
    owner: str,
    name: str,
    local_path: str,
    files: list[DiscoveredFile],
    language_counts: dict[str, int],
    total_chunks: int,
) -> None:
    """Write a partial metadata row for a Qdrant-only repository."""
    from app.db.repositories import StoredRepository, get_repository, save_repository

    if get_repository(repository_id) is not None:
        return  # a real row already exists; never downgrade it

    try:
        save_repository(
            StoredRepository(
                repository_id=repository_id,
                repository=repository,
                owner=owner,
                html_url=f"https://github.com/{owner}/{name or repository}",
                local_path=local_path,
                status="restored",
                indexed=True,
                supported_files=len(files),
                total_chunks=total_chunks,
                language_counts=language_counts,
                chunk_stats={"total_chunks": total_chunks},
            )
        )
    except Exception as exc:  # noqa: BLE001 - restore must not fail on this
        logging.getLogger(__name__).warning(
            "Could not backfill metadata for %s: %s", repository_id, exc
        )


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
