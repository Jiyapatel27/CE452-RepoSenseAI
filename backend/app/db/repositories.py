"""Repository metadata persistence (Step 10).

This closes a gap left by Step 8. Restoring repositories from Qdrant recovered
the file list and chunk count, but the ingestion statistics - files scanned,
skip reasons, parse counts - exist only in the analyse response, so a restarted
server reported them as zero. Persisting them here means a restore is complete
rather than partial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.db.database import connect, dumps, loads, utc_now

logger = logging.getLogger(__name__)


@dataclass
class StoredRepository:
    repository_id: str
    repository: str
    owner: str
    html_url: str
    local_path: str
    status: str
    branch: str | None = None
    authenticated: bool = False
    indexed: bool = False
    total_files_scanned: int = 0
    supported_files: int = 0
    total_chunks: int = 0
    language_counts: dict[str, int] = field(default_factory=dict)
    skipped_files: dict[str, int] = field(default_factory=dict)
    parse_stats: dict[str, Any] = field(default_factory=dict)
    chunk_stats: dict[str, Any] = field(default_factory=dict)
    analyzed_at: str = ""
    updated_at: str = ""


def save_repository(repo: StoredRepository) -> StoredRepository:
    """Insert or update a repository row, keyed by repository_id.

    Upsert rather than insert: re-analysing the same repository should refresh
    its statistics, not fail on a primary-key clash or create a duplicate.
    """
    now = utc_now()
    analyzed_at = repo.analyzed_at or now

    with connect() as connection:
        connection.execute(
            """INSERT INTO repositories
                   (repository_id, repository, owner, branch, html_url,
                    local_path, status, authenticated, indexed,
                    total_files_scanned, supported_files, total_chunks,
                    language_counts, skipped_files, parse_stats, chunk_stats,
                    analyzed_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (repository_id) DO UPDATE SET
                    repository          = excluded.repository,
                    owner               = excluded.owner,
                    branch              = excluded.branch,
                    html_url            = excluded.html_url,
                    local_path          = excluded.local_path,
                    status              = excluded.status,
                    authenticated       = excluded.authenticated,
                    indexed             = excluded.indexed,
                    total_files_scanned = excluded.total_files_scanned,
                    supported_files     = excluded.supported_files,
                    total_chunks        = excluded.total_chunks,
                    language_counts     = excluded.language_counts,
                    skipped_files       = excluded.skipped_files,
                    parse_stats         = excluded.parse_stats,
                    chunk_stats         = excluded.chunk_stats,
                    analyzed_at         = excluded.analyzed_at,
                    updated_at          = excluded.updated_at""",
            (
                repo.repository_id, repo.repository, repo.owner, repo.branch,
                repo.html_url, repo.local_path, repo.status,
                int(repo.authenticated), int(repo.indexed),
                repo.total_files_scanned, repo.supported_files,
                repo.total_chunks, dumps(repo.language_counts),
                dumps(repo.skipped_files), dumps(repo.parse_stats),
                dumps(repo.chunk_stats), analyzed_at, now,
            ),
        )

    repo.analyzed_at = analyzed_at
    repo.updated_at = now
    return repo


def _row_to_repository(row: Any) -> StoredRepository:
    return StoredRepository(
        repository_id=row["repository_id"],
        repository=row["repository"],
        owner=row["owner"],
        branch=row["branch"],
        html_url=row["html_url"],
        local_path=row["local_path"],
        status=row["status"],
        authenticated=bool(row["authenticated"]),
        indexed=bool(row["indexed"]),
        total_files_scanned=row["total_files_scanned"],
        supported_files=row["supported_files"],
        total_chunks=row["total_chunks"],
        language_counts=loads(row["language_counts"]),
        skipped_files=loads(row["skipped_files"]),
        parse_stats=loads(row["parse_stats"]),
        chunk_stats=loads(row["chunk_stats"]),
        analyzed_at=row["analyzed_at"],
        updated_at=row["updated_at"],
    )


def get_repository(repository_id: str) -> StoredRepository | None:
    with connect(readonly=True) as connection:
        row = connection.execute(
            "SELECT * FROM repositories WHERE repository_id = ?",
            (repository_id,),
        ).fetchone()
    return _row_to_repository(row) if row else None


def list_repositories() -> list[StoredRepository]:
    """All analysed repositories, most recently analysed first."""
    with connect(readonly=True) as connection:
        rows = connection.execute(
            "SELECT * FROM repositories ORDER BY analyzed_at DESC"
        ).fetchall()
    return [_row_to_repository(row) for row in rows]


def delete_repository_record(repository_id: str) -> bool:
    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM repositories WHERE repository_id = ?", (repository_id,)
        )
    return cursor.rowcount > 0
