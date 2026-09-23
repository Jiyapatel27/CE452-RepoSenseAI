"""Step 10: SQLite persistence.

Chat history has to survive a restart, so it cannot live in memory. SQLite is
the right size of tool here: one file, no server, no container, and `sqlite3` is
in the standard library - no new dependency.

Three SQLite behaviours are easy to get wrong, so they are handled explicitly in
`connect()`:

1. **Foreign keys are OFF by default**, per connection. Without the pragma,
   `ON DELETE CASCADE` silently does nothing and deleting a conversation would
   orphan all of its messages.
2. **Connections are not thread-safe.** FastAPI runs sync work in a threadpool,
   so a shared connection would eventually be used from the wrong thread. We
   open one per operation instead; SQLite is cheap to open and handles many
   concurrent readers.
3. **The default journal mode blocks readers during a write.** WAL lets the
   dashboard read while a chat message is being written.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import get_settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
-- Repositories that have been analysed. Mirrors what Step 8 could only
-- partially rebuild from Qdrant: the ingestion statistics (files scanned,
-- parse counts, skip reasons) exist nowhere else, so they are stored here.
CREATE TABLE IF NOT EXISTS repositories (
    repository_id       TEXT PRIMARY KEY,
    repository          TEXT NOT NULL,
    owner               TEXT NOT NULL,
    branch              TEXT,
    html_url            TEXT NOT NULL,
    local_path          TEXT NOT NULL,
    status              TEXT NOT NULL,
    authenticated       INTEGER NOT NULL DEFAULT 0,
    indexed             INTEGER NOT NULL DEFAULT 0,
    total_files_scanned INTEGER NOT NULL DEFAULT 0,
    supported_files     INTEGER NOT NULL DEFAULT 0,
    total_chunks        INTEGER NOT NULL DEFAULT 0,
    language_counts     TEXT NOT NULL DEFAULT '{}',
    skipped_files       TEXT NOT NULL DEFAULT '{}',
    parse_stats         TEXT NOT NULL DEFAULT '{}',
    chunk_stats         TEXT NOT NULL DEFAULT '{}',
    analyzed_at         TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

-- One chat thread against one repository.
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    title         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_repository
    ON conversations (repository_id, updated_at DESC);

-- Messages in a thread. ON DELETE CASCADE only works because connect()
-- enables the foreign_keys pragma.
CREATE TABLE IF NOT EXISTS messages (
    id                TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL
                      REFERENCES conversations (id) ON DELETE CASCADE,
    role              TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content           TEXT NOT NULL,
    created_at        TEXT NOT NULL,

    -- Assistant-only metadata. Null on user messages.
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    elapsed_ms        REAL,
    context_chunks    INTEGER,
    refused           INTEGER NOT NULL DEFAULT 0,
    truncated         INTEGER NOT NULL DEFAULT 0,
    -- The standalone question retrieval actually used, when a follow-up had to
    -- be rewritten (Step 11). Kept for debugging bad retrievals.
    resolved_question TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, created_at);

-- Files cited by an assistant message. Normalised rather than a JSON blob so
-- the dashboard can answer "most cited files" with a GROUP BY.
CREATE TABLE IF NOT EXISTS message_sources (
    message_id  TEXT NOT NULL REFERENCES messages (id) ON DELETE CASCADE,
    file_path   TEXT NOT NULL,
    score       REAL,
    start_line  INTEGER,
    end_line    INTEGER,
    language    TEXT,
    cited       INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_message_sources_message
    ON message_sources (message_id);
CREATE INDEX IF NOT EXISTS idx_message_sources_file
    ON message_sources (file_path);

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utc_now() -> str:
    """Timestamp as an ISO-8601 UTC string.

    Stored as text because SQLite has no native date type, and ISO strings sort
    chronologically - so ORDER BY on them is correct.
    """
    return datetime.now(timezone.utc).isoformat()


def dumps(value: Any) -> str:
    """JSON for a column, never NULL - simplifies every read path."""
    return json.dumps(value if value is not None else {})


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return {} if default is None else default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Corrupt JSON column, returning default")
        return {} if default is None else default


@contextmanager
def connect(*, readonly: bool = False) -> Iterator[sqlite3.Connection]:
    """Open a connection for one operation, then close it.

    Deliberately not cached: sqlite3 objects are bound to the thread that made
    them, and FastAPI hands sync work to arbitrary threadpool workers. Opening
    per operation costs microseconds and removes that whole class of bug.
    """
    settings = get_settings()
    path = Path(settings.database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=10.0)
    try:
        # Rows behave like dicts, so callers use column names, not indexes.
        connection.row_factory = sqlite3.Row
        # Must be per connection - see the module docstring.
        connection.execute("PRAGMA foreign_keys = ON")
        if not readonly:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        yield connection
        if not readonly:
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database() -> Path:
    """Create the schema if needed. Safe to call on every startup."""
    settings = get_settings()
    with connect() as connection:
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
    logger.info("Database ready at %s", settings.database_path)
    return Path(settings.database_path)


def get_schema_version() -> int:
    with connect(readonly=True) as connection:
        row = connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
    return int(row["value"]) if row else 0


def database_stats() -> dict[str, Any]:
    """Row counts and file size, for the dashboard's system panel."""
    settings = get_settings()
    path = Path(settings.database_path)

    with connect(readonly=True) as connection:
        counts = {
            table: connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608 - fixed names
            ).fetchone()["n"]
            for table in ("repositories", "conversations", "messages",
                          "message_sources")
        }

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "schema_version": get_schema_version(),
        **counts,
    }
