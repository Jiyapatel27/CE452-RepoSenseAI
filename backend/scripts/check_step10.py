"""Checks for Step 10 (SQLite persistence).

Runs against a throwaway database in a temp folder, so it never touches your
real data and needs nothing running.

Run from the `backend/` folder:
    python scripts/check_step10.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the settings at a temp database BEFORE anything imports them, so the
# real .reposense-data/reposense.db is never opened by this test.
_TEMP_DIR = tempfile.TemporaryDirectory()
import os  # noqa: E402

os.environ["DATABASE_PATH"] = str(Path(_TEMP_DIR.name) / "test.db")

from app.config import get_settings  # noqa: E402
from app.core.exceptions import ConversationNotFoundError  # noqa: E402
from app.db.conversations import (  # noqa: E402
    MessageSource,
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    get_recent_turns,
    list_conversations,
    make_title,
    rename_conversation,
)
from app.db.database import (  # noqa: E402
    connect,
    database_stats,
    get_schema_version,
    init_database,
)
from app.db.repositories import (  # noqa: E402
    StoredRepository,
    delete_repository_record,
    get_repository,
    list_repositories,
    save_repository,
)

failures: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"{'PASS ' if ok else 'FAIL '} {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool, note: str = "") -> None:
    ok = bool(actual)
    print(f"{'PASS ' if ok else 'FAIL '} {label}" + (f": {note}" if note else "")
          + ("" if ok else "  (expected True)"))
    if not ok:
        failures.append(label)


def main() -> int:
    settings = get_settings()
    check_true("using a temp database, not the real one",
               "test.db" in str(settings.database_path),
               note=str(settings.database_path))

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    path = init_database()
    check_true("database file created", path.exists())
    check("schema version recorded", get_schema_version(), 1)
    init_database()  # must be idempotent
    print("PASS  init_database is idempotent")

    with connect(readonly=True) as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    for table in ("repositories", "conversations", "messages",
                  "message_sources", "schema_meta"):
        check_true(f"table '{table}' exists", table in tables)

    # WAL matters: without it the dashboard would block while a chat writes.
    with connect(readonly=True) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    check("journal mode is WAL", str(mode).lower(), "wal")

    # Foreign keys are OFF by default in SQLite - if this regresses, cascades
    # silently stop working and deletes orphan their children.
    with connect(readonly=True) as connection:
        fk = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    check("foreign keys enabled", int(fk), 1)

    # ------------------------------------------------------------------
    # Repositories
    # ------------------------------------------------------------------
    repo = StoredRepository(
        repository_id="jiya2406__weatherwise",
        repository="weatherwise",
        owner="Jiya2406",
        html_url="https://github.com/Jiya2406/weatherwise",
        local_path="C:/data/repos/jiya2406__weatherwise",
        status="indexed",
        authenticated=True,
        indexed=True,
        total_files_scanned=46,
        supported_files=31,
        total_chunks=77,
        language_counts={"javascript": 26, "markdown": 3, "json": 2},
        skipped_files={"unsupported_extension": 13, "ignored_filename": 2},
        parse_stats={"files_parsed": 31, "total_lines": 1775},
        chunk_stats={"total_chunks": 77, "max_chunk_tokens": 396},
    )
    save_repository(repo)

    loaded = get_repository("jiya2406__weatherwise")
    check_true("repository saved and loaded", loaded is not None)
    check("owner round-trips", loaded.owner, "Jiya2406")
    check("authenticated round-trips as bool", loaded.authenticated, True)
    check("indexed round-trips as bool", loaded.indexed, True)
    # The gap Step 8 could not fill from Qdrant:
    check("total_files_scanned persisted", loaded.total_files_scanned, 46)
    check("skipped_files JSON round-trips",
          loaded.skipped_files["unsupported_extension"], 13)
    check("parse_stats JSON round-trips", loaded.parse_stats["total_lines"], 1775)
    check("language_counts JSON round-trips",
          loaded.language_counts["javascript"], 26)
    check("unknown repository returns None", get_repository("nope"), None)

    # Re-analysing must update, not duplicate or fail.
    repo.total_chunks = 91
    repo.status = "re-indexed"
    save_repository(repo)
    reloaded = get_repository("jiya2406__weatherwise")
    check("upsert updates chunk count", reloaded.total_chunks, 91)
    check("upsert updates status", reloaded.status, "re-indexed")
    check("upsert does not duplicate", len(list_repositories()), 1)

    # ------------------------------------------------------------------
    # Titles
    # ------------------------------------------------------------------
    check("short question becomes the title",
          make_title("Where is auth?"), "Where is auth?")
    check("whitespace is collapsed",
          make_title("  How   does\n login  work? "), "How does login work?")
    long_title = make_title("x" * 200)
    check_true("long title is truncated", len(long_title) <= 60,
               note=f"{len(long_title)} chars")
    check_true("truncated title is marked", long_title.endswith("…"))
    check("empty question still gets a title", make_title("   "),
          "New conversation")

    # ------------------------------------------------------------------
    # Conversations and messages
    # ------------------------------------------------------------------
    conversation = create_conversation(
        "jiya2406__weatherwise", make_title("Where is the DB initialized?")
    )
    check_true("conversation has an id", bool(conversation.id))
    check("title stored", conversation.title, "Where is the DB initialized?")

    add_message(conversation.id, "user", "Where is the DB initialized?")
    add_message(
        conversation.id, "assistant",
        "It is set up in server/config/db.js.",
        model="openai/gpt-oss-120b",
        prompt_tokens=1355, completion_tokens=206, elapsed_ms=2638.0,
        context_chunks=6,
        sources=[
            MessageSource("server/config/db.js", 0.631, 1, 18, "javascript"),
            MessageSource("server/app.js", 0.632, 1, 9, "javascript"),
        ],
    )

    messages = get_messages(conversation.id)
    check("both messages stored", len(messages), 2)
    check("messages are in order", [m.role for m in messages],
          ["user", "assistant"])
    check("assistant metadata stored", messages[1].prompt_tokens, 1355)
    check("elapsed_ms stored as float", messages[1].elapsed_ms, 2638.0)
    check("sources stored", len(messages[1].sources), 2)
    check("source file path", messages[1].sources[0].file_path,
          "server/config/db.js")
    check("source line range", (messages[1].sources[0].start_line,
                                messages[1].sources[0].end_line), (1, 18))
    check("user message has no sources", len(messages[0].sources), 0)
    check("user message has no model", messages[0].model, None)

    # Refusals must be distinguishable for the dashboard.
    add_message(conversation.id, "user", "Where is JWT auth?")
    add_message(conversation.id, "assistant",
                "The context does not contain any JWT authentication.",
                refused=True, model="openai/gpt-oss-120b",
                resolved_question="Where is JWT authentication in weatherwise?")
    messages = get_messages(conversation.id)
    check("refusal flag stored", messages[3].refused, True)
    check("resolved_question stored", messages[3].resolved_question,
          "Where is JWT authentication in weatherwise?")
    check("non-refusal stays false", messages[1].refused, False)

    # ------------------------------------------------------------------
    # Listing for the sidebar
    # ------------------------------------------------------------------
    second = create_conversation("jiya2406__weatherwise", "Routes question")
    add_message(second.id, "user", "Which files handle routes?")

    listed = list_conversations("jiya2406__weatherwise")
    check("both conversations listed", len(listed), 2)
    check_true("newest conversation first", listed[0].id == second.id,
               note=f"{listed[0].title!r} before {listed[1].title!r}")
    check("message_count computed", listed[1].message_count, 4)
    check("last_message preview present",
          listed[1].last_message,
          "The context does not contain any JWT authentication.")

    other = create_conversation("other__repo", "Different repo")
    check("filtering by repository works",
          len(list_conversations("jiya2406__weatherwise")), 2)
    check("all repositories when unfiltered", len(list_conversations()), 3)

    # ------------------------------------------------------------------
    # History window for Step 11
    # ------------------------------------------------------------------
    turns = get_recent_turns(conversation.id, limit=2)
    check("recent turns limited", len(turns), 2)
    check_true("recent turns are the latest, oldest first",
               turns[0].content == "Where is JWT auth?"
               and turns[1].role == "assistant")

    # ------------------------------------------------------------------
    # Rename, delete, cascade
    # ------------------------------------------------------------------
    renamed = rename_conversation(conversation.id, "DB setup")
    check("rename works", renamed.title, "DB setup")

    with connect(readonly=True) as connection:
        before = connection.execute(
            "SELECT COUNT(*) AS n FROM message_sources"
        ).fetchone()["n"]
    check_true("sources exist before delete", before > 0, note=f"{before} rows")

    delete_conversation(conversation.id)
    check("conversation deleted", len(list_conversations()), 2)
    check("messages cascaded", len(get_messages(conversation.id)), 0)

    with connect(readonly=True) as connection:
        after = connection.execute(
            "SELECT COUNT(*) AS n FROM message_sources"
        ).fetchone()["n"]
    # This is the assertion that proves the foreign_keys pragma is doing its
    # job - without it these rows would survive as orphans.
    check("message_sources cascaded too", after, 0)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    for label, call in (
        ("get_conversation", lambda: get_conversation("missing")),
        ("add_message", lambda: add_message("missing", "user", "hi")),
        ("rename_conversation", lambda: rename_conversation("missing", "x")),
        ("delete_conversation", lambda: delete_conversation("missing")),
    ):
        try:
            call()
            check(f"{label} raises on unknown id", "no error", "ConversationNotFound")
        except ConversationNotFoundError as exc:
            check(f"{label} raises 404", exc.status_code, 404)

    check_true("deleting an unknown repository returns False",
               delete_repository_record("nope") is False)
    check_true("deleting a real repository returns True",
               delete_repository_record("jiya2406__weatherwise") is True)

    # ------------------------------------------------------------------
    # Stats for the dashboard
    # ------------------------------------------------------------------
    delete_conversation(second.id)
    delete_conversation(other.id)
    stats = database_stats()
    print(f"\n      stats: {stats}")
    check("stats report zero conversations", stats["conversations"], 0)
    check_true("stats report a file size", stats["size_bytes"] > 0)
    check("stats report schema version", stats["schema_version"], 1)

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        _TEMP_DIR.cleanup()
