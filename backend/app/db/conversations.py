"""Conversation and message storage (Step 10).

Data access only - no prompting or retrieval logic. The conversational RAG in
Step 11 reads history from here and writes turns back.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import ConversationNotFoundError
from app.db.database import connect, utc_now

logger = logging.getLogger(__name__)

# A conversation is named after its first question. Long enough to recognise a
# thread in the sidebar, short enough not to wrap.
TITLE_MAX_CHARS = 60


@dataclass
class MessageSource:
    file_path: str
    score: float | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None
    cited: bool = True


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: float | None = None
    context_chunks: int | None = None
    refused: bool = False
    truncated: bool = False
    resolved_question: str | None = None
    sources: list[MessageSource] = field(default_factory=list)


@dataclass
class Conversation:
    id: str
    repository_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message: str | None = None


def make_title(question: str) -> str:
    """Derive a sidebar title from the opening question."""
    cleaned = " ".join(question.strip().split())
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned or "New conversation"
    return cleaned[: TITLE_MAX_CHARS - 1].rstrip() + "…"


# ----------------------------------------------------------------------
# Conversations
# ----------------------------------------------------------------------
def create_conversation(repository_id: str, title: str) -> Conversation:
    now = utc_now()
    conversation_id = str(uuid.uuid4())

    with connect() as connection:
        connection.execute(
            """INSERT INTO conversations
                   (id, repository_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, repository_id, title, now, now),
        )

    logger.info("Created conversation %s for %s", conversation_id, repository_id)
    return Conversation(
        id=conversation_id,
        repository_id=repository_id,
        title=title,
        created_at=now,
        updated_at=now,
    )


def _row_to_conversation(row: Any) -> Conversation:
    keys = row.keys()
    return Conversation(
        id=row["id"],
        repository_id=row["repository_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"] if "message_count" in keys else 0,
        last_message=row["last_message"] if "last_message" in keys else None,
    )


def list_conversations(
    repository_id: str | None = None, *, limit: int = 100, offset: int = 0
) -> list[Conversation]:
    """Conversations, most recently updated first.

    Message counts and last-message previews come from a single query with
    joins rather than a follow-up query per conversation - the sidebar would
    otherwise fire N+1 queries every time it renders.
    """
    where = "WHERE c.repository_id = ?" if repository_id else ""
    params: list[Any] = [repository_id] if repository_id else []

    query = f"""
        SELECT c.*,
               (SELECT COUNT(*) FROM messages m
                 WHERE m.conversation_id = c.id) AS message_count,
               (SELECT m.content FROM messages m
                 WHERE m.conversation_id = c.id
                 ORDER BY m.created_at DESC, m.rowid DESC LIMIT 1) AS last_message
          FROM conversations c
          {where}
      ORDER BY c.updated_at DESC
         LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    with connect(readonly=True) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_conversation(row) for row in rows]


def get_conversation(conversation_id: str) -> Conversation:
    with connect(readonly=True) as connection:
        row = connection.execute(
            """SELECT c.*,
                      (SELECT COUNT(*) FROM messages m
                        WHERE m.conversation_id = c.id) AS message_count
                 FROM conversations c WHERE c.id = ?""",
            (conversation_id,),
        ).fetchone()

    if row is None:
        raise ConversationNotFoundError(
            f"No conversation with id '{conversation_id}'."
        )
    return _row_to_conversation(row)


def rename_conversation(conversation_id: str, title: str) -> Conversation:
    with connect() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, utc_now(), conversation_id),
        )
        if cursor.rowcount == 0:
            raise ConversationNotFoundError(
                f"No conversation with id '{conversation_id}'."
            )
    return get_conversation(conversation_id)


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation. Messages and sources cascade."""
    with connect() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )
        if cursor.rowcount == 0:
            raise ConversationNotFoundError(
                f"No conversation with id '{conversation_id}'."
            )
    logger.info("Deleted conversation %s", conversation_id)


# ----------------------------------------------------------------------
# Messages
# ----------------------------------------------------------------------
def add_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    elapsed_ms: float | None = None,
    context_chunks: int | None = None,
    refused: bool = False,
    truncated: bool = False,
    resolved_question: str | None = None,
    sources: list[MessageSource] | None = None,
) -> Message:
    """Append a message and bump the conversation's updated_at.

    Both writes share one transaction, so a thread can never end up with a
    message whose parent still claims an older timestamp - which would sort it
    wrongly in the sidebar.
    """
    now = utc_now()
    message_id = str(uuid.uuid4())

    with connect() as connection:
        exists = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if exists is None:
            raise ConversationNotFoundError(
                f"No conversation with id '{conversation_id}'."
            )

        connection.execute(
            """INSERT INTO messages
                   (id, conversation_id, role, content, created_at, model,
                    prompt_tokens, completion_tokens, elapsed_ms,
                    context_chunks, refused, truncated, resolved_question)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message_id, conversation_id, role, content, now, model,
                prompt_tokens, completion_tokens, elapsed_ms, context_chunks,
                int(refused), int(truncated), resolved_question,
            ),
        )

        for source in sources or []:
            connection.execute(
                """INSERT INTO message_sources
                       (message_id, file_path, score, start_line, end_line,
                        language, cited)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id, source.file_path, source.score,
                    source.start_line, source.end_line, source.language,
                    int(source.cited),
                ),
            )

        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )

    return Message(
        id=message_id,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=now,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_ms=elapsed_ms,
        context_chunks=context_chunks,
        refused=refused,
        truncated=truncated,
        resolved_question=resolved_question,
        sources=sources or [],
    )


def get_messages(conversation_id: str) -> list[Message]:
    """Every message in a thread, oldest first, with its sources attached."""
    with connect(readonly=True) as connection:
        rows = connection.execute(
            """SELECT * FROM messages
                WHERE conversation_id = ?
             ORDER BY created_at ASC, rowid ASC""",
            (conversation_id,),
        ).fetchall()

        if not rows:
            return []

        # One query for all sources, then group in Python - avoids a query per
        # message when rendering a long thread.
        placeholders = ",".join("?" * len(rows))
        source_rows = connection.execute(
            f"""SELECT * FROM message_sources
                 WHERE message_id IN ({placeholders})""",  # noqa: S608
            [row["id"] for row in rows],
        ).fetchall()

    grouped: dict[str, list[MessageSource]] = {}
    for source in source_rows:
        grouped.setdefault(source["message_id"], []).append(
            MessageSource(
                file_path=source["file_path"],
                score=source["score"],
                start_line=source["start_line"],
                end_line=source["end_line"],
                language=source["language"],
                cited=bool(source["cited"]),
            )
        )

    return [
        Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            model=row["model"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            elapsed_ms=row["elapsed_ms"],
            context_chunks=row["context_chunks"],
            refused=bool(row["refused"]),
            truncated=bool(row["truncated"]),
            resolved_question=row["resolved_question"],
            sources=grouped.get(row["id"], []),
        )
        for row in rows
    ]


def get_recent_turns(conversation_id: str, limit: int = 6) -> list[Message]:
    """The last `limit` messages, oldest first.

    Step 11 feeds these to the model as conversation history. Bounded because
    an unbounded thread would eventually blow the context window, and old turns
    add little once a topic has moved on.
    """
    messages = get_messages(conversation_id)
    return messages[-limit:]
