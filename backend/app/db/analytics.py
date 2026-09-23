"""Dashboard aggregations (Step 12).

All computed in SQL rather than by loading rows into Python, so the dashboard
stays fast as history grows. Every query is read-only.

Dates are stored as ISO-8601 UTC strings, so `substr(created_at, 1, 10)` yields
`YYYY-MM-DD` and groups by day without needing SQLite's date functions - and it
sorts correctly as text.
"""

from __future__ import annotations

import logging
from typing import Any

from app.db.database import connect, loads

logger = logging.getLogger(__name__)


def overview() -> dict[str, Any]:
    """Headline totals for the top of the dashboard."""
    with connect(readonly=True) as connection:
        repositories = connection.execute(
            """SELECT COUNT(*) AS n,
                      COALESCE(SUM(supported_files), 0) AS files,
                      COALESCE(SUM(total_chunks), 0) AS chunks,
                      COALESCE(SUM(indexed), 0) AS indexed
                 FROM repositories"""
        ).fetchone()

        conversations = connection.execute(
            "SELECT COUNT(*) AS n FROM conversations"
        ).fetchone()["n"]

        messages = connection.execute(
            """SELECT
                   COUNT(*) AS total,
                   COALESCE(SUM(role = 'user'), 0) AS questions,
                   COALESCE(SUM(role = 'assistant'), 0) AS answers,
                   COALESCE(SUM(refused), 0) AS refusals,
                   COALESCE(SUM(truncated), 0) AS truncated,
                   COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   AVG(elapsed_ms) AS avg_elapsed_ms,
                   AVG(context_chunks) AS avg_context_chunks
                 FROM messages"""
        ).fetchone()

    answers = messages["answers"] or 0
    refusals = messages["refusals"] or 0

    return {
        "repositories": repositories["n"],
        "repositories_indexed": repositories["indexed"],
        "total_files": repositories["files"],
        "total_chunks": repositories["chunks"],
        "conversations": conversations,
        "questions": messages["questions"],
        "answers": answers,
        "refusals": refusals,
        # The share of questions the system honestly declined. Useful as a
        # quality signal: a high rate usually means retrieval is missing, not
        # that the model is unhelpful.
        "refusal_rate": round(refusals / answers, 3) if answers else 0.0,
        "truncated_answers": messages["truncated"] or 0,
        "prompt_tokens": messages["prompt_tokens"] or 0,
        "completion_tokens": messages["completion_tokens"] or 0,
        "avg_response_ms": round(messages["avg_elapsed_ms"] or 0, 1),
        "avg_context_chunks": round(messages["avg_context_chunks"] or 0, 1),
    }


def activity_by_day(days: int = 30) -> list[dict[str, Any]]:
    """Questions, answers and refusals per day, oldest first."""
    with connect(readonly=True) as connection:
        rows = connection.execute(
            """SELECT substr(created_at, 1, 10) AS day,
                      COALESCE(SUM(role = 'user'), 0) AS questions,
                      COALESCE(SUM(role = 'assistant'), 0) AS answers,
                      COALESCE(SUM(refused), 0) AS refusals
                 FROM messages
             GROUP BY day
             ORDER BY day DESC
                LIMIT ?""",
            (days,),
        ).fetchall()

    # Query descending to take the most recent N days, then flip so charts read
    # left-to-right in time order.
    return [dict(row) for row in reversed(rows)]


def repository_activity() -> list[dict[str, Any]]:
    """Per-repository usage, busiest first.

    Two subtleties, both of which cost a repository its row if ignored:

    * The id list is a UNION of `repositories` and the repository ids found on
      conversations. A repository indexed before Step 10 added persistence has
      no metadata row, but it can still have chat history - starting the query
      `FROM repositories` silently dropped it from the dashboard despite it
      being the most-used repository.
    * The joins are LEFT, so a repository that has been indexed but never asked
      about still appears, with zeros.
    """
    with connect(readonly=True) as connection:
        rows = connection.execute(
            """WITH known AS (
                     SELECT repository_id FROM repositories
                     UNION
                     SELECT DISTINCT repository_id FROM conversations
               )
               SELECT k.repository_id,
                      COALESCE(r.repository, k.repository_id) AS repository,
                      COALESCE(r.supported_files, 0) AS supported_files,
                      COALESCE(r.total_chunks, 0) AS total_chunks,
                      COALESCE(r.indexed, 0) AS indexed,
                      r.analyzed_at AS analyzed_at,
                      -- No metadata row means it predates persistence.
                      (r.repository_id IS NULL) AS metadata_missing,
                      COUNT(DISTINCT c.id) AS conversations,
                      COALESCE(SUM(m.role = 'user'), 0) AS questions,
                      COALESCE(SUM(m.refused), 0) AS refusals,
                      AVG(m.elapsed_ms) AS avg_elapsed_ms
                 FROM known k
            LEFT JOIN repositories r ON r.repository_id = k.repository_id
            LEFT JOIN conversations c ON c.repository_id = k.repository_id
            LEFT JOIN messages m ON m.conversation_id = c.id
             GROUP BY k.repository_id
             ORDER BY questions DESC, r.analyzed_at DESC"""
        ).fetchall()

    return [
        {
            **dict(row),
            "indexed": bool(row["indexed"]),
            "metadata_missing": bool(row["metadata_missing"]),
            "avg_elapsed_ms": round(row["avg_elapsed_ms"] or 0, 1),
        }
        for row in rows
    ]


def language_totals() -> dict[str, int]:
    """Combined language breakdown across every analysed repository.

    `language_counts` is a JSON column, so this is one place where the
    aggregation has to happen in Python.
    """
    with connect(readonly=True) as connection:
        rows = connection.execute(
            "SELECT language_counts FROM repositories"
        ).fetchall()

    totals: dict[str, int] = {}
    for row in rows:
        for language, count in loads(row["language_counts"]).items():
            totals[language] = totals.get(language, 0) + int(count)
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def top_cited_files(limit: int = 10) -> list[dict[str, Any]]:
    """Files the assistant cited most often.

    Filtered to `cited = 1`: a file merely retrieved into the context did not
    contribute to an answer, and counting it would overstate its relevance.
    """
    with connect(readonly=True) as connection:
        rows = connection.execute(
            """SELECT s.file_path,
                      s.language,
                      COUNT(*) AS citations,
                      AVG(s.score) AS avg_score
                 FROM message_sources s
                WHERE s.cited = 1
             GROUP BY s.file_path
             ORDER BY citations DESC, avg_score DESC
                LIMIT ?""",
            (limit,),
        ).fetchall()

    return [
        {**dict(row), "avg_score": round(row["avg_score"] or 0, 3)}
        for row in rows
    ]


def retrieval_quality() -> dict[str, Any]:
    """Similarity-score statistics, split by whether the answer was refused.

    The comparison is the interesting part: if refused answers score no lower
    than successful ones, retrieval is not separating relevant from irrelevant
    and top_k or chunking needs revisiting.
    """
    with connect(readonly=True) as connection:
        overall = connection.execute(
            """SELECT AVG(score) AS avg_score,
                      MIN(score) AS min_score,
                      MAX(score) AS max_score,
                      COUNT(*) AS samples
                 FROM message_sources
                WHERE score IS NOT NULL"""
        ).fetchone()

        # Best score per answer, grouped by outcome.
        split = connection.execute(
            """SELECT m.refused AS refused,
                      AVG(best.score) AS avg_best_score,
                      COUNT(*) AS answers
                 FROM messages m
                 JOIN (SELECT message_id, MAX(score) AS score
                         FROM message_sources
                        WHERE score IS NOT NULL
                     GROUP BY message_id) AS best
                   ON best.message_id = m.id
                WHERE m.role = 'assistant'
             GROUP BY m.refused"""
        ).fetchall()

    by_outcome = {
        ("refused" if row["refused"] else "answered"): {
            "avg_best_score": round(row["avg_best_score"] or 0, 3),
            "answers": row["answers"],
        }
        for row in split
    }

    return {
        "avg_score": round(overall["avg_score"] or 0, 3),
        "min_score": round(overall["min_score"] or 0, 3),
        "max_score": round(overall["max_score"] or 0, 3),
        "samples": overall["samples"],
        "answered": by_outcome.get("answered", {"avg_best_score": 0.0, "answers": 0}),
        "refused": by_outcome.get("refused", {"avg_best_score": 0.0, "answers": 0}),
    }


def recent_questions(limit: int = 15) -> list[dict[str, Any]]:
    """Latest questions with the outcome of each, newest first.

    The assistant reply is matched by "first assistant message after this user
    message in the same thread", which is exactly how turns are written.
    """
    with connect(readonly=True) as connection:
        rows = connection.execute(
            """SELECT q.id,
                      q.content AS question,
                      q.created_at,
                      c.repository_id,
                      c.id AS conversation_id,
                      c.title AS conversation_title,
                      a.refused AS refused,
                      a.elapsed_ms AS elapsed_ms,
                      a.context_chunks AS context_chunks,
                      a.resolved_question AS resolved_question
                 FROM messages q
                 JOIN conversations c ON c.id = q.conversation_id
            LEFT JOIN messages a
                   ON a.conversation_id = q.conversation_id
                  AND a.role = 'assistant'
                  AND a.created_at >= q.created_at
                  AND a.rowid > q.rowid
                  AND NOT EXISTS (
                      SELECT 1 FROM messages mid
                       WHERE mid.conversation_id = q.conversation_id
                         AND mid.role = 'assistant'
                         AND mid.rowid > q.rowid
                         AND mid.rowid < a.rowid
                  )
                WHERE q.role = 'user'
             ORDER BY q.created_at DESC, q.rowid DESC
                LIMIT ?""",
            (limit,),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "question": row["question"],
            "created_at": row["created_at"],
            "repository_id": row["repository_id"],
            "conversation_id": row["conversation_id"],
            "conversation_title": row["conversation_title"],
            "refused": bool(row["refused"]) if row["refused"] is not None else None,
            "elapsed_ms": round(row["elapsed_ms"] or 0, 1),
            "context_chunks": row["context_chunks"],
            "resolved_question": row["resolved_question"],
        }
        for row in rows
    ]
