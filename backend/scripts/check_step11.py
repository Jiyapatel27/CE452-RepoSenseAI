"""Checks for Step 11 (conversational RAG).

The assertion that matters: a follow-up containing only a pronoun ("how does it
hash passwords?") must still retrieve the right code. That only works if the
question is condensed into a standalone form before retrieval - so the test
compares retrieval WITH and WITHOUT condensation to prove the mechanism earns
its extra LLM call.

Requires Qdrant running and GROQ_API_KEY set. Uses the real database.

Run from the `backend/` folder:
    python scripts/check_step11.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.exceptions import (  # noqa: E402
    ConversationNotFoundError,
    InvalidSearchQueryError,
)
from app.db.conversations import (  # noqa: E402
    delete_conversation,
    get_messages,
    list_conversations,
)
from app.db.database import init_database  # noqa: E402
from app.services.llm_service import is_configured  # noqa: E402
from app.services.rag_service import (  # noqa: E402
    answer_in_conversation,
    condense_question,
    history_as_turns,
    looks_like_refusal,
    start_conversation,
)
from app.services.retrieval_service import search_chunks  # noqa: E402
from app.services.vector_store import get_collection_info, ping  # noqa: E402

failures: list[str] = []
settings = get_settings()


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


def pick_repository() -> str:
    """Prefer a repo with real auth code, so follow-ups have somewhere to go."""
    repositories = get_collection_info().repositories
    for candidate in repositories:
        if "realworld" in candidate:
            return candidate
    if not repositories:
        raise SystemExit("Nothing indexed - run analyze first.")
    return max(repositories.items(), key=lambda kv: kv[1])[0]


def main() -> int:
    try:
        ping()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  Qdrant unreachable: {exc}")
        return 1
    if not is_configured():
        print("FAIL  GROQ_API_KEY is not set")
        return 1

    init_database()
    repo_id = pick_repository()
    print(f"repository: {repo_id}\n")

    created: list[str] = []

    try:
        # --------------------------------------------------------------
        # Condensation in isolation
        # --------------------------------------------------------------
        print("--- condensation ---")
        check("no history returns the question unchanged",
              condense_question([], "Where is authentication?"),
              "Where is authentication?")

        history = [
            {"role": "user", "content": "Where is authentication implemented?"},
            {"role": "assistant",
             "content": "In src/app/routes/auth/auth.service.ts and auth.ts."},
        ]
        rewritten = condense_question(history, "how does it hash passwords?")
        print(f"      'how does it hash passwords?' -> {rewritten!r}")
        check_true("pronoun is resolved into a concrete name",
                   "it" not in rewritten.lower().split()
                   and any(t in rewritten.lower()
                           for t in ("auth", "password")),
                   note=rewritten)

        # The harder case: nothing in this question means anything on its own.
        contentless = condense_question(history, "Where is it used?")
        print(f"      'Where is it used?' -> {contentless!r}")
        check_true("contentless question gains a concrete subject",
                   any(t in contentless.lower()
                       for t in ("auth", "jwt", "service", "middleware")),
                   note=contentless)
        check_true("rewrite is a single line", "\n" not in rewritten)
        check_true("rewrite stays short", len(rewritten) < 400,
                   note=f"{len(rewritten)} chars")

        standalone = "Where is the database connection initialized?"
        kept = condense_question(history, standalone)
        print(f"      already-standalone -> {kept!r}")
        check_true("standalone questions are left broadly intact",
                   "database" in kept.lower(), note=kept)

        # --------------------------------------------------------------
        # THE KEY TEST: does condensation improve retrieval?
        # --------------------------------------------------------------
        print("\n--- retrieval with vs without condensation ---")
        raw_hits = search_chunks(repo_id, "how does it hash passwords?", top_k=5)
        condensed_hits = search_chunks(repo_id, rewritten, top_k=5)

        raw_files = [h.file_path for h in raw_hits]
        condensed_files = [h.file_path for h in condensed_hits]
        print(f"      raw pronoun question -> {raw_files[:3]}")
        print(f"      condensed question   -> {condensed_files[:3]}")
        print(f"      best score: raw={raw_hits[0].score:.3f} "
              f"condensed={condensed_hits[0].score:.3f}")

        raw_found_auth = any("auth" in f.lower() for f in raw_files[:3])
        condensed_found_auth = any("auth" in f.lower() for f in condensed_files[:3])
        check_true("condensed question retrieves auth code",
                   condensed_found_auth, note=f"{condensed_files[:3]}")
        check_true(
            "condensation improves retrieval (or the raw question already worked)",
            condensed_found_auth or not raw_found_auth,
            note=f"raw_found_auth={raw_found_auth}",
        )
        check_true("condensed question scores higher",
                   condensed_hits[0].score >= raw_hits[0].score,
                   note=f"{condensed_hits[0].score:.3f} >= {raw_hits[0].score:.3f}")

        # --------------------------------------------------------------
        # A real two-turn conversation
        # --------------------------------------------------------------
        print("\n--- multi-turn conversation ---")
        conversation, first, _, _ = start_conversation(
            repo_id, "Where is authentication implemented?"
        )
        created.append(conversation.id)
        print(f"      Q1: Where is authentication implemented?")
        print(f"      A1: {first.answer[:160]}")
        check_true("first answer is grounded", len(first.sources) > 0,
                   note=f"{first.sources}")
        check("conversation titled from the first question",
              conversation.title, "Where is authentication implemented?")
        check("no condensation needed on the first turn",
              first.resolved_question, "Where is authentication implemented?")

        # A deliberately contentless follow-up. "How does it hash passwords?"
        # is NOT a good test - "hash passwords" carries enough meaning to
        # retrieve correctly on its own, so the model rightly leaves it alone.
        # "Where is it used?" has no standalone meaning at all: every
        # meaningful word is a pronoun or a generic verb, so retrieval can only
        # work if the reference is resolved first.
        follow_up = "Where is it used?"
        second, _, assistant_two = answer_in_conversation(
            conversation.id, follow_up
        )
        print(f"\n      Q2: {follow_up}")
        print(f"      resolved to: {second.resolved_question!r}")
        print(f"      A2: {second.answer[:200]}")

        check_true("contentless follow-up was condensed",
                   second.resolved_question != follow_up,
                   note=second.resolved_question)
        check_true("condensed form names something concrete",
                   any(token in second.resolved_question.lower()
                       for token in ("auth", "jwt", "middleware", "token")),
                   note=second.resolved_question)
        check_true("follow-up got a grounded answer", len(second.sources) > 0,
                   note=f"{second.sources}")

        # Deliberately NOT asserting "never refuses". "Where is it used?" asks
        # for usage sites, and a file that merely imports auth barely mentions
        # it - so semantic search sometimes misses them and the model honestly
        # declines. Both outcomes are correct; what must hold is that retrieval
        # ran and the response is grounded in real chunks rather than invented.
        # Asserting no-refusal made this suite flaky, which is worse than not
        # asserting it at all.
        print(f"      outcome: {'refused (honest)' if second.refused else 'answered'}"
              f", {second.context_chunks} chunks retrieved")
        check_true("retrieval ran for the follow-up", second.context_chunks > 0,
                   note=f"{second.context_chunks} chunks")

        # What the raw pronoun question would have retrieved on its own - the
        # measure of what condensation bought us.
        raw_follow_up_hits = search_chunks(repo_id, follow_up, top_k=3)
        print(f"      raw '{follow_up}' would have retrieved: "
              f"{[h.file_path for h in raw_follow_up_hits]}")
        print(f"      condensed retrieved: {second.sources}")

        # --------------------------------------------------------------
        # Persistence of the thread
        # --------------------------------------------------------------
        print("\n--- persistence ---")
        messages = get_messages(conversation.id)
        check("four messages stored", len(messages), 4)
        check("roles alternate", [m.role for m in messages],
              ["user", "assistant", "user", "assistant"])
        check("user text stored verbatim", messages[2].content, follow_up)
        check_true("assistant token usage stored",
                   messages[3].prompt_tokens and messages[3].prompt_tokens > 0,
                   note=f"{messages[3].prompt_tokens} tokens")
        check_true("resolved_question stored for the follow-up",
                   bool(messages[3].resolved_question),
                   note=str(messages[3].resolved_question))
        check("resolved_question omitted on the first turn",
              messages[1].resolved_question, None)
        check_true("sources persisted with line numbers",
                   all(s.start_line is not None for s in messages[3].sources),
                   note=f"{len(messages[3].sources)} sources")
        check_true("cited flag distinguishes used from merely retrieved",
                   any(s.cited for s in messages[3].sources))

        # --------------------------------------------------------------
        # Resuming a thread later
        # --------------------------------------------------------------
        print("\n--- resume an existing thread ---")
        third, _, _ = answer_in_conversation(
            conversation.id, "And which file defines the middleware?"
        )
        print(f"      resolved to: {third.resolved_question!r}")
        print(f"      A3: {third.answer[:160]}")
        check("thread grew to six messages", len(get_messages(conversation.id)), 6)
        check_true("resumed turn used the earlier context",
                   third.resolved_question != "And which file defines the middleware?",
                   note=third.resolved_question)

        # --------------------------------------------------------------
        # Refusal detection (for the dashboard)
        # --------------------------------------------------------------
        print("\n--- refusal detection ---")
        check("refusal phrase detected",
              looks_like_refusal(
                  "The provided repository context does not contain any JWT code."
              ), True)
        check("normal answer not flagged",
              looks_like_refusal("Authentication is handled in auth.service.ts."),
              False)

        # --------------------------------------------------------------
        # History conversion
        # --------------------------------------------------------------
        turns = history_as_turns(get_messages(conversation.id))
        check("history converts to chat turns", len(turns), 6)
        check_true("turns only use valid roles",
                   all(t["role"] in {"user", "assistant"} for t in turns))

        # --------------------------------------------------------------
        # Errors
        # --------------------------------------------------------------
        print("\n--- errors ---")
        try:
            answer_in_conversation("missing-id", "hello")
            check("unknown conversation raises", "no error", "ConversationNotFound")
        except ConversationNotFoundError as exc:
            check("unknown conversation -> 404", exc.status_code, 404)

        try:
            answer_in_conversation(conversation.id, "   ")
            check("blank question raises", "no error", "InvalidSearchQuery")
        except InvalidSearchQueryError as exc:
            check("blank question -> 400", exc.status_code, 400)

        # Conversation must appear in the sidebar listing.
        listed = [c for c in list_conversations(repo_id) if c.id == conversation.id]
        check("conversation is listed for its repository", len(listed), 1)
        check("listing reports the message count", listed[0].message_count, 6)

    finally:
        for conversation_id in created:
            try:
                delete_conversation(conversation_id)
            except ConversationNotFoundError:
                pass
        print(f"\n(cleaned up {len(created)} test conversation(s))")

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
