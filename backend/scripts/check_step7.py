"""Checks for Step 7 (RAG + Groq).

The interesting assertions are not "does it answer" but:
  * does every cited file actually exist in the repository?
  * does it REFUSE when the repository has no such code?

Requires Qdrant running and GROQ_API_KEY set in backend/.env.

Run from the `backend/` folder:
    python scripts/check_step7.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.llm_service import is_configured  # noqa: E402
from app.services.rag_service import (  # noqa: E402
    NO_CONTEXT_ANSWER,
    build_context,
    normalise_answer,
)
from app.services.retrieval_service import SearchHit  # noqa: E402
from app.services.vector_store import get_collection_info, ping  # noqa: E402

failures: list[str] = []
settings = get_settings()
client = TestClient(app)

REFUSAL_MARKERS = (
    "does not", "doesn't", "no authentication", "not contain", "not present",
    "not shown", "not implemented", "no login", "cannot find", "could not find",
    "not include", "no code", "not appear", "nothing in", "absent", "lacks",
    "no such", "not found",
)


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


def ask(**body):
    return client.post("/api/chat", json=body)


def indexed_file_paths(repository_id: str) -> set[str]:
    """Every distinct file_path stored in Qdrant for a repository.

    Ground truth for "does this file exist?", read from the vector store so the
    check does not depend on the in-memory repository record.
    """
    from qdrant_client import models

    from app.services.vector_store import REPOSITORY_FIELD, get_client

    paths: set[str] = set()
    offset = None
    while True:
        points, offset = get_client().scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key=REPOSITORY_FIELD,
                    match=models.MatchValue(value=repository_id),
                )]
            ),
            limit=500,
            offset=offset,
            with_payload=["file_path"],
            with_vectors=False,
        )
        paths.update((p.payload or {}).get("file_path", "") for p in points)
        if offset is None:
            break
    return {p for p in paths if p}


def main() -> int:
    with client:
        return _run()


def _run() -> int:
    try:
        ping()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  Qdrant unreachable: {exc}")
        return 1

    if not is_configured():
        print("FAIL  GROQ_API_KEY is not set in backend/.env")
        return 1

    info = get_collection_info()
    if not info.repositories:
        print("FAIL  nothing indexed - run analyze + index first")
        return 1

    # The refusal assertions need a repository that genuinely has NO auth code,
    # otherwise "refuses to answer about JWT" would be the wrong expectation.
    # weatherwise qualifies; anything else, we skip that section rather than
    # assert something false.
    auth_free = [r for r in info.repositories if "weatherwise" in r]
    repo_id = auth_free[0] if auth_free else max(
        info.repositories.items(), key=lambda kv: kv[1]
    )[0]
    test_refusal = bool(auth_free)
    refusal_state = "enabled" if test_refusal else "SKIPPED (needs an auth-free repo)"
    print(f"repository: {repo_id}   model: {settings.groq_model}")
    print(f"refusal assertions: {refusal_state}\n")

    # ------------------------------------------------------------------
    # Offline: context builder (no API calls)
    # ------------------------------------------------------------------
    fake = [
        SearchHit("a/b.js", 0.9, "const x = 1;", "b.js", "javascript", 0, 1, 3, "id1"),
        SearchHit("c/d.py", 0.8, "def hi(): pass", "d.py", "python", 0, 10, 12, "id2"),
    ]
    text, used, truncated = build_context(fake, max_tokens=10_000)
    check("context uses all chunks when they fit", used, 2)
    check("context not truncated", truncated, False)
    check_true("context labels file paths", "a/b.js" in text and "c/d.py" in text)
    check_true("context labels line ranges", "lines 1-3" in text
               and "lines 10-12" in text)
    check("answer normalisation strips exotic spacing/hyphens",
          normalise_answer("lines 1‑018 and a–b"),
          "lines 1-018 and a-b")
    check("answer normalisation strips citation artifacts",
          normalise_answer("uses HS256【2†L1-L15】 for signing."),
          "uses HS256 for signing.")
    check("citation stripping tidies stray spacing",
          normalise_answer("the algorithm 【1†L5】."),
          "the algorithm.")
    check_true("context includes the code", "const x = 1;" in text)
    check_true("context numbers the chunks", "[1]" in text and "[2]" in text)

    # A tiny budget must keep at least one whole chunk, never half of one.
    _, used_small, truncated_small = build_context(fake, max_tokens=5)
    check("tiny budget keeps one whole chunk", used_small, 1)
    check("tiny budget reports truncation", truncated_small, True)

    # ------------------------------------------------------------------
    # An answerable question
    # ------------------------------------------------------------------
    r = ask(repository_id=repo_id,
            question="Where is the database connection initialized?")
    check("status 200", r.status_code, 200)
    body = r.json()

    for f in ("answer", "sources", "model", "context_chunks", "prompt_tokens"):
        check_true(f"response has '{f}'", f in body)

    print(f"\n  Q: Where is the database connection initialized?")
    print(f"  A: {body['answer'][:400]}")
    print(f"  sources: {body['sources']}")
    print(f"  tokens: {body['prompt_tokens']} in / {body['completion_tokens']} out"
          f"  ({body['elapsed_ms']} ms)\n")

    check_true("answer is non-empty", bool(body["answer"].strip()))
    check_true("answer carries no citation artifacts",
               "【" not in body["answer"] and "】" not in body["answer"])
    check_true("answer is ASCII-normalised",
               not any(ord(c) in (0x202F, 0x00A0, 0x2011) for c in body["answer"]))
    check("answer was not truncated", body["truncated_answer"], False)
    check_true("llm was called", body["llm_called"])
    check_true("sources returned", len(body["sources"]) > 0)
    check_true("context chunks used", body["context_chunks"] > 0)
    check_true("token usage reported", body["prompt_tokens"] > 0)
    check_true("model echoed back", bool(body["model"]))

    # The whole point of grounding: cited files must be real.
    # Read the ground truth from Qdrant payloads rather than the in-memory
    # store, so this check works after a restart too.
    real_files = indexed_file_paths(repo_id)
    print(f"  ({len(real_files)} distinct files indexed)")

    bogus = [s for s in body["sources"] if s not in real_files]
    check("every source is a real repository file", bogus, [])

    check_true(
        "sources are only files the answer cites",
        all(s in body["answer"] or s.rsplit("/", 1)[-1] in body["answer"]
            for s in body["sources"]),
        note=f"sources={body['sources']}",
    )

    # Any path-like string in the prose must correspond to a real file.
    mentioned = set(
        re.findall(r"[\w./-]+\.(?:js|jsx|ts|tsx|py|json|md)", body["answer"])
    )
    invented = [
        m for m in mentioned
        if not any(rf.endswith(m) or m.endswith(rf) for rf in real_files)
    ]
    check("answer invents no file paths", invented, [])
    print(f"  files mentioned in the answer: {sorted(mentioned)}")

    # Line numbers cited must fall inside the chunks that were supplied.
    supplied = {(c["file_path"], c["start_line"], c["end_line"])
                for c in body["context"]}
    max_line = max(end for _, _, end in supplied)
    # \s (not a literal space): the model may still use exotic spacing, and a
    # regex that silently matches nothing is worse than no check at all.
    cited_lines = [int(n) for n in re.findall(r"lines?\s*(\d+)", body["answer"])]
    check_true("line-number check actually found citations",
               bool(cited_lines), note=f"cited={cited_lines}")
    implausible = [n for n in cited_lines if n > max_line]
    check("cited line numbers are within the supplied chunks", implausible, [])

    # ------------------------------------------------------------------
    # THE KEY TEST: refuse when the repository has no such code
    # ------------------------------------------------------------------
    if test_refusal:
        print(f"--- refusal behaviour ({repo_id} has no auth code) ---")
        refusal_ok = 0
        refusal_total = 0
        for question in (
            "Where is JWT authentication implemented?",
            "How does user login work?",
            "How are passwords hashed?",
        ):
            body2 = ask(repository_id=repo_id, question=question).json()
            answer = body2["answer"]
            refused = any(m in answer.lower() for m in REFUSAL_MARKERS)
            refusal_total += 1
            refusal_ok += 1 if refused else 0
            print(f"\n  Q: {question}")
            print(f"  A: {answer[:300]}")
            print(f"  -> {'REFUSED (good)' if refused else 'DID NOT REFUSE (bad)'}")

        check("refuses every unanswerable question", refusal_ok, refusal_total)
    else:
        print("--- refusal behaviour SKIPPED (no auth-free repo indexed) ---")

    # ------------------------------------------------------------------
    # No context at all -> LLM skipped entirely
    # ------------------------------------------------------------------
    # min_score is not exposed on /chat, so drive the service directly.
    from app.services.rag_service import answer_question
    import app.services.rag_service as rag

    original = rag.search_chunks
    rag.search_chunks = lambda *a, **k: []          # simulate zero hits
    try:
        empty = answer_question(repo_id, "anything at all")
    finally:
        rag.search_chunks = original

    check("no hits -> canned answer", empty.answer, NO_CONTEXT_ANSWER)
    check("no hits -> llm skipped", empty.llm_called, False)
    check("no hits -> no sources", empty.sources, [])

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    check("unindexed repository -> 409",
          ask(repository_id="nope__missing", question="hi").status_code, 409)
    check("blank question -> 400",
          ask(repository_id=repo_id, question="   ").status_code, 400)
    check("missing question -> 422",
          client.post("/api/chat", json={"repository_id": repo_id}).status_code, 422)
    check("top_k above max -> 422",
          ask(repository_id=repo_id, question="hi", top_k=99).status_code, 422)

    # ------------------------------------------------------------------
    # Language filter reaches the LLM context
    # ------------------------------------------------------------------
    filtered = ask(repository_id=repo_id,
                   question="Which files handle API requests?",
                   languages=["javascript"]).json()
    check_true(
        "language filter applied to context",
        all(c["language"] == "javascript" for c in filtered["context"]),
        note=f"sources={filtered['sources']}",
    )

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
