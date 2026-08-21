"""Checks for Step 6 (semantic search, no LLM).

Runs the API end-to-end through FastAPI's TestClient, so it exercises the real
route, validation and error handling - not just the service function.

Requires Qdrant running (`docker compose up -d` from the project root).

Run from the `backend/` folder:
    python scripts/check_step6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.vector_store import (  # noqa: E402
    ensure_collection,
    get_collection_info,
    ping,
)

failures: list[str] = []
settings = get_settings()
client = TestClient(app)

# The five test questions from the project brief.
BRIEF_QUESTIONS = [
    "Where is authentication implemented?",
    "How is a user created?",
    "Where is the database connection initialized?",
    "How does login work?",
    "Which files handle API requests?",
]


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


def pick_indexed_repository() -> str:
    info = get_collection_info()
    if not info.repositories:
        raise SystemExit(
            "Nothing indexed yet. Run POST /api/repository/analyze then "
            "POST /api/repository/index first."
        )
    return max(info.repositories.items(), key=lambda kv: kv[1])[0]


def post_search(**body):
    return client.post("/api/search", json=body)


def main() -> int:
    # Entering the TestClient as a context manager runs the app's lifespan
    # hook, which preloads the embedding model - exactly what the real server
    # does. Without this, the first search would pay model-init time.
    with client:
        return _run()


def _run() -> int:
    try:
        ping()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  Qdrant unreachable: {exc}")
        print("Start it with `docker compose up -d` from the project root.")
        return 1

    # Makes sure the `language` payload index exists for the filter below.
    ensure_collection()

    repo_id = pick_indexed_repository()
    print(f"searching repository: {repo_id}\n")

    # ------------------------------------------------------------------
    # Response shape matches the brief
    # ------------------------------------------------------------------
    r = post_search(repository_id=repo_id,
                    query="Where is the database connection initialized?")
    check("status 200", r.status_code, 200)
    body = r.json()

    for field in ("repository_id", "query", "top_k", "result_count",
                  "results", "files", "elapsed_ms"):
        check_true(f"response has '{field}'", field in body)

    results = body["results"]
    check("default top_k applied", len(results), settings.search_top_k)

    first = results[0]
    for field in ("file_path", "score", "content"):
        check_true(f"result has '{field}' (required by the brief)", field in first)
    for field in ("start_line", "end_line", "language", "chunk_index"):
        check_true(f"result has '{field}'", field in first)

    check_true("content is non-empty", bool(first["content"].strip()))
    check_true("score in [0,1]", 0.0 <= first["score"] <= 1.0,
               note=f"score={first['score']}")
    check_true(
        "scores are descending",
        all(results[i]["score"] >= results[i + 1]["score"]
            for i in range(len(results) - 1)),
    )
    # The model is preloaded by the app's lifespan hook, so a search should be
    # fast even on the first call - not the ~12s of lazy model init.
    check_true("search is fast (model was preloaded)", body["elapsed_ms"] < 2000,
               note=f"{body['elapsed_ms']} ms")

    # ------------------------------------------------------------------
    # `files` is deduplicated and ordered
    # ------------------------------------------------------------------
    files = body["files"]
    check("files list is deduplicated", len(files), len(set(files)))
    check_true("files ordered by best hit", files[0] == results[0]["file_path"])
    check_true("files subset of result paths",
               set(files) == {r["file_path"] for r in results})
    print(f"      files -> {files}")

    # ------------------------------------------------------------------
    # top_k is honoured and capped
    # ------------------------------------------------------------------
    check("top_k=1 returns 1", len(post_search(
        repository_id=repo_id, query="database", top_k=1).json()["results"]), 1)
    check("top_k=3 returns 3", len(post_search(
        repository_id=repo_id, query="database", top_k=3).json()["results"]), 3)
    check("top_k above the schema max is rejected",
          post_search(repository_id=repo_id, query="x", top_k=999).status_code, 422)

    # ------------------------------------------------------------------
    # min_score filtering
    # ------------------------------------------------------------------
    filtered = post_search(repository_id=repo_id,
                           query="Where is the database connection initialized?",
                           min_score=0.6).json()["results"]
    check_true("min_score drops weak hits",
               all(hit["score"] >= 0.6 for hit in filtered),
               note=f"{len(filtered)} of {settings.search_top_k} survived 0.6")
    impossible = post_search(repository_id=repo_id, query="database",
                             min_score=0.99).json()
    check("min_score=0.99 returns nothing", impossible["result_count"], 0)
    check_true("empty results are not an error", impossible["results"] == []
               and impossible["files"] == [])

    # ------------------------------------------------------------------
    # Language filter
    # ------------------------------------------------------------------
    code_only = post_search(
        repository_id=repo_id,
        query="Which files handle API requests?",
        top_k=3,
        languages=["javascript", "typescript"],
    ).json()
    check_true(
        "language filter returns only those languages",
        all(hit["language"] in {"javascript", "typescript"}
            for hit in code_only["results"]),
        note=f"languages={[h['language'] for h in code_only['results']]}",
    )

    unfiltered = post_search(
        repository_id=repo_id, query="Which files handle API requests?", top_k=3
    ).json()
    print("\n  'Which files handle API requests?' unfiltered vs code-only:")
    for label, body_ in (("unfiltered", unfiltered), ("code only", code_only)):
        paths = [h["file_path"] for h in body_["results"]]
        print(f"    {label:11}: {paths}")

    check_true(
        "language filter excludes markdown/json noise",
        not any(h["language"] in {"markdown", "json"} for h in code_only["results"]),
    )
    check("nonexistent language returns nothing",
          post_search(repository_id=repo_id, query="anything",
                      languages=["cobol"]).json()["result_count"], 0)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    r = post_search(repository_id="nope__missing", query="anything")
    check("unindexed repository -> 409", r.status_code, 409)
    check("error code", r.json()["error"], "repository_not_indexed")

    check("blank query -> 400",
          post_search(repository_id=repo_id, query="   ").status_code, 400)
    check("missing query -> 422",
          client.post("/api/search", json={"repository_id": repo_id}).status_code, 422)

    # ------------------------------------------------------------------
    # Retrieval quality on the brief's five questions
    # ------------------------------------------------------------------
    print("\n--- the brief's five test questions ---")
    scores = {}
    for question in BRIEF_QUESTIONS:
        body = post_search(repository_id=repo_id, query=question, top_k=3).json()
        top = body["results"][0]
        scores[question] = top["score"]
        print(f"\n  Q: {question}")
        for hit in body["results"]:
            print(f"     {hit['score']:.3f}  {hit['file_path']}"
                  f":{hit['start_line']}-{hit['end_line']}")

    check_true("every question returns hits", all(s > 0 for s in scores.values()))

    # weatherwise has no auth/login code, so those two questions should score
    # below the ones the repository can actually answer.
    if "weatherwise" in repo_id:
        answerable = scores["Where is the database connection initialized?"]
        unanswerable = max(scores["Where is authentication implemented?"],
                           scores["How does login work?"])
        print(f"\n  answerable (db)   : {answerable:.3f}")
        print(f"  unanswerable (auth): {unanswerable:.3f}")
        check_true(
            "unanswerable questions score lower than answerable ones",
            unanswerable < answerable,
            note=f"{unanswerable:.3f} < {answerable:.3f}",
        )
        print("      NOTE: the margin is small - Step 7 must rely on prompt")
        print("      grounding, not a score threshold, to say 'I don't know'.")

    # ------------------------------------------------------------------
    # Same query is stable
    # ------------------------------------------------------------------
    a = post_search(repository_id=repo_id, query="how is weather data fetched")
    b = post_search(repository_id=repo_id, query="how is weather data fetched")
    check("repeated search is deterministic",
          [h["chunk_id"] for h in a.json()["results"]],
          [h["chunk_id"] for h in b.json()["results"]])

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
