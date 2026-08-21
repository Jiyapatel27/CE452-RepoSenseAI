"""Checks for Step 8 (backend integration).

Three things are verified:
  1. one /analyze call can clone -> parse -> chunk -> embed -> index
  2. repository records are restored from Qdrant after a "restart"
  3. /health reports each dependency separately

Requires Qdrant running and GROQ_API_KEY set in backend/.env.

Run from the `backend/` folder:
    python scripts/check_step8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.services.repository_store import (  # noqa: E402
    repository_store,
    restore_from_vector_store,
)
from app.services.vector_store import (  # noqa: E402
    count_repository_points,
    list_repository_summaries,
    ping,
)

failures: list[str] = []
settings = get_settings()
client = TestClient(app)

REPO_URL = "https://github.com/Jiya2406/weatherwise"
REPO_ID = "jiya2406__weatherwise"


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
    with client:
        return _run()


def _run() -> int:
    try:
        ping()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  Qdrant unreachable: {exc}")
        return 1

    # ------------------------------------------------------------------
    # 1. One-call pipeline
    # ------------------------------------------------------------------
    print("--- one-call analyze (auto_index) ---")
    r = client.post("/api/repository/analyze",
                    json={"repository_url": REPO_URL, "auto_index": True})
    check("analyze status 200", r.status_code, 200)
    body = r.json()
    print(f"      files={body['supported_files']} "
          f"chunks={body['chunk_stats']['total_chunks']} "
          f"indexed={body['indexed']} written={body['chunks_indexed']}")

    check("repository_id", body["repository_id"], REPO_ID)
    check_true("indexed in the same call", body["indexed"])
    check("chunks written matches chunk count",
          body["chunks_indexed"], body["chunk_stats"]["total_chunks"])
    check("status reflects indexing", body["status"], "indexed")
    check("vectors actually in Qdrant",
          count_repository_points(REPO_ID), body["chunk_stats"]["total_chunks"])

    # Searchable immediately - no second call needed.
    s = client.post("/api/search",
                    json={"repository_id": REPO_ID, "query": "database connection"})
    check("searchable straight after analyze", s.status_code, 200)
    check_true("search returns hits", len(s.json()["results"]) > 0)

    # ------------------------------------------------------------------
    # 2. auto_index=False keeps the old behaviour
    # ------------------------------------------------------------------
    print("\n--- analyze with auto_index=False ---")
    r2 = client.post("/api/repository/analyze",
                     json={"repository_url": REPO_URL, "auto_index": False})
    body2 = r2.json()
    check("not indexed when auto_index is false", body2["indexed"], False)
    check("nothing written", body2["chunks_indexed"], 0)
    check("chunk stats still computed",
          body2["chunk_stats"]["total_chunks"], body["chunk_stats"]["total_chunks"])
    # The vectors from the earlier run are untouched - analyze does not wipe them.
    check("existing vectors left alone",
          count_repository_points(REPO_ID), body["chunk_stats"]["total_chunks"])

    # ------------------------------------------------------------------
    # 3. Restore after a restart
    # ------------------------------------------------------------------
    print("\n--- restore from Qdrant (simulated restart) ---")
    before = {rec.repository_id for rec in repository_store.list_all()}
    check_true("records exist before the wipe", bool(before), note=f"{sorted(before)}")

    # Simulate a restart: drop the in-memory store, keep Qdrant.
    with repository_store._lock:  # noqa: SLF001 - test reaches in deliberately
        repository_store._records.clear()  # noqa: SLF001
    check("store is empty after the wipe", len(repository_store.list_all()), 0)

    # Before restoring, the API should admit it knows nothing.
    check("status 404 while the store is empty",
          client.get(f"/api/repository/status?repository_id={REPO_ID}").status_code,
          404)
    # ...but search still works, because it reads Qdrant directly.
    check("search still works with an empty store",
          client.post("/api/search",
                      json={"repository_id": REPO_ID, "query": "database"}).status_code,
          200)

    restored = restore_from_vector_store()
    expected_repos = len(list_repository_summaries())
    check("restored every repository in Qdrant", restored, expected_repos)

    record = repository_store.get(REPO_ID)
    check_true("weatherwise restored", record is not None)
    check("restored record is marked as such", record.status, "restored")
    check("restored record is indexed", record.indexed, True)
    check("restored chunk count matches Qdrant",
          record.total_chunks, count_repository_points(REPO_ID))
    check_true("restored file list is populated",
               len(record.files) > 0, note=f"{len(record.files)} files")
    check_true("restored languages populated",
               bool(record.language_counts), note=f"{record.language_counts}")
    check_true("restored local_path points at the clone",
               record.repo_root.name == REPO_ID)

    # Ingestion stats are genuinely unknown after a restore - they must read as
    # zero rather than being invented.
    check("unknown scan count is 0, not faked", record.total_files_scanned, 0)
    check("unknown parse stats are 0", record.parse_summary.files_parsed, 0)

    # Endpoints that need the record work again.
    check("status works after restore",
          client.get(f"/api/repository/status?repository_id={REPO_ID}").status_code,
          200)
    check("files endpoint works after restore",
          client.get(f"/api/repository/files?repository_id={REPO_ID}").status_code,
          200)
    chunks = client.get(f"/api/repository/chunks?repository_id={REPO_ID}&limit=2")
    check("chunks endpoint works after restore", chunks.status_code, 200)
    check_true("restored record can re-chunk from disk",
               chunks.json()["total"] > 0,
               note=f"{chunks.json()['total']} chunks")

    # Restoring twice must not duplicate or clobber.
    again = restore_from_vector_store()
    check("second restore is a no-op", again, 0)

    # ------------------------------------------------------------------
    # 4. /health reports dependencies separately
    # ------------------------------------------------------------------
    print("\n--- /health ---")
    h = client.get("/health").json()
    print(f"      status={h['status']} ready_for_chat={h['ready_for_chat']} "
          f"qdrant={h['qdrant']['reachable']} points={h['qdrant']['points']}")

    for key in ("status", "ready_for_chat", "qdrant", "groq", "embeddings"):
        check_true(f"health has '{key}'", key in h)
    check("health reports qdrant reachable", h["qdrant"]["reachable"], True)
    check("health reports groq configured", h["groq"]["configured"], True)
    check("health reports the model", h["groq"]["model"], settings.groq_model)
    check_true("health counts stored points", h["qdrant"]["points"] > 0)
    check("health is ready for chat", h["ready_for_chat"], True)
    check_true("health lists indexed repositories",
               REPO_ID in h["qdrant"]["repositories"])

    # ------------------------------------------------------------------
    # 5. Full pipeline end to end, one question
    # ------------------------------------------------------------------
    print("\n--- end to end: analyze -> chat ---")
    ans = client.post("/api/chat", json={
        "repository_id": REPO_ID,
        "question": "Where is the database connection initialized?",
    }).json()
    print(f"  A: {ans['answer'][:200]}")
    check_true("chat answers after the one-call pipeline",
               bool(ans["answer"].strip()) and ans["llm_called"])
    check_true("chat cites real sources", len(ans["sources"]) > 0,
               note=f"{ans['sources']}")

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
