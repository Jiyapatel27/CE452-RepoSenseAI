"""Checks for Step 5 (Qdrant storage).

Requires Qdrant to be running:
    docker compose up -d      (from the project root)

Run from the `backend/` folder:
    python scripts/check_step5.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.exceptions import VectorStoreUnavailableError  # noqa: E402
from app.services.chunker_service import iter_chunks  # noqa: E402
from app.services.embedding_service import count_tokens, embed_query  # noqa: E402
from app.services.github_service import scan_repository_files  # noqa: E402
from app.services.parser_service import iter_parsed_files  # noqa: E402
from app.services.vector_store import (  # noqa: E402
    REPOSITORY_FIELD,
    count_repository_points,
    delete_repository,
    ensure_collection,
    get_client,
    get_collection_info,
    index_chunks,
    ping,
)

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
    print(f"{'PASS ' if ok else 'FAIL '} {label}"
          + (f": {note}" if note else "")
          + ("" if ok else " (expected True)"))
    if not ok:
        failures.append(label)


def pick_repo() -> tuple[str, Path]:
    """Use weatherwise if cloned, else any repo under the storage dir."""
    preferred = settings.repo_storage_dir / "jiya2406__weatherwise"
    if preferred.exists():
        return preferred.name, preferred
    for path in sorted(settings.repo_storage_dir.glob("*")):
        if path.is_dir():
            return path.name, path
    raise SystemExit(
        "No cloned repository found. Run POST /api/repository/analyze first."
    )


def main() -> int:
    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------
    print(f"Qdrant URL: {settings.qdrant_url}")
    try:
        ping()
        print("PASS  Qdrant is reachable")
    except VectorStoreUnavailableError as exc:
        print(f"FAIL  Qdrant unreachable: {exc.message}")
        print(f"      {exc.detail}")
        print("\nStart it with `docker compose up -d` from the project root.")
        return 1

    # ------------------------------------------------------------------
    # Collection setup is idempotent
    # ------------------------------------------------------------------
    name = ensure_collection()
    check("collection name", name, settings.qdrant_collection)
    ensure_collection()  # must not raise on a second call
    print("PASS  ensure_collection is idempotent")

    info = get_collection_info()
    check("collection exists", info.exists, True)
    check("vector dimension is 384", info.dimension, 384)
    check("distance is cosine", (info.distance or "").lower(), "cosine")

    # The payload index is what keeps repository filtering fast.
    raw = get_client().get_collection(name)
    schema = raw.payload_schema or {}
    check_true(
        f"payload index on '{REPOSITORY_FIELD}' exists",
        REPOSITORY_FIELD in schema,
        note=f"indexed fields: {sorted(schema)}",
    )

    # ------------------------------------------------------------------
    # Index a real repository
    # ------------------------------------------------------------------
    repo_id, repo_root = pick_repo()
    print(f"\n--- indexing {repo_id} from {repo_root.name} ---")

    scan = scan_repository_files(
        repo_root,
        max_file_size_bytes=settings.max_file_size_bytes,
        max_files=settings.max_files_per_repo,
    )

    def chunk_stream():
        return iter_chunks(
            iter_parsed_files(
                repo_root,
                scan.files,
                # Human-readable name vs the id used for filtering - the real
                # API passes these separately too.
                repository=repo_id.split("__")[-1],
                repository_id=repo_id,
            ),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
            length_function=count_tokens,
        )

    expected_chunks = sum(1 for _ in chunk_stream())
    print(f"repository produces {expected_chunks} chunks")

    started = time.perf_counter()
    result = index_chunks(repo_id, chunk_stream())
    elapsed = time.perf_counter() - started
    print(f"indexed in {elapsed:.1f}s "
          f"({result.chunks_indexed / max(elapsed, 0.01):.1f} chunks/sec)")

    check("all chunks indexed", result.chunks_indexed, expected_chunks)
    check("dimension reported", result.dimension, 384)
    check_true("used more than one batch or fits in one",
               result.batches >= 1, note=f"batches={result.batches}")
    check("stored point count matches",
          count_repository_points(repo_id), expected_chunks)

    # ------------------------------------------------------------------
    # Re-indexing must not duplicate
    # ------------------------------------------------------------------
    second = index_chunks(repo_id, chunk_stream())
    check("re-index deletes old vectors first",
          second.vectors_deleted_first, expected_chunks)
    check("re-index leaves the same count",
          count_repository_points(repo_id), expected_chunks)
    print("PASS  re-indexing is idempotent (no duplicates)")

    # ------------------------------------------------------------------
    # Payload completeness - Steps 6/7 rely on these fields
    # ------------------------------------------------------------------
    points, _ = get_client().scroll(
        collection_name=name, limit=1, with_payload=True, with_vectors=True
    )
    payload = points[0].payload or {}
    for field in (
        REPOSITORY_FIELD, "repository", "file_path", "file_name", "language",
        "chunk_index", "content", "start_line", "end_line",
    ):
        check_true(f"payload has '{field}'", field in payload)
    check_true("payload content is non-empty", bool(payload.get("content", "").strip()))
    check("stored vector has 384 dims", len(points[0].vector), 384)

    # ------------------------------------------------------------------
    # Repository isolation via the payload filter
    # ------------------------------------------------------------------
    other_id = f"{repo_id}__isolation_probe"
    probe = list(chunk_stream())[:3]
    for chunk in probe:
        chunk.repository_id = other_id
        chunk.chunk_id = chunk.chunk_id.replace(chunk.chunk_id[0], "0", 1)
    index_chunks(other_id, iter(probe))

    check("probe repo stored separately", count_repository_points(other_id), 3)
    check("original repo count unchanged",
          count_repository_points(repo_id), expected_chunks)

    # ------------------------------------------------------------------
    # Filtered search returns only the requested repository
    # ------------------------------------------------------------------
    from qdrant_client import models

    query_vector = embed_query("How is the database connection initialized?")
    hits = get_client().query_points(
        collection_name=name,
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key=REPOSITORY_FIELD, match=models.MatchValue(value=repo_id)
            )]
        ),
        limit=5,
        with_payload=True,
    ).points

    print("\n--- filtered search results ---")
    for hit in hits:
        p = hit.payload or {}
        print(f"  {hit.score:.3f}  {p.get('file_path')}"
              f":{p.get('start_line')}-{p.get('end_line')}")

    check_true("search returns hits", len(hits) > 0)
    check_true(
        "every hit belongs to the filtered repository",
        all((h.payload or {}).get(REPOSITORY_FIELD) == repo_id for h in hits),
    )
    check_true("scores are descending",
               all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)))
    check_true(
        "cosine scores are in [-1, 1]",
        all(-1.0001 <= h.score <= 1.0001 for h in hits),
    )

    # ------------------------------------------------------------------
    # Cleanup of the probe repository
    # ------------------------------------------------------------------
    removed = delete_repository(other_id)
    check("probe cleanup removed its vectors", removed, 3)
    check("probe repo is gone", count_repository_points(other_id), 0)
    check("original repo still intact",
          count_repository_points(repo_id), expected_chunks)

    final = get_collection_info()
    print(f"\ncollection '{final.collection}': {final.points_count} points")
    print(f"per repository: {final.repositories}")

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
