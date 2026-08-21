"""Offline checks for Step 4 (local embeddings).

Beyond mechanical checks (dimension, normalisation, determinism) this verifies
the two things that silently break retrieval:

  * the BGE query/passage asymmetry is actually applied
  * real chunks fit inside the model's 512-token limit, so nothing is truncated

Finally it runs a real retrieval test over the cloned weatherwise repo using
plain cosine similarity - no Qdrant needed - which previews Step 6 and proves
the embeddings are semantically useful, not merely well-shaped.

First run downloads the model (~130 MB) and needs internet.

Run from the `backend/` folder:
    python scripts/check_step4.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

failures: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"{'PASS ' if ok else 'FAIL '} {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool, note: str = "") -> None:
    ok = bool(actual)
    print(f"{'PASS ' if ok else 'FAIL '} {label}{f': {note}' if note else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    import numpy as np

    from app.config import get_settings
    from app.services.embedding_service import (
        count_tokens,
        embed_documents,
        embed_query,
        get_model_info,
    )

    settings = get_settings()

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------
    print("Loading model (first run downloads ~130 MB)...")
    started = time.perf_counter()
    info = get_model_info()
    print(f"loaded in {time.perf_counter() - started:.1f}s\n")

    print(f"model  : {info.model_name}")
    print(f"dim    : {info.dimension}")
    print(f"max len: {info.max_sequence_length} tokens")
    print(f"device : {info.device}\n")

    check("model name", info.model_name, "BAAI/bge-small-en-v1.5")
    check("dimension is 384", info.dimension, 384)
    check("max sequence length is 512", info.max_sequence_length, 512)

    # ------------------------------------------------------------------
    # Vector shape / normalisation / determinism
    # ------------------------------------------------------------------
    texts = [
        "const connectDB = async () => { await mongoose.connect(uri); };",
        "def add(a, b):\n    return a + b",
        "# Weather app README",
    ]
    vectors = embed_documents(texts)

    check("one vector per input", len(vectors), 3)
    check_true("every vector has 384 floats",
               all(len(v) == 384 for v in vectors))
    check_true("values are plain floats",
               all(isinstance(x, float) for x in vectors[0]))

    norms = [float(np.linalg.norm(v)) for v in vectors]
    check_true(
        "vectors are L2-normalised",
        all(abs(n - 1.0) < 1e-5 for n in norms),
        f"norms={[round(n, 6) for n in norms]}",
    )

    again = embed_documents(texts)
    check_true(
        "embeddings are deterministic",
        np.allclose(np.array(vectors), np.array(again), atol=1e-6),
    )

    # Batching must not change results.
    one_at_a_time = [embed_documents([t])[0] for t in texts]
    check_true(
        "batch size does not change vectors",
        np.allclose(np.array(vectors), np.array(one_at_a_time), atol=1e-5),
    )
    check("empty input -> empty list", embed_documents([]), [])

    # ------------------------------------------------------------------
    # BGE query/passage asymmetry
    # ------------------------------------------------------------------
    same_text = "how is the database connection initialised"
    as_document = embed_documents([same_text])[0]
    as_query = embed_query(same_text)
    check_true(
        "query embedding differs from passage embedding",
        not np.allclose(np.array(as_document), np.array(as_query), atol=1e-4),
        "the instruction prefix is being applied to queries only",
    )
    check_true("query prefix is configured", bool(settings.embedding_query_prefix))
    check_true("query vector is normalised",
               abs(float(np.linalg.norm(as_query)) - 1.0) < 1e-5)

    # ------------------------------------------------------------------
    # Token limits on REAL chunks - validates the Step 3 chunk_size choice
    # ------------------------------------------------------------------
    from app.services.chunker_service import iter_chunks
    from app.services.github_service import scan_repository_files
    from app.services.parser_service import iter_parsed_files

    repo_root = settings.repo_storage_dir / "jiya2406__weatherwise"
    if not repo_root.exists():
        print(f"\nSKIP retrieval tests - clone not found at {repo_root}")
        print("Run POST /api/repository/analyze for weatherwise first.\n")
    else:
        scan = scan_repository_files(
            repo_root,
            max_file_size_bytes=settings.max_file_size_bytes,
            max_files=settings.max_files_per_repo,
        )
        chunks = list(
            iter_chunks(
                iter_parsed_files(
                    repo_root,
                    scan.files,
                    repository="weatherwise",
                    repository_id="jiya2406__weatherwise",
                ),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                min_chunk_chars=settings.min_chunk_chars,
                # chunk_size is in tokens, so the chunker needs the tokeniser.
                length_function=count_tokens,
            )
        )

        token_counts = [count_tokens(c.content) for c in chunks]
        max_tokens = max(token_counts)
        over_limit = [
            (c.file_path, t)
            for c, t in zip(chunks, token_counts)
            if t > info.max_sequence_length
        ]
        print(f"\n--- token counts over {len(chunks)} real chunks ---")
        print(f"  max      : {max_tokens} tokens (limit {info.max_sequence_length})")
        print(f"  average  : {sum(token_counts) // len(token_counts)} tokens")
        print(f"  chars/token ~ {sum(c.char_count for c in chunks) / sum(token_counts):.2f}")
        print()
        check("no chunk exceeds the token limit", over_limit, [])

        # ------------------------------------------------------------------
        # Real retrieval test (cosine similarity, no vector DB)
        # ------------------------------------------------------------------
        print("Embedding all chunks...")
        started = time.perf_counter()
        matrix = np.array(embed_documents([c.content for c in chunks]))
        elapsed = time.perf_counter() - started
        print(f"embedded {len(chunks)} chunks in {elapsed:.1f}s "
              f"({len(chunks) / elapsed:.1f} chunks/sec)\n")

        def search(question: str, top_k: int = 3):
            # Unit vectors, so a dot product IS the cosine similarity.
            scores = matrix @ np.array(embed_query(question))
            ranked = np.argsort(-scores)[:top_k]
            return [(chunks[i].file_path, float(scores[i])) for i in ranked]

        answerable = {
            "Where is the database connection initialized?": "server/config/db.js",
            "How are weather API requests made?": "weather",
            "How is search history saved?": "istory",
        }

        print("--- retrieval results ---")
        top_scores = []
        for question, expected_fragment in answerable.items():
            results = search(question)
            print(f'\n  Q: "{question}"')
            for path, score in results:
                print(f"     {score:.3f}  {path}")
            hit = any(expected_fragment in path for path, _ in results)
            check_true(
                f"top-3 contains {expected_fragment!r}",
                hit,
                f"best={results[0][0]}",
            )
            top_scores.append(results[0][1])

        # weatherwise has no auth code at all. An unanswerable question should
        # score lower than answerable ones - this is what a Step 6/7 relevance
        # threshold will rely on to say "context is insufficient".
        unanswerable = "Where is JWT authentication and user login implemented?"
        miss = search(unanswerable)
        print(f'\n  Q (nothing to find): "{unanswerable}"')
        for path, score in miss:
            print(f"     {score:.3f}  {path}")
        print()
        check_true(
            "unanswerable question scores below answerable ones",
            miss[0][1] < min(top_scores),
            f"unanswerable={miss[0][1]:.3f} vs answerable min={min(top_scores):.3f}",
        )

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
