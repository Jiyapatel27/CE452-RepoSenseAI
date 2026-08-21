"""Step 6: semantic search over the stored chunks.

    query -> embedding -> Qdrant (filtered by repository) -> top-K chunks

No LLM involved. This is the retrieval half of RAG, kept separate so it can be
tested and judged on its own - if retrieval is wrong, no amount of prompting in
Step 7 will save the answer.

One deliberate choice: search reads **only** Qdrant, and does not require the
repository to be present in the in-memory store. Vectors survive a server
restart; the store does not. Requiring both would force a pointless re-analyse
just to search code that is already indexed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import models

from app.config import get_settings
from app.core.exceptions import (
    InvalidSearchQueryError,
    RepositoryNotIndexedError,
)
from app.services.embedding_service import embed_query
from app.services.vector_store import (
    REPOSITORY_FIELD,
    count_repository_points,
    get_client,
    wrap_vector_store_error,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One retrieved chunk, with everything needed to cite it."""

    file_path: str
    score: float
    content: str
    file_name: str
    language: str
    chunk_index: int
    start_line: int
    end_line: int
    chunk_id: str


def _build_filter(
    repository_id: str, languages: list[str] | None
) -> models.Filter:
    """Repository filter, optionally narrowed to specific languages.

    The language filter is a genuinely useful knob: prose describing a feature
    often out-ranks the code implementing it, so a question like "which files
    handle API requests?" can surface README.md and package.json above the
    actual route handlers. Restricting to code languages fixes that, and it is
    left explicit rather than guessed from the query wording.
    """
    conditions: list[models.Condition] = [
        models.FieldCondition(
            key=REPOSITORY_FIELD, match=models.MatchValue(value=repository_id)
        )
    ]
    if languages:
        conditions.append(
            models.FieldCondition(
                key="language", match=models.MatchAny(any=list(languages))
            )
        )
    return models.Filter(must=conditions)


def search_chunks(
    repository_id: str,
    query: str,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    languages: list[str] | None = None,
) -> list[SearchHit]:
    """Return the chunks most semantically similar to `query`.

    Raises:
        InvalidSearchQueryError: blank query.
        RepositoryNotIndexedError: nothing stored for this repository.
        VectorStoreUnavailableError: Qdrant unreachable.
    """
    settings = get_settings()

    if not query or not query.strip():
        raise InvalidSearchQueryError("Query must not be empty.")

    limit = top_k if top_k is not None else settings.search_top_k
    limit = max(1, min(limit, settings.search_max_top_k))
    threshold = min_score if min_score is not None else settings.search_min_score

    # Fail with a clear 409 rather than silently returning nothing, so the
    # caller knows to index rather than assuming its question was too obscure.
    stored = count_repository_points(repository_id)
    if stored == 0:
        raise RepositoryNotIndexedError(
            f"Repository '{repository_id}' has no indexed vectors.",
            detail="Call POST /api/repository/index first.",
        )

    # embed_query applies the BGE instruction prefix; passages were embedded
    # WITHOUT it in Step 5. That asymmetry is what the model was trained for -
    # using the same function on both sides measurably degrades ranking.
    vector = embed_query(query)

    try:
        response = get_client().query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            query_filter=_build_filter(repository_id, languages),
            limit=limit,
            score_threshold=threshold if threshold > 0 else None,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise wrap_vector_store_error("search", exc) from exc

    hits: list[SearchHit] = []
    for point in response.points:
        payload = point.payload or {}
        content = payload.get("content")
        if not content:
            # A point without its text is useless downstream; skip rather than
            # hand Step 7 an empty context block.
            logger.warning("Point %s has no content payload", point.id)
            continue

        hits.append(
            SearchHit(
                file_path=payload.get("file_path", "unknown"),
                score=round(float(point.score), 4),
                content=content,
                file_name=payload.get("file_name", ""),
                language=payload.get("language", ""),
                chunk_index=int(payload.get("chunk_index", 0)),
                start_line=int(payload.get("start_line", 0)),
                end_line=int(payload.get("end_line", 0)),
                chunk_id=str(point.id),
            )
        )

    logger.info(
        "Search '%s' in %s -> %d hits (best=%.3f)",
        query[:60],
        repository_id,
        len(hits),
        hits[0].score if hits else 0.0,
    )
    return hits


def unique_files(hits: list[SearchHit]) -> list[str]:
    """File paths from `hits`, deduplicated, best-scoring first.

    Several chunks often come from the same file, so the raw hit list is a poor
    "relevant files" list for the UI. Order is preserved (hits arrive sorted by
    score), so the first mention of each file wins.
    """
    seen: list[str] = []
    for hit in hits:
        if hit.file_path not in seen:
            seen.append(hit.file_path)
    return seen
