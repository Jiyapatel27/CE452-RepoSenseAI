"""Step 5: Qdrant vector store.

Stores one point per code chunk: the 384-dim embedding plus a payload carrying
everything needed to answer with citations (content, file path, line range,
language, repository).

Design choices worth knowing:

*One collection, not one per repository.* The embedding model has a fixed
dimension, so a single collection suffices. Repositories are separated by a
`repository_id` payload field with a **keyword index** on it - without that
index, every filtered search would scan the whole collection.

*Deterministic point ids.* Chunk ids are UUID5s of
`repository_id:file_path:chunk_index` (Step 3), so re-indexing overwrites the
same points instead of inserting duplicates.

*Delete-then-upsert on re-index.* Stable ids alone are not enough: if a file is
deleted from the repository, or shrinks so it yields fewer chunks, the old
points would linger and keep being retrieved. Clearing the repository's points
first guarantees the index matches the current code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Iterator, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import get_settings
from app.core.exceptions import (
    RepoSenseError,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from app.services.chunker_service import CodeChunk
from app.services.embedding_service import embed_documents, get_embedding_dimension

logger = logging.getLogger(__name__)

# Payload field used to separate repositories. Indexed for fast filtering.
REPOSITORY_FIELD = "repository_id"


@dataclass
class IndexResult:
    """Outcome of indexing one repository."""

    repository_id: str
    collection: str
    chunks_indexed: int
    vectors_deleted_first: int
    dimension: int
    batches: int


@dataclass
class RepositoryVectorSummary:
    """What Qdrant knows about a repository, without the in-memory record.

    Enough to rebuild a usable repository entry after a server restart: the
    vectors outlive the process, so the API should not pretend the repository
    was never analysed.
    """

    repository_id: str
    repository: str
    chunk_count: int
    # file_path -> language, for every file that contributed a chunk.
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class CollectionInfo:
    collection: str
    exists: bool
    dimension: int | None
    distance: str | None
    points_count: int
    repositories: dict[str, int]


@lru_cache
def get_client() -> QdrantClient:
    """Cached Qdrant client.

    Cached because each client owns an HTTP connection pool; building one per
    request would leak sockets under load.
    """
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout_seconds,
    )


def wrap_vector_store_error(action: str, exc: Exception) -> RepoSenseError:
    """Translate qdrant-client exceptions into our API errors."""
    if isinstance(exc, ResponseHandlingException):
        settings = get_settings()
        return VectorStoreUnavailableError(
            f"Cannot reach Qdrant at {settings.qdrant_url}.",
            detail=(
                "Start it with `docker compose up -d` from the project root, "
                "then check http://localhost:6333/healthz."
            ),
        )
    if isinstance(exc, UnexpectedResponse):
        return VectorStoreError(
            f"Qdrant rejected the {action} request.", detail=str(exc)
        )
    return VectorStoreError(f"Unexpected error during {action}.", detail=str(exc))


def ping() -> bool:
    """True when Qdrant answers. Raises VectorStoreUnavailableError if not."""
    try:
        get_client().get_collections()
        return True
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
        raise wrap_vector_store_error("health check", exc) from exc


def ensure_collection(*, recreate: bool = False) -> str:
    """Create the collection and its payload index if missing.

    Idempotent, so it is safe to call before every indexing run.
    """
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_collection
    dimension = get_embedding_dimension()

    try:
        exists = client.collection_exists(name)

        if exists and recreate:
            logger.info("Dropping collection %s", name)
            client.delete_collection(name)
            exists = False

        if not exists:
            logger.info("Creating collection %s (dim=%d, cosine)", name, dimension)
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    # Cosine matches the normalised vectors from Step 4.
                    distance=models.Distance.COSINE,
                ),
            )

        # Without these indexes the corresponding filters would full-scan.
        # create_payload_index is idempotent for an identical schema.
        # - repository_id: every search filters on it
        # - language: optional filter used to exclude docs/config from
        #   code-focused questions (Step 6)
        for field in (REPOSITORY_FIELD, "language"):
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        return name
    except Exception as exc:  # noqa: BLE001
        raise wrap_vector_store_error("collection setup", exc) from exc


def _repository_filter(repository_id: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=REPOSITORY_FIELD,
                match=models.MatchValue(value=repository_id),
            )
        ]
    )


def count_repository_points(repository_id: str) -> int:
    """How many vectors are stored for a repository (0 if none)."""
    settings = get_settings()
    client = get_client()
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return 0
        result = client.count(
            collection_name=settings.qdrant_collection,
            count_filter=_repository_filter(repository_id),
            exact=True,
        )
        return int(result.count)
    except Exception as exc:  # noqa: BLE001
        raise wrap_vector_store_error("count", exc) from exc


def delete_repository(repository_id: str) -> int:
    """Remove every vector belonging to a repository. Returns how many."""
    settings = get_settings()
    client = get_client()
    try:
        if not client.collection_exists(settings.qdrant_collection):
            return 0
        existing = count_repository_points(repository_id)
        if existing:
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=_repository_filter(repository_id)
                ),
                wait=True,
            )
            logger.info("Deleted %d old vectors for %s", existing, repository_id)
        return existing
    except Exception as exc:  # noqa: BLE001
        raise wrap_vector_store_error("delete", exc) from exc


def _chunk_to_point(chunk: CodeChunk, vector: Sequence[float]) -> models.PointStruct:
    """Build a Qdrant point.

    The chunk's text lives in the payload so retrieval returns the code itself -
    Steps 6 and 7 never need to touch the filesystem again.
    """
    return models.PointStruct(
        id=chunk.chunk_id,
        vector=list(vector),
        payload={
            REPOSITORY_FIELD: chunk.repository_id,
            "repository": chunk.repository,
            "file_path": chunk.file_path,
            "file_name": chunk.file_name,
            "language": chunk.language,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "char_count": chunk.char_count,
            "token_count": chunk.token_count,
        },
    )


def _batched(items: Iterable[CodeChunk], size: int) -> Iterator[list[CodeChunk]]:
    batch: list[CodeChunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def index_chunks(
    repository_id: str,
    chunks: Iterable[CodeChunk],
    *,
    replace_existing: bool = True,
) -> IndexResult:
    """Embed `chunks` and upsert them into Qdrant.

    `chunks` may be a generator - it is consumed in batches, so a large
    repository never has all its chunks or vectors in memory at once.
    """
    settings = get_settings()
    collection = ensure_collection()
    client = get_client()

    deleted = delete_repository(repository_id) if replace_existing else 0

    total = 0
    batches = 0
    for batch in _batched(chunks, settings.qdrant_batch_size):
        # embed_documents deliberately does NOT apply the BGE query prefix -
        # these are passages, and prefixing them would hurt retrieval.
        vectors = embed_documents([chunk.content for chunk in batch])
        points = [
            _chunk_to_point(chunk, vector) for chunk, vector in zip(batch, vectors)
        ]
        try:
            client.upsert(collection_name=collection, points=points, wait=True)
        except Exception as exc:  # noqa: BLE001
            raise wrap_vector_store_error("upsert", exc) from exc

        total += len(points)
        batches += 1
        logger.info(
            "Indexed batch %d (%d points, %d total) for %s",
            batches,
            len(points),
            total,
            repository_id,
        )

    return IndexResult(
        repository_id=repository_id,
        collection=collection,
        chunks_indexed=total,
        vectors_deleted_first=deleted,
        dimension=get_embedding_dimension(),
        batches=batches,
    )


def list_repository_summaries() -> list[RepositoryVectorSummary]:
    """Summarise every repository stored in the collection.

    Scrolls the whole collection once. Fine at this scale (hundreds to a few
    thousand points); a production system would keep this in its own metadata
    store rather than deriving it from payloads.
    """
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_collection

    try:
        if not client.collection_exists(name):
            return []

        summaries: dict[str, RepositoryVectorSummary] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name,
                limit=1_000,
                offset=offset,
                with_payload=[
                    REPOSITORY_FIELD, "repository", "file_path", "language"
                ],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                key = payload.get(REPOSITORY_FIELD)
                if not key:
                    continue

                summary = summaries.get(key)
                if summary is None:
                    summary = RepositoryVectorSummary(
                        repository_id=key,
                        repository=payload.get("repository") or key,
                        chunk_count=0,
                    )
                    summaries[key] = summary

                summary.chunk_count += 1
                path = payload.get("file_path")
                if path:
                    summary.files[path] = payload.get("language") or ""

            if offset is None:
                break

        return sorted(summaries.values(), key=lambda s: -s.chunk_count)
    except Exception as exc:  # noqa: BLE001
        raise wrap_vector_store_error("repository summaries", exc) from exc


def get_collection_info() -> CollectionInfo:
    """Describe the collection, including a per-repository point count."""
    settings = get_settings()
    client = get_client()
    name = settings.qdrant_collection

    try:
        if not client.collection_exists(name):
            return CollectionInfo(
                collection=name,
                exists=False,
                dimension=None,
                distance=None,
                points_count=0,
                repositories={},
            )

        info = client.get_collection(name)
        vectors = info.config.params.vectors
        dimension = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)

        # Qdrant has no "group by" for counts, so page through ids and tally
        # the payload field. Fine at this scale; a production system would keep
        # these counts in its own metadata store.
        repositories: dict[str, int] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name,
                limit=1_000,
                offset=offset,
                with_payload=[REPOSITORY_FIELD],
                with_vectors=False,
            )
            for point in points:
                key = (point.payload or {}).get(REPOSITORY_FIELD, "unknown")
                repositories[key] = repositories.get(key, 0) + 1
            if offset is None:
                break

        return CollectionInfo(
            collection=name,
            exists=True,
            dimension=int(dimension) if dimension else None,
            distance=str(distance.value if hasattr(distance, "value") else distance),
            points_count=int(info.points_count or 0),
            repositories=dict(
                sorted(repositories.items(), key=lambda item: -item[1])
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise wrap_vector_store_error("collection info", exc) from exc
