"""Step 5: vector store routes - indexing chunks into Qdrant."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.exceptions import (
    RepositoryFilesMissingError,
    RepositoryNotFoundError,
    VectorStoreUnavailableError,
)
from app.models.schemas import (
    IndexRepositoryRequest,
    IndexRepositoryResponse,
    VectorStoreInfoResponse,
)
from app.services.chunker_service import iter_chunks
from app.services.embedding_service import count_tokens
from app.services.parser_service import iter_parsed_files
from app.services.repository_store import repository_store
from app.services.vector_store import (
    get_collection_info,
    index_chunks,
    ping,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["vector store"])


@router.post("/repository/index", response_model=IndexRepositoryResponse)
async def index_repository(
    payload: IndexRepositoryRequest,
) -> IndexRepositoryResponse:
    """Embed an analysed repository's chunks and store them in Qdrant."""
    settings = get_settings()

    record = repository_store.get(payload.repository_id)
    if record is None:
        raise RepositoryNotFoundError(
            f"No analysed repository with id '{payload.repository_id}'.",
            detail="Call POST /api/repository/analyze first.",
        )

    if not record.repo_root.exists():
        raise RepositoryFilesMissingError(
            "The local copy of this repository is gone.",
            detail="Re-run POST /api/repository/analyze.",
        )

    def _run():
        # Re-parse and re-chunk straight into the indexer. Everything streams,
        # so a large repository never materialises all its chunks at once.
        chunk_stream = iter_chunks(
            iter_parsed_files(
                record.repo_root,
                record.files,
                repository=record.repository,
                repository_id=record.repository_id,
            ),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
            length_function=count_tokens,
        )
        return index_chunks(
            record.repository_id,
            chunk_stream,
            replace_existing=payload.replace_existing,
        )

    started = time.perf_counter()
    result = await run_in_threadpool(_run)
    elapsed = time.perf_counter() - started

    record.indexed = result.chunks_indexed > 0
    record.status = "indexed" if record.indexed else record.status
    repository_store.save(record)

    logger.info(
        "Indexed %s: %d chunks into '%s' in %.1fs",
        result.repository_id,
        result.chunks_indexed,
        result.collection,
        elapsed,
    )

    return IndexRepositoryResponse(
        repository_id=result.repository_id,
        collection=result.collection,
        chunks_indexed=result.chunks_indexed,
        vectors_deleted_first=result.vectors_deleted_first,
        dimension=result.dimension,
        batches=result.batches,
        indexed=record.indexed,
        elapsed_seconds=round(elapsed, 2),
    )


@router.get("/vectorstore/info", response_model=VectorStoreInfoResponse)
async def vectorstore_info() -> VectorStoreInfoResponse:
    """Report Qdrant connectivity and what is currently stored."""
    settings = get_settings()

    try:
        await run_in_threadpool(ping)
    except VectorStoreUnavailableError:
        # Reported as data rather than a 503: "is Qdrant up?" is exactly what
        # this endpoint exists to answer, so it should still return a body.
        return VectorStoreInfoResponse(
            qdrant_url=settings.qdrant_url,
            reachable=False,
            collection=settings.qdrant_collection,
            exists=False,
        )

    info = await run_in_threadpool(get_collection_info)
    return VectorStoreInfoResponse(
        qdrant_url=settings.qdrant_url,
        reachable=True,
        collection=info.collection,
        exists=info.exists,
        dimension=info.dimension,
        distance=info.distance,
        points_count=info.points_count,
        repositories=info.repositories,
    )
