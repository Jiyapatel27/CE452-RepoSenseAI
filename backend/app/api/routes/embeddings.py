"""Embedding model routes (Step 4)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.models.schemas import EmbeddingInfoResponse
from app.services.embedding_service import get_model_info

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.get("/info", response_model=EmbeddingInfoResponse)
async def embedding_info() -> EmbeddingInfoResponse:
    """Report the embedding model in use.

    The first call loads the model (downloading it if needed), so it can take a
    few seconds; later calls are instant. Loading is blocking, hence the
    threadpool.
    """
    settings = get_settings()
    info = await run_in_threadpool(get_model_info)

    return EmbeddingInfoResponse(
        model_name=info.model_name,
        dimension=info.dimension,
        max_sequence_length=info.max_sequence_length,
        device=info.device,
        batch_size=settings.embedding_batch_size,
        query_prefix=settings.embedding_query_prefix,
    )
