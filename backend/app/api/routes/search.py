"""Step 6: semantic search endpoint (no LLM)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.retrieval_service import search_chunks, unique_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    """Find the code chunks most relevant to a natural-language query.

    Pure retrieval: embed the query, filter Qdrant to this repository, return
    the top-K chunks by cosine similarity. No LLM is involved, so the results
    show exactly what Step 7 will be given as context.
    """
    settings = get_settings()

    started = time.perf_counter()
    hits = await run_in_threadpool(
        search_chunks,
        payload.repository_id,
        payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        languages=payload.languages,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    return SearchResponse(
        repository_id=payload.repository_id,
        query=payload.query,
        top_k=payload.top_k or settings.search_top_k,
        result_count=len(hits),
        results=[SearchResult(**vars(hit)) for hit in hits],
        files=unique_files(hits),
        elapsed_ms=round(elapsed_ms, 1),
    )
