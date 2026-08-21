"""Step 7: RAG chat endpoint (retrieval + Groq)."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import ChatRequest, ChatResponse, ChatSource
from app.services.rag_service import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Answer a question about a repository, grounded in its indexed code.

    Pipeline: question -> embedding -> Qdrant -> top-K chunks -> prompt -> Groq.

    The response includes the context chunks that were used, so any claim in
    the answer can be checked against the code it came from.
    """
    started = time.perf_counter()
    result = await run_in_threadpool(
        answer_question,
        payload.repository_id,
        payload.question,
        top_k=payload.top_k,
        languages=payload.languages,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        repository_id=payload.repository_id,
        question=payload.question,
        model=result.model,
        context_chunks=result.context_chunks,
        truncated_context=result.truncated_context,
        truncated_answer=result.truncated_answer,
        llm_called=result.llm_called,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        elapsed_ms=round(elapsed_ms, 1),
        context=[
            ChatSource(
                file_path=hit.file_path,
                score=hit.score,
                start_line=hit.start_line,
                end_line=hit.end_line,
                language=hit.language,
            )
            for hit in result.hits
        ],
    )
