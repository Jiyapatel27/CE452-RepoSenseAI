"""Step 12: dashboard analytics."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.db import analytics
from app.db.database import database_stats
from app.models.schemas import AnalyticsOverviewResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _system_panel() -> dict:
    """Vector store, model and storage status.

    Qdrant failures are reported as data rather than raised: the dashboard
    should render with an "offline" indicator, not return an error page.
    """
    settings = get_settings()
    panel: dict = {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.groq_model,
        "groq_configured": bool(settings.groq_api_key),
        "chunk_size_tokens": settings.chunk_size,
        "chunk_overlap_tokens": settings.chunk_overlap,
        "database": database_stats(),
        "qdrant": {"url": settings.qdrant_url, "reachable": False},
    }

    try:
        from app.services.vector_store import get_collection_info

        info = get_collection_info()
        panel["qdrant"] = {
            "url": settings.qdrant_url,
            "reachable": True,
            "collection": info.collection,
            "points": info.points_count,
            "dimension": info.dimension,
            "distance": info.distance,
            "repositories": info.repositories,
        }
    except Exception as exc:  # noqa: BLE001 - surfaced as data
        logger.warning("Qdrant unavailable for the dashboard: %s", exc)

    return panel


@router.get("/overview", response_model=AnalyticsOverviewResponse)
async def overview(
    days: int = Query(30, ge=1, le=365, description="Days of activity history."),
    top_files: int = Query(10, ge=1, le=50),
    recent: int = Query(15, ge=1, le=100),
) -> AnalyticsOverviewResponse:
    """Everything the dashboard needs, in one request.

    One endpoint rather than seven: the dashboard renders as a unit, and seven
    round trips would make it flash in stages as each panel arrived.
    """

    def _collect() -> AnalyticsOverviewResponse:
        return AnalyticsOverviewResponse(
            overview=analytics.overview(),
            activity=analytics.activity_by_day(days),
            repositories=analytics.repository_activity(),
            languages=analytics.language_totals(),
            top_files=analytics.top_cited_files(top_files),
            retrieval=analytics.retrieval_quality(),
            recent_questions=analytics.recent_questions(recent),
            system=_system_panel(),
        )

    return await run_in_threadpool(_collect)
