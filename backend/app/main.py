"""RepoSense AI - FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    analytics,
    chat,
    conversations,
    embeddings,
    repository,
    search,
    vectorstore,
)
from app.config import get_settings
from app.core.exceptions import RepoSenseError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)

# qdrant-client and huggingface_hub talk over httpx, which logs every single
# request at INFO. Indexing one repository produces dozens of those lines and
# buries our own progress logs, so raise httpx to WARNING.
for _noisy in ("httpx", "httpcore", "sentence_transformers.base.model"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the embedding model before serving traffic.

    Loading it lazily makes the *first* request pay ~12s of PyTorch and model
    init - long enough to look like a hang. Paying it at startup instead means
    every request is fast. Disable with PRELOAD_EMBEDDING_MODEL=false when the
    constant reloads of `--reload` make the wait annoying.
    """
    settings = get_settings()
    if settings.preload_embedding_model:
        from app.services.embedding_service import get_model_info

        logger.info("Preloading embedding model (set PRELOAD_EMBEDDING_MODEL=false to skip)")
        info = await run_in_threadpool(get_model_info)
        logger.info(
            "Embedding model warm: %s (dim=%d, device=%s)",
            info.model_name,
            info.dimension,
            info.device,
        )

    # Create the SQLite schema before anything reads it.
    from app.db.database import init_database

    await run_in_threadpool(init_database)

    # Rebuild repository records so a restarted server does not claim it knows
    # nothing about code it can already search. SQLite is authoritative;
    # Qdrant covers anything indexed before persistence existed. Never fatal -
    # /analyze still works if this finds nothing.
    try:
        from app.services.repository_store import restore_repositories

        restored = await run_in_threadpool(restore_repositories)
        total = restored["from_database"] + restored["from_vectors"]
        if total:
            logger.info(
                "Restored %d repository record(s) (%d from database, "
                "%d from Qdrant only)",
                total,
                restored["from_database"],
                restored["from_vectors"],
            )
        else:
            logger.info("No previously analysed repositories found")
    except Exception as exc:  # noqa: BLE001 - startup must not fail on this
        logger.warning("Could not restore repositories: %s", exc)

    yield


app = FastAPI(
    title="RepoSense AI",
    description="AI-powered GitHub repository understanding system.",
    version="0.1.0",
    lifespan=lifespan,
)

# The Next.js dev server (Step 9) runs on port 3000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RepoSenseError)
async def reposense_error_handler(_: Request, exc: RepoSenseError) -> JSONResponse:
    """Turn our domain errors into clean JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


app.include_router(repository.router)
app.include_router(embeddings.router)
app.include_router(vectorstore.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(analytics.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, object]:
    """System readiness, so the UI can tell a setup problem from a bug.

    Always returns 200 even when a dependency is down - a health check that
    500s tells you nothing about *which* part is broken. Read the booleans.
    """
    settings = get_settings()

    qdrant_up = False
    collection_points = 0
    indexed_repositories: dict[str, int] = {}
    try:
        from app.services.vector_store import get_collection_info, ping

        await run_in_threadpool(ping)
        qdrant_up = True
        info = await run_in_threadpool(get_collection_info)
        collection_points = info.points_count
        indexed_repositories = info.repositories
    except Exception:  # noqa: BLE001 - reported as data, not an exception
        pass

    from app.services.repository_store import repository_store

    ready = qdrant_up and bool(settings.groq_api_key)
    return {
        "status": "ok" if ready else "degraded",
        "ready_for_chat": ready,
        "qdrant": {
            "url": settings.qdrant_url,
            "reachable": qdrant_up,
            "collection": settings.qdrant_collection,
            "points": collection_points,
            "repositories": indexed_repositories,
        },
        "groq": {
            "configured": bool(settings.groq_api_key),
            "model": settings.groq_model,
        },
        "embeddings": {
            "model": settings.embedding_model,
            "preloaded": settings.preload_embedding_model,
        },
        "repositories_in_memory": len(repository_store.list_all()),
        "repo_storage_dir": str(settings.repo_storage_dir),
    }
