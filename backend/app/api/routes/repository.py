"""Repository ingestion routes (Step 1)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.core.exceptions import (
    FileNotInRepositoryError,
    RepositoryFilesMissingError,
    RepositoryNotFoundError,
)
from app.models.schemas import (
    AnalyzeRepositoryRequest,
    AnalyzeRepositoryResponse,
    ChunkListResponse,
    ChunkStats,
    CodeChunkResponse,
    ParsedFileResponse,
    ParseStats,
    RepositoryFile,
    RepositoryFileListResponse,
    RepositoryStatus,
    RepositoryStatusListResponse,
)
from app.services.chunker_service import ChunkSummary, iter_chunks
from app.services.embedding_service import count_tokens
from app.services.vector_store import IndexResult, index_chunks
from app.services.github_service import (
    clone_repository,
    parse_repository_url,
    scan_repository_files,
)
from app.services.parser_service import (
    iter_parsed_files,
    parse_file,
    summarise_repository,
)
from app.services.repository_store import RepositoryRecord, repository_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/repository", tags=["repository"])

# How many files to echo back in the analyze response (the full list stays
# server-side in the store).
FILE_PREVIEW_LIMIT = 50


def _chunk_stream(repo_path, files, repository, repository_id, settings, summary=None):
    """Parse + chunk a repository as a generator.

    `length_function=count_tokens` sizes chunks in tokens, guaranteeing none
    exceeds the embedding model's 512-token limit. Passing `summary` collects
    statistics while the chunks stream past.
    """
    parsed_files = iter_parsed_files(
        repo_path,
        files,
        repository=repository,
        repository_id=repository_id,
    )
    return iter_chunks(
        parsed_files,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
        length_function=count_tokens,
        summary=summary,
    )


def _chunk_repository_stats(
    repo_path,
    files,
    repository: str,
    repository_id: str,
    settings,
) -> ChunkSummary:
    """Parse + chunk a whole repository, keeping only the statistics."""
    summary = ChunkSummary()
    for _ in _chunk_stream(
        repo_path, files, repository, repository_id, settings, summary
    ):
        pass  # chunk text intentionally discarded
    return summary


def _chunk_and_index(
    repo_path,
    files,
    repository: str,
    repository_id: str,
    settings,
) -> tuple[ChunkSummary, IndexResult]:
    """Chunk and index in a SINGLE pass.

    Chunking is not free - it tokenises every file - so doing it once for stats
    and again for indexing would double the cost of an analyse-and-index call.
    `iter_chunks` fills the summary as the indexer consumes the stream, so one
    traversal yields both.
    """
    summary = ChunkSummary()
    stream = _chunk_stream(
        repo_path, files, repository, repository_id, settings, summary
    )
    result = index_chunks(repository_id, stream)
    return summary, result


@router.post("/analyze", response_model=AnalyzeRepositoryResponse)
async def analyze_repository(
    payload: AnalyzeRepositoryRequest,
) -> AnalyzeRepositoryResponse:
    """Validate the URL, clone the repository, and list its supported files."""
    settings = get_settings()

    # Raises InvalidRepositoryURLError on bad input.
    ref = parse_repository_url(payload.repository_url)

    # A token in the request wins over the server-wide one in .env, so the UI
    # can analyse a private repo the server has no standing access to.
    token = (payload.github_token or settings.github_token or "").strip() or None

    # git clone and the directory walk are blocking, so keep them off the event
    # loop - otherwise the whole server freezes during a long clone.
    repo_path = await run_in_threadpool(
        clone_repository,
        ref,
        settings.repo_storage_dir,
        token=token,
        timeout_seconds=settings.clone_timeout_seconds,
    )

    scan = await run_in_threadpool(
        scan_repository_files,
        repo_path,
        max_file_size_bytes=settings.max_file_size_bytes,
        max_files=settings.max_files_per_repo,
    )

    # Step 2: decode every file once, up front. This validates the whole set
    # (catching unreadable/undecodable files before chunking) and yields the
    # line/character totals. Contents are discarded, not cached.
    parse_summary = await run_in_threadpool(
        summarise_repository,
        repo_path,
        scan.files,
        repository=ref.name,
        repository_id=ref.repository_id,
    )

    # Steps 3-5. With auto_index the chunks are embedded and stored in the same
    # pass that measures them, so the UI needs one request instead of two.
    index_result: IndexResult | None = None
    if payload.auto_index:
        chunk_summary, index_result = await run_in_threadpool(
            _chunk_and_index,
            repo_path,
            scan.files,
            ref.name,
            ref.repository_id,
            settings,
        )
    else:
        chunk_summary = await run_in_threadpool(
            _chunk_repository_stats,
            repo_path,
            scan.files,
            ref.name,
            ref.repository_id,
            settings,
        )

    record = repository_store.save(
        RepositoryRecord(
            repository_id=ref.repository_id,
            repository=ref.name,
            owner=ref.owner,
            branch=ref.branch,
            html_url=ref.html_url,
            local_path=str(repo_path),
            status="indexed" if index_result else "files_parsed",
            indexed=index_result is not None and index_result.chunks_indexed > 0,
            total_files_scanned=scan.total_files_seen,
            files=scan.files,
            skipped_files=scan.skipped,
            language_counts=scan.language_counts,
            truncated=scan.truncated,
            authenticated=token is not None,
            parse_summary=parse_summary,
            chunk_summary=chunk_summary,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )

    logger.info(
        "Analysed %s: %d/%d files supported, %d parsed (%d lines), %d chunks, "
        "indexed=%s",
        record.repository_id,
        record.supported_files,
        record.total_files_scanned,
        parse_summary.files_parsed,
        parse_summary.total_lines,
        chunk_summary.total_chunks,
        record.indexed,
    )

    return AnalyzeRepositoryResponse(
        repository_id=record.repository_id,
        repository=record.repository,
        owner=record.owner,
        branch=record.branch,
        html_url=record.html_url,
        local_path=record.local_path,
        status=record.status,
        authenticated=record.authenticated,
        total_files_scanned=record.total_files_scanned,
        supported_files=record.supported_files,
        skipped_files=record.skipped_files,
        language_counts=record.language_counts,
        truncated=record.truncated,
        parse_stats=_to_parse_stats(record),
        chunk_stats=_to_chunk_stats(record),
        indexed=record.indexed,
        chunks_indexed=index_result.chunks_indexed if index_result else 0,
        files=[
            RepositoryFile(**vars(file))
            for file in record.files[:FILE_PREVIEW_LIMIT]
        ],
    )


def _to_parse_stats(record: RepositoryRecord) -> ParseStats:
    summary = record.parse_summary
    return ParseStats(
        files_parsed=summary.files_parsed,
        files_failed=summary.files_failed,
        total_lines=summary.total_lines,
        total_characters=summary.total_characters,
        failures=summary.failures,
    )


def _to_chunk_stats(record: RepositoryRecord) -> ChunkStats:
    summary = record.chunk_summary
    return ChunkStats(
        total_chunks=summary.total_chunks,
        files_chunked=summary.files_chunked,
        files_without_chunks=summary.files_without_chunks,
        average_chunk_chars=summary.average_chunk_chars,
        min_chunk_chars=summary.min_chunk_chars,
        max_chunk_chars=summary.max_chunk_chars,
        average_chunk_tokens=summary.average_chunk_tokens,
        max_chunk_tokens=summary.max_chunk_tokens,
        chunks_per_language=summary.chunks_per_language,
        chunk_size=record.chunk_size,
        chunk_overlap=record.chunk_overlap,
    )


def _require_record(repository_id: str) -> RepositoryRecord:
    record = repository_store.get(repository_id)
    if record is None:
        raise RepositoryNotFoundError(
            f"No analysed repository with id '{repository_id}'.",
            detail="Call POST /api/repository/analyze first.",
        )
    return record


def _to_status(record: RepositoryRecord) -> RepositoryStatus:
    return RepositoryStatus(
        repository_id=record.repository_id,
        repository=record.repository,
        owner=record.owner,
        branch=record.branch,
        html_url=record.html_url,
        status=record.status,
        total_files_scanned=record.total_files_scanned,
        supported_files=record.supported_files,
        language_counts=record.language_counts,
        analyzed_at=record.analyzed_at,
        parse_stats=_to_parse_stats(record),
        chunk_stats=_to_chunk_stats(record),
        total_chunks=record.total_chunks,
        indexed=record.indexed,
    )


@router.get("/status", response_model=RepositoryStatusListResponse)
async def repository_status(
    repository_id: str | None = Query(
        None, description="Omit to list every analysed repository."
    ),
) -> RepositoryStatusListResponse:
    """Report ingestion status for one repository, or for all of them."""
    if repository_id:
        records = [_require_record(repository_id)]
    else:
        records = repository_store.list_all()

    return RepositoryStatusListResponse(
        count=len(records),
        repositories=[_to_status(record) for record in records],
    )


@router.get("/files", response_model=RepositoryFileListResponse)
async def list_repository_files(
    repository_id: str = Query(..., description="Id returned by /analyze."),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> RepositoryFileListResponse:
    """Paginated list of the files discovered in a repository (metadata only)."""
    record = _require_record(repository_id)
    page = record.files[offset : offset + limit]

    return RepositoryFileListResponse(
        repository_id=record.repository_id,
        total=record.supported_files,
        offset=offset,
        limit=limit,
        files=[RepositoryFile(**vars(file)) for file in page],
    )


@router.get("/chunks", response_model=ChunkListResponse)
async def list_repository_chunks(
    repository_id: str = Query(..., description="Id returned by /analyze."),
    file_path: str | None = Query(
        None, description="Restrict to a single file's chunks."
    ),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> ChunkListResponse:
    """Inspect the chunks a repository produces.

    Chunks are regenerated from disk on each call rather than cached, so this
    always reflects the current chunk settings and costs no memory between
    requests.
    """
    settings = get_settings()
    record = _require_record(repository_id)

    if not record.repo_root.exists():
        raise RepositoryFilesMissingError(
            "The local copy of this repository is gone.",
            detail="Re-run POST /api/repository/analyze.",
        )

    files = record.files
    if file_path:
        normalised = file_path.replace("\\", "/").lstrip("/")
        files = [file for file in files if file.file_path == normalised]
        if not files:
            raise FileNotInRepositoryError(
                "That file is not part of the analysed file set.", detail=file_path
            )

    def _collect() -> tuple[list[CodeChunkResponse], int]:
        parsed_files = iter_parsed_files(
            record.repo_root,
            files,
            repository=record.repository,
            repository_id=record.repository_id,
        )
        chunk_stream = iter_chunks(
            parsed_files,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
            length_function=count_tokens,
        )

        page: list[CodeChunkResponse] = []
        total = 0
        for chunk in chunk_stream:
            # Streaming means we can count everything while only materialising
            # the requested page.
            if offset <= total < offset + limit:
                page.append(CodeChunkResponse(**vars(chunk)))
            total += 1
        return page, total

    page, total = await run_in_threadpool(_collect)

    return ChunkListResponse(
        repository_id=record.repository_id,
        total=total,
        offset=offset,
        limit=limit,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunks=page,
    )


@router.get("/file", response_model=ParsedFileResponse)
async def get_parsed_file(
    repository_id: str = Query(..., description="Id returned by /analyze."),
    file_path: str = Query(..., description="Path relative to the repository root."),
) -> ParsedFileResponse:
    """Return one file parsed into the Step 2 structure, including its content."""
    record = _require_record(repository_id)

    if not record.repo_root.exists():
        raise RepositoryFilesMissingError(
            "The local copy of this repository is gone.",
            detail="Re-run POST /api/repository/analyze.",
        )

    # Only serve files that passed our filters - this also means `file_path`
    # is matched against a known-good list rather than trusted directly.
    normalised = file_path.replace("\\", "/").lstrip("/")
    discovered = next(
        (file for file in record.files if file.file_path == normalised), None
    )
    if discovered is None:
        raise FileNotInRepositoryError(
            "That file is not part of the analysed file set.",
            detail=file_path,
        )

    parsed = await run_in_threadpool(
        parse_file,
        record.repo_root,
        discovered,
        repository=record.repository,
        repository_id=record.repository_id,
    )
    if parsed is None:
        raise FileNotInRepositoryError(
            "File could not be read from disk.", detail=file_path
        )

    return ParsedFileResponse(**vars(parsed))
