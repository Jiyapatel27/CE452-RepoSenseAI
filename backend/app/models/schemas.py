"""Pydantic request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# POST /api/repository/analyze
# ----------------------------------------------------------------------
class AnalyzeRepositoryRequest(BaseModel):
    repository_url: str = Field(
        ...,
        description="GitHub repository URL (public or private).",
        examples=["https://github.com/pallets/flask"],
    )
    github_token: str | None = Field(
        None,
        description=(
            "GitHub Personal Access Token, required only for private "
            "repositories. Overrides GITHUB_TOKEN from .env. Never stored or "
            "logged."
        ),
    )
    auto_index: bool = Field(
        True,
        description=(
            "Embed and store the chunks in Qdrant as part of this call, so the "
            "repository is immediately searchable. Set false to only inspect "
            "files and chunks without paying the embedding cost."
        ),
    )


class RepositoryFile(BaseModel):
    file_path: str
    file_name: str
    extension: str
    language: str
    size_bytes: int


class ParseStats(BaseModel):
    """Step 2: results of decoding every supported file."""

    files_parsed: int
    files_failed: int
    total_lines: int
    total_characters: int
    failures: list[str] = Field(
        default_factory=list, description="First few files that could not be parsed."
    )


class ChunkStats(BaseModel):
    """Step 3: results of splitting the repository into chunks."""

    total_chunks: int
    files_chunked: int
    files_without_chunks: int
    average_chunk_chars: int
    min_chunk_chars: int
    max_chunk_chars: int
    average_chunk_tokens: int
    max_chunk_tokens: int = Field(
        ..., description="Must stay under the embedding model's 512-token limit."
    )
    chunks_per_language: dict[str, int] = Field(default_factory=dict)
    chunk_size: int = Field(..., description="Configured chunk size in TOKENS.")
    chunk_overlap: int = Field(..., description="Configured overlap in TOKENS.")


class AnalyzeRepositoryResponse(BaseModel):
    repository_id: str
    repository: str
    owner: str
    branch: str | None = None
    html_url: str
    local_path: str
    status: str
    authenticated: bool = Field(
        False,
        description=(
            "True if a GitHub token was supplied for the clone. Note this does "
            "not by itself prove the repository is private - public repos also "
            "accept a token."
        ),
    )

    total_files_scanned: int = Field(
        ..., description="Every file git delivered, before filtering."
    )
    supported_files: int = Field(
        ..., description="Files that passed the extension/size/binary filters."
    )
    skipped_files: dict[str, int] = Field(
        default_factory=dict, description="Skip reason -> count."
    )
    language_counts: dict[str, int] = Field(default_factory=dict)
    truncated: bool = Field(
        False, description="True when the max-file limit stopped the scan early."
    )

    parse_stats: ParseStats
    chunk_stats: ChunkStats
    indexed: bool = Field(
        False, description="True when the chunks are stored in Qdrant and searchable."
    )
    chunks_indexed: int = Field(
        0, description="Vectors written to Qdrant during this call."
    )

    files: list[RepositoryFile] = Field(
        default_factory=list,
        description="Preview of the discovered files (capped for readability).",
    )


# ----------------------------------------------------------------------
# GET /api/repository/status
# ----------------------------------------------------------------------
class RepositoryStatus(BaseModel):
    repository_id: str
    repository: str
    owner: str
    branch: str | None = None
    html_url: str
    status: str
    total_files_scanned: int
    supported_files: int
    language_counts: dict[str, int] = Field(default_factory=dict)
    analyzed_at: str
    parse_stats: ParseStats
    chunk_stats: ChunkStats
    total_chunks: int = 0

    # Filled in by Step 5 (Qdrant indexing).
    indexed: bool = False


class RepositoryStatusListResponse(BaseModel):
    count: int
    repositories: list[RepositoryStatus]


# ----------------------------------------------------------------------
# GET /api/repository/files  and  GET /api/repository/file
# ----------------------------------------------------------------------
class RepositoryFileListResponse(BaseModel):
    repository_id: str
    total: int
    offset: int
    limit: int
    files: list[RepositoryFile]


class CodeChunkResponse(BaseModel):
    """One chunk - the Step 3 output shape, and what Step 5 stores in Qdrant."""

    chunk_id: str
    repository: str
    repository_id: str
    file_path: str
    file_name: str
    language: str
    chunk_index: int
    content: str
    char_count: int
    start_line: int
    end_line: int
    token_count: int | None = None


class ChunkListResponse(BaseModel):
    repository_id: str
    total: int
    offset: int
    limit: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[CodeChunkResponse]


class ParsedFileResponse(BaseModel):
    """One fully parsed file - the Step 2 output shape."""

    repository: str
    repository_id: str
    file_path: str
    file_name: str
    extension: str
    language: str
    line_count: int
    char_count: int
    encoding: str
    content: str


# ----------------------------------------------------------------------
# GET /api/embeddings/info
# ----------------------------------------------------------------------
class EmbeddingInfoResponse(BaseModel):
    model_name: str
    dimension: int = Field(..., description="Vector size, e.g. 384.")
    max_sequence_length: int = Field(
        ..., description="Tokens beyond this are truncated by the model."
    )
    device: str
    batch_size: int
    query_prefix: str = Field(
        ..., description="Instruction prefix applied to queries, not to passages."
    )


# ----------------------------------------------------------------------
# Step 5: Qdrant vector store
# ----------------------------------------------------------------------
class IndexRepositoryRequest(BaseModel):
    repository_id: str = Field(..., description="Id returned by /analyze.")
    replace_existing: bool = Field(
        True,
        description=(
            "Delete the repository's existing vectors first. Keep this true so "
            "chunks from deleted files cannot linger in the index."
        ),
    )


class IndexRepositoryResponse(BaseModel):
    repository_id: str
    collection: str
    chunks_indexed: int
    vectors_deleted_first: int
    dimension: int
    batches: int
    indexed: bool
    elapsed_seconds: float


class VectorStoreInfoResponse(BaseModel):
    qdrant_url: str
    reachable: bool
    collection: str
    exists: bool
    dimension: int | None = None
    distance: str | None = None
    points_count: int = 0
    repositories: dict[str, int] = Field(
        default_factory=dict, description="repository_id -> stored vector count."
    )


# ----------------------------------------------------------------------
# Step 6: POST /api/search
# ----------------------------------------------------------------------
class SearchRequest(BaseModel):
    repository_id: str = Field(..., description="Id returned by /analyze.")
    query: str = Field(
        ...,
        description="Natural-language question about the code.",
        examples=["Where is user authentication implemented?"],
    )
    top_k: int | None = Field(
        None, ge=1, le=50, description="How many chunks to return. Defaults to 5."
    )
    min_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Drop results below this cosine score. Off by default - the gap "
            "between relevant and irrelevant is often only ~0.05, so a cutoff "
            "is unreliable."
        ),
    )
    languages: list[str] | None = Field(
        None,
        description=(
            "Restrict results to these languages, e.g. "
            '["javascript","typescript"]. Useful because prose describing a '
            "feature often out-ranks the code implementing it - filtering to "
            "code languages keeps README.md and package.json out of the way."
        ),
        examples=[["javascript", "typescript"]],
    )


class SearchResult(BaseModel):
    file_path: str
    score: float
    content: str
    file_name: str
    language: str
    chunk_index: int
    start_line: int
    end_line: int
    chunk_id: str


class SearchResponse(BaseModel):
    repository_id: str
    query: str
    top_k: int
    result_count: int
    results: list[SearchResult]
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Unique file paths from the results, best match first - ready for "
            "the UI's 'Relevant Files' list."
        ),
    )
    elapsed_ms: float


# ----------------------------------------------------------------------
# Step 7: POST /api/chat
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    repository_id: str = Field(..., description="Id returned by /analyze.")
    question: str = Field(
        ...,
        description="Natural-language question about the repository.",
        examples=["How does authentication work?"],
    )
    top_k: int | None = Field(
        None, ge=1, le=20, description="Chunks of context to use. Defaults to 6."
    )
    languages: list[str] | None = Field(
        None,
        description=(
            "Restrict context to these languages. Useful when docs out-rank "
            "the code, e.g. [\"javascript\"]."
        ),
    )


class ChatSource(BaseModel):
    """A chunk that was actually placed in the prompt."""

    file_path: str
    score: float
    start_line: int
    end_line: int
    language: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list, description="File paths used, best match first."
    )
    repository_id: str
    question: str
    model: str
    context_chunks: int = Field(
        0, description="How many chunks were placed in the prompt."
    )
    truncated_context: bool = Field(
        False, description="True if some retrieved chunks did not fit."
    )
    truncated_answer: bool = Field(
        False,
        description=(
            "True if the answer hit the token limit and stops mid-sentence. "
            "The UI should say so rather than present it as complete."
        ),
    )
    llm_called: bool = Field(
        True, description="False when retrieval found nothing and the LLM was skipped."
    )
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    context: list[ChatSource] = Field(
        default_factory=list,
        description="Details of the context chunks, so an answer can be verified.",
    )


# ----------------------------------------------------------------------
# Step 12: conversations
# ----------------------------------------------------------------------
class StartConversationRequest(BaseModel):
    repository_id: str = Field(..., description="Id returned by /analyze.")
    question: str = Field(..., description="The opening question.")
    top_k: int | None = Field(None, ge=1, le=20)
    languages: list[str] | None = None


class ContinueConversationRequest(BaseModel):
    question: str = Field(..., description="Follow-up question.")
    top_k: int | None = Field(None, ge=1, le=20)
    languages: list[str] | None = None


class RenameConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class MessageSourceResponse(BaseModel):
    file_path: str
    score: float | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None
    cited: bool = True


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: float | None = None
    context_chunks: int | None = None
    refused: bool = False
    truncated: bool = False
    resolved_question: str | None = Field(
        None,
        description=(
            "The standalone question used for retrieval. Present only when a "
            "follow-up had to be rewritten."
        ),
    )
    sources: list[MessageSourceResponse] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    id: str
    repository_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message: str | None = None


class ConversationListResponse(BaseModel):
    count: int
    conversations: list[ConversationResponse]


class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: list[MessageResponse]


class ConversationTurnResponse(BaseModel):
    """The result of asking a question inside a thread."""

    conversation: ConversationResponse
    user_message: MessageResponse
    assistant_message: MessageResponse
    # Convenience mirrors of the assistant message, so a simple client does not
    # have to dig into the message object.
    answer: str
    sources: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Step 12: dashboard analytics
# ----------------------------------------------------------------------
class AnalyticsOverviewResponse(BaseModel):
    overview: dict
    activity: list[dict] = Field(
        default_factory=list, description="Questions/answers/refusals per day."
    )
    repositories: list[dict] = Field(default_factory=list)
    languages: dict[str, int] = Field(default_factory=dict)
    top_files: list[dict] = Field(default_factory=list)
    retrieval: dict = Field(default_factory=dict)
    recent_questions: list[dict] = Field(default_factory=list)
    system: dict = Field(default_factory=dict)


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str | None = None
