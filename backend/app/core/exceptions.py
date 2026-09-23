"""Application-specific errors.

Every error carries the HTTP status code the API should return, so route
handlers stay free of try/except noise (see the handler in `app/main.py`).
"""


class RepoSenseError(Exception):
    """Base class for all expected RepoSense AI failures."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class InvalidRepositoryURLError(RepoSenseError):
    status_code = 400
    error_code = "invalid_repository_url"


class RepositoryNotFoundError(RepoSenseError):
    status_code = 404
    error_code = "repository_not_found"


class RepositoryAccessDeniedError(RepoSenseError):
    """A token was supplied but GitHub rejected it, or it lacks access."""

    status_code = 403
    error_code = "repository_access_denied"


class RepositoryDownloadError(RepoSenseError):
    """Network problem, timeout, or git failure while cloning."""

    status_code = 502
    error_code = "repository_download_failed"


class EmptyRepositoryError(RepoSenseError):
    status_code = 422
    error_code = "empty_repository"


class NoSupportedFilesError(RepoSenseError):
    status_code = 422
    error_code = "no_supported_files"


class GitNotInstalledError(RepoSenseError):
    status_code = 500
    error_code = "git_not_installed"


class EmbeddingModelError(RepoSenseError):
    """The embedding model could not be loaded (e.g. first-run download failed)."""

    status_code = 503
    error_code = "embedding_model_unavailable"


class VectorStoreUnavailableError(RepoSenseError):
    """Qdrant is not reachable."""

    status_code = 503
    error_code = "vector_store_unavailable"


class VectorStoreError(RepoSenseError):
    """Qdrant is reachable but rejected the operation."""

    status_code = 500
    error_code = "vector_store_error"


class LLMNotConfiguredError(RepoSenseError):
    """GROQ_API_KEY is missing."""

    status_code = 500
    error_code = "llm_not_configured"


class LLMRateLimitError(RepoSenseError):
    """Groq rate limit hit - the caller should retry later."""

    status_code = 429
    error_code = "llm_rate_limited"


class LLMUnavailableError(RepoSenseError):
    """Groq unreachable, timed out, or returned an error."""

    status_code = 502
    error_code = "llm_unavailable"


class ConversationNotFoundError(RepoSenseError):
    """No chat thread with that id."""

    status_code = 404
    error_code = "conversation_not_found"


class InvalidSearchQueryError(RepoSenseError):
    status_code = 400
    error_code = "invalid_search_query"


class RepositoryNotIndexedError(RepoSenseError):
    """The repository has no vectors stored yet."""

    status_code = 409
    error_code = "repository_not_indexed"


class FileNotInRepositoryError(RepoSenseError):
    """Requested file path is missing, or escapes the repository root."""

    status_code = 404
    error_code = "file_not_found"


class RepositoryFilesMissingError(RepoSenseError):
    """The clone was deleted from disk after it was analysed."""

    status_code = 410
    error_code = "repository_files_missing"
