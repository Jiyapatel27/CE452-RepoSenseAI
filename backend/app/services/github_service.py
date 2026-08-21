"""Step 1: GitHub repository handling.

Responsibilities
    1. Validate/parse a GitHub repository URL.
    2. Shallow-clone the repository into local storage.
    3. Walk the clone and return the list of supported source files.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlparse

from app.core.exceptions import (
    EmptyRepositoryError,
    GitNotInstalledError,
    InvalidRepositoryURLError,
    NoSupportedFilesError,
    RepoSenseError,
    RepositoryAccessDeniedError,
    RepositoryDownloadError,
    RepositoryNotFoundError,
)
from app.core.file_filters import (
    SkipReason,
    detect_language,
    is_ignored_directory,
    is_ignored_filename,
    looks_binary,
)

logger = logging.getLogger(__name__)

# GitHub allows letters, digits, hyphens, underscores and dots in these.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_ALLOWED_HOSTS = {"github.com", "www.github.com"}


# ----------------------------------------------------------------------
# 1. URL parsing / validation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RepositoryRef:
    """A validated reference to a GitHub repository."""

    owner: str
    name: str
    branch: str | None = None

    @property
    def repository_id(self) -> str:
        """Stable id used as the Qdrant partition key later on."""
        return f"{self.owner}__{self.name}".lower()

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


def parse_repository_url(url: str) -> RepositoryRef:
    """Turn a user-supplied GitHub URL into a `RepositoryRef`.

    Accepted forms::

        https://github.com/owner/repo
        https://github.com/owner/repo.git
        https://github.com/owner/repo/tree/some-branch
        github.com/owner/repo

    Raises:
        InvalidRepositoryURLError: anything else.
    """
    if not url or not url.strip():
        raise InvalidRepositoryURLError("Repository URL must not be empty.")

    cleaned = url.strip()

    # Allow users to paste "github.com/owner/repo" without a scheme.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = f"https://{cleaned}"

    parsed = urlparse(cleaned)

    if parsed.scheme not in {"http", "https"}:
        raise InvalidRepositoryURLError(
            "Only http(s) GitHub URLs are supported.",
            detail=f"Got scheme '{parsed.scheme}'.",
        )

    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise InvalidRepositoryURLError(
            "Only github.com repositories are supported right now.",
            detail=f"Got host '{host or 'unknown'}'.",
        )

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        raise InvalidRepositoryURLError(
            "URL must point to a repository, e.g. https://github.com/owner/repo."
        )

    owner, name = segments[0], segments[1]
    if name.lower().endswith(".git"):
        name = name[: -len(".git")]

    if not _NAME_PATTERN.match(owner) or not _NAME_PATTERN.match(name):
        raise InvalidRepositoryURLError(
            "Owner and repository name contain unsupported characters."
        )

    # Optional ".../tree/<branch>" suffix -> use that branch.
    branch: str | None = None
    extra = segments[2:]
    if extra:
        if extra[0] == "tree" and len(extra) > 1:
            branch = "/".join(extra[1:])
        else:
            raise InvalidRepositoryURLError(
                "URL must be a repository root or a /tree/<branch> URL.",
                detail=f"Unexpected path segments: {'/'.join(extra)}",
            )

    return RepositoryRef(owner=owner, name=name, branch=branch)


# ----------------------------------------------------------------------
# 2. Cloning
# ----------------------------------------------------------------------
def _force_delete(path: Path, *, attempts: int = 5) -> bool:
    """Best-effort recursive delete. Returns True when `path` is gone.

    On Windows two things get in the way: git marks files inside `.git` as
    read-only, and it can still hold a handle on the directory for a moment
    after exiting. So we clear the read-only bit and retry a few times.
    """
    if not path.exists():
        return True

    def _on_error(func, failing_path, _exc):  # noqa: ANN001
        os.chmod(failing_path, stat.S_IWRITE)
        func(failing_path)

    for attempt in range(attempts):
        try:
            shutil.rmtree(path, onexc=_on_error)
            return True
        except OSError as exc:
            if attempt == attempts - 1:
                logger.warning("Could not delete %s: %s", path, exc)
                return not path.exists()
            time.sleep(0.3 * (attempt + 1))

    return not path.exists()


def redact(text: str, token: str | None) -> str:
    """Remove a token from text before it is logged or returned to a client.

    git echoes the remote URL in its error output, and that URL carries the
    token, so every path out of this module (logs *and* API error details) must
    go through here.
    """
    if not text:
        return ""
    cleaned = text
    if token:
        cleaned = cleaned.replace(token, "***")
        cleaned = cleaned.replace(quote(token, safe=""), "***")
    # Belt and braces: strip any "user:secret@" credentials still present.
    return re.sub(r"://[^/@\s]*:[^/@\s]*@", "://***:***@", cleaned)


# git's wording differs between "no such repo" and "bad credentials", and both
# appear for private repos when no token is supplied.
_AUTH_FAILURE_MARKERS = (
    "repository not found",
    "could not read username",
    "could not read password",
    "authentication failed",
    "terminal prompts disabled",
    "invalid username or password",
    "403 forbidden",
)


def _looks_like_auth_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _AUTH_FAILURE_MARKERS)


def _classify_git_failure(
    ref: RepositoryRef, stderr: str, *, token_used: bool
) -> RepoSenseError:
    """Map git's stderr to a meaningful application error.

    `stderr` must already be redacted by the caller.
    """
    lowered = stderr.lower()
    detail = stderr.strip() or None

    if _looks_like_auth_failure(stderr):
        if token_used:
            return RepositoryAccessDeniedError(
                f"GitHub denied access to '{ref.owner}/{ref.name}'. The token is "
                "invalid, expired, or lacks read access to this repository.",
                detail=detail,
            )
        return RepositoryNotFoundError(
            f"Repository '{ref.owner}/{ref.name}' was not found. If it is "
            "private, supply a GitHub token to analyse it.",
            detail=detail,
        )

    if "remote branch" in lowered and "not found" in lowered:
        return InvalidRepositoryURLError(
            f"Branch '{ref.branch}' does not exist in {ref.owner}/{ref.name}.",
            detail=detail,
        )

    network_markers = ("could not resolve host", "failed to connect", "timed out")
    if any(marker in lowered for marker in network_markers):
        return RepositoryDownloadError(
            "Could not reach github.com. Check your internet connection.",
            detail=detail,
        )

    return RepositoryDownloadError("git clone failed.", detail=detail)


def build_clone_url(ref: RepositoryRef, token: str | None) -> str:
    """Clone URL, with the token embedded as credentials when one is given.

    `x-access-token` is the username GitHub expects for token auth; the token
    itself goes in the password field. It is percent-encoded so a token
    containing URL-special characters cannot corrupt the URL.
    """
    if not token:
        return ref.clone_url
    return (
        f"https://x-access-token:{quote(token, safe='')}"
        f"@github.com/{ref.owner}/{ref.name}.git"
    )


def _run_clone(
    ref: RepositoryRef,
    destination: Path,
    *,
    token: str | None,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run a single `git clone` attempt. Never logs or returns the raw token."""
    # `-c credential.helper=` disables Git Credential Manager, which would
    # otherwise pop up a GUI login window and hang the request forever.
    command = [
        "git",
        "-c",
        "credential.helper=",
        "clone",
        "--depth",
        "1",
        "--single-branch",
    ]
    if ref.branch:
        command += ["--branch", ref.branch]
    command += [build_clone_url(ref, token), str(destination)]

    # GIT_TERMINAL_PROMPT=0 makes private/missing repos fail fast instead of
    # blocking the server on a credential prompt.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}

    # Log the clean URL, never `command` - that contains the token.
    logger.info(
        "Cloning %s%s", ref.clone_url, " (with token)" if token else " (anonymous)"
    )
    return subprocess.run(  # noqa: S603 - fixed command, validated args
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
        check=False,
    )


def clone_repository(
    ref: RepositoryRef,
    storage_dir: Path,
    *,
    token: str | None = None,
    timeout_seconds: int = 300,
) -> Path:
    """Shallow-clone `ref` into `storage_dir/<repository_id>` and return the path.

    Works for public and private repositories. When `token` is supplied it is
    tried first; if GitHub rejects it we retry anonymously, so a stale token in
    `.env` cannot break analysis of a public repository.

    An existing clone of the same repository is replaced, so re-analysing a
    repository always picks up the latest code.
    """
    if shutil.which("git") is None:
        raise GitNotInstalledError(
            "git is not installed or not on PATH. Install Git and restart the server."
        )

    destination = storage_dir / ref.repository_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    if not _force_delete(destination):
        raise RepositoryDownloadError(
            "Could not clear the previous copy of this repository.",
            detail=f"Delete '{destination}' manually and try again.",
        )

    # With a token: try it, then fall back to anonymous. Without: one attempt.
    candidates: list[str | None] = [token, None] if token else [None]
    last_error: RepoSenseError | None = None
    token_error: RepoSenseError | None = None

    for attempt_token in candidates:
        try:
            result = _run_clone(
                ref,
                destination,
                token=attempt_token,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _force_delete(destination)
            raise RepositoryDownloadError(
                f"Cloning timed out after {timeout_seconds}s. "
                "Try a smaller repository."
            ) from exc

        if result.returncode == 0:
            last_error = None
            break

        # Redact before the output touches a log or an HTTP response.
        stderr = redact(result.stderr or result.stdout, attempt_token)
        _force_delete(destination)

        last_error = _classify_git_failure(
            ref, stderr, token_used=bool(attempt_token)
        )
        if attempt_token:
            token_error = last_error

        # Only a credential rejection is worth retrying without the token - a
        # network error or a bad branch would fail identically either way.
        if not (attempt_token and _looks_like_auth_failure(stderr)):
            break

        logger.info("Token rejected for %s, retrying anonymously", ref.clone_url)

    if last_error is not None:
        # If a token was supplied, its 403 is the more actionable message than
        # the anonymous attempt's "not found".
        raise token_error or last_error

    # The git history is not needed for analysis - and dropping it also removes
    # `.git/config`, where git stored the tokenised remote URL.
    _force_delete(destination / ".git")

    if not any(destination.iterdir()):
        _force_delete(destination)
        raise EmptyRepositoryError(
            f"Repository '{ref.owner}/{ref.name}' appears to be empty."
        )

    return destination


# ----------------------------------------------------------------------
# 3. Walking the clone
# ----------------------------------------------------------------------
@dataclass
class DiscoveredFile:
    """A repository file that passed all filters."""

    file_path: str  # POSIX path relative to the repository root
    file_name: str
    extension: str
    language: str
    size_bytes: int


@dataclass
class ScanResult:
    files: list[DiscoveredFile] = field(default_factory=list)
    total_files_seen: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def record_skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def language_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for file in self.files:
            counts[file.language] = counts.get(file.language, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))


def scan_repository_files(
    repo_path: Path,
    *,
    max_file_size_bytes: int = 1_000_000,
    max_files: int = 5_000,
) -> ScanResult:
    """Collect every supported source file inside `repo_path`."""
    result = ScanResult()

    for current_root, directory_names, file_names in os.walk(repo_path):
        # Mutating `directory_names` in place prunes the walk (os.walk contract),
        # so we never even enter node_modules & friends.
        kept_directories = []
        for name in sorted(directory_names):
            if is_ignored_directory(name):
                result.record_skip(SkipReason.IGNORED_DIRECTORY)
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for file_name in sorted(file_names):
            result.total_files_seen += 1

            if len(result.files) >= max_files:
                result.truncated = True
                return result

            language = detect_language(file_name)
            if language is None:
                result.record_skip(SkipReason.UNSUPPORTED_EXTENSION)
                continue

            if is_ignored_filename(file_name):
                result.record_skip(SkipReason.IGNORED_FILENAME)
                continue

            absolute = Path(current_root) / file_name
            try:
                size = absolute.stat().st_size
            except OSError:
                result.record_skip(SkipReason.UNREADABLE)
                continue

            if size == 0:
                result.record_skip(SkipReason.EMPTY)
                continue

            if size > max_file_size_bytes:
                result.record_skip(SkipReason.TOO_LARGE)
                continue

            if looks_binary(absolute):
                result.record_skip(SkipReason.BINARY)
                continue

            relative = absolute.relative_to(repo_path).as_posix()
            result.files.append(
                DiscoveredFile(
                    file_path=relative,
                    file_name=file_name,
                    extension=absolute.suffix.lower(),
                    language=language,
                    size_bytes=size,
                )
            )

    if not result.files:
        raise NoSupportedFilesError(
            "No supported source files were found in this repository.",
            detail=f"Scanned {result.total_files_seen} files.",
        )

    return result
