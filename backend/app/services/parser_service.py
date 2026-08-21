"""Step 2: File parsing.

Turns each discovered file into a structured record:

    {
      "repository":  "example-repo",
      "file_path":   "src/auth/AuthService.js",
      "language":    "javascript",
      "content":     "..."
    }

Contents are read from disk on demand (see `iter_parsed_files`) rather than
cached in memory, so a large repository does not pin hundreds of megabytes for
the lifetime of the server.
"""

from __future__ import annotations

import codecs
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from app.services.github_service import DiscoveredFile

logger = logging.getLogger(__name__)

# Tried in order. utf-8-sig transparently strips a UTF-8 byte-order mark, which
# is common in files created on Windows.
_ENCODINGS = ("utf-8-sig", "utf-16", "latin-1")


@dataclass
class ParsedFile:
    """A single source file, decoded and normalised."""

    repository: str
    repository_id: str
    file_path: str
    file_name: str
    extension: str
    language: str
    content: str
    line_count: int
    char_count: int
    encoding: str


@dataclass
class ParseSummary:
    """Aggregate result of parsing a whole repository."""

    files_parsed: int = 0
    files_failed: int = 0
    total_lines: int = 0
    total_characters: int = 0
    failures: list[str] = field(default_factory=list)

    def record_failure(self, file_path: str, reason: str) -> None:
        self.files_failed += 1
        # Keep the list bounded so one broken repo cannot balloon the response.
        if len(self.failures) < 20:
            self.failures.append(f"{file_path}: {reason}")

    def record_success(self, parsed: ParsedFile) -> None:
        self.files_parsed += 1
        self.total_lines += parsed.line_count
        self.total_characters += parsed.char_count


def _normalise_newlines(text: str) -> str:
    """Collapse CRLF/CR to LF so chunk boundaries are platform-independent."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_bytes(raw: bytes) -> tuple[str, str]:
    """Decode file bytes to text, returning (text, encoding_used).

    latin-1 never fails, so this always succeeds for non-empty input; the
    fallback keeps odd-but-readable files in the index instead of dropping them.
    """
    for encoding in _ENCODINGS:
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        # utf-8-sig also decodes plain UTF-8, so only claim "-sig" for real BOMs.
        if encoding == "utf-8-sig" and not raw.startswith(codecs.BOM_UTF8):
            return text, "utf-8"
        return text, encoding
    # Last resort: replace undecodable bytes rather than lose the file.
    return raw.decode("utf-8", errors="replace"), "utf-8+replace"


def parse_file(
    repo_root: Path,
    discovered: DiscoveredFile,
    *,
    repository: str,
    repository_id: str,
) -> ParsedFile | None:
    """Read and decode one file. Returns None if it has no usable content."""
    absolute = repo_root / discovered.file_path

    try:
        raw = absolute.read_bytes()
    except OSError as exc:
        logger.debug("Could not read %s: %s", discovered.file_path, exc)
        return None

    text, encoding = decode_bytes(raw)
    content = _normalise_newlines(text)

    # Whitespace-only files carry no meaning for retrieval.
    if not content.strip():
        return None

    return ParsedFile(
        repository=repository,
        repository_id=repository_id,
        file_path=discovered.file_path,
        file_name=discovered.file_name,
        extension=discovered.extension,
        language=discovered.language,
        content=content,
        # splitlines() rather than count("\n") + 1, so a trailing newline does
        # not inflate the count by a phantom final line.
        line_count=len(content.splitlines()),
        char_count=len(content),
        encoding=encoding,
    )


def iter_parsed_files(
    repo_root: Path,
    files: Iterable[DiscoveredFile],
    *,
    repository: str,
    repository_id: str,
    summary: ParseSummary | None = None,
) -> Iterator[ParsedFile]:
    """Lazily parse every file, optionally recording stats into `summary`.

    A generator on purpose: Step 3 chunks each file and throws the text away, so
    only one file's content is ever held in memory.
    """
    for discovered in files:
        parsed = parse_file(
            repo_root,
            discovered,
            repository=repository,
            repository_id=repository_id,
        )
        if parsed is None:
            if summary is not None:
                summary.record_failure(discovered.file_path, "unreadable or empty")
            continue

        if summary is not None:
            summary.record_success(parsed)
        yield parsed


def summarise_repository(
    repo_root: Path,
    files: Iterable[DiscoveredFile],
    *,
    repository: str,
    repository_id: str,
) -> ParseSummary:
    """Parse everything once to validate it and collect totals, discarding text."""
    summary = ParseSummary()
    for _ in iter_parsed_files(
        repo_root,
        files,
        repository=repository,
        repository_id=repository_id,
        summary=summary,
    ):
        pass  # content intentionally dropped - we only want the statistics
    return summary
