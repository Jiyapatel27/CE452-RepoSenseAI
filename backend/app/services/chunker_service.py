"""Step 3: code chunking.

Splits each parsed file into embedding-sized pieces using LangChain's
`RecursiveCharacterTextSplitter`.

Why language-aware splitting matters: the generic splitter breaks on blank
lines and spaces, which happily cuts a function in half. `from_language()`
supplies separators for the language (``\\nclass ``, ``\\ndef ``, ``\\nfunction ``,
...) so the splitter *prefers* boundaries between top-level definitions and only
falls back to crude splits when a single definition is longer than a chunk.
That keeps whole functions inside one chunk, which is what makes retrieval
answer "where is X implemented?" correctly.

Each chunk carries the metadata Qdrant needs in Step 5, plus line numbers so
answers in Step 7 can cite `path/to/file.js:42`.

Chunk size is measured in **tokens**, not characters, using the embedding
model's own tokeniser. Character-based sizing cannot bound the token count:
measured across two real repositories, the worst-case density is ~1.4-2.0
chars/token while the average is ~3.0, so a 1000-character chunk can be anywhere
from 330 to 700 tokens. Anything over the model's 512-token limit is silently
truncated - the chunk still produces a valid-looking vector representing only
its opening fragment. Sizing in tokens makes that structurally impossible.

A note on `chunk_overlap`, because the behaviour surprises people: overlap does
**not** appear between every pair of chunks. LangChain only carries text forward
when it had to cut *inside* a unit. If a chunk boundary falls cleanly between two
functions, the overlap is 0 - correctly, since each chunk is already a complete
semantic unit. Overlap shows up exactly where it is needed: when a single
function is longer than `chunk_size` and must be split mid-body.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Iterable, Iterator

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.services.parser_service import ParsedFile

logger = logging.getLogger(__name__)

# Our language labels -> LangChain's splitter profiles.
# "json" is intentionally absent: LangChain has no JSON profile, so those files
# fall through to the generic character splitter.
_LANGUAGE_PROFILES: dict[str, Language] = {
    "python": Language.PYTHON,
    "javascript": Language.JS,
    "typescript": Language.TS,
    "java": Language.JAVA,
    "cpp": Language.CPP,
    "c": Language.C,
    "go": Language.GO,
    "rust": Language.RUST,
    "markdown": Language.MARKDOWN,
}

# Fixed namespace so chunk ids are reproducible across server restarts. Qdrant
# requires point ids to be UUIDs or unsigned ints, and stable ids make
# re-indexing a repository an idempotent upsert rather than a duplicate insert.
_CHUNK_NAMESPACE = uuid.UUID("7f9a1d02-4c3b-4f77-9a1e-0d5b2c8e6a11")


@dataclass
class CodeChunk:
    """One embedding-sized piece of a source file."""

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
    # None when chunking by characters (no tokeniser supplied).
    token_count: int | None = None


@dataclass
class ChunkSummary:
    """Aggregate statistics for a chunked repository."""

    total_chunks: int = 0
    files_chunked: int = 0
    files_without_chunks: int = 0
    total_characters: int = 0
    min_chunk_chars: int = 0
    max_chunk_chars: int = 0
    total_tokens: int = 0
    max_chunk_tokens: int = 0
    chunks_per_language: dict[str, int] = field(default_factory=dict)

    @property
    def average_chunk_chars(self) -> int:
        if not self.total_chunks:
            return 0
        return round(self.total_characters / self.total_chunks)

    @property
    def average_chunk_tokens(self) -> int:
        if not self.total_chunks or not self.total_tokens:
            return 0
        return round(self.total_tokens / self.total_chunks)

    def record(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            self.files_without_chunks += 1
            return

        self.files_chunked += 1
        for chunk in chunks:
            self.total_chunks += 1
            self.total_characters += chunk.char_count
            self.chunks_per_language[chunk.language] = (
                self.chunks_per_language.get(chunk.language, 0) + 1
            )
            if not self.min_chunk_chars or chunk.char_count < self.min_chunk_chars:
                self.min_chunk_chars = chunk.char_count
            self.max_chunk_chars = max(self.max_chunk_chars, chunk.char_count)
            if chunk.token_count is not None:
                self.total_tokens += chunk.token_count
                self.max_chunk_tokens = max(self.max_chunk_tokens, chunk.token_count)


@lru_cache(maxsize=64)
def _get_splitter(
    language: str,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int] | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build (and cache) a splitter for a language/size combination.

    `length_function` decides the *unit* of `chunk_size`. Pass a tokeniser-based
    counter to size chunks in tokens (the default in the pipeline); leave it None
    to size in characters.

    Cached because constructing a splitter compiles its separator list, and we
    would otherwise rebuild an identical one for every file in the repository.
    """
    options: dict = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # Gives each chunk its character offset in the original file, which we
        # convert into line numbers below.
        "add_start_index": True,
    }
    if length_function is not None:
        options["length_function"] = length_function

    profile = _LANGUAGE_PROFILES.get(language)
    if profile is not None:
        # Build directly rather than via from_language(), so the language's
        # separators combine with a custom length_function.
        try:
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(
                profile
            )
            return RecursiveCharacterTextSplitter(separators=separators, **options)
        except (ValueError, KeyError):  # pragma: no cover - defensive
            logger.warning(
                "No LangChain profile for %s, using the generic splitter", language
            )

    return RecursiveCharacterTextSplitter(**options)


def make_chunk_id(repository_id: str, file_path: str, chunk_index: int) -> str:
    """Deterministic UUID for a chunk, used as its Qdrant point id."""
    return str(
        uuid.uuid5(_CHUNK_NAMESPACE, f"{repository_id}:{file_path}:{chunk_index}")
    )


def chunk_parsed_file(
    parsed: ParsedFile,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    min_chunk_chars: int = 40,
    length_function: Callable[[str], int] | None = None,
) -> list[CodeChunk]:
    """Split one parsed file into `CodeChunk`s.

    `chunk_size`/`chunk_overlap` are measured with `length_function` - tokens
    when a tokeniser is passed, characters when it is None.
    """
    splitter = _get_splitter(
        parsed.language, chunk_size, chunk_overlap, length_function
    )
    documents = splitter.create_documents([parsed.content])

    # `min_chunk_chars` exists to discard noise *fragments* produced by a split
    # (a lone "}", a single import). A file small enough to fit in one chunk is
    # not a fragment - it is the whole file - so it is always kept, otherwise a
    # short-but-real source file would vanish from the index entirely.
    is_single_chunk_file = len(documents) == 1

    chunks: list[CodeChunk] = []
    for document in documents:
        content = document.page_content
        if not is_single_chunk_file and len(content.strip()) < min_chunk_chars:
            continue

        start_index = int(document.metadata.get("start_index", 0) or 0)
        # Lines are 1-based; count the newlines preceding this chunk.
        start_line = parsed.content.count("\n", 0, start_index) + 1
        end_line = start_line + content.count("\n")

        # Index is assigned after filtering, so indices stay contiguous
        # (0, 1, 2, ...) even when short chunks were dropped.
        chunk_index = len(chunks)
        chunks.append(
            CodeChunk(
                chunk_id=make_chunk_id(
                    parsed.repository_id, parsed.file_path, chunk_index
                ),
                repository=parsed.repository,
                repository_id=parsed.repository_id,
                file_path=parsed.file_path,
                file_name=parsed.file_name,
                language=parsed.language,
                chunk_index=chunk_index,
                content=content,
                char_count=len(content),
                start_line=start_line,
                end_line=end_line,
                token_count=(
                    length_function(content) if length_function is not None else None
                ),
            )
        )

    return chunks


def iter_chunks(
    parsed_files: Iterable[ParsedFile],
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    min_chunk_chars: int = 40,
    length_function: Callable[[str], int] | None = None,
    summary: ChunkSummary | None = None,
) -> Iterator[CodeChunk]:
    """Stream chunks for many files, optionally collecting stats.

    A generator so Step 4 can embed batches without ever materialising every
    chunk of a large repository in memory.
    """
    for parsed in parsed_files:
        chunks = chunk_parsed_file(
            parsed,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
            length_function=length_function,
        )
        if summary is not None:
            summary.record(chunks)
        yield from chunks


def summarise_chunks(
    parsed_files: Iterable[ParsedFile],
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    min_chunk_chars: int = 40,
    length_function: Callable[[str], int] | None = None,
) -> ChunkSummary:
    """Count and measure chunks without keeping their text."""
    summary = ChunkSummary()
    for _ in iter_chunks(
        parsed_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
        length_function=length_function,
        summary=summary,
    ):
        pass  # text intentionally discarded - only statistics are kept
    return summary
