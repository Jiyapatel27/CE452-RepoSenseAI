"""Offline checks for Step 3 (code chunking).

Verifies chunk metadata, contiguous indices, deterministic ids, line-number
tracking, overlap, and that language-aware splitting keeps whole functions
together. No server, no network.

Run from the `backend/` folder:
    python scripts/check_step3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunker_service import (  # noqa: E402
    chunk_parsed_file,
    make_chunk_id,
    summarise_chunks,
)
from app.services.parser_service import ParsedFile  # noqa: E402

failures: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"{'PASS ' if ok else 'FAIL '} {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool) -> None:
    check(label, bool(actual), True)


def _overlap_between(first: str, second: str) -> int:
    """Length of the longest suffix of `first` that is a prefix of `second`."""
    for size in range(min(len(first), len(second)), 0, -1):
        if first[-size:] == second[:size]:
            return size
    return 0


def make_parsed(content: str, *, language: str, file_path: str) -> ParsedFile:
    return ParsedFile(
        repository="demo-repo",
        repository_id="demo__repo",
        file_path=file_path,
        file_name=file_path.rsplit("/", 1)[-1],
        extension="." + file_path.rsplit(".", 1)[-1],
        language=language,
        content=content,
        line_count=len(content.splitlines()),
        char_count=len(content),
        encoding="utf-8",
    )


# Six small Python functions, each well under one chunk.
PY_SOURCE = "\n\n".join(
    f"def function_{n}(value):\n"
    f'    """Docstring for function {n}."""\n'
    f"    result = value * {n}\n"
    f"    return result"
    for n in range(1, 13)
)

JS_SOURCE = """\
import express from 'express';

export function login(username, password) {
  const user = findUser(username);
  return checkPassword(user, password);
}

export function logout(session) {
  session.destroy();
}
"""


def main() -> int:
    # ------------------------------------------------------------------
    # Small file -> single chunk
    # ------------------------------------------------------------------
    small = make_parsed("def hi():\n    return 'hello world'\n", language="python",
                        file_path="src/small.py")
    chunks = chunk_parsed_file(small, chunk_size=1000, chunk_overlap=150)
    check("small file -> 1 chunk", len(chunks), 1)
    check("chunk_index starts at 0", chunks[0].chunk_index, 0)
    check("start_line", chunks[0].start_line, 1)
    check("metadata: repository", chunks[0].repository, "demo-repo")
    check("metadata: repository_id", chunks[0].repository_id, "demo__repo")
    check("metadata: file_path", chunks[0].file_path, "src/small.py")
    check("metadata: language", chunks[0].language, "python")
    check("metadata: char_count", chunks[0].char_count, len(chunks[0].content))

    # ------------------------------------------------------------------
    # Larger file -> several chunks
    # ------------------------------------------------------------------
    big = make_parsed(PY_SOURCE, language="python", file_path="src/big.py")
    chunks = chunk_parsed_file(big, chunk_size=300, chunk_overlap=60)
    print(f"\n--- {len(chunks)} chunks from {len(PY_SOURCE)} chars (size=300) ---")
    for c in chunks:
        first = c.content.splitlines()[0] if c.content.splitlines() else ""
        print(f"  [{c.chunk_index}] lines {c.start_line:>3}-{c.end_line:<3} "
              f"{c.char_count:>4} chars | {first[:44]}")
    print()

    check_true("big file -> multiple chunks", len(chunks) > 1)
    check(
        "chunk indices contiguous",
        [c.chunk_index for c in chunks],
        list(range(len(chunks))),
    )
    check_true(
        "every chunk within size limit",
        all(c.char_count <= 300 for c in chunks),
    )
    check_true("no empty chunks", all(c.content.strip() for c in chunks))
    check_true(
        "line numbers increase",
        all(
            chunks[i].start_line <= chunks[i + 1].start_line
            for i in range(len(chunks) - 1)
        ),
    )
    check_true("end_line >= start_line", all(c.end_line >= c.start_line for c in chunks))
    check_true(
        "last chunk ends at/near file end",
        chunks[-1].end_line <= len(PY_SOURCE.splitlines()) + 1,
    )

    # Line numbers must point at the real location in the original file.
    source_lines = PY_SOURCE.splitlines()
    mismatches = []
    for c in chunks:
        first_line = c.content.splitlines()[0].strip()
        actual_line = source_lines[c.start_line - 1].strip()
        if first_line and not actual_line.startswith(first_line[:20]):
            mismatches.append((c.chunk_index, c.start_line, first_line[:30], actual_line[:30]))
    check("start_line points at real source line", mismatches, [])

    # ------------------------------------------------------------------
    # Deterministic ids
    # ------------------------------------------------------------------
    again = chunk_parsed_file(big, chunk_size=300, chunk_overlap=60)
    check(
        "chunk ids are deterministic",
        [c.chunk_id for c in chunks],
        [c.chunk_id for c in again],
    )
    check_true("chunk ids unique", len({c.chunk_id for c in chunks}) == len(chunks))
    check(
        "chunk id is a stable uuid5",
        chunks[0].chunk_id,
        make_chunk_id("demo__repo", "src/big.py", 0),
    )
    check_true(
        "different repo -> different id",
        make_chunk_id("other__repo", "src/big.py", 0) != chunks[0].chunk_id,
    )

    # ------------------------------------------------------------------
    # Overlap
    #
    # Overlap is NOT expected between every pair of chunks. It exists to avoid
    # losing context at an *arbitrary* cut point, so LangChain only carries text
    # forward when it had to split inside a unit. When chunks break cleanly
    # between whole functions, overlap is 0 - and that is correct, because each
    # chunk is already a complete semantic unit.
    # ------------------------------------------------------------------
    # Case A: many small functions -> clean boundaries, so every chunk should
    # begin at a function definition rather than mid-function.
    check_true(
        "clean boundaries: chunks start at a definition",
        all(c.content.lstrip().startswith("def ") for c in chunks),
    )

    # Case B: a single function longer than chunk_size must be cut mid-body,
    # and there overlap must appear.
    long_function = make_parsed(
        "def process(data):\n"
        + "\n".join(f"    step_{n} = transform(data, {n})" for n in range(1, 40))
        + "\n    return step_39\n",
        language="python",
        file_path="src/long.py",
    )
    long_chunks = chunk_parsed_file(long_function, chunk_size=300, chunk_overlap=60)
    overlaps = [
        _overlap_between(long_chunks[i].content, long_chunks[i + 1].content)
        for i in range(len(long_chunks) - 1)
    ]
    print(f"--- mid-function split: {len(long_chunks)} chunks, "
          f"overlaps = {overlaps} ---\n")
    check_true("forced mid-unit split -> multiple chunks", len(long_chunks) > 1)
    check_true("mid-unit splits do overlap", all(o > 0 for o in overlaps))

    no_overlap_chunks = chunk_parsed_file(
        long_function, chunk_size=300, chunk_overlap=0
    )
    check(
        "overlap=0 -> no overlap",
        [
            _overlap_between(
                no_overlap_chunks[i].content, no_overlap_chunks[i + 1].content
            )
            for i in range(len(no_overlap_chunks) - 1)
        ],
        [0] * (len(no_overlap_chunks) - 1),
    )

    # ------------------------------------------------------------------
    # Language-aware splitting keeps functions intact
    # ------------------------------------------------------------------
    js = make_parsed(JS_SOURCE, language="javascript", file_path="src/auth.js")
    js_chunks = chunk_parsed_file(js, chunk_size=160, chunk_overlap=20)
    print(f"--- javascript: {len(js_chunks)} chunks (size=160) ---")
    for c in js_chunks:
        print(f"  [{c.chunk_index}] {c.char_count:>4} chars | "
              f"{c.content.splitlines()[0][:50]!r}")
    print()
    whole_login = any(
        "function login" in c.content and "checkPassword" in c.content
        for c in js_chunks
    )
    check_true("js: login() kept in one chunk", whole_login)

    # JSON has no LangChain profile - must still chunk via the generic splitter.
    js_on = make_parsed('{"a": ' + '"x",' * 300 + '"b": 1}', language="json",
                        file_path="data.json")
    json_chunks = chunk_parsed_file(js_on, chunk_size=300, chunk_overlap=50)
    check_true("json falls back to generic splitter", len(json_chunks) > 1)

    # ------------------------------------------------------------------
    # min_chunk_chars filtering
    # ------------------------------------------------------------------
    noisy = make_parsed("x = 1\n\n\n}\n\n\n" + PY_SOURCE, language="python",
                        file_path="src/noisy.py")
    filtered = chunk_parsed_file(noisy, chunk_size=300, chunk_overlap=0,
                                min_chunk_chars=100)
    check_true(
        "min_chunk_chars drops tiny chunks",
        all(len(c.content.strip()) >= 100 for c in filtered),
    )
    check(
        "indices still contiguous after filtering",
        [c.chunk_index for c in filtered],
        list(range(len(filtered))),
    )

    # ------------------------------------------------------------------
    # Configurable size actually changes the result
    # ------------------------------------------------------------------
    few = chunk_parsed_file(big, chunk_size=2000, chunk_overlap=100)
    many = chunk_parsed_file(big, chunk_size=200, chunk_overlap=20)
    print(f"size=2000 -> {len(few)} chunks | size=200 -> {len(many)} chunks\n")
    check_true("bigger chunk_size -> fewer chunks", len(few) < len(many))

    # ------------------------------------------------------------------
    # Custom length_function -> chunk_size measured in that unit
    #
    # The pipeline passes the embedding model's tokeniser so chunk_size means
    # tokens. Here a cheap stand-in (words) proves the plumbing works without
    # loading PyTorch, keeping this suite fast and model-independent.
    # ------------------------------------------------------------------
    def word_count(text: str) -> int:
        return len(text.split())

    word_chunks = chunk_parsed_file(
        big, chunk_size=20, chunk_overlap=4, length_function=word_count
    )
    check_true("custom length_function -> multiple chunks", len(word_chunks) > 1)
    check_true(
        "no chunk exceeds size in the custom unit",
        all(word_count(c.content) <= 20 for c in word_chunks),
    )
    check_true(
        "token_count populated when length_function given",
        all(c.token_count is not None for c in word_chunks),
    )
    check_true(
        "token_count matches the length_function",
        all(c.token_count == word_count(c.content) for c in word_chunks),
    )
    check_true(
        "token_count is None without a length_function",
        all(c.token_count is None for c in chunks),
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = summarise_chunks([small, big, js], chunk_size=300, chunk_overlap=60)
    check("summary: files_chunked", summary.files_chunked, 3)
    check("summary: files_without_chunks", summary.files_without_chunks, 0)
    check_true("summary: total_chunks > 0", summary.total_chunks > 0)
    check_true(
        "summary: average within bounds",
        0 < summary.average_chunk_chars <= 300,
    )
    check_true("summary: max <= chunk_size", summary.max_chunk_chars <= 300)
    check_true(
        "summary: languages counted",
        set(summary.chunks_per_language) == {"python", "javascript"},
    )

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
