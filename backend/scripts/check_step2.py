"""Offline checks for Step 2 (file parsing) and the Step 1 filters.

Builds a small fake repository in a temp folder containing the awkward cases
(BOM, CRLF, binary, empty, oversized, node_modules, lock files) and verifies
that scanning + parsing handle each one correctly. No server, no network.

Run from the `backend/` folder:
    python scripts/check_step2.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.file_filters import SkipReason  # noqa: E402
from app.services.github_service import scan_repository_files  # noqa: E402
from app.services.parser_service import (  # noqa: E402
    iter_parsed_files,
    summarise_repository,
)

failures: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"{'PASS ' if ok else 'FAIL '} {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        failures.append(label)


def build_fake_repo(root: Path) -> None:
    """Create a repository that exercises every filter and parser branch."""
    (root / "src" / "auth").mkdir(parents=True)
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "__pycache__").mkdir(parents=True)
    (root / ".github" / "workflows").mkdir(parents=True)

    # --- files that SHOULD be parsed ---
    (root / "src" / "auth" / "AuthService.js").write_text(
        "function login(user) {\n  return jwt.sign(user);\n}\n", encoding="utf-8"
    )
    # CRLF line endings -> must be normalised to LF.
    (root / "src" / "app.py").write_bytes(b"import os\r\nprint('hi')\r\n")
    # UTF-8 BOM -> must be stripped, not left as a stray character.
    (root / "README.md").write_bytes(
        b"\xef\xbb\xbf# Title\n\nHello \xc3\xa9\xc3\xa9\n"
    )
    (root / "package.json").write_text('{"name":"demo"}\n', encoding="utf-8")
    (root / "src" / "main.go").write_text("package main\n", encoding="utf-8")

    # --- files that should be SKIPPED ---
    (root / "package-lock.json").write_text('{"lockfileVersion":3}\n', encoding="utf-8")
    (root / "bundle.min.js").write_text("var a=1;\n", encoding="utf-8")
    (root / "types.d.ts").write_text("declare const x: number;\n", encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    (root / "notes.txt").write_text("unsupported extension\n", encoding="utf-8")
    (root / "empty.py").write_text("", encoding="utf-8")
    (root / "whitespace.py").write_text("   \n\n  \n", encoding="utf-8")
    (root / "huge.py").write_text("x = 1\n" * 100_000, encoding="utf-8")  # ~600 KB
    (root / "fake_binary.py").write_bytes(b"import os\x00\x00truncated")
    (root / "node_modules" / "left-pad" / "index.js").write_text("m", encoding="utf-8")
    (root / "__pycache__" / "app.cpython-312.pyc").write_text("x", encoding="utf-8")
    (root / ".github" / "workflows" / "ci.json").write_text("{}", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "demo-repo"
        root.mkdir()
        build_fake_repo(root)

        # max_file_size_bytes=100_000 makes huge.py (~600 KB) exceed the limit.
        scan = scan_repository_files(
            root, max_file_size_bytes=100_000, max_files=1000
        )

        found = sorted(f.file_path for f in scan.files)
        print("\n--- discovered files ---")
        for path in found:
            print(f"  {path}")
        print()

        check(
            "discovered file set",
            found,
            [
                "README.md",
                "package.json",
                "src/app.py",
                "src/auth/AuthService.js",
                "src/main.go",
                "whitespace.py",
            ],
        )

        # whitespace.py passes the size/binary filters but has no real content,
        # so the parser is the layer that drops it.
        check("scan: node_modules pruned", "node_modules/left-pad/index.js" in found, False)
        check("scan: lock file skipped", "package-lock.json" in found, False)
        check("scan: minified skipped", "bundle.min.js" in found, False)
        check("scan: .d.ts skipped", "types.d.ts" in found, False)
        check("scan: binary skipped", "logo.png" in found, False)
        check("scan: fake binary skipped", "fake_binary.py" in found, False)
        check("scan: oversized skipped", "huge.py" in found, False)
        check("scan: empty skipped", "empty.py" in found, False)
        check("scan: unsupported ext skipped", "notes.txt" in found, False)
        check("scan: too_large counted", scan.skipped.get(SkipReason.TOO_LARGE), 1)
        check("scan: empty counted", scan.skipped.get(SkipReason.EMPTY), 1)
        # Only fake_binary.py reaches the NUL-byte check; logo.png is dropped
        # earlier and more cheaply by the extension filter.
        check("scan: binary counted", scan.skipped.get(SkipReason.BINARY), 1)
        check("scan: truncated flag", scan.truncated, False)
        check(
            "scan: languages",
            scan.language_counts,
            {"python": 2, "javascript": 1, "markdown": 1, "json": 1, "go": 1},
        )

        # --- parsing ---
        parsed = {
            p.file_path: p
            for p in iter_parsed_files(
                root, scan.files, repository="demo-repo", repository_id="demo__repo"
            )
        }

        print("\n--- parsed files ---")
        for path, p in sorted(parsed.items()):
            preview = p.content[:40].replace("\n", "\\n")
            print(f"  {path:26} {p.language:11} {p.line_count:>3} lines  "
                  f"enc={p.encoding:10} {preview!r}")
        print()

        check("parse: whitespace-only dropped", "whitespace.py" in parsed, False)
        check("parse: file count", len(parsed), 5)

        py = parsed["src/app.py"]
        check("parse: CRLF normalised", "\r" in py.content, False)
        check("parse: CRLF content", py.content, "import os\nprint('hi')\n")

        readme = parsed["README.md"]
        check("parse: BOM stripped", readme.content.startswith("# Title"), True)
        check("parse: BOM encoding label", readme.encoding, "utf-8-sig")
        check("parse: plain utf-8 label", py.encoding, "utf-8")
        check("parse: non-ascii preserved", "éé" in readme.content, True)

        js = parsed["src/auth/AuthService.js"]
        check("parse: language mapped", js.language, "javascript")
        check("parse: repository name", js.repository, "demo-repo")
        check("parse: repository id", js.repository_id, "demo__repo")
        check("parse: line count", js.line_count, 3)
        check("parse: char count", js.char_count, len(js.content))

        # --- summary (contents discarded) ---
        summary = summarise_repository(
            root, scan.files, repository="demo-repo", repository_id="demo__repo"
        )
        check("summary: files_parsed", summary.files_parsed, 5)
        check("summary: files_failed", summary.files_failed, 1)  # whitespace.py
        check("summary: total_lines > 0", summary.total_lines > 0, True)
        check(
            "summary: failure recorded",
            any("whitespace.py" in f for f in summary.failures),
            True,
        )

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
