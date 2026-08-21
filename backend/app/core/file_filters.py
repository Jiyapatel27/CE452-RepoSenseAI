"""Rules that decide which files of a repository we care about.

Kept in one place so Step 2 (parsing) and Step 3 (chunking) reuse the same
definitions instead of re-inventing them.
"""

from pathlib import Path

# Extension -> language label used everywhere downstream (metadata, prompts).
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".py": "python",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".md": "markdown",
    ".json": "json",
}

# Directory names that are never worth reading (dependencies, build output,
# caches, editor config, virtual environments).
IGNORED_DIRECTORIES: set[str] = {
    ".git",
    ".github",
    ".svn",
    ".hg",
    "node_modules",
    "bower_components",
    "vendor",
    "dist",
    "build",
    "out",
    "target",
    ".next",
    ".nuxt",
    ".turbo",
    ".cache",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "venv",
    ".venv",
    "env",
    ".env",
    "site-packages",
    ".idea",
    ".vscode",
    ".gradle",
    "bin",
    "obj",
}

# Exact file names to skip even though the extension is supported.
IGNORED_FILENAMES: set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
    "composer.lock",
    "poetry.lock",
    "Cargo.lock",
    "tsconfig.tsbuildinfo",
}

# Name endings that mark generated / minified / bundled artifacts.
IGNORED_FILENAME_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".map",
    ".d.ts",
    ".lock.json",
)


class SkipReason:
    """Why a file was excluded (used for reporting/debugging)."""

    IGNORED_DIRECTORY = "ignored_directory"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    IGNORED_FILENAME = "ignored_filename"
    TOO_LARGE = "too_large"
    EMPTY = "empty_file"
    BINARY = "binary_content"
    UNREADABLE = "unreadable"


def is_ignored_directory(name: str) -> bool:
    """True for directory names we never walk into (hidden dirs included)."""
    if name in IGNORED_DIRECTORIES:
        return True
    # Skip other hidden directories (.husky, .circleci, ...) but not "." / "..".
    return name.startswith(".") and name not in {".", ".."}


def detect_language(path: Path | str) -> str | None:
    """Map a file path to a language label, or None if unsupported."""
    return SUPPORTED_EXTENSIONS.get(Path(path).suffix.lower())


def is_ignored_filename(name: str) -> bool:
    if name in IGNORED_FILENAMES:
        return True
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in IGNORED_FILENAME_SUFFIXES)


def looks_binary(path: Path, sample_size: int = 4096) -> bool:
    """Cheap binary check: a NUL byte in the first few KB means binary."""
    try:
        with path.open("rb") as handle:
            return b"\x00" in handle.read(sample_size)
    except OSError:
        return True
