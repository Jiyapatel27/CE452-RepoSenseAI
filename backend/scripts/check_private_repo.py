"""Offline checks for private-repository support (token handling + redaction).

The important property here is that a GitHub token must never appear in a log
line or an API error response. git echoes the remote URL - which carries the
token - in its error output, so this verifies the redaction path.

Run from the `backend/` folder:
    python scripts/check_private_repo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.exceptions import (  # noqa: E402
    RepositoryAccessDeniedError,
    RepositoryNotFoundError,
)
from app.services.github_service import (  # noqa: E402
    _classify_git_failure,
    build_clone_url,
    parse_repository_url,
    redact,
)

TOKEN = "ghp_SuperSecret1234567890abcdefGHIJKL"
failures: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    ok = actual == expected
    print(f"{'PASS ' if ok else 'FAIL '} {label}: {actual!r}"
          + ("" if ok else f" (expected {expected!r})"))
    if not ok:
        failures.append(label)


def check_true(label: str, actual: bool) -> None:
    check(label, bool(actual), True)


def main() -> int:
    ref = parse_repository_url("https://github.com/acme/private-repo")

    # --- clone URL construction ---
    check(
        "public clone url has no credentials",
        build_clone_url(ref, None),
        "https://github.com/acme/private-repo.git",
    )
    check(
        "token embedded as x-access-token",
        build_clone_url(ref, TOKEN),
        f"https://x-access-token:{TOKEN}@github.com/acme/private-repo.git",
    )
    # A token with URL-special characters must not corrupt the URL.
    weird = "abc/def:ghi@jkl?mno"
    url = build_clone_url(ref, weird)
    check_true("special chars percent-encoded", "abc%2Fdef%3Aghi%40jkl%3Fmno" in url)
    check_true("url still points at the right repo", url.endswith("@github.com/acme/private-repo.git"))

    # --- redaction ---
    git_stderr = (
        f"Cloning into 'C:\\repos\\x'...\n"
        f"remote: Invalid username or password.\n"
        f"fatal: Authentication failed for "
        f"'https://x-access-token:{TOKEN}@github.com/acme/private-repo.git/'\n"
    )
    cleaned = redact(git_stderr, TOKEN)
    print("\n--- redacted git output ---")
    print("  " + cleaned.replace("\n", "\n  ").rstrip())
    print()

    check_true("token absent after redaction", TOKEN not in cleaned)
    check_true("useful message preserved", "Authentication failed" in cleaned)
    check_true("credentials masked", "***" in cleaned)
    check("redact handles empty input", redact("", TOKEN), "")
    check("redact handles no token", redact("plain text", None), "plain text")
    # Even an unknown token still gets caught by the user:pass@ pattern.
    check_true(
        "unknown credentials masked too",
        "othersecret" not in redact("https://u:othersecret@github.com/a/b", None),
    )

    # --- error classification ---
    with_token = _classify_git_failure(ref, cleaned, token_used=True)
    check_true("token failure -> access denied", isinstance(with_token, RepositoryAccessDeniedError))
    check("access denied is 403", with_token.status_code, 403)
    check_true("403 detail is redacted", TOKEN not in (with_token.detail or ""))
    check_true("403 message is redacted", TOKEN not in with_token.message)

    anon = _classify_git_failure(
        ref, "remote: Repository not found.\nfatal: repository not found", token_used=False
    )
    check_true("anon failure -> not found", isinstance(anon, RepositoryNotFoundError))
    check("not found is 404", anon.status_code, 404)
    check_true("404 message suggests a token", "token" in anon.message.lower())

    print(
        "\nAll checks passed."
        if not failures
        else f"\n{len(failures)} check(s) failed: {failures}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
