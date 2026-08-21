"""Offline sanity checks for Step 1 (no network, no server needed).

Run from the `backend/` folder:
    python scripts/check_step1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow "python scripts/check_step1.py" to import the `app` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.exceptions import InvalidRepositoryURLError  # noqa: E402
from app.services.github_service import parse_repository_url  # noqa: E402

VALID_CASES = [
    ("https://github.com/pallets/flask", "pallets", "flask", None),
    ("https://github.com/pallets/flask.git", "pallets", "flask", None),
    ("https://github.com/pallets/flask/", "pallets", "flask", None),
    ("github.com/pallets/flask", "pallets", "flask", None),
    ("https://www.github.com/pallets/flask", "pallets", "flask", None),
    ("https://github.com/pallets/flask/tree/main", "pallets", "flask", "main"),
]

INVALID_CASES = [
    "",
    "   ",
    "not a url",
    "https://gitlab.com/owner/repo",
    "https://github.com/only-owner",
    "ftp://github.com/owner/repo",
    "https://github.com/owner/repo/issues/12",
]


def main() -> int:
    failures = 0

    for url, owner, name, branch in VALID_CASES:
        try:
            ref = parse_repository_url(url)
        except InvalidRepositoryURLError as exc:
            print(f"FAIL  {url!r} -> unexpectedly rejected: {exc.message}")
            failures += 1
            continue

        ok = (ref.owner, ref.name, ref.branch) == (owner, name, branch)
        print(
            f"{'PASS ' if ok else 'FAIL '} {url!r} -> "
            f"{ref.owner}/{ref.name} branch={ref.branch} id={ref.repository_id}"
        )
        failures += 0 if ok else 1

    for url in INVALID_CASES:
        try:
            parse_repository_url(url)
        except InvalidRepositoryURLError as exc:
            print(f"PASS  {url!r} -> rejected: {exc.message}")
        else:
            print(f"FAIL  {url!r} -> should have been rejected")
            failures += 1

    print("\nAll checks passed." if not failures else f"\n{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
