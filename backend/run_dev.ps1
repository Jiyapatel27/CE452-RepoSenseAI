# Start the RepoSense AI backend in development mode.
#
# Usage (from the backend/ folder):
#     .\run_dev.ps1
#
# Note on --reload: uvicorn ALWAYS watches the current working directory, even
# when --reload-dir is given, so a `git clone` into a folder under backend/
# would trigger a reload mid-request and kill the server on Windows
# (OSError: [WinError 87]). Cloned repos therefore live in
# ../.reposense-data/repos (see app/config.py). The excludes below are a second
# line of defence in case you point the storage dir somewhere else.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app `
    --host 127.0.0.1 `
    --port 8000 `
    --reload `
    --reload-dir app `
    --reload-exclude "data/*" `
    --reload-exclude ".reposense-data/*"
