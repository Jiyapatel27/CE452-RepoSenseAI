# RepoSense AI

AI-powered GitHub repository understanding system.

Pipeline (built step by step):

```
GitHub URL -> Download -> Filter -> Parse -> Chunk -> Embed -> Qdrant
           -> Semantic Search -> Groq + Llama -> Grounded Answer -> Next.js UI
```

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Repository ingestion (clone + file filtering) | Done |
| 2 | File parsing (content + metadata) | Done |
| 3 | Code chunking | Done |
| 4 | Local embeddings (BAAI/bge-small-en-v1.5) | Done |
| 5 | Qdrant insertion | Done |
| 6 | Semantic search (no LLM) | Done |
| 7 | Groq RAG (see model note) | Done |
| 8 | Backend integration | Done |
| 9 | Next.js UI | Done |

---

## Step 1: Repository ingestion

### What it does

`POST /api/repository/analyze` validates a GitHub URL, shallow-clones the
repository, deletes the `.git` history, then walks the tree and reports every
supported source file. `GET /api/repository/status` reports what has been
ingested so far.

Supported extensions: `.js .jsx .ts .tsx .py .java .cpp .c .go .rs .md .json`

Ignored: `.git`, `node_modules`, `dist`, `build`, `.next`, `__pycache__`, other
hidden/dependency/build folders, lock files, minified and generated files,
binary files, empty files, and files over 1 MB.

### Requirements

- Python 3.10+ (tested on 3.12)
- Git on your `PATH`

Docker is **not** needed until Step 5 (Qdrant).

### One-time setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` can stay empty for Step 1 - `GROQ_API_KEY` is only used from Step 7.

### Run the server

```powershell
cd backend
.\run_dev.ps1
```

Leave that terminal open. The API is at http://localhost:8000 and interactive
docs at http://localhost:8000/docs.

> **Why `run_dev.ps1` instead of plain `uvicorn --reload`?**
> uvicorn always watches the current working directory, so if cloned repos are
> stored under `backend/`, each `git clone` triggers a reload mid-request and
> kills the server on Windows (`OSError: [WinError 87]`). Clones therefore go to
> `.reposense-data/repos/` at the project root, outside the watched folder.

### Test it

Offline URL-validation checks (no server, no network needed):

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step1.py
```

Then, with the server running, in a **second** terminal:

```powershell
# 1. Health check
Invoke-RestMethod http://localhost:8000/health

# 2. Analyze a medium-sized public repo
$body = '{"repository_url":"https://github.com/gothinkster/node-express-realworld-example-app"}'
$r = Invoke-RestMethod http://localhost:8000/api/repository/analyze -Method Post -ContentType application/json -Body $body
$r | Select-Object repository_id, total_files_scanned, supported_files, truncated
$r.language_counts
$r.files | Select-Object -First 10 file_path, language, size_bytes

# 3. Check status
Invoke-RestMethod http://localhost:8000/api/repository/status
```

Expected for that repo: ~66 files scanned, ~50 supported, mostly TypeScript.

### Error handling

| Case | Response |
|------|----------|
| Not a github.com URL | 400 `invalid_repository_url` |
| Missing `repository_url` | 422 (FastAPI validation) |
| Repo missing, or private with no token | 404 `repository_not_found` |
| Token rejected or lacks access | 403 `repository_access_denied` |
| Branch does not exist | 400 `invalid_repository_url` |
| Network down / clone timeout | 502 `repository_download_failed` |
| Empty repository | 422 `empty_repository` |
| No supported files | 422 `no_supported_files` |
| Git not installed | 500 `git_not_installed` |

Try one:

```powershell
try {
  Invoke-RestMethod http://localhost:8000/api/repository/analyze -Method Post `
    -ContentType application/json -Body '{"repository_url":"https://gitlab.com/a/b"}'
} catch { $_.ErrorDetails.Message }
```

---

## Private repositories

Public **and** private repositories both work. Private ones need a GitHub
Personal Access Token.

> This is token-based cloning, not GitHub OAuth login. There are no user
> accounts or sessions - that remains out of scope.

### Creating a token

1. Go to https://github.com/settings/personal-access-tokens (fine-grained) or
   https://github.com/settings/tokens (classic).
2. Fine-grained: select the repositories you need, then grant
   **Repository permissions -> Contents -> Read-only**.
   Classic: tick the **`repo`** scope.
3. Copy the token (`github_pat_...` or `ghp_...`). GitHub shows it once.

For an organisation repo, an owner may need to approve the token, and if the org
enforces SAML SSO you must authorise the token for that org.

### Two ways to supply it

**A. Server-wide** - put it in `backend/.env`, good for your own demo repos:

```
GITHUB_TOKEN=github_pat_xxxxxxxxxxxx
```

**B. Per request** - send it in the request body. This is what the Step 9 UI will
use, so any user can analyse their own private repo without the server holding
standing credentials:

```powershell
$body = @{
  repository_url = "https://github.com/your-org/your-private-repo"
  github_token   = "github_pat_xxxxxxxxxxxx"
} | ConvertTo-Json

$r = Invoke-RestMethod http://localhost:8000/api/repository/analyze `
     -Method Post -ContentType application/json -Body $body
$r | Select-Object repository_id, authenticated, supported_files
```

A request token overrides `GITHUB_TOKEN`. If a token is rejected, the clone is
retried anonymously, so a stale `.env` token cannot break public repositories.

### Token safety

- Tokens are **never stored** - not in the repository record, not on disk.
- Tokens are **never logged**. Only the clean URL is logged, plus
  `(with token)` / `(anonymous)`.
- git echoes the remote URL (including credentials) in its error output, so all
  git output passes through `redact()` before reaching a log line or an API
  response. Verified by `scripts/check_private_repo.py`.
- `.git/` is deleted after cloning, which also removes the tokenised remote URL
  git wrote into `.git/config`.
- Tokens are percent-encoded into the URL, so special characters cannot corrupt
  it.

One residual exposure: while `git clone` runs, the token is present in that
process's command line, which other processes on the same machine could read.
Acceptable for local development; a production deployment should switch to a
credential helper or `http.extraHeader`.

### Errors

| Case | Response |
|---|---|
| Private repo, no token | 404 `repository_not_found` (message suggests a token) |
| Token invalid / expired / lacks access | 403 `repository_access_denied` |

### Test it

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_private_repo.py
```

Expect `All checks passed.` (18 assertions, focused on token redaction).

---

## Step 2: File parsing

### What it does

Every discovered file is read, decoded and normalised into this shape:

```json
{
  "repository": "example-repo",
  "repository_id": "owner__example-repo",
  "file_path": "src/auth/AuthService.js",
  "file_name": "AuthService.js",
  "extension": ".js",
  "language": "javascript",
  "line_count": 42,
  "char_count": 1180,
  "encoding": "utf-8",
  "content": "..."
}
```

Handled while decoding:

- **UTF-8 BOM** stripped (common on Windows), reported as `utf-8-sig`
- **CRLF / CR** normalised to `LF`, so chunk boundaries in Step 3 do not depend
  on the platform the code was written on
- **UTF-16** and **latin-1** fallbacks, so an oddly-encoded file is still
  indexed instead of silently dropped
- **Whitespace-only files** dropped (they carry no retrievable meaning)

`POST /api/repository/analyze` now also parses everything once and returns
`parse_stats`, so decode problems surface immediately rather than during
embedding.

> **Design note:** file contents are *not* kept in memory. The repository record
> stores metadata and totals only; contents are re-read from disk through a
> generator (`iter_parsed_files`). A 5000-file repo would otherwise pin hundreds
> of megabytes for the lifetime of the server.

### New endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/repository/files` | Paginated file metadata (`limit`, `offset`) |
| `GET /api/repository/file` | One file parsed, **including content** |

### Test it

Offline checks — builds a synthetic repo covering BOM, CRLF, binary, empty,
oversized, `node_modules` and lock files, then asserts on the results:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step2.py
```

Expect `All checks passed.` (33 assertions).

With the server running, in a second terminal:

```powershell
# Analyze, then look at the new parse_stats block
$body = '{"repository_url":"https://github.com/gothinkster/node-express-realworld-example-app"}'
$r = Invoke-RestMethod http://localhost:8000/api/repository/analyze -Method Post -ContentType application/json -Body $body
$id = $r.repository_id
$r.parse_stats

# List files
(Invoke-RestMethod "http://localhost:8000/api/repository/files?repository_id=$id&limit=5").files

# Fetch one file's parsed content
$path = "src/app/routes/auth/auth.service.ts"
$f = Invoke-RestMethod "http://localhost:8000/api/repository/file?repository_id=$id&file_path=$path"
$f | Select-Object file_path, language, line_count, char_count, encoding
$f.content.Substring(0, 300)
```

That file is 4248 bytes on disk but reports `char_count` 4065 - the 183 CRLF
pairs became single LF characters, which is the normalisation doing its job.

If you would rather not hard-code a path, let the API pick one for you:

```powershell
$path = (Invoke-RestMethod "http://localhost:8000/api/repository/files?repository_id=$id&limit=500").files |
        Where-Object { $_.language -eq 'typescript' } | Select-Object -First 1 -ExpandProperty file_path
```

### Error handling

| Case | Response |
|------|----------|
| Unknown `repository_id` | 404 `repository_not_found` |
| File not in the analysed set | 404 `file_not_found` |
| Clone deleted from disk after analysis | 410 `repository_files_missing` |

`file_path` is matched against the stored list of discovered files rather than
being joined onto the filesystem path, so `../../secrets` cannot escape the
repository root.

---

## Step 3: Code chunking

### What it does

Each parsed file is split into embedding-sized chunks with LangChain's
`RecursiveCharacterTextSplitter`:

```json
{
  "chunk_id": "af4e279b-...",
  "repository": "weatherwise",
  "repository_id": "jiya2406__weatherwise",
  "file_path": "server/config/db.js",
  "file_name": "db.js",
  "language": "javascript",
  "chunk_index": 0,
  "content": "const mongoose = require(\"mongoose\");\n...",
  "char_count": 621,
  "start_line": 1,
  "end_line": 18
}
```

### Why language-aware splitting

We use `RecursiveCharacterTextSplitter.from_language()`, not the plain splitter.
The generic splitter breaks on blank lines and spaces, which cheerfully cuts a
function in half. The language profile supplies separators (`\nclass `, `\ndef `,
`\nfunction `, ...) so the splitter *prefers* boundaries between definitions and
only falls back to crude splits when one definition exceeds `chunk_size`.

That is what makes "where is X implemented?" answerable - the whole function
stays in one chunk. Measured on `weatherwise`, `server/config/db.js` becomes a
single 621-char chunk containing the entire `connectDB` function plus its
docstring.

`json` has no LangChain profile, so those files use the generic splitter.

### About `chunk_overlap` (this surprises people)

Overlap does **not** appear between every pair of chunks. LangChain only carries
text forward when it had to cut *inside* a unit:

| Situation | Overlap |
|---|---|
| Boundary falls between whole functions | **0 chars** |
| One function longer than `chunk_size`, split mid-body | **~30 chars** |

Zero overlap at a clean boundary is correct - the chunk is already a complete
semantic unit. Overlap appears exactly where context would otherwise be lost.

### Chunk ids

`chunk_id` is a deterministic UUID5 of `repository_id:file_path:chunk_index`.
Stable ids make re-indexing a repository in Step 5 an idempotent upsert instead
of inserting duplicates.

### Configuration - sizes are in TOKENS

```
CHUNK_SIZE=400        # tokens
CHUNK_OVERLAP=60      # tokens
MIN_CHUNK_CHARS=40    # characters (a noise filter, not a size limit)
```

**Why tokens and not characters.** `bge-small-en-v1.5` truncates at **512
tokens**. Character sizing cannot bound that, because token density varies
enormously across code. Measured over two real repositories:

| | chars/token |
|---|---|
| Average | ~3.0 |
| **Worst-case chunk** | **~1.4-2.0** |

So a 1000-character chunk is anywhere from ~330 to ~700 tokens. Measured with
character sizing at `chunk_size=1000`:

| repo | max tokens | result |
|---|---|---|
| weatherwise | 451 | OK |
| realworld-ts | **514** | **1 chunk truncated** |

A truncated chunk is the worst kind of bug: it still produces a valid-looking
384-float vector, but that vector represents only the chunk's opening fragment
while the API keeps reporting the full text. Nothing errors; retrieval just
quietly gets worse.

Sizing in tokens - via the embedding model's own tokeniser as the splitter's
`length_function` - makes this structurally impossible. Measured after the
change:

| CHUNK_SIZE (tokens) | chunks (weatherwise) | max tokens | over 512 |
|---|---|---|---|
| 256 | 117 | 244 | 0 |
| **400** | **77** | **396** | **0** |
| 480 | 64 | 460 | 0 |
| 512 | 62 | 499 | 0 |

Zero truncation at every setting, on both repositories. Token sizing also adapts
sensibly: at 400 tokens a dense code chunk is ~800 characters while prose-like
markdown reaches ~1500.

`MIN_CHUNK_CHARS` stays in characters - it only drops noise fragments (a lone
`}`) produced by a split. A file small enough to fit in one chunk is always kept
regardless, so short-but-real source files never vanish from the index.

Because chunk sizes are token-based, chunking loads the embedding model. The
chunker itself stays model-agnostic: `length_function` is injected, defaulting to
character counting when omitted (which keeps `check_step3.py` fast and free of
PyTorch).

### New endpoint

`GET /api/repository/chunks` - inspect chunks, optionally for one file:

```powershell
$id = "jiya2406__weatherwise"
# All chunks, first page
$c = Invoke-RestMethod "http://localhost:8000/api/repository/chunks?repository_id=$id&limit=5"
$c.total
$c.chunks | Select-Object chunk_index, file_path, start_line, end_line, char_count

# Chunks of one file
$c = Invoke-RestMethod "http://localhost:8000/api/repository/chunks?repository_id=$id&file_path=server/config/db.js"
$c.chunks[0].content
```

Chunks are regenerated from disk per call rather than cached, so the endpoint
always reflects current settings and costs no memory between requests.

### Test it

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step3.py
```

Expect `All checks passed.` (34 assertions: metadata, contiguous indices,
deterministic ids, line-number accuracy, overlap behaviour, language-aware
boundaries, size configurability).

---

## Step 4: Local embeddings

### What it does

Turns chunk text into 384-dimensional vectors using
**BAAI/bge-small-en-v1.5** via `sentence-transformers`. Runs entirely on your
machine - no paid embedding API. The model (~130 MB) downloads once on first use
and is cached by Hugging Face.

```
Code chunk -> bge-small-en-v1.5 -> [0.021, -0.118, ...]  (384 floats)
```

### Two details that quietly ruin retrieval

**1. BGE is asymmetric.** A search *query* must be prefixed with
`"Represent this sentence for searching relevant passages: "`; stored *passages*
must not be. Embedding both sides identically still produces vectors and
plausible-looking scores, so the mistake is invisible - ranking quality just
drops. Hence two separate functions:

| Function | Prefix applied | Use for |
|---|---|---|
| `embed_documents()` | No | chunks going into Qdrant |
| `embed_query()` | Yes | the user's question |

**2. Vectors are L2-normalised.** With unit vectors, cosine similarity equals the
dot product, so scores land in a predictable range and Qdrant's `COSINE`
distance behaves consistently.

### Loading is lazy

Importing `sentence-transformers` pulls in PyTorch, which is slow. The model
loads on first use behind a lock, not at import time - otherwise every server
start and every unrelated test would pay for it.

### Configuration

```
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=auto          # auto | cpu | cuda
```

### New endpoint

```powershell
Invoke-RestMethod http://localhost:8000/api/embeddings/info
```

Returns model name, `dimension` (384), `max_sequence_length` (512), device,
batch size and the query prefix. **The first call takes a few seconds** (or
longer on first ever run, while the model downloads); later calls are instant.

### Test it

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step4.py
```

This does more than check vector shapes. It also:

- counts tokens of **real** chunks to prove none hit the 512-token limit
  (a truncated chunk still yields a vector, so this is the only way to catch it)
- runs a **real retrieval test** over the cloned repo with plain cosine
  similarity - no Qdrant needed - which previews Step 6
- checks an **unanswerable** question scores lower than answerable ones, which
  is what the Step 7 "context is insufficient" behaviour will depend on

### Measured results on `weatherwise`

```
dim 384 | max_seq_len 512 | device cpu
91 chunks embedded in 10.1s (9.0 chunks/sec, CPU)
max 396 tokens / chunk (limit 512)
```

| Question | Top result | Score |
|---|---|---|
| Where is the database connection initialized? | `server/app.js` (`db.js` 2nd) | 0.632 |
| How are weather API requests made? | `server/services/weatherService.js` | 0.770 |
| How is search history saved? | `server/controllers/historyController.js` | 0.790 |
| *Where is JWT authentication implemented?* (no such code) | `server/app.js` | **0.578** |

The last row matters: `weatherwise` has no auth code, and the best match scores
**0.578** against **0.632-0.790** for answerable questions. The ordering is
right, but the margin is only ~0.05 - so Step 7 should not lean on a hard score
threshold to decide "context is insufficient". Prompt-level grounding
instructions are the more reliable mechanism, with the score as a weak signal.

Also note `README.md` and `CLAUDE.md` compete strongly for most questions, since
prose describing a feature resembles a question about it more than the code does.

---

## Step 5: Qdrant vector store

### Start Qdrant

From the **project root** (the folder holding `docker-compose.yml`):

```powershell
docker compose up -d
docker compose ps          # expect: Up (healthy)
```

Docker Desktop must be running first. Dashboard: http://localhost:6333/dashboard

> **Windows note.** Docker Desktop 29.x installs to
> `%LOCALAPPDATA%\Programs\DockerDesktop` and may create no Start Menu shortcut.
> If `docker` is "not recognized", open a **new** terminal - PATH updates only
> reach newly launched shells. If you get `no configuration file provided`, you
> are not in the project root.

### What it does

`POST /api/repository/index` re-parses, re-chunks, embeds, and upserts a
repository into Qdrant. Each chunk becomes one point: a 384-dim vector plus a
payload carrying everything needed to answer with citations.

```json
{
  "repository_id": "jiya2406__weatherwise",
  "repository": "weatherwise",
  "file_path": "server/config/db.js",
  "file_name": "db.js",
  "language": "javascript",
  "chunk_index": 0,
  "content": "const mongoose = require(\"mongoose\");\n...",
  "start_line": 1,
  "end_line": 18,
  "char_count": 621,
  "token_count": 158
}
```

Chunk text is stored in the payload, so Steps 6 and 7 never touch the filesystem
again - a search result already contains the code to show and to feed the LLM.

### Design decisions

**One collection, not one per repository.** The embedding dimension is fixed, so
a single collection suffices. Repositories are separated by a `repository_id`
payload field with a **keyword index** on it. That index is not optional -
without it, every filtered search degrades to a full scan.

**Cosine distance**, matching the L2-normalised vectors from Step 4.

**Deterministic point ids.** Chunk ids are UUID5s of
`repository_id:file_path:chunk_index`, so re-indexing overwrites the same points
rather than inserting duplicates.

**Delete-then-upsert on re-index.** Stable ids alone are not enough: if a file is
deleted from the repository, or shrinks so it produces fewer chunks, the stale
points would linger and keep being retrieved. Clearing the repository's points
first guarantees the index matches the current code. Verified by the test - a
second index run reports `vectors_deleted_first: 77` and the count stays at 77.

**Batched upserts** (`QDRANT_BATCH_SIZE`, default 128). One request per chunk
would make indexing network-bound instead of embedding-bound.

**Streaming.** Parse -> chunk -> embed -> upsert is a generator pipeline, so a
large repository never holds all its chunks or vectors in memory.

### Configuration

```
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=reposense_chunks
QDRANT_BATCH_SIZE=128
QDRANT_TIMEOUT_SECONDS=30
QDRANT_API_KEY=              # only for Qdrant Cloud; local needs none
```

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/repository/index` | Embed + store a repository's chunks |
| `GET /api/vectorstore/info` | Qdrant health, collection config, per-repo counts |

```powershell
$body = '{"repository_url":"https://github.com/Jiya2406/weatherwise"}'
$r = Invoke-RestMethod http://localhost:8000/api/repository/analyze -Method Post -ContentType application/json -Body $body

$ix = @{ repository_id = $r.repository_id } | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/api/repository/index -Method Post -ContentType application/json -Body $ix

Invoke-RestMethod http://localhost:8000/api/vectorstore/info
```

Measured on `weatherwise`: **77 chunks indexed in ~10.5s** (~7 chunks/sec, CPU),
one batch.

`GET /api/vectorstore/info` deliberately returns `reachable: false` with HTTP 200
rather than a 503 when Qdrant is down - "is Qdrant up?" is the question this
endpoint exists to answer, so it should still return a body.

### Errors

| Case | Response |
|---|---|
| Qdrant not running | 503 `vector_store_unavailable` (with the `docker compose up -d` hint) |
| Qdrant rejects the request | 500 `vector_store_error` |
| Unknown `repository_id` | 404 `repository_not_found` |
| Clone deleted from disk | 410 `repository_files_missing` |

### Test it

Needs Qdrant running:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step5.py
```

Expect `All checks passed.` (30 assertions). It verifies collection config,
the payload index exists, idempotent re-indexing, payload completeness,
**repository isolation** (indexes a second probe repo and confirms filtered
search never leaks across it), score ordering, and cleanup.

---

## Step 6: Semantic search (no LLM)

```
query -> embedding -> Qdrant (filtered by repository) -> top-K chunks
```

### `POST /api/search`

```json
{
  "repository_id": "jiya2406__weatherwise",
  "query": "Where is the database connection initialized?",
  "top_k": 5,
  "languages": ["javascript"],
  "min_score": 0.0
}
```

Response:

```json
{
  "repository_id": "jiya2406__weatherwise",
  "query": "Where is the database connection initialized?",
  "top_k": 5,
  "result_count": 5,
  "results": [
    {
      "file_path": "server/config/db.js",
      "score": 0.631,
      "content": "const mongoose = require(\"mongoose\");\n...",
      "language": "javascript",
      "start_line": 1,
      "end_line": 18,
      "chunk_index": 0,
      "chunk_id": "af4e279b-..."
    }
  ],
  "files": ["server/app.js", "server/config/db.js", "CLAUDE.md"],
  "elapsed_ms": 195.6
}
```

`file_path`, `score` and `content` are the three fields the brief requires; the
rest are additions that Step 7 and the UI need. `files` is the deduplicated
path list, best match first - ready for the UI's "Relevant Files" section, since
several chunks usually come from the same file.

### Design decisions

**Search reads only Qdrant.** It does *not* require the repository to be in the
in-memory store. Vectors survive a server restart; the store does not. Coupling
them would force a pointless re-analyse just to search already-indexed code.

**Query and passage embeddings are asymmetric.** Queries go through
`embed_query` (which applies the BGE instruction prefix); passages were embedded
in Step 5 *without* it. That is what the model was trained for - using one
function for both measurably degrades ranking.

**The model is preloaded at startup.** Lazy loading made the first search take
**12.3 seconds** while PyTorch and the model initialised - long enough to look
like a hang. Warming it in the app's lifespan hook brings that to **~195 ms**.
Set `PRELOAD_EMBEDDING_MODEL=false` if the `--reload` restarts get annoying
during development.

**`min_score` is off by default.** See the numbers below - the gap between
answerable and unanswerable questions is only ~0.05, far too narrow for a
hard cutoff.

### The `languages` filter matters more than it looks

Prose describing a feature often out-ranks the code implementing it. Measured on
`weatherwise`:

| *"Which files handle API requests?"* | Top 3 |
|---|---|
| No filter | `README.md`, `server/package.json`, `client/package.json` |
| `languages: ["javascript"]` | `server/app.js`, `server/services/weatherService.js` x2 |

Unfiltered results are useless for that question - `package.json` matches because
it *lists* an API library, not because it handles requests. The filter is
explicit rather than guessed from the query, and both `repository_id` and
`language` have keyword payload indexes so filtering stays fast.

### Retrieval quality on the brief's five questions

Measured against `weatherwise`, which has **no authentication code at all**:

| Question | Best hit | Score |
|---|---|---|
| Where is the database connection initialized? | `server/app.js`, then `config/db.js` | 0.632 |
| Which files handle API requests? | `README.md` (`server/app.js` with filter) | 0.695 |
| *Where is authentication implemented?* | `CLAUDE.md` | **0.587** |
| *How does login work?* | `server/app.js` | **0.560** |
| *How is a user created?* | `SearchBar.jsx` | **0.559** |

The last three have no real answer in this repository, and they do score lower -
but only by ~0.05. **This is the key input to Step 7:** a score threshold cannot
reliably detect "I don't know", so the prompt must instruct the model to refuse
when the context does not support an answer.

### Errors

| Case | Response |
|---|---|
| Repository has no vectors | 409 `repository_not_indexed` |
| Blank query | 400 `invalid_search_query` |
| Missing `query` field | 422 (FastAPI validation) |
| `top_k` above 50 | 422 |
| Qdrant down | 503 `vector_store_unavailable` |
| Nothing matched | **200 with `results: []`** - not an error |

### Test it

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step6.py
```

Expect `All checks passed.` (36 assertions). It runs through FastAPI's
`TestClient`, so it exercises the real route, validation and error handling -
including the lifespan hook that preloads the model.

Manually:

```powershell
$body = @{
  repository_id = "jiya2406__weatherwise"
  query         = "Where is the database connection initialized?"
  top_k         = 3
} | ConvertTo-Json

$r = Invoke-RestMethod http://localhost:8000/api/search -Method Post -ContentType application/json -Body $body
$r.results | Select-Object score, file_path, start_line, end_line
$r.files
```

---

## Step 7: RAG answers (Groq)

```
question -> embedding -> Qdrant -> top-K chunks -> prompt -> Groq -> answer
```

### Important: Groq no longer serves Llama chat models

The brief specified "Groq API + Llama 3.x". As of this build that is **not
possible**: `llama-3.3-70b-versatile` returns HTTP 404 "model not found", and the
only remaining Llama entries are `llama-prompt-guard-2-22m/86m` - 512-token
safety classifiers, not chat models.

Chat models Groq currently serves:

| Model | Context | Max output |
|---|---|---|
| **`openai/gpt-oss-120b`** (our default) | 131k | 65k |
| `openai/gpt-oss-20b` | 131k | 65k |
| `qwen/qwen3.6-27b` | 131k | 16k |
| `groq/compound`, `groq/compound-mini` | 131k | 8k |
| `allam-2-7b` | 4k | 4k |

Only the model *name* changed - no code depends on which model it is. Set
`GROQ_MODEL` in `.env` to switch. Check what is live:

```powershell
.\.venv\Scripts\python.exe -c "from groq import Groq; print([m.id for m in Groq().models.list().data])"
```

A retired model name produces a clear error rather than a mystery 404:

```
502 llm_unavailable
message: Groq returned an error (HTTP 404).
detail : Model 'llama-3.3-70b-versatile' may no longer be available.
         List current models with: ...
```

### `POST /api/chat`

```json
{
  "repository_id": "jiya2406__weatherwise",
  "question": "Where is the database connection initialized?",
  "top_k": 6,
  "languages": ["javascript"]
}
```

Response:

```json
{
  "answer": "The MongoDB connection is set up in `server/config/db.js` ...",
  "sources": ["server/app.js", "server/config/db.js"],
  "model": "openai/gpt-oss-120b",
  "context_chunks": 6,
  "prompt_tokens": 1355,
  "completion_tokens": 215,
  "elapsed_ms": 2980.3,
  "context": [
    {"file_path": "server/config/db.js", "score": 0.631,
     "start_line": 1, "end_line": 18, "language": "javascript"}
  ]
}
```

`context` lists exactly what the model was shown, so every claim in `answer` can
be checked against the code it came from.

### Making it refuse instead of inventing

This is the hard part, and it is a **prompt** problem, not a threshold problem.
Step 6 measured why: on a repository with no authentication code, the best
"where is authentication implemented?" match still scored **0.587**, against
**0.632** for a question the code genuinely answers. A ~0.05 margin cannot
separate the two, so any score cutoff would either discard good results or admit
junk.

What actually works:

1. **Labelled context.** Each chunk is prefixed
   `[1] server/config/db.js (lines 1-18, javascript)`. The model has exact
   strings to cite, and an invented path is visibly absent from the context.
2. **A system prompt that forbids outside knowledge**, requires every identifier
   to appear verbatim in the context, and demands an explicit
   "the context does not show..." when the answer is not there.
3. **`temperature=0`.** Creativity here manifests as invented filenames.
4. **Skip the LLM entirely when retrieval returns nothing** - answer from a
   constant instead of asking the model to work from an empty context.

Measured on `weatherwise` (which has **no** auth code):

| Question | Behaviour |
|---|---|
| Where is the database connection initialized? | Answers, cites `server/config/db.js` + `server/app.js` |
| Where is JWT authentication implemented? | **Refuses**, and lists the files it did see |
| How does user login work? | **Refuses** |
| How are passwords hashed? | **Refuses** |

The refusals even enumerate which files were searched, which is far more useful
than a bare "I don't know".

`scripts/check_step7.py` verifies grounding mechanically, not by eye:

- every path in `sources` exists in the indexed file set (read from Qdrant)
- every file path mentioned in the prose corresponds to a real file
- every cited line number falls inside a chunk that was actually supplied
- all three unanswerable questions are refused

### `sources` means "cited", not "retrieved"

Six chunks go into the prompt, but a question about the database still retrieves
`README.md` as filler. Listing all six as sources would make the UI's "Relevant
Files" section misleading, so `sources` is filtered to files the answer actually
references (by full path, then by bare filename), falling back to the full
context only when nothing matched - which is the normal case for a refusal.

### Answer normalisation

Two real artifacts of this model, both cleaned in `normalise_answer`:

1. **Exotic typography.** `gpt-oss` writes ranges as `lines 1‑18` using a narrow
   no-break space (U+202F) and a non-breaking hyphen (U+2011). Typographically
   correct, but it breaks naive `\s`/`-` handling downstream and renders as
   "118" in consoles that cannot display those characters - which looks
   alarmingly like an invented line number.
2. **Citation markers.** It sometimes emits its own reference tokens, e.g.
   `...the HS256 algorithm【2†L1-L15】.` These are an artifact of its training
   format and meaningless to a user. We supply our own file/line citations, so
   they are stripped (along with the stray spacing left behind).

### Answer truncation is reported

`groq_max_tokens` defaults to 1600. At 1024 a "how does login work?" answer that
walks through both a controller and a service quoted enough code to hit the cap
and stop mid-sentence. When it does happen, the response sets
`truncated_answer: true` so the UI can say so rather than present a cut-off
explanation as complete.

### Verified on a repository that *has* auth

Refusal is only half the proof - a system that refuses everything would pass
that test. So the same five questions were run against
`gothinkster/node-express-realworld-example-app` (95 chunks), which really does
implement JWT auth:

| Question | Result |
|---|---|
| Where is authentication implemented? | Answers: `auth.ts` (JWT middleware, `express-jwt`, HS256) + `auth.service.ts` (`generateToken`, bcrypt) |
| How is a user created? | Answers, quoting the `POST /users` handler from `auth.controller.ts` |
| How does login work? | Answers, tracing controller -> service |
| Which files handle API requests? | Answers: `src/main.ts` registers the router |
| Where is the database connection initialized? | **Refuses** - see below |

That last row is instructive. The repo *does* have
`src/prisma/prisma-client.ts`, but retrieval did not surface it, and the model
said so precisely: *"The seed script uses a `prisma` instance, but the file where
that instance is created is not included in the context."* It refused rather than
guessing - so the weakness is retrieval **recall**, not grounding. Raising
`top_k` or filtering by language usually fixes cases like this.

### Configuration

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b
GROQ_TEMPERATURE=0.0
GROQ_MAX_TOKENS=1024
CHAT_TOP_K=6
CHAT_MAX_CONTEXT_TOKENS=12000
```

### Errors

| Case | Response |
|---|---|
| `GROQ_API_KEY` missing or rejected | 500 `llm_not_configured` |
| Groq rate limit | 429 `llm_rate_limited` (the SDK retries first) |
| Groq unreachable / retired model | 502 `llm_unavailable` |
| Repository not indexed | 409 `repository_not_indexed` |
| Blank question | 400 `invalid_search_query` |
| Retrieval found nothing | 200, canned answer, `llm_called: false` |

### Test it

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step7.py
```

Expect `All checks passed.` (33 assertions).

```powershell
$body = @{
  repository_id = "jiya2406__weatherwise"
  question      = "How is search history saved?"
} | ConvertTo-Json

$r = Invoke-RestMethod http://localhost:8000/api/chat -Method Post -ContentType application/json -Body $body
$r.answer
$r.sources
```

---

## Step 8: Backend integration

Three papercuts removed before the UI is built on top.

### 1. One call does the whole pipeline

`POST /api/repository/analyze` now takes `auto_index` (default **true**) and
runs clone -> filter -> parse -> chunk -> embed -> index in a single request:

```json
{
  "repository_url": "https://github.com/Jiya2406/weatherwise",
  "auto_index": true
}
```

```json
{
  "repository_id": "jiya2406__weatherwise",
  "supported_files": 31,
  "chunk_stats": { "total_chunks": 77, ... },
  "indexed": true,
  "chunks_indexed": 77,
  "status": "indexed"
}
```

The repository is searchable the moment this returns - no second call. Pass
`auto_index: false` to inspect files and chunks without paying the embedding
cost.

**Chunking still happens only once.** Tokenising every file is the expensive
part, so measuring the chunks and indexing them share one pass:
`iter_chunks(summary=...)` fills the statistics while the indexer consumes the
same generator. Doing it naively would double the work of every analyse call.

### 2. Repository records survive a restart

Vectors live in Qdrant and outlive the process; the repository store is
in-memory and does not. Previously that meant a restarted server answered
"no analysed repository" for code it could already search, and you had to
re-clone to get `/chunks` or `/files` back.

The app now rebuilds those records from Qdrant payloads on startup:

```
INFO  app.main | Restored 2 repository record(s) from Qdrant
```

Restored records are deliberately **partial and labelled**. Qdrant stores chunk
payloads, not ingestion statistics, so `total_files_scanned`, `skipped_files`
and the parse counts are genuinely unknown - they read **0** and `status` is
`"restored"` rather than quietly inventing plausible numbers. Re-running
`/analyze` fills them in.

What *is* recovered: file list, languages, chunk count, `indexed`, and the clone
path - enough for `/status`, `/files` and `/chunks` to work again.

### 3. `/health` reports each dependency separately

```json
{
  "status": "ok",
  "ready_for_chat": true,
  "qdrant":     { "reachable": true, "points": 172, "repositories": {...} },
  "groq":       { "configured": true, "model": "openai/gpt-oss-120b" },
  "embeddings": { "model": "BAAI/bge-small-en-v1.5", "preloaded": true },
  "repositories_in_memory": 2
}
```

It always returns **200**, even when a dependency is down - a health check that
500s tells you nothing about *which* part broke. The UI reads the booleans, so it
can say "start Qdrant" or "add your Groq key" instead of showing a generic
error.

### Test it

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_step8.py
```

Expect `All checks passed.` (44 assertions), including a **simulated restart**:
the test wipes the in-memory store, confirms `/status` correctly 404s while
`/search` still works (because search reads Qdrant directly), then restores and
confirms every dependent endpoint works again - and that a second restore is a
no-op.

---

## Step 9: Next.js frontend

Next.js 16 (App Router) + React 19 + Tailwind v4, plain JavaScript. One page,
no state library, no component library.

### Run it

Three things must be running. **Terminal 1** - Qdrant (from the project root):

```powershell
docker compose up -d
```

**Terminal 2** - backend:

```powershell
cd backend
.\run_dev.ps1
```

**Terminal 3** - frontend:

```powershell
cd frontend
npm install     # first time only
npm run dev
```

Then open **http://localhost:3000**.

### What the page does

```
REPOSENSE AI

GITHUB REPOSITORY URL
[ https://github.com/owner/repository ]
Private repository? Add a GitHub token
[ Analyze Repository ]

REPOSITORY STATUS
gothinkster__node-express-realworld-example-app
Files: 50 | Chunks: 95 | Indexed: OK
typescript 37 | json 11 | markdown 1 | javascript 1

ASK ABOUT YOUR REPOSITORY
[ Where is authentication implemented? ]  [ Ask ]

AI RESPONSE
Authentication is handled in the auth route files:
  - src/app/routes/auth/auth.service.ts - contains the logic that validates
    credentials, hashes passwords and creates a JWT token with
    generateToken(user.id) (see lines 183-211).
  ...

RELEVANT FILES
- src/app/routes/auth/auth.service.ts
- src/app/routes/auth/auth.ts
> Show the exact code the model was given (6 chunks)
```

Beyond the brief's layout, four things earned their place:

**A GitHub token field**, collapsed by default. Private repositories were a hard
requirement, and a server-side `.env` token cannot cover a repository the server
has no access to. It is sent with that one request and never stored.

**A setup banner.** The page calls `/health` on load; if Qdrant is down or
`GROQ_API_KEY` is missing it says exactly that, with the command to fix it,
instead of failing mysteriously on first use.

**An "already indexed" list.** Vectors survive a backend restart (Step 8), so
previously analysed repositories are offered as one-click choices - no re-clone
needed to ask a question.

**A progress explanation.** Analyse takes 10-30s because embeddings run locally
on CPU. Without a note saying so, that reads as a hang.

**Collapsible context.** `Show the exact code the model was given` lists every
chunk with its file, line range and score, so any answer can be checked against
its source.

### Rendering the answer

The model replies in Markdown, and it consistently uses three constructs:
`**bold**`, `` `inline code` ``, and `- ` bullets. Rendered as plain text those
show up as literal asterisks and backticks, which makes the primary output look
broken.

`app/AnswerText.js` handles exactly those three - about 40 lines, no Markdown
dependency (the brief says avoid unnecessary technologies). It builds React
elements rather than setting `innerHTML`, so model output can never inject
markup.

One subtlety: the model nests them, writing ``**`src/app/auth.ts`**`` - bold
wrapping code. A single-pass splitter renders that with visible backticks
*inside* the bold text, so the bold branch recurses once to handle the inner
span.

### Verified in a real browser

Driven headlessly with Playwright against the running stack:

| Check | Result |
|---|---|
| Invalid URL (`gitlab.com`) | Readable error, no crash |
| Repo with no supported files | "No supported source files were found" |
| Analyse a real repo | `Files: 50 | Chunks: 95 | Indexed: OK` |
| Ask a question | Answer + 2 cited files rendered |
| Markdown | No literal `**` or backticks; `<strong>`/`<code>` present |
| JavaScript errors | none |

Note: the browser console logs the deliberate 400/422 error-path responses as
"Failed to load resource". Those are handled by the app - only genuine JS faults
matter.

### Files

```
frontend/
  app/
    page.js         the whole UI
    AnswerText.js   minimal Markdown renderer
    layout.js       metadata
    globals.css     Tailwind import + base styles
  lib/api.js        backend client, error-shape normalisation
  .env.local        NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### Layout

```
backend/
  app/
    main.py                     FastAPI app, CORS, error handler
    config.py                   settings from .env
    core/
      exceptions.py             typed errors -> HTTP status codes
      file_filters.py           extensions, ignore rules, language map
    models/schemas.py           request/response models
    services/
      github_service.py         URL parsing, cloning, file discovery
      parser_service.py         decoding, normalising, parse stats
      chunker_service.py        LangChain splitting, chunk metadata
      embedding_service.py      local bge-small embeddings, token counting
      vector_store.py           Qdrant collection, upserts, filtering
      retrieval_service.py      query embedding, filtered search, dedup
      llm_service.py            Groq client, typed error mapping
      rag_service.py            prompt building, grounding, citations
      repository_store.py       in-memory registry of analysed repos
    api/routes/repository.py    /api/repository/* endpoints
    api/routes/embeddings.py    /api/embeddings/* endpoints
    api/routes/vectorstore.py   /api/repository/index, /api/vectorstore/info
    api/routes/search.py        /api/search
    api/routes/chat.py          /api/chat
  scripts/check_step1.py        offline URL-validation checks
  scripts/check_step2.py        offline scan + parse checks
  scripts/check_step3.py        offline chunking checks
  scripts/check_step4.py        embedding + real retrieval checks
  scripts/check_step5.py        Qdrant checks (needs the container)
  scripts/check_step6.py        search API checks (needs the container)
  scripts/check_step7.py        RAG grounding checks (needs Qdrant + Groq key)
  scripts/check_step8.py        integration + restart-restore checks
  scripts/check_private_repo.py offline token/redaction checks
  run_dev.ps1                   dev server launcher
  requirements.txt
  .env.example
docker-compose.yml              Qdrant container
.reposense-data/
  repos/                        cloned repositories (gitignored)
  qdrant/                       vector storage (gitignored)
```
