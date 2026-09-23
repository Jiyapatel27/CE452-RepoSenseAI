"""Application settings, loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Step 1: repository ingestion
    # ------------------------------------------------------------------
    # Where cloned repositories are stored on disk.
    #
    # This deliberately lives OUTSIDE backend/ and starts with a dot. uvicorn's
    # --reload watcher always watches the current working directory (it appends
    # Path.cwd() to the watch list regardless of --reload-dir), so storing
    # clones under backend/ made every `git clone` trigger a reload and kill the
    # dev server mid-request. Keeping the data dir out of the CWD, and hidden,
    # avoids that entirely.
    repo_storage_dir: Path = BACKEND_DIR.parent / ".reposense-data" / "repos"

    # Abort a clone that takes longer than this (seconds).
    clone_timeout_seconds: int = 300

    # Skip source files bigger than this (bytes). Huge files are usually
    # generated/minified and are useless for code understanding.
    max_file_size_bytes: int = 1_000_000

    # Safety limit so a giant monorepo cannot blow up the pipeline.
    max_files_per_repo: int = 5_000

    # ------------------------------------------------------------------
    # Step 3: code chunking
    # ------------------------------------------------------------------
    # Chunk size in TOKENS, measured with the embedding model's own tokeniser.
    #
    # Tokens, not characters: bge-small-en-v1.5 truncates at 512 tokens, and
    # character sizing cannot bound that. Measured over two real repositories,
    # worst-case density is ~1.4-2.0 chars/token against a ~3.0 average, so a
    # 1000-char chunk ranges from ~330 to ~700 tokens. Chunks above 512 are
    # silently truncated - still producing a plausible vector built from only
    # the opening fragment. 400 leaves headroom below the limit.
    chunk_size: int = 400

    # Overlap keeps a function that straddles a boundary retrievable from
    # either side. ~15% of chunk_size is a good rule of thumb.
    chunk_overlap: int = 60

    # Drop chunks shorter than this - a stray "}" or import line embeds to
    # noise and pollutes search results.
    min_chunk_chars: int = 40

    # ------------------------------------------------------------------
    # Step 4: embeddings (local, no paid API)
    # ------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # BGE models are trained asymmetrically: a search query must be prefixed
    # with an instruction, while stored passages must NOT be. Using the wrong
    # one on either side measurably degrades retrieval.
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # How many chunks to encode per forward pass. Higher is faster but uses
    # more RAM; 32 is comfortable on CPU.
    embedding_batch_size: int = 32

    # "auto" picks CUDA when available, otherwise CPU. Force with "cpu"/"cuda".
    embedding_device: str = "auto"

    # GitHub Personal Access Token used to clone PRIVATE repositories.
    # Optional: public repositories work without it. A token sent in the
    # /analyze request body takes priority over this one.
    github_token: str = ""

    # ------------------------------------------------------------------
    # Phase 2 / Step 10: persistence
    # ------------------------------------------------------------------
    # SQLite, deliberately: chat history needs to survive a restart, and a
    # single file needs no server, no container and no connection pooling.
    # Lives beside the vectors so all local state is in one gitignored folder.
    database_path: Path = (
        BACKEND_DIR.parent / ".reposense-data" / "reposense.db"
    )

    # ------------------------------------------------------------------
    # Step 11: conversational chat
    # ------------------------------------------------------------------
    # How many prior messages to feed back to the model. 6 is three exchanges -
    # enough for follow-ups to resolve, bounded so a long thread cannot grow
    # the prompt without limit.
    chat_history_turns: int = 6

    # Budget for rewriting a follow-up into a standalone question.
    #
    # This looks generous for a one-line output, and it has to be: gpt-oss is a
    # REASONING model, so max_tokens covers its internal reasoning *and* the
    # visible content. Measured on this exact prompt it emits ~565 characters
    # of reasoning before the question. At 120 the whole budget went to
    # reasoning, finish_reason came back "length", and content was EMPTY -
    # which silently disabled condensation altogether. 512 leaves ample room
    # (actual usage is ~135 tokens).
    condense_max_tokens: int = 512

    # Optional, model-specific. gpt-oss accepts "low" | "medium" | "high" and
    # "low" roughly halves the reasoning tokens for a mechanical rewrite like
    # condensation. Left empty by default because other models reject the
    # parameter outright - set it only if your GROQ_MODEL supports it.
    groq_reasoning_effort: str = ""

    # ------------------------------------------------------------------
    # Step 5: Qdrant vector store
    # ------------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"

    # Optional - only needed for Qdrant Cloud. Local Docker needs no auth.
    qdrant_api_key: str = ""

    # One collection holds every repository, separated by a `repository_id`
    # payload field with an index on it. Simpler than a collection per repo:
    # the model dimension is fixed, so one collection is enough, and dropping a
    # repository is a filtered delete.
    qdrant_collection: str = "reposense_chunks"

    # Points per upsert request. Batching matters: one request per chunk would
    # make indexing network-bound rather than embedding-bound.
    qdrant_batch_size: int = 128

    # Fail fast when Qdrant is down instead of hanging the API request.
    qdrant_timeout_seconds: int = 30

    # ------------------------------------------------------------------
    # Step 6: semantic search
    # ------------------------------------------------------------------
    # How many chunks to retrieve by default. 5 is a good balance: enough
    # context for Step 7's prompt without burning LLM tokens on weak matches.
    search_top_k: int = 5

    # Upper bound on a caller-supplied top_k, so one request cannot pull the
    # whole collection into memory.
    search_max_top_k: int = 50

    # Minimum cosine score to return. Deliberately 0.0 (disabled) by default:
    # measured on a repository with no auth code, the best "unanswerable"
    # match scored 0.578 against 0.632-0.790 for answerable questions. A ~0.05
    # margin is far too narrow to hard-code a cutoff - it would discard good
    # results on some repositories and admit junk on others. Step 7 handles
    # "insufficient context" through prompt grounding instead.
    search_min_score: float = 0.0

    # Load the embedding model during startup instead of on the first request.
    # Without this the first search pays ~12s of model init and looks broken.
    # Set false for faster dev reloads (uvicorn --reload restarts on every edit).
    preload_embedding_model: bool = True

    # ------------------------------------------------------------------
    # Step 7: RAG answer generation (Groq)
    # ------------------------------------------------------------------
    groq_api_key: str = ""

    # Groq has retired its Llama chat models - `llama-3.3-70b-versatile` and
    # friends now return "model not found"; only llama-prompt-guard (a 512-token
    # safety classifier) carries the Llama name. gpt-oss-120b is the strongest
    # general chat model Groq currently serves. Override in .env at any time:
    # the code never depends on which model this is.
    groq_model: str = "openai/gpt-oss-120b"

    # 0.0 for grounded answers. Any creativity here shows up as invented file
    # names and function signatures, which is the one thing we must avoid.
    groq_temperature: float = 0.0

    # Cap on the generated answer. 1024 proved too tight: a "how does login
    # work?" answer that walks through a controller and a service quoted enough
    # code to hit the limit and stop mid-sentence. 1600 clears that with room
    # spare, and `truncated_answer` in the response reports it if it ever bites.
    groq_max_tokens: int = 1_600

    groq_timeout_seconds: int = 60

    # How many chunks to feed the model. Slightly more than the search default:
    # extra context helps the answer, and the model ignores weak chunks better
    # than a human skimming search results would.
    chat_top_k: int = 6

    # Safety valve on prompt size. Retrieved chunks are ~400 tokens each, so 6
    # chunks is ~2.4k - far under the model's 131k window. This only matters if
    # someone raises chat_top_k a long way.
    chat_max_context_tokens: int = 12_000


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance (read the .env file only once)."""
    return Settings()
