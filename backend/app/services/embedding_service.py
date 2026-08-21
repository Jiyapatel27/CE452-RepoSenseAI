"""Step 4: local embeddings with sentence-transformers.

Model: BAAI/bge-small-en-v1.5 - 384 dimensions, 512-token limit, runs on CPU,
downloaded once (~130 MB) and cached by Hugging Face. No paid API.

Two details that are easy to get wrong and quietly ruin retrieval:

1. **BGE is asymmetric.** A search *query* must be prefixed with
   "Represent this sentence for searching relevant passages: ", while stored
   *passages* must not be. Embedding both sides the same way still "works" -
   you get vectors and plausible-looking scores - but ranking quality drops.
   Hence two distinct functions: `embed_documents()` and `embed_query()`.

2. **Vectors are L2-normalised.** With unit vectors, cosine similarity equals
   the dot product, so similarity scores land in a predictable 0-1 range and
   Qdrant's COSINE distance behaves consistently.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

from app.config import get_settings
from app.core.exceptions import EmbeddingModelError

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Loading the model is not thread-safe and takes a few seconds, so the first
# caller holds this lock while the rest wait rather than all loading copies.
_load_lock = threading.Lock()


@dataclass(frozen=True)
class EmbeddingModelInfo:
    model_name: str
    dimension: int
    max_sequence_length: int
    device: str


def _resolve_device(requested: str) -> str:
    """Turn "auto" into a concrete device string."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing or broken
        return "cpu"


@lru_cache(maxsize=1)
def _load_model() -> "SentenceTransformer":
    """Load the model once per process.

    Deliberately lazy: importing sentence-transformers pulls in PyTorch, which
    is slow, so we must not do it at module import time or every server start
    (and every unrelated unit test) would pay for it.
    """
    settings = get_settings()
    device = _resolve_device(settings.embedding_device)

    with _load_lock:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "sentence-transformers is not installed.",
                detail="Run: pip install -r requirements.txt",
            ) from exc

        logger.info(
            "Loading embedding model %s on %s (first run downloads ~130 MB)",
            settings.embedding_model,
            device,
        )
        try:
            model = SentenceTransformer(settings.embedding_model, device=device)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Could not load embedding model '{settings.embedding_model}'.",
                detail=(
                    "The first run needs internet access to download the model. "
                    f"Underlying error: {exc}"
                ),
            ) from exc

    logger.info(
        "Embedding model ready: dim=%d, max_seq_len=%d",
        _dimension_of(model),
        model.max_seq_length,
    )
    return model


def _dimension_of(model: "SentenceTransformer") -> int:
    """Vector size, across sentence-transformers versions.

    v6 renamed `get_sentence_embedding_dimension()` to `get_embedding_dimension()`
    and warns on the old name; older versions only have the old one.
    """
    getter = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    return int(getter())


def get_model_info() -> EmbeddingModelInfo:
    """Describe the loaded model (loads it if needed)."""
    settings = get_settings()
    model = _load_model()
    return EmbeddingModelInfo(
        model_name=settings.embedding_model,
        dimension=_dimension_of(model),
        max_sequence_length=int(model.max_seq_length),
        device=str(model.device),
    )


def get_embedding_dimension() -> int:
    """Vector size, needed to create the Qdrant collection in Step 5."""
    return _dimension_of(_load_model())


def _encode(texts: Sequence[str], *, batch_size: int | None = None) -> list[list[float]]:
    settings = get_settings()
    model = _load_model()

    vectors = model.encode(
        list(texts),
        batch_size=batch_size or settings.embedding_batch_size,
        # Unit vectors: cosine similarity == dot product.
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_documents(
    texts: Sequence[str], *, batch_size: int | None = None
) -> list[list[float]]:
    """Embed passages for storage. No query prefix - that is the BGE contract."""
    if not texts:
        return []
    return _encode(texts, batch_size=batch_size)


def embed_query(query: str) -> list[float]:
    """Embed a single search query, with the BGE instruction prefix applied."""
    settings = get_settings()
    prefixed = f"{settings.embedding_query_prefix}{query}"
    return _encode([prefixed])[0]


def embed_queries(queries: Sequence[str]) -> list[list[float]]:
    """Batch version of `embed_query`."""
    if not queries:
        return []
    settings = get_settings()
    prefixed = [f"{settings.embedding_query_prefix}{q}" for q in queries]
    return _encode(prefixed)


def count_tokens(text: str) -> int:
    """Token count for `text` under the model's own tokenizer.

    Used both to size chunks in Step 3 and to verify nothing exceeds the
    512-token limit - a truncated chunk still produces a vector, so counting is
    the only way to notice the tail was dropped.

    `verbose=False` suppresses transformers' "sequence longer than maximum"
    warning: we are counting tokens, not running them through the model, and
    counting a long file is exactly what the chunker needs to do.
    """
    model = _load_model()
    return len(
        model.tokenizer.encode(text, add_special_tokens=True, verbose=False)
    )
