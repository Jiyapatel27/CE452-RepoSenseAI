"""Step 7 (part 1): the Groq LLM client.

Thin wrapper around the official Groq SDK. Kept separate from the RAG logic so
prompt building and retrieval can be tested without a network call, and so
swapping providers later touches only this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.config import get_settings
from app.core.exceptions import (
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMReply:
    """A completion plus the metadata worth surfacing to the caller."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@lru_cache
def get_groq_client() -> Groq:
    """Cached Groq client. Raises if no API key is configured."""
    settings = get_settings()
    key = (settings.groq_api_key or "").strip()
    if not key:
        raise LLMNotConfiguredError(
            "GROQ_API_KEY is not set.",
            detail=(
                "Get a free key at https://console.groq.com, add it to "
                "backend/.env as GROQ_API_KEY=..., then restart the server "
                "(.env is read once at startup)."
            ),
        )
    return Groq(api_key=key, timeout=settings.groq_timeout_seconds)


def is_configured() -> bool:
    """True when a key is present, without raising."""
    return bool((get_settings().groq_api_key or "").strip())


def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMReply:
    """Send one chat completion request to Groq.

    Every failure mode is translated into a typed error so the API returns a
    useful status code instead of a 500 with a stack trace.
    """
    settings = get_settings()
    client = get_groq_client()
    model_name = model or settings.groq_model

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=(
                temperature if temperature is not None else settings.groq_temperature
            ),
            max_tokens=max_tokens or settings.groq_max_tokens,
        )
    except AuthenticationError as exc:
        raise LLMNotConfiguredError(
            "Groq rejected the API key.",
            detail="Check GROQ_API_KEY in backend/.env, then restart the server.",
        ) from exc
    except RateLimitError as exc:
        raise LLMRateLimitError(
            "Groq rate limit reached. Wait a moment and try again.",
            detail=str(exc),
        ) from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise LLMUnavailableError(
            "Could not reach the Groq API.",
            detail="Check your internet connection and try again.",
        ) from exc
    except APIStatusError as exc:
        # The most common cause here is a retired model name, so say so
        # explicitly - "model not found" is otherwise a confusing 404.
        hint = (
            f"Model '{model_name}' may no longer be available. "
            "List current models with: python -c \"from groq import Groq; "
            "print([m.id for m in Groq().models.list().data])\""
            if exc.status_code == 404
            else str(exc)
        )
        raise LLMUnavailableError(
            f"Groq returned an error (HTTP {exc.status_code}).", detail=hint
        ) from exc

    choice = response.choices[0]
    usage = response.usage

    return LLMReply(
        content=(choice.message.content or "").strip(),
        model=response.model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        finish_reason=str(choice.finish_reason or ""),
    )
