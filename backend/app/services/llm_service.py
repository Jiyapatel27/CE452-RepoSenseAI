"""Step 7 (part 1): the Groq LLM client.

Thin wrapper around the official Groq SDK. Kept separate from the RAG logic so
prompt building and retrieval can be tested without a network call, and so
swapping providers later touches only this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

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
    history: list[dict[str, str]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMReply:
    """Send one chat completion request to Groq.

    `history` is a list of prior `{"role", "content"}` turns, inserted between
    the system prompt and the new question. Passing real conversation turns
    rather than flattening them into one string lets the model see who said
    what - which is what makes "it" and "that file" resolvable in a follow-up.

    Every failure mode is translated into a typed error so the API returns a
    useful status code instead of a 500 with a stack trace.
    """
    settings = get_settings()
    client = get_groq_client()
    model_name = model or settings.groq_model

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history or []:
        # Guard against anything unexpected reaching the API: Groq rejects
        # roles outside this set, and a stray one would 400 the whole request.
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    extra: dict[str, Any] = {}
    if settings.groq_reasoning_effort:
        extra["reasoning_effort"] = settings.groq_reasoning_effort

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=(
                temperature if temperature is not None else settings.groq_temperature
            ),
            max_tokens=max_tokens or settings.groq_max_tokens,
            **extra,
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
    content = (choice.message.content or "").strip()

    # A reasoning model spends max_tokens on reasoning *before* it writes
    # anything, so too small a budget yields finish_reason="length" with
    # EMPTY content. That is silent and very confusing downstream - a caller
    # with a fallback path just quietly stops working. Say so loudly.
    if not content:
        logger.error(
            "Groq returned EMPTY content (model=%s, finish_reason=%s, "
            "completion_tokens=%s, max_tokens=%s). For a reasoning model this "
            "usually means max_tokens was too small: the budget covers "
            "reasoning as well as the reply. Raise it.",
            response.model,
            choice.finish_reason,
            getattr(usage, "completion_tokens", "?"),
            max_tokens or settings.groq_max_tokens,
        )

    return LLMReply(
        content=content,
        model=response.model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        finish_reason=str(choice.finish_reason or ""),
    )
