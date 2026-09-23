"""Step 12: conversation routes - start, list, resume, rename, delete."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Response
from fastapi.concurrency import run_in_threadpool

from app.db import conversations as store
from app.models.schemas import (
    ContinueConversationRequest,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationTurnResponse,
    MessageResponse,
    MessageSourceResponse,
    RenameConversationRequest,
    StartConversationRequest,
)
from app.services.rag_service import answer_in_conversation, start_conversation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _to_conversation(conversation: store.Conversation) -> ConversationResponse:
    return ConversationResponse(**vars(conversation))


def _to_message(message: store.Message) -> MessageResponse:
    data = vars(message).copy()
    data.pop("conversation_id", None)
    data["sources"] = [
        MessageSourceResponse(**vars(source)) for source in message.sources
    ]
    return MessageResponse(**data)


@router.post("", response_model=ConversationTurnResponse)
async def start(payload: StartConversationRequest) -> ConversationTurnResponse:
    """Start a thread and answer its first question in one call.

    Combined rather than split into create-then-ask: a conversation with no
    messages is not useful to show, and the title is derived from the first
    question anyway.
    """
    conversation, result, user_message, assistant_message = await run_in_threadpool(
        start_conversation,
        payload.repository_id,
        payload.question,
        top_k=payload.top_k,
        languages=payload.languages,
    )

    # message_count is computed by the store, so re-read it for an accurate
    # response rather than reporting the stale zero from creation.
    refreshed = await run_in_threadpool(store.get_conversation, conversation.id)

    return ConversationTurnResponse(
        conversation=_to_conversation(refreshed),
        user_message=_to_message(user_message),
        assistant_message=_to_message(assistant_message),
        answer=result.answer,
        sources=result.sources,
    )


@router.get("", response_model=ConversationListResponse)
async def list_all(
    repository_id: str | None = Query(
        None, description="Omit to list threads across every repository."
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ConversationListResponse:
    """Threads for the history sidebar, most recently updated first."""
    conversations = await run_in_threadpool(
        store.list_conversations, repository_id, limit=limit, offset=offset
    )
    return ConversationListResponse(
        count=len(conversations),
        conversations=[_to_conversation(c) for c in conversations],
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def detail(conversation_id: str) -> ConversationDetailResponse:
    """A whole thread, for resuming it. Raises 404 if unknown."""
    conversation = await run_in_threadpool(store.get_conversation, conversation_id)
    messages = await run_in_threadpool(store.get_messages, conversation_id)
    return ConversationDetailResponse(
        conversation=_to_conversation(conversation),
        messages=[_to_message(message) for message in messages],
    )


@router.post("/{conversation_id}/messages", response_model=ConversationTurnResponse)
async def continue_thread(
    conversation_id: str, payload: ContinueConversationRequest
) -> ConversationTurnResponse:
    """Ask a follow-up in an existing thread.

    The repository is taken from the conversation, not the request, so a thread
    cannot switch codebases halfway through.
    """
    result, user_message, assistant_message = await run_in_threadpool(
        answer_in_conversation,
        conversation_id,
        payload.question,
        top_k=payload.top_k,
        languages=payload.languages,
    )
    conversation = await run_in_threadpool(store.get_conversation, conversation_id)

    return ConversationTurnResponse(
        conversation=_to_conversation(conversation),
        user_message=_to_message(user_message),
        assistant_message=_to_message(assistant_message),
        answer=result.answer,
        sources=result.sources,
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def rename(
    conversation_id: str, payload: RenameConversationRequest
) -> ConversationResponse:
    """Rename a thread."""
    conversation = await run_in_threadpool(
        store.rename_conversation, conversation_id, payload.title.strip()
    )
    return _to_conversation(conversation)


@router.delete("/{conversation_id}", status_code=204)
async def delete(conversation_id: str) -> Response:
    """Delete a thread. Its messages and sources cascade."""
    await run_in_threadpool(store.delete_conversation, conversation_id)
    return Response(status_code=204)
