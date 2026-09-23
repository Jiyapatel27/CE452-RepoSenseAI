"""Step 7 (part 2): retrieval-augmented answer generation.

    question -> embedding -> Qdrant -> top-K chunks -> prompt -> Groq -> answer

The hard part is not wiring this up; it is making the model refuse when the
repository has no answer. Step 6 measured why a score threshold cannot do that
job: on a repository with no authentication code, the best "where is
authentication implemented?" match still scored 0.587 against 0.632 for a
question the code *does* answer. A ~0.05 margin is not separable.

So refusal is enforced through the prompt instead:

* every chunk is labelled with its real file path and line range, so the model
  has something concrete to cite and inventing a path is visibly wrong;
* the system prompt forbids outside knowledge and demands an explicit
  "not in this repository" when the context does not support an answer;
* temperature is 0, because creativity here manifests as invented filenames.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.core.exceptions import InvalidSearchQueryError, RepoSenseError
from app.db.conversations import (
    Conversation,
    Message,
    MessageSource,
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_recent_turns,
    make_title,
)
from app.services.embedding_service import count_tokens
from app.services.llm_service import LLMReply, complete
from app.services.retrieval_service import SearchHit, search_chunks, unique_files

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are RepoSense AI, a repository analysis assistant.

Rules:
1. Answer ONLY using the repository context provided in the user message.
2. Never use general programming knowledge to fill gaps, and never guess.
3. Never invent file paths, function names, variables, or behaviour. Every
   identifier you mention must appear verbatim in the context.
4. If the context does not contain the answer, say so plainly - for example:
   "The provided repository context does not show any authentication code."
   Do not pad a non-answer with generic advice about how such code is usually
   written.
5. Cite the file paths you relied on, and line numbers when useful, e.g.
   `server/config/db.js` (lines 1-18).
6. Partial answers are fine: state what the context does show, then say
   explicitly what is missing.
7. Be concise and concrete. Prefer naming real functions over describing
   patterns in the abstract.\
"""

# Returned instead of calling the LLM when retrieval found nothing at all.
NO_CONTEXT_ANSWER = (
    "I could not find any relevant code in this repository for that question."
)


CONDENSE_PROMPT = """\
You rewrite follow-up questions so they can stand alone.

Given a conversation and a new question, output a single question that carries
all the context needed to search a codebase, with no reference to the
conversation.

Rules:
1. Replace pronouns and vague references ("it", "that file", "this function")
   with the concrete names they refer to in the conversation.
2. If the question already stands alone, output it UNCHANGED.
3. Keep it short and keep the user's intent. Do not add detail they did not ask
   for, and do not narrow the question.
4. Output ONLY the question. No preamble, no explanation, no quotes.

Examples:

Conversation:
user: Where is authentication implemented?
assistant: In src/app/routes/auth/auth.service.ts and auth.ts.
Question: how does it hash passwords?
Output: How does auth.service.ts hash passwords?

Conversation:
user: Which files handle API requests?
assistant: src/main.ts registers the router.
Question: Where is the database connection initialized?
Output: Where is the database connection initialized?\
"""

# Phrases that indicate the model declined to answer. Used only for dashboard
# statistics, so an approximate signal is acceptable - it is never shown to the
# user or used to alter the answer. Asking the model for a structured
# answered/not-answered flag would be more reliable but risks mangling code
# blocks inside a JSON payload.
_REFUSAL_MARKERS = (
    "does not contain",
    "does not show",
    "doesn't contain",
    "doesn't show",
    "not contain any",
    "no relevant code",
    "could not find",
    "cannot find",
    "not included in the context",
    "not present in the",
    "no code related to",
)


def looks_like_refusal(answer: str) -> bool:
    """Heuristic: did the model decline for lack of context?"""
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def history_as_turns(messages: list) -> list[dict[str, str]]:
    """Convert stored Message rows into Groq chat turns."""
    return [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role in {"user", "assistant"} and message.content
    ]


def condense_question(history: list[dict[str, str]], question: str) -> str:
    """Rewrite a follow-up into a standalone question for retrieval.

    This is the piece that makes multi-turn chat work. Embedding "how does it
    hash passwords?" on its own retrieves noise, because "it" has no meaning in
    vector space - the pronoun carries the entire subject and the embedding
    model never sees it. Resolving the reference first is what lets retrieval
    find the right chunks.

    Falls back to the original question if the rewrite fails or looks wrong:
    a degraded search beats a failed request.
    """
    if not history:
        return question

    transcript = "\n".join(
        f"{turn['role']}: {turn['content'][:400]}" for turn in history
    )
    prompt = f"Conversation:\n{transcript}\nQuestion: {question}\nOutput:"

    budget = get_settings().condense_max_tokens
    try:
        reply = complete(
            CONDENSE_PROMPT, prompt, max_tokens=budget, temperature=0.0
        )

        # A condensed question is always one short line, so empty content can
        # only mean the reasoning ate the whole budget. Reasoning length is not
        # bounded, so no fixed budget is safe - retry once with a much larger
        # one rather than silently degrading retrieval.
        if not reply.content.strip() and reply.finish_reason == "length":
            logger.warning(
                "Condensation hit the %d-token budget before producing output; "
                "retrying with %d",
                budget,
                budget * 4,
            )
            reply = complete(
                CONDENSE_PROMPT, prompt, max_tokens=budget * 4, temperature=0.0
            )
    except RepoSenseError as exc:
        logger.warning("Condensation failed (%s); using the original question", exc)
        return question

    rewritten = normalise_answer(reply.content).strip().strip('"').strip()

    # Sanity-check the rewrite. A model that ignores the instructions and starts
    # explaining instead of rewriting would otherwise poison retrieval, and an
    # empty response would search for nothing.
    if not rewritten:
        # Loud, because falling back here means condensation is switched off
        # and follow-up questions will quietly retrieve the wrong code. The
        # usual cause is condense_max_tokens being too small for a reasoning
        # model to get past its own reasoning.
        logger.error(
            "Condensation returned nothing; falling back to the raw question "
            "%r. Follow-up retrieval will be degraded - check "
            "CONDENSE_MAX_TOKENS (currently %d).",
            question,
            get_settings().condense_max_tokens,
        )
        return question
    if len(rewritten) > 400 or "\n" in rewritten.strip():
        logger.warning("Condensation returned prose, not a question; ignoring it")
        return question

    if rewritten != question:
        logger.info("Condensed %r -> %r", question, rewritten)
    return rewritten


@dataclass
class RagAnswer:
    """A grounded answer plus everything needed to verify it."""

    answer: str
    sources: list[str]
    hits: list[SearchHit] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_chunks: int = 0
    truncated_context: bool = False
    # True when the model ran into max_tokens, so the answer stops mid-thought.
    truncated_answer: bool = False
    llm_called: bool = True
    # Heuristic flag for dashboard statistics - see looks_like_refusal().
    refused: bool = False
    # The standalone question retrieval actually used. Differs from what the
    # user typed only when a follow-up had to be rewritten.
    resolved_question: str = ""


def build_context(hits: list[SearchHit], max_tokens: int) -> tuple[str, int, bool]:
    """Render retrieved chunks as a numbered, citable context block.

    Returns (context_text, chunks_used, was_truncated).

    Each block is labelled with its real path and line range. That is what makes
    grounding checkable: the model has exact strings to cite, and a reader can
    open the file and confirm.
    """
    blocks: list[str] = []
    used = 0
    running_tokens = 0
    truncated = False

    for index, hit in enumerate(hits, start=1):
        header = (
            f"[{index}] {hit.file_path} "
            f"(lines {hit.start_line}-{hit.end_line}, {hit.language})"
        )
        block = f"{header}\n```{hit.language}\n{hit.content}\n```"

        block_tokens = count_tokens(block)
        if used and running_tokens + block_tokens > max_tokens:
            # Keep whole chunks only - a half-chunk of code is worse than none,
            # since the model may reason about a function it cannot fully see.
            truncated = True
            break

        blocks.append(block)
        running_tokens += block_tokens
        used += 1

    return "\n\n".join(blocks), used, truncated


# Typographic characters models like to emit, mapped to plain ASCII.
# gpt-oss writes ranges as "lines 1‑18" - a narrow no-break space and
# a non-breaking hyphen. Perfectly correct text, but it breaks naive `\s`/`-`
# handling downstream (and renders as "118" in consoles that cannot show it),
# so normalise before the answer leaves this module.
# CAUTION: the keys below are literal invisible/lookalike characters
# (U+202F, U+00A0, U+2011, U+2013). They are correct - verified by code point -
# but an editor will not show the difference from a normal space or hyphen, so
# do not retype them by hand. Check with:
#   python -c "import app.services.rag_service as r; \
#              print([hex(ord(k)) for k in r._ANSWER_REPLACEMENTS])"
_ANSWER_REPLACEMENTS = {
    " ": " ",  # narrow no-break space
    " ": " ",  # no-break space
    "‑": "-",  # non-breaking hyphen
    "–": "-",  # en dash
}


# gpt-oss sometimes emits its own citation markers - "the HS256 algorithm
# [2 L1-L15]" with CJK bracket characters. They are an artifact of the model's
# training format, meaningless to a user, and would render as gibberish in the
# UI. We supply our own file/line citations, so strip these.
_CITATION_ARTIFACT = re.compile(r"【[^】]*】")


def normalise_answer(text: str) -> str:
    """Clean model output: ASCII spacing/hyphens, no citation artifacts."""
    for source, target in _ANSWER_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = _CITATION_ARTIFACT.sub("", text)
    # Stripping a marker can leave " ." or doubled spaces behind.
    text = re.sub(r" +([.,;:)])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def cited_sources(answer: str, hits: list[SearchHit]) -> list[str]:
    """File paths the answer actually refers to, best match first.

    Every file placed in the prompt is *available*, but only some get used - a
    question about the database still retrieves README.md as filler. Listing all
    of them as "sources" would make the UI's Relevant Files section misleading,
    so match against the answer text and fall back to the full context only when
    nothing matched (e.g. a refusal, where no file is cited).
    """
    ordered = unique_files(hits)
    cited = [path for path in ordered if path in answer]
    if cited:
        return cited

    # Also try bare file names: models often write `db.js` rather than the
    # full `server/config/db.js`.
    by_name = [
        path for path in ordered if path.rsplit("/", 1)[-1] in answer
    ]
    return by_name or ordered


def build_user_prompt(question: str, context: str) -> str:
    """Assemble the user message.

    The question is repeated after the context: with long contexts, models
    attend better to instructions placed last.
    """
    return (
        "Repository context:\n"
        "-------------------\n"
        f"{context}\n"
        "-------------------\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. If it does not contain the "
        "answer, say so explicitly. Cite the file paths you used."
    )


def answer_question(
    repository_id: str,
    question: str,
    *,
    top_k: int | None = None,
    languages: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
) -> RagAnswer:
    """Retrieve context and generate a grounded answer.

    When `history` is supplied the question is first condensed into a
    standalone form for retrieval, and the prior turns are passed to the model
    so it can resolve references naturally.

    Raises:
        InvalidSearchQueryError: blank question.
        RepositoryNotIndexedError: repository has no vectors.
        LLMNotConfiguredError / LLMRateLimitError / LLMUnavailableError.
    """
    settings = get_settings()

    if not question or not question.strip():
        raise InvalidSearchQueryError("Question must not be empty.")

    # Retrieval uses the standalone form; the model still sees what the user
    # actually typed, plus the history, so the reply reads naturally.
    resolved = condense_question(history or [], question)

    hits = search_chunks(
        repository_id,
        resolved,
        top_k=top_k or settings.chat_top_k,
        languages=languages,
    )

    # No context at all: answer directly rather than asking the model to
    # invent something from nothing. Saves an API call and removes the risk.
    if not hits:
        logger.info("No hits for %r in %s - skipping the LLM", resolved, repository_id)
        return RagAnswer(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            hits=[],
            model=settings.groq_model,
            llm_called=False,
            refused=True,
            resolved_question=resolved,
        )

    context, used, truncated = build_context(hits, settings.chat_max_context_tokens)
    reply: LLMReply = complete(
        SYSTEM_PROMPT,
        build_user_prompt(question, context),
        history=history,
    )
    answer = normalise_answer(reply.content)

    # finish_reason "length" means max_tokens cut the answer off mid-sentence.
    # Surfacing it beats presenting a truncated explanation as complete.
    hit_token_limit = reply.finish_reason == "length"
    if hit_token_limit:
        logger.warning(
            "Answer hit the %d-token limit and was truncated (question: %r)",
            settings.groq_max_tokens,
            question[:60],
        )

    logger.info(
        "Answered %r on %s using %d chunks (%d in / %d out tokens)",
        question[:60],
        repository_id,
        used,
        reply.prompt_tokens,
        reply.completion_tokens,
    )

    # Sources are drawn only from chunks that made it into the prompt, so a
    # truncated context cannot claim credit for chunks the model never saw.
    return RagAnswer(
        answer=answer,
        sources=cited_sources(answer, hits[:used]),
        hits=hits[:used],
        model=reply.model,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        context_chunks=used,
        truncated_context=truncated,
        truncated_answer=hit_token_limit,
        refused=looks_like_refusal(answer),
        resolved_question=resolved,
    )


def answer_in_conversation(
    conversation_id: str,
    question: str,
    *,
    top_k: int | None = None,
    languages: list[str] | None = None,
) -> tuple[RagAnswer, Message, Message]:
    """Answer inside an existing thread, persisting both turns.

    Returns (answer, user_message, assistant_message).

    The repository comes from the conversation rather than the caller, so a
    thread can never silently switch which codebase it is talking about
    halfway through - which would make its earlier answers misleading.

    Raises:
        ConversationNotFoundError: unknown thread.
        InvalidSearchQueryError: blank question.
        RepositoryNotIndexedError: the thread's repository has no vectors.
    """
    settings = get_settings()

    if not question or not question.strip():
        raise InvalidSearchQueryError("Question must not be empty.")

    conversation = get_conversation(conversation_id)
    history = history_as_turns(
        get_recent_turns(conversation_id, limit=settings.chat_history_turns)
    )

    # Timed here rather than in the route, because this is the value that gets
    # persisted and the dashboard averages. Measuring only in the route left
    # every stored message with elapsed_ms NULL, so "average response time"
    # read as 0 ms.
    started = time.perf_counter()
    result = answer_question(
        conversation.repository_id,
        question,
        top_k=top_k,
        languages=languages,
        history=history,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    # Persist the user turn first so the thread reads in the right order even
    # if the assistant write were to fail.
    user_message = add_message(conversation_id, "user", question.strip())
    assistant_message = add_message(
        conversation_id,
        "assistant",
        result.answer,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        elapsed_ms=elapsed_ms,
        context_chunks=result.context_chunks,
        refused=result.refused,
        truncated=result.truncated_answer,
        # Only worth storing when it differs - otherwise it is noise.
        resolved_question=(
            result.resolved_question
            if result.resolved_question != question.strip()
            else None
        ),
        sources=[
            MessageSource(
                file_path=hit.file_path,
                score=hit.score,
                start_line=hit.start_line,
                end_line=hit.end_line,
                language=hit.language,
                cited=hit.file_path in result.sources,
            )
            for hit in result.hits
        ],
    )

    return result, user_message, assistant_message


def start_conversation(
    repository_id: str,
    question: str,
    *,
    top_k: int | None = None,
    languages: list[str] | None = None,
) -> tuple[Conversation, RagAnswer, Message, Message]:
    """Create a thread titled after the first question, then answer in it."""
    if not question or not question.strip():
        raise InvalidSearchQueryError("Question must not be empty.")

    conversation = create_conversation(repository_id, make_title(question))

    # The thread row is committed before the answer is attempted, so a failure
    # here (unindexed repository, Groq down) would otherwise strand an empty
    # conversation - visible in the sidebar, counted on the dashboard, and
    # impossible to use. Roll it back.
    try:
        result, user_message, assistant_message = answer_in_conversation(
            conversation.id, question, top_k=top_k, languages=languages
        )
    except Exception:
        try:
            delete_conversation(conversation.id)
        except Exception:  # noqa: BLE001 - never mask the original failure
            logger.warning(
                "Could not clean up empty conversation %s", conversation.id
            )
        raise

    return conversation, result, user_message, assistant_message
