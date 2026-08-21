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
from dataclasses import dataclass, field

from app.config import get_settings
from app.core.exceptions import InvalidSearchQueryError
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
) -> RagAnswer:
    """Retrieve context and generate a grounded answer.

    Raises:
        InvalidSearchQueryError: blank question.
        RepositoryNotIndexedError: repository has no vectors.
        LLMNotConfiguredError / LLMRateLimitError / LLMUnavailableError.
    """
    settings = get_settings()

    if not question or not question.strip():
        raise InvalidSearchQueryError("Question must not be empty.")

    hits = search_chunks(
        repository_id,
        question,
        top_k=top_k or settings.chat_top_k,
        languages=languages,
    )

    # No context at all: answer directly rather than asking the model to
    # invent something from nothing. Saves an API call and removes the risk.
    if not hits:
        logger.info("No hits for %r in %s - skipping the LLM", question, repository_id)
        return RagAnswer(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            hits=[],
            model=settings.groq_model,
            llm_called=False,
        )

    context, used, truncated = build_context(hits, settings.chat_max_context_tokens)
    reply: LLMReply = complete(SYSTEM_PROMPT, build_user_prompt(question, context))
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
    )
