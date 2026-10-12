"""One-shot legal Q&A grounded in retrieved Labour Code articles.

Pipeline: question -> retrieve -> relevance guardrail -> LLM ->
citation guardrail -> answer. Every legal-facing output ends with the
centralized disclaimer, and every failure path returns the fixed
Azerbaijani fallback instead of an invented answer.

Usage from the repo root::

    python -m src.generation.qa_prompts "Mən haqsız işdən çıxarılmışam"
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.guardrails import (
    add_legal_disclaimer,
    is_relevant,
    validate_article_citations,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_TOKEN = "NOT_FOUND"

SYSTEM_PROMPT = (
    "Sən Azərbaycan Respublikasının Əmək Məcəlləsi üzrə hüquqi "
    "məlumat köməkçisisən. Yalnız aşağıda verilən kontekst "
    "maddələrinə əsaslanaraq cavab ver. Kontekstdə olmayan maddə "
    "nömrəsi, müddət, məbləğ və ya hüquqi iddia yazma. Hər hüquqi "
    "iddiaya 'Maddə N' formatında istinad et. "
    "Aşağıdakı hallarda kontekstdəki maddələri tətbiq edərək cavab ver: "
    "1) İstifadəçi birbaşa hüquqi sual versə; "
    "2) İstifadəçi öz vəziyyətini izah etsə (məs. 'işdən çıxarıblar'); "
    "3) İstifadəçi dolayı və ya gündəlik sual versə (məs. 'doğum günümdə "
    "işə getməliyəm?', 'bayramda işləsəm nə olar?') — bu zaman aidiyyatlı "
    "məzuniyyət, istirahət günü və ya iş vaxtı maddələrini tətbiq edərək "
    "izah et. "
    "Maddənin tətbiq şərtlərini saxla. İşçinin təşəbbüsü ilə xitam "
    "qaydasını işəgötürənin təşəbbüsü ilə xitama tətbiq etmə. "
    "Müqavilənin müddətli olduğunu və ya işçinin istefa verdiyini "
    "istifadəçi deməyibsə, fərz etmə. Kontekstdə məhkəməyə müraciət "
    "qaydası yoxdursa, belə hüquqi tövsiyə əlavə etmə. "
    "Sual bir neçə məsələni əhatə edirsə, kontekstin əsaslandırdığı "
    "hissələrə cavab ver, digər hissələr üçün məlumatın yetərli "
    "olmadığını açıq bildir. Xitamın səbəbi məlum deyilsə, verilən "
    "ümumi əsasları izah et, konkret əsasın mövcud olduğunu fərz etmə. "
    "Sual əmək hüququ ilə bağlı olsa belə, verilən kontekstdə "
    "tətbiq olunan hüquqi əsas yoxdursa NOT_FOUND yaz. "
    "Cavabı Azərbaycan dilində, qısa və aydın yaz."
)

USER_TEMPLATE = "Kontekst:\n{context}\n\nSual: {question}"

LLMFn = Callable[[str, str], str]
RetrieveFn = Callable[..., List[Any]]


@dataclass
class QAResult:
    """Outcome of one QA call.

    Attributes:
        answer: Final text shown to the user, disclaimer included.
        grounded: True only if an LLM answer passed all guardrails.
        sources: Retrieved chunks used as context (empty on fallback).
        flagged_articles: Article numbers the LLM cited that were not
            present in the retrieved context.
    """

    answer: str
    grounded: bool
    sources: List[Any] = field(default_factory=list)
    flagged_articles: List[str] = field(default_factory=list)
    failure_reason: Optional[str] = None


def format_context(chunks: List[Any]) -> str:
    """Renders retrieved chunks as a labelled context block.

    Args:
        chunks: RetrievedChunk objects with ``article_number``,
            ``title`` and ``text`` attributes.

    Returns:
        Text with one ``[Maddə N]`` block per chunk.
    """
    blocks = []
    for chunk in chunks:
        title = f"[Maddə {chunk.article_number}]"
        if chunk.title:
            title += f" ({chunk.title})"
        blocks.append(f"{title}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_prompts(question: str, chunks: List[Any]) -> Tuple[str, str]:
    """Builds the (system, user) prompt pair for a question.

    Args:
        question: The user's legal question.
        chunks: Retrieved article chunks.

    Returns:
        Tuple of system prompt and user prompt strings.
    """
    user = USER_TEMPLATE.format(
        context=format_context(chunks), question=question.strip()
    )
    return SYSTEM_PROMPT, user


_client = None


def _get_client():
    """Returns a cached genai.Client instance."""
    global _client
    if _client is None:
        from google import genai

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (.env).")
        _client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client


def _is_retryable_error(exc: BaseException) -> bool:
    """Returns True if exception is a transient network or rate-limit spike."""
    err_str = str(exc).lower()
    retryable_markers = (
        "429", "503", "500", "overloaded", "quota", "resource_exhausted",
        "rate limit", "spikes", "timeout", "unavailable",
    )
    return any(marker in err_str for marker in retryable_markers)


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _generate_with_retry(
    client: Any,
    model: str,
    system: str,
    user: str,
    response_json_schema: Optional[dict] = None,
) -> str:
    """Generates content with automatic exponential backoff retries."""
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            # Headroom: reasoning models count thinking tokens here.
            max_output_tokens=2048,
            response_mime_type=("application/json"
                                if response_json_schema else None),
            response_json_schema=response_json_schema,
        ),
    )
    return response.text or ""


def call_llm(system: str, user: str, *,
             response_json_schema: Optional[dict] = None) -> str:
    """Calls the configured Gemini model with exponential backoff retries.

    Uses ``settings.GEMINI_API_KEY`` and ``settings.LLM_MODEL_NAME``.

    Args:
        system: System prompt.
        user: User prompt.
        response_json_schema: Optional strict JSON schema for case providers.

    Returns:
        The model's text output (empty string if nothing was
        returned, e.g. a blocked or truncated response).

    Raises:
        RuntimeError: If no Gemini API key is configured.
    """
    client = _get_client()
    return _generate_with_retry(
        client=client,
        model=settings.LLM_MODEL_NAME,
        system=system,
        user=user,
        response_json_schema=response_json_schema,
    )


def _fallback(flagged: Optional[List[str]] = None, *,
              failure_reason: Optional[str] = None) -> QAResult:
    """Builds an ungrounded result carrying the fixed fallback text."""
    return QAResult(
        answer=add_legal_disclaimer(settings.NO_ARTICLE_FALLBACK_AZ),
        grounded=False,
        flagged_articles=flagged or [], failure_reason=failure_reason,
    )


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    llm_fn: Optional[LLMFn] = None,
    retrieve_fn: Optional[RetrieveFn] = None,
) -> QAResult:
    """Answers one legal question, grounded strictly in retrieved text.

    Args:
        question: The user's question in Azerbaijani.
        top_k: Number of chunks to retrieve (config default if None).
        llm_fn: Callable ``(system, user) -> str``; defaults to
            :func:`call_llm`. Injectable for tests.
        retrieve_fn: Retrieval callable; defaults to Phase 3
            ``retrieve``. Injectable for tests.

    Returns:
        A QAResult. On weak retrieval, LLM refusal or ungrounded
        article citations the fixed fallback is returned.
    """
    llm_fn = llm_fn or call_llm
    if retrieve_fn is None:
        # Imported lazily so tests do not load embedding models.
        from src.retrieval import retrieve as retrieve_fn

    chunks = retrieve_fn(question, top_k=top_k)
    if not chunks or not is_relevant(question, chunks):
        logger.info("Retrieval below threshold; returning fallback.")
        return _fallback(failure_reason="insufficient_retrieval")

    system, user = build_prompts(question, chunks)
    try:
        raw = (llm_fn(system, user) or "").strip()
    except AssertionError:
        raise
    except Exception:
        logger.error("LLM call failed after retries.")
        return _fallback(failure_reason="provider_error")

    if not raw or NOT_FOUND_TOKEN in raw:
        logger.info("LLM reported insufficient context.")
        reason = "empty_response" if not raw else "provider_refusal"
        return _fallback(failure_reason=reason)

    check = validate_article_citations(raw, chunks)
    if not check.is_valid:
        logger.warning(
            "Ungrounded articles cited: %s", check.unsupported_articles
        )
        return _fallback(check.unsupported_articles,
                         failure_reason="unsupported_citations")

    return QAResult(add_legal_disclaimer(raw), True, chunks)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint: prints the answer and cited sources."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Usage: python -m src.generation.qa_prompts <sual>")
        return
    result = answer_question(" ".join(args))
    print(result.answer)
    if result.sources:
        print("\nMənbələr:")
        for chunk in result.sources:
            print(f"  - Maddə {chunk.article_number} "
                  f"(score={chunk.score:.3f})")


if __name__ == "__main__":
    main()
