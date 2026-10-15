"""Validate explicit article references against retrieved context."""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.retrieval.retriever import RetrievedChunk


@dataclass
class CitationValidationResult:
    """Cited and unsupported numbers, each in first-mention order."""

    is_valid: bool
    cited_articles: List[str]
    unsupported_articles: List[str]


# Ordinal suffixes cover forms such as 70-ci, 99-cu, 6-cı, and 3-cü,
# and amendment articles like 76-1-ci.
_ORDINAL = r"(?:[0-9]+(?:-[0-9]+)?)\s*-\s*(?:cı|ci|cu|cü)"
_CITATION_PATTERN = re.compile(
    rf"(?<![\w.]){_ORDINAL}"
    rf"(?:\s*(?:,\s*(?:və\s+)?|və\s+){_ORDINAL})*"
    r"\s+maddə\w*\b"
    r"|\bmaddə\s+[0-9]+(?:[.-][0-9]+)?\b",
    re.IGNORECASE,
)


def validate_article_citations(
    answer: str, results: List["RetrievedChunk"]
) -> CitationValidationResult:
    """Check ordinal references and 'Maddə N' references using regex only.

    Supports ordinal lists joined by commas or 'və' before 'maddə',
    including inflected forms such as 'maddəsinə' and 'maddələrə'.
    This MVP checks explicit whole-number article references, not all legal
    citation grammar or whether an article supports the answer's claims.
    An answer with no recognized citations is valid.
    """
    retrieved_articles = set()
    for result in results:
        number = str(result.article_number).strip()
        norm_num = str(int(number)) if number.isdecimal() else number
        retrieved_articles.add(norm_num)

    cited_articles = []
    for match in _CITATION_PATTERN.finditer(answer):
        text = match.group()
        m_direct = re.match(
            r"\bmaddə\s+([0-9]+(?:-[0-9]+)?)(?:\.[0-9]+)?\b",
            text,
            re.IGNORECASE,
        )
        if m_direct:
            raw_num = m_direct.group(1)
            norm_num = str(int(raw_num)) if raw_num.isdecimal() else raw_num
            if norm_num not in cited_articles:
                cited_articles.append(norm_num)
        else:
            for raw_num in re.findall(
                r"([0-9]+(?:-[0-9]+)?)\s*-\s*(?:cı|ci|cu|cü)",
                text,
                re.IGNORECASE,
            ):
                norm_num = (
                    str(int(raw_num)) if raw_num.isdecimal() else raw_num
                )
                if norm_num not in cited_articles:
                    cited_articles.append(norm_num)

    unsupported_articles = [
        number for number in cited_articles if number not in retrieved_articles
    ]
    return CitationValidationResult(
        is_valid=not unsupported_articles,
        cited_articles=cited_articles,
        unsupported_articles=unsupported_articles,
    )
