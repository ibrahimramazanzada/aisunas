"""Hybrid retrieval module combining BM25 and dense vector search.

Implements Reciprocal Rank Fusion (RRF) and weighted score fusion across
BM25 lexical matching and ChromaDB dense cosine similarity for legal queries.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.config import settings
from src.retrieval.case_query import enrich_query
from src.indexing.bm25_indexer import BM25Manager
from src.indexing.vector_store import EmbeddingManager, VectorStoreManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved article chunk with hybrid ranking scores.

    Attributes:
        chunk_id: Unique chunk identifier.
        article_number: Azerbaijani Labour Code article number (e.g. '70').
        title: Title of the legal article.
        section: Section (Bölmə) heading.
        chapter: Chapter (Fəsil) heading.
        text: Extracted legal text content.
        score: Combined hybrid confidence score in [0.0, 1.0].
        dense_score: Raw dense cosine similarity score.
        bm25_score: Normalized BM25 lexical score.
        rank: 1-indexed hybrid rank position.
        page_start: Source PDF starting page.
        page_end: Source PDF ending page.
    """
    chunk_id: str
    article_number: str
    title: str
    section: str
    chapter: str
    text: str
    score: float
    dense_score: float
    bm25_score: float
    rank: int
    page_start: int
    page_end: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance to dictionary representation."""
        return asdict(self)


class LegalRetriever:
    """Hybrid retriever combining lexical BM25 and dense embedding search."""

    def __init__(
        self,
        vector_mgr: Optional[VectorStoreManager] = None,
        bm25_mgr: Optional[BM25Manager] = None,
        k_rrf: int = 60,
    ):
        """Initializes retriever with search managers and fusion parameters.

        Args:
            vector_mgr: Optional initialized VectorStoreManager.
            bm25_mgr: Optional initialized BM25Manager.
            k_rrf: Smoothing constant for Reciprocal Rank Fusion (default 60).
        """
        self.vector_mgr = vector_mgr or VectorStoreManager()
        self.bm25_mgr = bm25_mgr or BM25Manager()
        self.k_rrf = k_rrf
        self._bm25_loaded = False

    def _ensure_bm25_loaded(self) -> None:
        """Lazily loads BM25 index from disk if not already in memory."""
        if not self._bm25_loaded and self.bm25_mgr.bm25 is None:
            logger.info("Loading persisted BM25 index for retrieval...")
            self.bm25_mgr.load()
            self._bm25_loaded = True

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        case_state: Optional[Any] = None,
    ) -> List[RetrievedChunk]:
        """Executes hybrid retrieval over dense vectors and BM25 index.

        Args:
            query: User's legal query string in Azerbaijani.
            top_k: Number of ranked chunks to return (defaults to config).
            dense_weight: Relative weight for dense similarity.
            bm25_weight: Relative weight for lexical match.
            case_state: Optional CaseState to enrich the retrieval query.

        Returns:
            List of RetrievedChunk instances ordered by hybrid score.
        """
        k = top_k or settings.RETRIEVAL_TOP_K
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        effective_query = enrich_query(query, case_state)

        # Expand colloquial labour dispute phrasings
        # Replace Azerbaijani-specific uppercase letters BEFORE .lower() so
        # that locale-insensitive lowercasing does not leave İ/I intact.
        normalized_q = (
            effective_query
            .replace("İ", "i")
            .replace("I", "ı")
            .lower()
        )
        colloquial_expansions = []
        colloquial_dismissal_words = (
            "qov", "qovul", "qovdular",
            "çıxarıldım", "çıxardılar", "çıxarıblar",
        )
        if any(w in normalized_q for w in colloquial_dismissal_words):
            colloquial_expansions.append(
                "işdən çıxarılma əmək müqaviləsinə xitam ləğv"
            )
        if any(w in normalized_q for w in (
            "maaşımı vermirlər", "maaş gecikir", "pul vermirlər"
        )):
            colloquial_expansions.append(
                "əmək haqqının ödənilməsi müddəti gecikdirilməsi"
            )
        if colloquial_expansions:
            effective_query = (
                f"{effective_query} {' '.join(colloquial_expansions)}"
            )

        self._ensure_bm25_loaded()

        # Check if query directly references a specific statutory article
        art_match = re.search(
            r"(?:maddə\s*(\d+)|\b(\d+)-(?:ci|cı|cu|cü)\s*maddə)",
            query,
            re.IGNORECASE,
        )
        targeted_art = (
            art_match.group(1) or art_match.group(2) if art_match else None
        )

        # Fetch more candidates for rank fusion
        candidate_k = max(k * 3, 15)

        # 1. Dense vector retrieval
        dense_results = self.vector_mgr.query(
            query_text=effective_query, top_k=candidate_k
        )

        # 2. BM25 lexical retrieval
        bm25_results = self.bm25_mgr.query(
            query_text=effective_query, top_k=candidate_k
        )

        # 3. Reciprocal Rank Fusion & Combined Scoring
        fused_chunks = self._fuse_results(
            dense_results=dense_results,
            bm25_results=bm25_results,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            top_k=k,
        )

        if targeted_art:
            matching = [
                c for c in self.bm25_mgr.chunks
                if str(c.article_number) == targeted_art
            ]
            targeted_chunks = [
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    article_number=str(c.article_number),
                    title=c.title,
                    section=c.section,
                    chapter=c.chapter,
                    text=c.text,
                    score=1.0,
                    dense_score=1.0,
                    bm25_score=1.0,
                    rank=idx,
                    page_start=c.page_start,
                    page_end=c.page_end,
                )
                for idx, c in enumerate(matching, start=1)
            ]
            if targeted_chunks:
                existing_cids = {c.chunk_id for c in targeted_chunks}
                combined = targeted_chunks + [
                    c for c in fused_chunks if c.chunk_id not in existing_cids
                ]
                return combined[:k]

        return fused_chunks

    def _fuse_results(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        dense_weight: float,
        bm25_weight: float,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Merges dense and BM25 results using RRF and weighted normalization.
        """
        chunk_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}
        dense_score_map: Dict[str, float] = {}
        bm25_score_map: Dict[str, float] = {}

        # Process dense rankings
        for rank_idx, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            dense_score_map[cid] = item.get("score", 0.0)
            rrf_val = dense_weight * (1.0 / (self.k_rrf + rank_idx))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_val

        # Process BM25 rankings
        for rank_idx, item in enumerate(bm25_results, start=1):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            bm25_score_map[cid] = item.get("score", 0.0)
            rrf_val = bm25_weight * (1.0 / (self.k_rrf + rank_idx))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_val

        if not rrf_scores:
            return []

        # Normalize combined RRF score to [0.0, 1.0]
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        raw_sorted_cids = sorted(
            rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True
        )

        # Select top candidates with article diversification (max 2 chunks/art)
        sorted_cids: List[str] = []
        article_counts: Dict[str, int] = {}
        for cid in raw_sorted_cids:
            art_num = str(chunk_map[cid].get("article_number", ""))
            if article_counts.get(art_num, 0) < 2:
                sorted_cids.append(cid)
                article_counts[art_num] = article_counts.get(art_num, 0) + 1
            if len(sorted_cids) >= top_k:
                break

        if len(sorted_cids) < top_k:
            for cid in raw_sorted_cids:
                if cid not in sorted_cids:
                    sorted_cids.append(cid)
                if len(sorted_cids) >= top_k:
                    break

        output_chunks: List[RetrievedChunk] = []
        for rank_pos, cid in enumerate(sorted_cids, start=1):
            item = chunk_map[cid]
            norm_score = rrf_scores[cid] / max_rrf if max_rrf > 0 else 0.0

            # Weighted linear blend of raw normalized scores
            d_score = dense_score_map.get(cid, 0.0)
            b_score = bm25_score_map.get(cid, 0.0)
            linear_confidence = (
                (dense_weight * d_score + bm25_weight * b_score) /
                (dense_weight + bm25_weight)
            )

            # Combined confidence metric blends rank fusion & direct similarity
            final_confidence = 0.5 * norm_score + 0.5 * linear_confidence

            output_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    article_number=str(item.get("article_number", "")),
                    title=item.get("title", ""),
                    section=item.get("section", ""),
                    chapter=item.get("chapter", ""),
                    text=item.get("text", ""),
                    score=round(final_confidence, 4),
                    dense_score=round(d_score, 4),
                    bm25_score=round(b_score, 4),
                    rank=rank_pos,
                    page_start=int(item.get("page_start", 1)),
                    page_end=int(item.get("page_end", 1)),
                )
            )

        # Sort by final confidence score and assign 1-indexed ranks
        output_chunks.sort(key=lambda x: x.score, reverse=True)
        for idx, chunk in enumerate(output_chunks, start=1):
            chunk.rank = idx

        return output_chunks

    def _extract_case_state_terms(self, case_state: Any) -> str:
        """Returns Azerbaijani confirmed case terms for existing callers."""
        return enrich_query("", case_state).strip()


# Global module-level convenience function
_retriever_instance: Optional[LegalRetriever] = None


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    case_state: Optional[Any] = None,
) -> List[RetrievedChunk]:
    """Retrieves top-k relevant legal chunks for a query string.

    Args:
        query: Search query in Azerbaijani.
        top_k: Number of ranked chunks to retrieve.
        case_state: Optional dialogue CaseState to enrich query.

    Returns:
        List of RetrievedChunk instances.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = LegalRetriever()
    return _retriever_instance.retrieve(
        query=query, top_k=top_k, case_state=case_state
    )
