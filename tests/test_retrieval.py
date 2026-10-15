"""Unit and integration tests for hybrid retrieval over Azerbaijani Law.

Tests lexical BM25 matching, dense semantic matching, Reciprocal Rank Fusion,
and query behavior against persisted ChromaDB and BM25 indices.
"""

import sys
import unittest
from unittest.mock import Mock

from src.retrieval.retriever import LegalRetriever, retrieve


def test_hybrid_retrieval_exact_keyword():
    """Tests retrieval with exact legal keywords in Azerbaijani."""
    retriever = LegalRetriever()

    # Query for employer dismissal grounds
    results = retriever.retrieve(
        "Maddə 70 işəgötürən tərəfindən xitam əsasları", top_k=2
    )
    assert len(results) > 0, "Expected at least one retrieved chunk"

    top = results[0]
    assert top.article_number == "70", (
        f"Expected article 70, got {top.article_number}"
    )
    assert top.rank == 1
    assert 0.0 <= top.score <= 1.0
    assert 0.0 <= top.dense_score <= 1.0
    assert 0.0 <= top.bm25_score <= 1.0
    assert "xitam" in top.text.lower()
    print("PASS: Exact keyword retrieval for Maddə 70")


def test_hybrid_retrieval_semantic_query():
    """Tests semantic retrieval where keywords may be paraphrased."""
    retriever = LegalRetriever()

    # Query regarding military draft termination
    query = "işçi orduya hərbi xidmətə çağırıldıqda əmək müqaviləsi"
    results = retriever.retrieve(query, top_k=2)
    assert len(results) > 0, "Expected results for military service query"

    top = results[0]
    assert top.article_number == "74", (
        f"Expected article 74, got {top.article_number}"
    )
    assert "hərbi" in top.text.lower()
    print("PASS: Semantic retrieval for Maddə 74")


def test_retrieval_empty_query():
    """Tests that empty or whitespace query safely returns empty list."""
    retriever = LegalRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    print("PASS: Empty query edge case")


def test_convenience_function():
    """Tests the global retrieve() module-level function."""
    results = retrieve("müəssisə ləğv edildikdə", top_k=1)
    assert len(results) == 1
    assert results[0].article_number == "70"
    print("PASS: Module-level retrieve() entrypoint")


class TestEmployerDismissalQueries(unittest.TestCase):
    """Keeps employer dismissal distinct from resignation and negation."""

    def query_used(self, message):
        """Captures queries sent to both real retrieval interfaces."""
        dense = Mock()
        dense.query.return_value = []
        lexical = Mock()
        lexical.query.return_value = []
        retriever = LegalRetriever(vector_mgr=dense, bm25_mgr=lexical)
        retriever.retrieve(message)
        query = dense.query.call_args.kwargs["query_text"]
        self.assertEqual(query, lexical.query.call_args.kwargs["query_text"])
        return query

    def test_explicit_employer_termination(self):
        """Expands an explicit employer action without replacing user facts."""
        message = "İşəgötürən əmək müqaviləmə xitam verib."
        query = self.query_used(message)
        self.assertTrue(query.startswith(message))
        self.assertIn("işəgötürən tərəfindən", query)

    def test_employee_resignation_is_not_employer_dismissal(self):
        """Resignation mentioning an employer does not change its actor."""
        message = "İşəgötürənə bildirdim ki, özüm müqaviləyə xitam verirəm."
        self.assertEqual(self.query_used(message), message)

    def test_negated_employer_termination_is_not_expanded(self):
        """Negation must not become a statement that dismissal occurred."""
        message = "İşəgötürən müqaviləmə xitam verməyib."
        self.assertEqual(self.query_used(message), message)

    def test_maternity_leave_expansion(self):
        """Expands colloquial analıq / dekret queries with statutory terms."""
        message = "analıq məzuniyyətinə necə çıxa bilərəm"
        query = self.query_used(message)
        self.assertTrue(query.startswith(message))
        self.assertIn("hamiləliyə və doğuşa görə məzuniyyət", query)


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    test_hybrid_retrieval_exact_keyword()
    test_hybrid_retrieval_semantic_query()
    test_retrieval_empty_query()
    test_convenience_function()
    print("ALL RETRIEVAL TESTS PASSED SUCCESSFULLY!")
