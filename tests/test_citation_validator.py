"""Citation unit tests without models, databases, or network calls."""

import unittest
from unittest.mock import MagicMock, patch

from src.guardrails.citation_validator import validate_article_citations


class TestCitationValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the real dataclass without loading retrieval dependencies.
        with patch.dict("sys.modules", {
            "src.indexing.bm25_indexer": MagicMock(),
            "src.indexing.vector_store": MagicMock(),
        }):
            from src.retrieval.retriever import RetrievedChunk
            cls.chunk_type = RetrievedChunk

    def make_results(self, *numbers):
        return [
            self.chunk_type(
                chunk_id=f"chunk_{rank}",
                article_number=number,
                title="Example article",
                section="",
                chapter="",
                text="Example legal text.",
                score=0.7,
                dense_score=0.6,
                bm25_score=0.8,
                rank=rank,
                page_start=1,
                page_end=1,
            )
            for rank, number in enumerate(numbers, start=1)
        ]

    def check_answer(self, answer, retrieved, cited, unsupported):
        results = self.make_results(*retrieved)
        result = validate_article_citations(answer, results)
        self.assertEqual(result.cited_articles, cited)
        self.assertEqual(result.unsupported_articles, unsupported)
        self.assertEqual(result.is_valid, not unsupported)

    def test_no_citation(self):
        self.check_answer("Hüquqi yardım üçün müraciət edin.", [], [], [])

    def test_ordinal_article(self):
        self.check_answer("70-ci maddə", ["70"], ["70"], [])

    def test_article_before_number(self):
        self.check_answer("Maddə 70", ["70"], ["70"], [])

    def test_lowercase_article_before_number(self):
        self.check_answer("maddə 70", ["70"], ["70"], [])

    def test_dative_article(self):
        self.check_answer(
            "Azərbaycan Respublikası Əmək Məcəlləsinin "
            "70-ci maddəsinə əsasən...",
            ["70"], ["70"], [],
        )

    def test_genitive_article(self):
        self.check_answer("70-ci maddənin", ["70"], ["70"], [])

    def test_joined_articles(self):
        self.check_answer(
            "70-ci və 78-ci maddələr", ["70", "78"], ["70", "78"], []
        )

    def test_unsupported_article(self):
        self.check_answer(
            "70-ci və 99-cu maddələrə əsasən...",
            ["70"], ["70", "99"], ["99"],
        )

    def test_duplicate_references(self):
        self.check_answer("Maddə 70 və 70-ci maddə", ["70"], ["70"], [])

    def test_unrelated_number(self):
        self.check_answer(
            "İşçi 30 gün ərzində müraciət edə bilər.", [], [], []
        )

    def test_empty_retrieved_results(self):
        self.check_answer("70-ci maddə", [], ["70"], ["70"])

    def test_multiple_valid_citations(self):
        self.check_answer(
            "Maddə 77. 70-ci və 78-ci maddələrə əsasən...",
            ["70", "77", "78"], ["77", "70", "78"], [],
        )

    def test_deterministic_first_mention_order(self):
        for retrieved in (["70", "77"], ["77", "70", "70"]):
            with self.subTest(retrieved=retrieved):
                self.check_answer(
                    "99-cu və 78-ci maddələr. Maddə 70. "
                    "Maddə 99. 78-ci maddə.",
                    retrieved, ["99", "78", "70"], ["99", "78"],
                )

    def test_comma_separated_articles(self):
        self.check_answer(
            "70-ci, 77-ci və 78-ci maddələr",
            ["70", "77", "78"], ["70", "77", "78"], [],
        )

    def test_normalized_article_numbers(self):
        self.check_answer("Maddə 070 və 70-ci maddə", [" 70 "], ["70"], [])

    def test_other_ordinal_suffixes_and_uppercase(self):
        self.check_answer(
            "6-cı və 3-cü MADDƏLƏR", ["6", "3"], ["6", "3"], []
        )

    def test_unrelated_number_next_to_citation(self):
        self.check_answer(
            "30 gün ərzində 70-ci maddəyə əsasən", ["70"], ["70"], []
        )

    def test_subclause_period_citation(self):
        self.check_answer(
            "Maddə 51.1-ə əsasən sınaq müddəti təyin edilir.",
            ["51"], ["51"], [],
        )

    def test_hyphenated_amendment_article(self):
        self.check_answer(
            "Maddə 76-1 və 76-1-ci maddə üzrə tənzimləmə",
            ["76-1"], ["76-1"], [],
        )


if __name__ == "__main__":
    unittest.main()
