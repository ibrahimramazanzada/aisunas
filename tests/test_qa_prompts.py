"""Offline tests for Phase 5 QA (no API, no models, no index)."""

import unittest
from unittest.mock import MagicMock, patch

from src.config import settings
from src.generation import qa_prompts
from src.guardrails import add_legal_disclaimer


class TestQAPrompts(unittest.TestCase):
    """Checks the guardrail wiring of answer_question."""

    @classmethod
    def setUpClass(cls):
        # Import the real dataclass without loading retrieval models.
        with patch.dict("sys.modules", {
            "src.indexing.bm25_indexer": MagicMock(),
            "src.indexing.vector_store": MagicMock(),
        }):
            from src.retrieval.retriever import RetrievedChunk
            cls.chunk_type = RetrievedChunk

    def make_chunk(self, number, dense_score=0.9):
        """Builds a real RetrievedChunk for the given article."""
        return self.chunk_type(
            chunk_id=f"c{number}", article_number=str(number),
            title="Nümunə", section="", chapter="",
            text="Nümunə hüquqi mətn.", score=0.9,
            dense_score=dense_score, bm25_score=0.5, rank=1,
            page_start=1, page_end=1,
        )

    def retrieve(self, dense_score=0.9):
        """Returns a fake retriever yielding articles 70 and 74."""
        chunks = [self.make_chunk(70, dense_score),
                  self.make_chunk(74, dense_score)]
        return lambda query, top_k=None: chunks

    def test_grounded_answer_passes(self):
        text = "Bu, Maddə 74 ilə tənzimlənir."
        result = qa_prompts.answer_question(
            "sual", llm_fn=lambda s, u: text,
            retrieve_fn=self.retrieve(),
        )
        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, add_legal_disclaimer(text))

    def test_structured_provider_configuration_preserves_schema(self):
        """Gemini gets the strict schema without changing the return format."""
        client = MagicMock()
        client.models.generate_content.return_value.text = '{"legal_basis":[]}'
        from src.generation.case_prompts import DRAFT_RESPONSE_SCHEMA
        raw = qa_prompts._generate_with_retry.__wrapped__(
            client, "test-model", "system", "user", DRAFT_RESPONSE_SCHEMA)
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, DRAFT_RESPONSE_SCHEMA)
        self.assertEqual(raw, '{"legal_basis":[]}')

    def test_quick_provider_remains_plain_text(self):
        """Optional structured output does not affect one-shot QA callers."""
        client = MagicMock()
        client.models.generate_content.return_value.text = "Adi cavab"
        raw = qa_prompts._generate_with_retry.__wrapped__(
            client, "test-model", "system", "user")
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertIsNone(config.response_mime_type)
        self.assertIsNone(config.response_json_schema)
        self.assertEqual(raw, "Adi cavab")

    def test_ungrounded_article_triggers_fallback(self):
        result = qa_prompts.answer_question(
            "sual", llm_fn=lambda s, u: "Bax: Maddə 999.",
            retrieve_fn=self.retrieve(),
        )
        self.assertFalse(result.grounded)
        self.assertIn(settings.NO_ARTICLE_FALLBACK_AZ, result.answer)
        self.assertEqual(result.flagged_articles, ["999"])

    def test_ordinal_citation_is_checked(self):
        result = qa_prompts.answer_question(
            "sual", llm_fn=lambda s, u: "999-cu maddəyə bax.",
            retrieve_fn=self.retrieve(),
        )
        self.assertFalse(result.grounded)

    def test_llm_not_found_returns_fallback(self):
        result = qa_prompts.answer_question(
            "sual", llm_fn=lambda s, u: qa_prompts.NOT_FOUND_TOKEN,
            retrieve_fn=self.retrieve(),
        )
        self.assertFalse(result.grounded)
        self.assertIn(settings.NO_ARTICLE_FALLBACK_AZ, result.answer)

    def test_low_relevance_skips_llm(self):
        def boom(system, user):
            raise AssertionError("LLM must not be called")

        result = qa_prompts.answer_question(
            "sual", llm_fn=boom, retrieve_fn=self.retrieve(0.1),
        )
        self.assertFalse(result.grounded)

    def test_empty_retrieval_skips_llm(self):
        def boom(system, user):
            raise AssertionError("LLM must not be called")

        result = qa_prompts.answer_question(
            "sual", llm_fn=boom, retrieve_fn=lambda q, top_k=None: [],
        )
        self.assertFalse(result.grounded)
        self.assertTrue(result.answer.endswith(settings.LEGAL_DISCLAIMER_AZ))

    def test_safe_failure_reasons_distinguish_provider_and_retrieval(self):
        """Distinguishes service failures from missing legal evidence."""
        provider = MagicMock(side_effect=TimeoutError("private payload"))
        with patch.object(qa_prompts.logger, "error") as logging:
            result = qa_prompts.answer_question(
                "sual", llm_fn=provider, retrieve_fn=self.retrieve())
        self.assertEqual(result.failure_reason, "provider_error")
        self.assertNotIn("private", str(logging.call_args))
        for raw, reason in (("", "empty_response"),
                            ("NOT_FOUND", "provider_refusal")):
            result = qa_prompts.answer_question(
                "sual", llm_fn=lambda s, u: raw, retrieve_fn=self.retrieve())
            self.assertEqual(result.failure_reason, reason)
        result = qa_prompts.answer_question(
            "sual", retrieve_fn=lambda q, top_k=None: [])
        self.assertEqual(result.failure_reason, "insufficient_retrieval")


if __name__ == "__main__":
    unittest.main()
