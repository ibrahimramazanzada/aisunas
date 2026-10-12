"""Offline Phase 8 dialogue, extraction and retrieval-boundary tests."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config import settings
from src.dialogue.case_state import CaseState
from src.dialogue.category_classifier import CategoryPrediction
from src.dialogue.slot_filling import DialogueFlow, SlotExtractor
from src.dialogue.slot_schema import (
    DisputeCategory as DC, category_label, required_slots,
)
from src.retrieval.case_query import enrich_query
from src.retrieval.retriever import LegalRetriever, RetrievedChunk, retrieve


def record(value, confidence=0.95, correction=False):
    """Builds a mock LLM slot record."""
    return {"value": value, "confidence": confidence,
            "correction": correction}


class TestExtraction(unittest.TestCase):
    """Verifies validation, provenance and prompt boundaries."""

    def setUp(self):
        """Creates a confirmed wages case."""
        self.state = CaseState()
        self.state.confirm_category(DC.UNPAID_WAGES)

    def extract(self, records, message="yanvar 1000 manat"):
        """Extracts from a deterministic mock JSON response."""
        llm = Mock(return_value=json.dumps({"slots": records}))
        result = SlotExtractor(llm).extract(message, self.state,
                                            "unpaid_period")
        payload = json.loads(llm.call_args.args[1])
        self.assertEqual(payload["pending_slot"], "unpaid_period")
        self.assertNotIn("dismissal_date", payload["slots"])
        return result

    def test_multiple_values(self):
        """Verifies multiple values."""
        result = self.extract({"unpaid_period": record("yanvar"),
                               "amount_owed_approx": record("1000 manat")})
        self.assertEqual(set(result), {"unpaid_period", "amount_owed_approx"})

    def test_invalid_values_and_confidence(self):
        """Verifies invalid values and confidence."""
        invalid = [None, "", "invented", False, 1000, {}, []]
        for value in invalid:
            with self.subTest(value=value):
                result = self.extract({"unpaid_period": record(value)})
                self.assertEqual(result, {})
        for confidence in (None, True, "0.9", -1, 0.79, 1.1,
                           float("nan"), float("inf")):
            with self.subTest(confidence=confidence):
                self.assertEqual(self.extract({
                    "unpaid_period": record("yanvar", confidence)}), {})

    def test_unknown_inapplicable_and_malformed_records(self):
        """Verifies unknown inapplicable and malformed records."""
        self.assertEqual(self.extract({"unknown": record("yanvar"),
                                       "dismissal_date": record("yanvar"),
                                       "unpaid_period": []}), {})
        for data in ([], {}, {"slots": []}):
            llm = Mock(return_value=json.dumps(data))
            self.assertEqual(SlotExtractor(llm).extract("yanvar", self.state),
                             {})

    def test_no_extraction_before_confirmation_or_empty(self):
        """Verifies no extraction before confirmation or empty."""
        llm = Mock()
        extractor = SlotExtractor(llm)
        self.assertEqual(extractor.extract("yanvar", CaseState()), {})
        self.assertEqual(extractor.extract(" ", self.state), {})
        llm.assert_not_called()

    def test_dates_preserved_and_threshold_validation(self):
        """Verifies dates preserved and threshold validation."""
        value = "keçən ay"
        result = self.extract({"last_paid_date": record(value)}, value)
        self.assertEqual(result["last_paid_date"]["value"], value)
        with self.assertRaises(ValueError):
            SlotExtractor(threshold=2)

    def test_uncertain_clauses_do_not_discard_independent_facts(self):
        """Rejects uncertain amounts while retaining a separate clear fact."""
        for value, message in (("bilmirəm.", "bilmirəm."),
                               ("1000 manat", "Bəlkə 1000 manat"),
                               ("1000 manat", "1000 manat, dəqiq bilmirəm")):
            with self.subTest(message=message):
                result = self.extract({"amount_owed_approx": record(value)},
                                      message)
                self.assertEqual(result, {})
        result = self.extract({"amount_owed_approx": record("1000 manat"),
                               "unpaid_period": record("yanvar")},
                              "Bəlkə 1000 manat; yanvar")
        self.assertEqual(set(result), {"unpaid_period"})

    def test_duplicate_json_is_rejected(self):
        """Rejects contradictory duplicate confidence keys."""
        llm = Mock(return_value='{"slots":{"unpaid_period":'
                   '{"value":"yanvar","confidence":0.1,"confidence":1}}}')
        with self.assertRaises(ValueError):
            SlotExtractor(llm).extract("yanvar", self.state)

    def test_oversized_confidence_does_not_block_other_slots(self):
        """Rejects an oversized number without losing another valid record."""
        result = self.extract({"amount_owed_approx": record("1000", 10**400),
                               "unpaid_period": record("yanvar")})
        self.assertEqual(set(result), {"unpaid_period"})


class TestDialogue(unittest.TestCase):
    """Exercises persisted dialogue transitions with all services mocked."""

    def setUp(self):
        """Creates isolated service mocks and relevant source fixtures."""
        self.classifier = Mock(threshold=0.32)
        self.classifier.predict.return_value = CategoryPrediction(
            DC.UNPAID_WAGES, 0.9)
        self.llm = Mock(return_value='{"slots": {}}')
        self.source = RetrievedChunk(
            "fixture", "fixture-article", "Fixture title", "Section",
            "Chapter", "Fixture source text", 0.9, 0.8, 0.7, 1, 3, 4)
        self.retrieval = Mock(return_value=[self.source])
        self.flow = DialogueFlow(classifier=self.classifier,
                                 extractor=SlotExtractor(self.llm),
                                 retrieve_fn=self.retrieval)

    def choose(self, category=DC.UNPAID_WAGES):
        """Explicitly selects a category through the natural label."""
        return self.flow.handle_message(category_label(category))

    def answer(self, result, message, records):
        """Sends an answer and mock extraction for an existing session."""
        self.llm.return_value = json.dumps({"slots": records})
        return self.flow.handle_message(message, result.session_id)

    def complete(self):
        """Completes a wages case in one multi-slot user answer."""
        result = self.choose()
        values = {"employment_status": "işləyirəm",
                  "last_paid_date": "iyun", "unpaid_period": "iyul",
                  "amount_owed_approx": "1000 manat"}
        return self.answer(result, "; ".join(values.values()),
                           {k: record(v) for k, v in values.items()})

    def test_prediction_confirmation_initial_facts(self):
        """Verifies prediction confirmation initial facts."""
        result = self.flow.handle_message("Maaşımı vermirlər, işləyirəm")
        self.assertEqual(result.status, "category_confirmation")
        self.assertFalse(result.state.category_confirmed)
        self.llm.assert_not_called()
        self.llm.return_value = json.dumps({"slots": {
            "employment_status": record("işləyirəm")}})
        result = self.flow.handle_message("bəli", result.session_id)
        self.assertTrue(result.state.category_confirmed)
        self.assertEqual(result.next_slot, "last_paid_date")
        self.assertEqual(len(result.state.conversation_history), 4)
        self.classifier.predict.assert_called_once()

    def test_all_categories_and_order_no_repeat(self):
        """Verifies all categories and order no repeat."""
        for category in DC:
            with self.subTest(category=category):
                result = self.choose(category)
                for index, name in enumerate(required_slots(category)):
                    self.assertEqual(result.next_slot, name)
                    value = f"istifadəçi faktı {index}"
                    result = self.answer(result, value, {name: record(value)})
                self.assertEqual(result.status, "ready")
                self.assertTrue(result.state.is_ready_for_draft())

    def test_multi_slot_completion_metadata_and_cached_sources(self):
        """Verifies multi slot completion metadata and cached sources."""
        result = self.complete()
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.state.retrieved_articles,
                         [self.source.to_dict()])
        self.assertIn(settings.LEGAL_DISCLAIMER_AZ, result.message)
        self.assertEqual(self.retrieval.call_count, 1)
        self.llm.return_value = '{"slots": {}}'
        result = self.flow.handle_message("davam", result.session_id)
        self.assertEqual(result.status, "ready")
        self.assertEqual(self.retrieval.call_count, 1)

    def test_correction_requires_explicit_intent(self):
        """Verifies correction requires explicit intent."""
        result = self.choose()
        result = self.answer(result, "iyul", {"unpaid_period": record("iyul")})
        result = self.answer(result, "avqust", {
            "unpaid_period": record("avqust", correction=True)})
        self.assertEqual(result.state.filled_slots["unpaid_period"], "iyul")
        result = self.answer(result, "Düzəliş: avqust", {
            "unpaid_period": record("avqust", correction=True)})
        self.assertEqual(result.state.filled_slots["unpaid_period"], "avqust")

    def test_completed_case_correction_refreshes_sources(self):
        """Verifies completed case correction refreshes sources."""
        result = self.complete()
        result = self.answer(result, "Düzəliş: 2000 manat", {
            "amount_owed_approx": record("2000 manat", correction=True)})
        self.assertEqual(result.status, "ready")
        self.assertEqual(self.retrieval.call_count, 2)

    def test_category_change_retains_only_applicable_facts(self):
        """Verifies category change retains only applicable facts."""
        result = self.complete()
        count = len(result.state.conversation_history)
        result = self.flow.handle_message(
            category_label(DC.UNPAID_OVERTIME), result.session_id)
        self.assertEqual(result.next_slot, "contracted_hours")
        self.assertEqual(result.state.filled_slots, {"unpaid_period": "iyul"})
        self.assertEqual(result.state.retrieved_articles, [])
        self.assertEqual(len(result.state.conversation_history), count + 2)

    def test_low_confidence_retries_and_explicit_selection(self):
        """Verifies low confidence retries and explicit selection."""
        self.classifier.predict.return_value = None
        result = self.flow.handle_message("problem var")
        self.assertEqual(result.status, "category_selection")
        result = self.flow.handle_message("aydın deyil", result.session_id)
        self.assertEqual(result.status, "category_selection")
        self.assertEqual(self.classifier.predict.call_count, 2)
        result = self.flow.handle_message("seçirəm", result.session_id,
                                          category=DC.UNPAID_WAGES)
        self.assertEqual(result.next_slot, "employment_status")
        self.assertEqual(self.classifier.predict.call_count, 3)

    def test_proposal_rejection_does_not_reconfirm(self):
        """Verifies proposal rejection does not reconfirm."""
        result = self.flow.handle_message("Maaş problemi")
        result = self.flow.handle_message("xeyr", result.session_id)
        self.assertEqual(result.status, "category_selection")
        self.assertIsNone(result.state.dispute_category)
        self.classifier.predict.assert_called_once()

    def test_short_negative_answer_is_a_fact_after_confirmation(self):
        """Verifies short negative answer is a fact after confirmation."""
        result = self.choose(DC.UNLAWFUL_DISMISSAL)
        values = {name: f"fakt {index}" for index, name in enumerate(
            required_slots(DC.UNLAWFUL_DISMISSAL)[:-1])}
        result = self.answer(result, "; ".join(values.values()),
                             {k: record(v) for k, v in values.items()})
        self.assertEqual(result.next_slot, "signed_documents")
        result = self.answer(result, "xeyr", {
            "signed_documents": record("xeyr")})
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.state.filled_slots["signed_documents"], "xeyr")

    def test_explicit_selection_preserves_initial_message_facts(self):
        """Verifies explicit selection preserves initial message facts."""
        self.classifier.predict.return_value = None
        result = self.flow.handle_message("Maaşımı vermirlər, işləyirəm")
        self.llm.return_value = json.dumps({"slots": {
            "employment_status": record("işləyirəm")}})
        result = self.flow.handle_message(
            category_label(DC.UNPAID_WAGES), result.session_id)
        self.assertEqual(result.next_slot, "last_paid_date")

    def test_classifier_errors_unknown_and_invalid_probabilities(self):
        """Verifies classifier errors unknown and invalid probabilities."""
        for prediction in (CategoryPrediction("UNKNOWN", 0.9),
                           CategoryPrediction(DC.UNPAID_WAGES, 0.1),
                           CategoryPrediction(DC.UNPAID_WAGES, float("nan"))):
            self.classifier.predict.return_value = prediction
            self.assertEqual(self.flow.handle_message("problem").status,
                             "category_selection")
        self.classifier.predict.side_effect = RuntimeError("model failure")
        self.assertEqual(self.flow.handle_message("problem").status,
                         "category_selection")

    def test_model_loaded_once_and_missing_model_fallback(self):
        """Verifies model loaded once and missing model fallback."""
        with patch('src.dialogue.category_classifier.CategoryClassifier.load',
                   return_value=self.classifier) as load:
            self.classifier.predict.return_value = None
            flow = DialogueFlow(extractor=SlotExtractor(self.llm))
            result = flow.handle_message("problem")
            flow.handle_message("yenə problem", result.session_id)
            load.assert_called_once()
        with patch('src.dialogue.category_classifier.CategoryClassifier.load',
                   side_effect=FileNotFoundError) as load:
            flow = DialogueFlow()
            result = flow.handle_message("problem")
            flow.handle_message("yenə", result.session_id)
            load.assert_called_once()

    def test_ambiguous_malformed_and_extraction_errors_reask(self):
        """Verifies ambiguous malformed and extraction errors reask."""
        result = self.choose()
        for response in ('{"slots": {}}', 'bad json', '[]',
                         '{"slots": {"employment_status": '
                         '{"value": "bilmirəm", "confidence": 0.2}}}'):
            self.llm.return_value = response
            result = self.flow.handle_message("bilmirəm", result.session_id)
            self.assertEqual(result.next_slot, "employment_status")
            self.assertEqual(result.state.filled_slots, {})
        self.llm.side_effect = RuntimeError("API unavailable")
        result = self.flow.handle_message("işləyirəm", result.session_id)
        self.assertEqual(result.next_slot, "employment_status")
        self.retrieval.assert_not_called()

    def test_empty_invalid_unknown_session_and_invalid_category(self):
        """Verifies empty invalid unknown session and invalid category."""
        for message in (None, 1, "", " "):
            self.assertEqual(self.flow.handle_message(message).status,
                             "invalid_input")
        self.classifier.predict.assert_not_called()
        with self.assertRaises(KeyError):
            self.flow.handle_message("test", "does-not-exist")
        result = self.flow.handle_message("test", category="UNKNOWN")
        self.assertEqual(result.status, "category_selection")

    def test_session_isolation_and_detached_result(self):
        """Verifies session isolation and detached result."""
        first = self.choose()
        second = self.choose()
        first = self.answer(first, "işləyirəm", {
            "employment_status": record("işləyirəm")})
        self.assertEqual(self.flow.store.get_session(second.session_id)
                         .filled_slots, {})
        first.state.filled_slots.clear()
        self.assertTrue(self.flow.store.get_session(first.session_id)
                        .filled_slots)

    def test_retrieval_empty_weak_error_and_retry(self):
        """Verifies retrieval empty weak error and retry."""
        for error in (False, True):
            with self.subTest(error=error):
                self.retrieval.side_effect = RuntimeError() if error else None
                self.retrieval.return_value = []
                result = self.complete()
                self.assertEqual(result.status,
                                 "retrieval_error" if error
                                 else "retrieval_empty")
                self.assertTrue(result.state.is_ready_for_draft())
                self.assertEqual(result.state.retrieved_articles, [])
                self.retrieval.side_effect = None
                self.retrieval.return_value = [self.source]
                self.llm.return_value = '{"slots": {}}'
                result = self.flow.handle_message("yenidən", result.session_id)
                self.assertEqual(result.status, "ready")
        self.source.dense_score = 0.1
        result = self.complete()
        self.assertEqual(result.status, "retrieval_empty")

    def test_empty_turn_preserves_question_context(self):
        """Keeps the pending slot after an empty message."""
        result = self.choose()
        result = self.flow.handle_message(" ", result.session_id)
        self.assertEqual(result.next_slot, "employment_status")
        result = self.answer(result, "işləyirəm", {
            "employment_status": record("işləyirəm")})
        payload = json.loads(self.llm.call_args.args[1])
        self.assertEqual(payload["pending_slot"], "employment_status")
        self.assertEqual(result.next_slot, "last_paid_date")

    def test_selection_and_category_change_extract_current_facts(self):
        """Keeps facts accompanying an explicit category argument."""
        self.llm.return_value = json.dumps({"slots": {
            "employment_status": record("işləyirəm")}})
        result = self.flow.handle_message(
            "işləyirəm", category=DC.UNPAID_WAGES)
        self.assertEqual(result.next_slot, "last_paid_date")
        self.llm.return_value = json.dumps({"slots": {
            "contracted_hours": record("40 saat")}})
        result = self.flow.handle_message("40 saat", result.session_id,
                                          category=DC.UNPAID_OVERTIME)
        self.assertEqual(result.next_slot, "actual_hours_worked_pattern")
        self.assertEqual(result.state.filled_slots,
                         {"contracted_hours": "40 saat"})

    def test_invalid_injected_records_do_not_block_valid_slots(self):
        """Skips malformed records and preserves later valid facts."""
        self.flow.extractor = Mock()
        self.flow.extractor.extract.return_value = {
            "unknown": {"value": "işləyirəm"},
            "last_paid_date": [], "amount_owed_approx": {"value": ""},
            "employment_status": {"value": "işləyirəm"},
        }
        result = self.flow.handle_message(
            "işləyirəm", category=DC.UNPAID_WAGES)
        self.assertEqual(result.state.filled_slots,
                         {"employment_status": "işləyirəm"})


class TestCaseRetrieval(unittest.TestCase):
    """Verifies both retrieval entrypoints without models or index writes."""

    def test_confirmed_query_category_values_and_no_mutation(self):
        """Verifies confirmed query category values and no mutation."""
        state = CaseState()
        state.confirm_category(DC.UNPAID_WAGES)
        state.update_slot("amount_owed_approx", 0)
        state.update_slot("employment_status", False)
        snapshot = state.to_dict()
        query = enrich_query("maaş", state)
        self.assertIn(category_label(DC.UNPAID_WAGES), query)
        self.assertIn("0", query)
        self.assertIn("False", query)
        self.assertNotIn("DisputeCategory", query)
        self.assertEqual(snapshot, state.to_dict())
        state.category_confirmed = False
        self.assertEqual(enrich_query(" maaş ", state), "maaş")
        self.assertEqual(enrich_query("maaş"), "maaş")

    def test_both_backends_receive_enriched_query(self):
        """Verifies both backends receive enriched query."""
        state = CaseState()
        state.confirm_category(DC.UNPAID_WAGES)
        state.update_slot("unpaid_period", "iyul")
        vector = Mock()
        bm25 = Mock()
        vector.query.return_value = []
        bm25.query.return_value = []
        retriever = LegalRetriever(vector, bm25)
        self.assertEqual(retriever.retrieve("maaş", case_state=state), [])
        for manager in (vector, bm25):
            query = manager.query.call_args.kwargs["query_text"]
            self.assertIn("iyul", query)
            self.assertIn(category_label(DC.UNPAID_WAGES), query)
        self.assertEqual(retriever.retrieve(" ", case_state=state), [])

    def test_module_entrypoint_forwards_case_state(self):
        """Verifies module entrypoint forwards case state."""
        state = CaseState()
        with patch('src.retrieval.retriever._retriever_instance') as backend:
            backend.retrieve.return_value = []
            self.assertEqual(retrieve("maaş", top_k=2, case_state=state), [])
            backend.retrieve.assert_called_once_with(
                query="maaş", top_k=2, case_state=state)

    def test_missing_or_truthy_confirmation_does_not_enrich(self):
        """Requires boolean confirmation rather than assuming it."""
        state = SimpleNamespace(dispute_category=DC.UNPAID_WAGES,
                                filled_slots={"unpaid_period": "secret"})
        self.assertEqual(enrich_query("maaş", state), "maaş")
        for flag in (False, "false", "true", 1, None):
            state.category_confirmed = flag
            self.assertEqual(enrich_query("maaş", state), "maaş")

    def test_existing_positional_callers_and_real_metadata(self):
        """Keeps original call shapes, query text and backend metadata."""
        source = {
            "chunk_id": "fixture-70", "article_number": "70",
            "title": "Source title", "text": "Source text",
            "section": "Source section", "chapter": "Source chapter",
            "page_start": 5, "page_end": 6, "score": 0.8,
        }
        vector, bm25 = Mock(), Mock()
        vector.query.return_value = [source]
        bm25.query.return_value = [source]
        backend = LegalRetriever(vector, bm25)
        chunks = backend.retrieve("maaş", 2, 0.5, 0.5)
        for manager in (vector, bm25):
            self.assertEqual(manager.query.call_args.kwargs["query_text"],
                             "maaş")
        for name in ("chunk_id", "article_number", "title", "text", "section",
                     "chapter", "page_start", "page_end"):
            self.assertEqual(getattr(chunks[0], name), source[name])
        with patch('src.retrieval.retriever._retriever_instance', backend):
            self.assertEqual(retrieve("maaş", 2)[0].to_dict(),
                             chunks[0].to_dict())


if __name__ == "__main__":
    unittest.main()
