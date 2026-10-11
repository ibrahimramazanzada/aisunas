"""
Phase 10 backend and real Streamlit AppTest with external services mocked.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from app.backend import AppSession
from src.config import settings
from src.dialogue.category_classifier import CategoryPrediction
from src.dialogue.slot_filling import DialogueFlow, SlotExtractor
from src.dialogue.slot_schema import DisputeCategory as DC, category_label
from src.generation.case_prompts import draft_session
from src.generation.qa_prompts import QAResult
from src.retrieval.retriever import RetrievedChunk


def fixture_chunk():
    """Creates an explicitly synthetic source used only by automated tests."""
    return RetrievedChunk(
        "fixture-ui", "123", "Synthetic UI test source", "Sınaq bölməsi",
        "Sınaq fəsli", "Bu mətn yalnız tətbiqin sınağı üçün hazırlanmışdır.",
        0.9, 0.8, 0.7, 1, 3, 4)


def make_session():
    """Builds actual backend orchestration with mocked expensive services."""
    classifier = Mock(threshold=0.32)
    classifier.predict.return_value = CategoryPrediction(DC.UNPAID_WAGES, 0.9)
    extractor_llm = Mock(return_value='{"slots": {}}')
    chunk = fixture_chunk()
    retrieval = Mock(return_value=[chunk])
    flow = DialogueFlow(classifier=classifier,
                        extractor=SlotExtractor(extractor_llm),
                        retrieve_fn=retrieval)
    provider = Mock(return_value=json.dumps({"legal_basis": [{
        "chunk_id": chunk.chunk_id, "quote": chunk.text}]}))

    def draft(session_id, store):
        """Uses real Phase 9 validation and rendering with a mock provider."""
        return draft_session(session_id, store, provider)

    qa = Mock(return_value=QAResult("Sınaq cavabı", True, [chunk]))
    session = AppSession(flow, qa, draft)
    return session, classifier, extractor_llm, retrieval, provider, qa


def set_wages(extractor_llm):
    """Configures a multi-slot user answer without inventing any values."""
    values = {"employment_status": "işləyirəm", "last_paid_date": "iyun",
              "unpaid_period": "iyul", "amount_owed_approx": "1000 manat"}
    extractor_llm.return_value = json.dumps({"slots": {
        name: {"value": value, "confidence": 0.95}
        for name, value in values.items()}})
    return "; ".join(values.values())


class TestAppBackend(unittest.TestCase):
    """Checks per-browser continuity, safe failures and draft invalidation."""

    def setUp(self):
        """Creates an isolated controller with actual backend components."""
        (self.session, self.classifier, self.llm, self.retrieval,
         self.provider, self.qa) = make_session()

    def complete(self):
        """Completes a confirmed wages case through actual dialogue methods."""
        self.session.submit_case(category_label(DC.UNPAID_WAGES))
        self.session.submit_case(set_wages(self.llm))

    def test_confirmation_readiness_and_no_early_retrieval(self):
        """
        Classifier proposals cannot establish readiness or trigger search.
        """
        self.session.submit_case("Maaşımı vermirlər")
        self.assertFalse(self.session.case_state.category_confirmed)
        self.assertIsNone(self.session.generate())
        self.retrieval.assert_not_called()
        self.session.submit_case("bəli")
        self.assertTrue(self.session.case_state.category_confirmed)
        self.assertFalse(self.session.case_state.is_ready_for_draft())

    def test_completion_generate_cache_and_correction(self):
        """
        Reuses unchanged drafts and invalidates them on explicit correction.
        """
        self.complete()
        result = self.session.generate()
        self.assertEqual(result.status, "drafted")
        self.session.generate()
        self.provider.assert_called_once()
        self.retrieval.assert_called_once()
        self.llm.return_value = json.dumps({"slots": {
            "amount_owed_approx": {"value": "1200 manat", "confidence": 0.95,
                                   "correction": True}}})
        self.session.submit_case("Düzəliş: 1200 manat")
        self.assertIsNone(self.session.complaint)
        self.assertEqual(self.session.case_state.filled_slots[
            "amount_owed_approx"], "1200 manat")
        self.assertEqual(self.retrieval.call_count, 2)

    def test_category_change_and_reset(self):
        """
        Changes category explicitly; reset clears histories and old facts.
        """
        self.complete()
        self.session.generate()
        self.session.submit_case(category_label(DC.UNPAID_OVERTIME))
        self.assertEqual(self.session.case_state.dispute_category,
                         DC.UNPAID_OVERTIME)
        self.assertEqual(self.session.case_state.filled_slots,
                         {"unpaid_period": "iyul"})
        self.assertIsNone(self.session.complaint)
        old_id = self.session.session_id
        self.session.reset()
        self.assertNotEqual(self.session.session_id, old_id)
        self.assertEqual(self.session.case_state.filled_slots, {})
        self.assertEqual(self.session.quick_history, [])

    def test_errors_and_timeouts_keep_facts_and_hide_details(self):
        """Unexpected dialogue and provider failures cannot expose secrets."""
        self.complete()
        before = self.session.case_state.to_dict()
        self.session.flow.handle_message = Mock(
            side_effect=RuntimeError("secret exception"))
        self.assertIsNone(self.session.submit_case("private message"))
        self.assertEqual(self.session.case_state.to_dict(), before)
        self.assertNotIn("secret", self.session.notice[1])
        self.provider.side_effect = TimeoutError("private provider error")
        self.assertEqual(self.session.generate().status, "provider_error")
        self.assertEqual(self.session.case_state.to_dict(), before)

    def test_empty_retrieval_classifier_failure_and_low_confidence(self):
        """
        Manual selection survives classifier failure; empty search is safe.
        """
        self.classifier.predict.side_effect = RuntimeError("unavailable")
        result = self.session.submit_case("problem")
        self.assertEqual(result.status, "category_selection")
        self.retrieval.return_value = []
        self.complete()
        self.assertEqual(self.session.notice[0], "warning")
        result = self.session.generate()
        self.assertEqual(result.status, "insufficient_grounding")
        self.provider.assert_not_called()
        other, classifier, *_ = make_session()
        classifier.predict.return_value = None
        self.assertEqual(other.submit_case("problem").status,
                         "category_selection")

    def test_quick_mode_failures_and_isolation(self):
        """Quick QA never changes case facts; failures return safe messages."""
        self.session.submit_quick("Əmək hüququ ilə bağlı sual")
        self.qa.assert_called_once()
        self.assertIn(settings.LEGAL_DISCLAIMER_AZ,
                      self.session.quick_history[-1]["content"])
        self.assertTrue(self.session.quick_sources)
        self.assertEqual(self.session.case_state.filled_slots, {})
        self.qa.side_effect = TimeoutError("secret provider response")
        self.session.submit_quick("başqa sual")
        self.assertEqual(self.session.quick_sources, [])
        self.assertNotIn("secret", self.session.quick_history[-1]["content"])
        self.session.submit_quick(" ")
        self.assertEqual(self.session.notice[0], "warning")

    def test_browser_sessions_are_independent(self):
        """
        Independent controllers cannot share cases or conversation history.
        """
        second, *_ = make_session()
        self.complete()
        self.assertEqual(second.case_state.filled_slots, {})
        detached = self.session.case_state
        detached.filled_slots.clear()
        self.assertTrue(self.session.case_state.filled_slots)


class TestStreamlitUI(unittest.TestCase):
    """Runs actual Streamlit rendering and widgets with injected backends."""

    def setUp(self):
        """Initializes AppTest and private mock-backed application services."""
        (self.session, self.classifier, self.llm, self.retrieval,
         self.provider, self.qa) = make_session()
        entry = (Path(__file__).resolve().parents[1]
                 / "app" / "streamlit_app.py")
        self.at = AppTest.from_file(str(entry), default_timeout=15)
        self.at.session_state["app_session"] = self.session
        self.at.run()

    def button(self, label):
        """Finds rendered buttons by their Azerbaijani labels."""
        return next(button for button in self.at.button
                    if button.label == label)

    def case_mode(self):
        """Switches to the multi-turn mode through the actual widget."""
        self.at.radio(key="mode").set_value("İşin hazırlanması").run()

    def send(self, message):
        """Submits an actual rendered chat input and reruns the application."""
        self.at.chat_input[0].set_value(message).run()
        self.assertEqual(len(self.at.exception), 0)

    def test_initialization_disclaimer_and_quick_chat(self):
        """
        Welcome, legal disclaimer and grounded quick chat render correctly.
        """
        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(self.at.title[0].value, "Əmək hüququ köməkçisi")
        self.assertTrue(any(settings.LEGAL_DISCLAIMER_AZ in item.value
                            for item in self.at.warning))
        self.send("İş vaxtı haqqında sual")
        self.qa.assert_called_once_with("İş vaxtı haqqında sual")
        self.assertEqual(len(self.at.chat_message), 2)
        self.assertTrue(any("Maddə 123" == item.label
                            for item in self.at.expander))

    def test_ui_confirmation_multislot_sources_and_complaint(self):
        """Completes the actual widgets-to-backend-to-document workflow."""
        self.case_mode()
        self.send("Maaşımı vermirlər")
        self.assertFalse(
            self.at.session_state["case_state"].category_confirmed)
        self.assertFalse(any(b.label == "Ərizə layihəsini hazırla"
                             for b in self.at.button))
        self.send("bəli")
        self.send(set_wages(self.llm))
        self.assertTrue(
            self.at.session_state["case_state"].is_ready_for_draft())
        self.button("Ərizə layihəsini hazırla").click().run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertTrue(any("ƏRİZƏ LAYİHƏSİ" in item.value
                            for item in self.at.text))
        self.assertTrue(any("3–4" in item.value for item in self.at.caption))

    def test_manual_selection_and_new_conversation(self):
        """
        Form selection explicitly confirms; new conversation clears state.
        """
        self.case_mode()
        self.at.selectbox(key="category_choice").set_value(DC.UNPAID_WAGES)
        self.button("Kateqoriyanı təsdiq et").click().run()
        self.assertTrue(self.at.session_state["case_state"].category_confirmed)
        old_id = self.at.session_state["session_id"]
        self.button("Yeni söhbət").click().run()
        self.assertNotEqual(self.at.session_state["session_id"], old_id)
        self.assertFalse(
            self.at.session_state["case_state"].category_confirmed)
        self.assertEqual(len(self.at.exception), 0)

    def test_mode_switch_preserves_case_and_quick_history(self):
        """Switching modes does not erase either independent conversation."""
        self.send("sual")
        self.case_mode()
        self.send("iş haqqında problem")
        old_id = self.at.session_state["session_id"]
        self.at.radio(key="mode").set_value("Sürətli sual").run()
        self.assertEqual(len(self.at.chat_message), 2)
        self.case_mode()
        self.assertEqual(self.at.session_state["session_id"], old_id)
        self.assertEqual(len(self.at.chat_message), 2)

    def test_empty_sources_and_provider_failure_render_warnings(self):
        """Fallback warnings preserve complete case information in the UI."""
        self.retrieval.return_value = []
        self.case_mode()
        self.send(category_label(DC.UNPAID_WAGES))
        self.send(set_wages(self.llm))
        self.button("Ərizə layihəsini hazırla").click().run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertTrue(any(settings.NO_ARTICLE_FALLBACK_AZ in item.value
                            for item in self.at.warning))
        self.assertEqual(len(self.at.expander), 0)

    def test_untrusted_text_is_not_rendered_as_html(self):
        """Source and user strings are rendered as text rather than markup."""
        chunk = fixture_chunk()
        chunk.text = '<script>alert("fixture")</script>'
        self.qa.return_value = QAResult("plain answer", True, [chunk])
        self.send("<b>user text</b>")
        self.assertTrue(any(chunk.text == item.value for item in self.at.text))
        self.assertFalse(any(chunk.text in item.value
                             for item in self.at.markdown))

    def test_default_initialization_without_model_loading(self):
        """The real default entrypoint creates a private empty session."""
        entry = (Path(__file__).resolve().parents[1]
                 / "app" / "streamlit_app.py")
        app = AppTest.from_file(str(entry), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["session_id"])
        self.assertEqual(app.session_state["case_state"].filled_slots, {})
        self.assertNotEqual(app.session_state["session_id"],
                            self.at.session_state["session_id"])

    def test_correction_and_category_change_clear_stale_draft(self):
        """Actual widgets invalidate drafts after changing confirmed facts."""
        self.case_mode()
        self.send(category_label(DC.UNPAID_WAGES))
        self.send(set_wages(self.llm))
        self.button("Ərizə layihəsini hazırla").click().run()
        self.llm.return_value = json.dumps({"slots": {
            "amount_owed_approx": {"value": "1200 manat", "confidence": 0.95,
                                   "correction": True}}})
        self.send("Düzəliş: 1200 manat")
        self.assertIsNone(self.session.complaint)
        self.assertTrue(any("1200 manat" in item.value
                            for item in self.at.text))
        self.at.selectbox(key="category_choice").set_value(DC.UNPAID_OVERTIME)
        self.button("Kateqoriyanı təsdiq et").click().run()
        self.assertEqual(self.session.case_state.dispute_category,
                         DC.UNPAID_OVERTIME)
        self.assertFalse(any(b.label == "Ərizə layihəsini hazırla"
                             for b in self.at.button))
        self.assertEqual(len(self.at.expander), 0)

    def test_provider_timeout_keeps_ui_case_available(self):
        """Provider errors render a safe warning without losing user facts."""
        self.case_mode()
        self.send(category_label(DC.UNPAID_WAGES))
        self.send(set_wages(self.llm))
        before = self.session.case_state.to_dict()
        self.provider.side_effect = TimeoutError("secret provider details")
        self.button("Ərizə layihəsini hazırla").click().run()
        self.assertEqual(len(self.at.exception), 0)
        self.assertEqual(self.session.case_state.to_dict(), before)
        self.assertEqual(self.session.complaint.status, "provider_error")
        self.assertTrue(self.at.warning)
        self.assertFalse(any("secret" in item.value
                             for item in self.at.warning))


if __name__ == "__main__":
    unittest.main()
