"""Offline Phase 7 state and classifier-boundary tests."""

import json
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from src.dialogue.case_state import CaseState, PHASE7_MODEL_PATH
from src.dialogue.category_classifier import (
    CategoryClassifier, CategoryPrediction,
)
from src.dialogue.slot_schema import DisputeCategory as DC, required_slots


class TestCaseState(unittest.TestCase):
    """Verifies facts, serialization, category proposals and readiness."""

    def test_initial_state(self):
        state = CaseState()
        self.assertIsNone(state.dispute_category)
        self.assertEqual(state.filled_slots, {})
        self.assertEqual(state.conversation_history, [])
        self.assertEqual(state.retrieved_articles, [])
        self.assertNotEqual(state.session_id, CaseState().session_id)
        self.assertEqual(state.missing_required_slots(), [])
        self.assertFalse(state.is_ready_for_draft())

    def test_required_order_and_optional_completion(self):
        for category in DC:
            with self.subTest(category=category):
                state = CaseState()
                state.confirm_category(category)
                names = required_slots(category)
                self.assertEqual(state.missing_required_slots(), names)
                for name in names:
                    state.update_slot(name, "İstifadəçinin faktı")
                self.assertTrue(state.is_ready_for_draft())
                self.assertEqual(state.missing_required_slots(), [])

    def test_corrections_preserve_other_facts(self):
        state = CaseState(dispute_category=DC.UNPAID_WAGES)
        state.update_slot("unpaid_period", "yanvar")
        state.update_slot("last_paid_date", "keçən ay")
        state.update_slot("unpaid_period", "fevral")
        self.assertEqual(state.filled_slots["unpaid_period"], "fevral")
        self.assertEqual(state.filled_slots["last_paid_date"], "keçən ay")

    def test_slot_errors_are_atomic(self):
        state = CaseState(dispute_category=DC.UNPAID_WAGES)
        with self.assertRaises(KeyError):
            state.update_slot("invented", "fact")
        with self.assertRaises(ValueError):
            state.update_slot("dismissal_date", "today")
        for value in [None, "", "  ", [], {}, float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                state.update_slot("unpaid_period", value)
        self.assertEqual(state.filled_slots, {})

    def test_false_and_zero_are_facts(self):
        state = CaseState(dispute_category=DC.UNPAID_WAGES)
        state.update_slot("amount_owed_approx", 0)
        state.update_slot("employment_status", False)
        self.assertNotIn("amount_owed_approx", state.missing_required_slots())
        self.assertNotIn("employment_status", state.missing_required_slots())

    def test_serialization_detaches_and_preserves(self):
        state = CaseState()
        state.confirm_category(DC.UNPAID_WAGES)
        state.update_slot("evidence_available", ["mesaj"])
        state.conversation_history.append({"role": "user", "content": "Salam"})
        state.retrieved_articles.append({"article_number": "70"})
        snapshot = json.loads(json.dumps(state.to_dict()))
        restored = CaseState.from_dict(snapshot)
        self.assertEqual(restored.to_dict(), state.to_dict())
        restored.filled_slots["evidence_available"].append("bank")
        snapshot["conversation_history"].clear()
        self.assertEqual(state.filled_slots["evidence_available"], ["mesaj"])
        self.assertEqual(len(restored.conversation_history), 1)

    def test_malformed_state_rejected(self):
        for data in [{"dispute_category": "OTHER"}, {"session_id": ""},
                     {"category_confirmed": True},
                     {"filled_slots": {"bad": "fact"}},
                     {"conversation_history": ["text"]}]:
            with self.subTest(data=data), self.assertRaises(
                (ValueError, KeyError)
            ):
                CaseState.from_dict(data)

    def test_category_change_keeps_shared_facts(self):
        state = CaseState(dispute_category=DC.UNPAID_WAGES)
        state.update_slot("unpaid_period", "yanvar")
        state.update_slot("last_paid_date", "dekabr")
        state.retrieved_articles.append({"article_number": "70"})
        state.conversation_history.append({"role": "user", "content": "fakt"})
        state.confirm_category(DC.UNPAID_OVERTIME)
        self.assertEqual(state.filled_slots, {"unpaid_period": "yanvar"})
        self.assertEqual(state.retrieved_articles, [])
        self.assertEqual(len(state.conversation_history), 1)
        self.assertFalse(state.is_ready_for_draft())

    def test_proposal_requires_confirmation_even_when_complete(self):
        state = CaseState()
        classifier = Mock()
        classifier.predict.return_value = CategoryPrediction(
            DC.UNPAID_WAGES, 0.35)
        prediction = state.init_from_user_message("Maaş verilmir", classifier)
        self.assertEqual(prediction.confidence, 0.35)
        self.assertEqual(state.dispute_category, DC.UNPAID_WAGES)
        for name in required_slots(DC.UNPAID_WAGES):
            state.update_slot(name, "fakt")
        self.assertFalse(state.is_ready_for_draft())
        state.confirm_category(DC.UNPAID_WAGES)
        self.assertTrue(state.is_ready_for_draft())

    def test_none_prediction_remains_unknown(self):
        state = CaseState()
        classifier = Mock()
        classifier.predict.return_value = None
        self.assertIsNone(state.init_from_user_message("Salam", classifier))
        self.assertIsNone(state.dispute_category)
        self.assertFalse(state.is_ready_for_draft())
        self.assertEqual(state.conversation_history[0]["content"], "Salam")

    def test_off_topic_prediction_is_not_confirmed(self):
        for message in ["Bakıda hava necədir?", "İşdə problemim var."]:
            classifier = Mock()
            classifier.predict.return_value = CategoryPrediction(
                DC.UNPAID_OVERTIME, 0.34)
            state = CaseState()
            state.init_from_user_message(message, classifier)
            self.assertFalse(state.category_confirmed)
            self.assertFalse(state.is_ready_for_draft())

    def test_empty_input_never_loads_model(self):
        with patch.object(CategoryClassifier, "load") as load:
            state = CaseState()
            for message in ["", " \t\n"]:
                self.assertIsNone(state.init_from_user_message(message))
            load.assert_not_called()
            self.assertEqual(state.conversation_history, [])
            with self.assertRaises(TypeError):
                state.init_from_user_message(None)

    def test_explicit_candidate_path(self):
        with patch.object(CategoryClassifier, "load") as load:
            load.return_value.predict.return_value = None
            CaseState().init_from_user_message("Salam")
            load.assert_called_once_with(PHASE7_MODEL_PATH)
        self.assertEqual(PHASE7_MODEL_PATH.name,
                         "category_classifier_candidate.pkl")

    def test_actual_candidate_load_without_embedding(self):
        embed = Mock(side_effect=AssertionError("Must not embed"))
        clf = CategoryClassifier.load(PHASE7_MODEL_PATH, embed_fn=embed)
        self.assertEqual(clf.threshold, 0.32)
        self.assertEqual(set(clf.classes), set(DC))
        embed.assert_not_called()

    def test_model_failure_preserves_state(self):
        state = CaseState()
        before = state.to_dict()
        classifier = Mock()
        classifier.predict.side_effect = RuntimeError("Unavailable")
        with self.assertRaises(RuntimeError):
            state.init_from_user_message("fakt", classifier)
        self.assertEqual(state.to_dict(), before)

    def test_existing_category_not_silently_replaced(self):
        classifier = Mock()
        state = CaseState(dispute_category=DC.UNPAID_WAGES)
        with self.assertRaises(ValueError):
            state.init_from_user_message("new", classifier)
        classifier.predict.assert_not_called()

    def test_load_errors_preserve_state(self):
        for error in [FileNotFoundError("missing"), ValueError("corrupt")]:
            state = CaseState()
            before = state.to_dict()
            with patch.object(CategoryClassifier, "load", side_effect=error):
                with self.assertRaises(type(error)):
                    state.init_from_user_message("Maaş verilmir")
            self.assertEqual(state.to_dict(), before)

    def test_custom_model_path(self):
        path = Path("models/custom.pkl")
        with patch.object(CategoryClassifier, "load") as load:
            load.return_value.predict.return_value = None
            CaseState().init_from_user_message("Salam", model_path=path)
            load.assert_called_once_with(path)

    def test_invalid_confirmation_preserves_state(self):
        state = CaseState()
        state.confirm_category(DC.UNPAID_WAGES)
        state.update_slot("unpaid_period", "yanvar")
        before = state.to_dict()
        with self.assertRaises(ValueError):
            state.confirm_category("UNKNOWN")
        self.assertEqual(state.to_dict(), before)

    def test_nonserializable_nested_values_rejected(self):
        state = CaseState()
        for value in [{"item": object()}, [object()]]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                state.update_slot("evidence_available", value)
        self.assertEqual(state.filled_slots, {})
