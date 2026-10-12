"""Offline tests of Phase 7 persistence and session isolation."""

import unittest

from src.dialogue.case_state import CaseState
from src.dialogue import session_store as sessions
from src.dialogue.slot_schema import DisputeCategory as DC


class TestSessionStore(unittest.TestCase):
    """Checks explicit saving, independent sessions and error handling."""

    def setUp(self):
        """Creates an isolated store for each scenario."""
        self.store = sessions.SessionStore()

    def test_new_sessions_are_unique_and_empty(self):
        a = self.store.create_session()
        b = self.store.create_session()
        self.assertNotEqual(a, b)
        self.assertEqual(self.store.get_session(a).session_id, a)
        self.assertFalse(self.store.get_session(a).is_ready_for_draft())

    def test_multiturn_save_and_isolation(self):
        a = self.store.create_session()
        b = self.store.create_session()
        state = self.store.get_session(a)
        state.confirm_category(DC.UNPAID_WAGES)
        state.update_slot("unpaid_period", "yanvar")
        self.store.save_session(a, state)
        state.update_slot("unpaid_period", "unsaved")
        restored = self.store.get_session(a)
        self.assertEqual(restored.filled_slots["unpaid_period"], "yanvar")
        restored.update_slot("last_paid_date", "dekabr")
        self.store.save_session(a, restored)
        self.assertEqual(len(self.store.get_session(a).filled_slots), 2)
        self.assertEqual(self.store.get_session(b).filled_slots, {})

    def test_unknown_ids_and_mismatch(self):
        with self.assertRaises(KeyError):
            self.store.get_session("missing")
        with self.assertRaises(KeyError):
            self.store.save_session("missing", CaseState())
        session_id = self.store.create_session()
        with self.assertRaises(ValueError):
            self.store.save_session(session_id, CaseState())
        with self.assertRaises(TypeError):
            self.store.save_session(session_id, {})

    def test_invalid_mutation_cannot_corrupt_saved_snapshot(self):
        session_id = self.store.create_session()
        state = self.store.get_session(session_id)
        state.filled_slots["invented"] = "value"
        with self.assertRaises(KeyError):
            self.store.save_session(session_id, state)
        self.assertEqual(self.store.get_session(session_id).filled_slots, {})

    def test_module_interface(self):
        session_id = sessions.create_session()
        state = sessions.get_session(session_id)
        state.confirm_category(DC.UNPAID_WAGES)
        sessions.save_session(session_id, state)
        self.assertTrue(sessions.get_session(session_id).category_confirmed)

    def test_nested_history_articles_and_slots_are_isolated(self):
        session_id = self.store.create_session()
        state = self.store.get_session(session_id)
        state.update_slot("evidence_available", ["mesaj"])
        state.conversation_history.append({
            "role": "user", "content": {"text": "Salam"}})
        state.retrieved_articles.append({
            "article_number": "70", "metadata": {"pages": [1]}})
        self.store.save_session(session_id, state)
        restored = self.store.get_session(session_id)
        restored.filled_slots["evidence_available"].append("bank")
        restored.conversation_history[0]["content"]["text"] = "changed"
        restored.retrieved_articles[0]["metadata"]["pages"].append(2)
        state.retrieved_articles[0]["metadata"]["pages"].append(3)
        stored = self.store.get_session(session_id)
        self.assertEqual(stored.filled_slots["evidence_available"], ["mesaj"])
        self.assertEqual(stored.conversation_history[0]["content"]["text"],
                         "Salam")
        self.assertEqual(stored.retrieved_articles[0]["metadata"]["pages"],
                         [1])

    def test_independent_store_instances(self):
        session_id = self.store.create_session()
        with self.assertRaises(KeyError):
            sessions.SessionStore().get_session(session_id)
