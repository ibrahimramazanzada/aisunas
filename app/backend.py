"""Per-browser application state over the existing Phases 5 and 7–9 APIs."""

from typing import Callable, Optional
from copy import deepcopy

from src.config import settings
from src.dialogue.session_store import SessionStore
from src.dialogue.slot_filling import DialogueFlow
from src.generation.case_prompts import draft_session
from src.guardrails import add_legal_disclaimer


class AppSession:
    """Owns one browser's flow, histories and optional complaint result.

    No user state is globally cached. Dependencies are injectable for offline
    UI tests; production always uses the existing real backend interfaces.
    """

    def __init__(self, flow: Optional[DialogueFlow] = None,
                 qa_fn: Optional[Callable] = None,
                 draft_fn: Optional[Callable] = None) -> None:
        """Creates a private flow and session without loading any model."""
        self.flow = flow if flow is not None else DialogueFlow()
        self.qa_fn = qa_fn
        self.draft_fn = draft_fn if draft_fn is not None else draft_session
        self.session_id = self.flow.store.create_session()
        self.quick_history = []
        self.quick_sources = []
        self.complaint = None
        self.notice = None

    @property
    def case_state(self):
        """Reads a fresh detached CaseState from the flow's SessionStore."""
        return self.flow.store.get_session(self.session_id)

    def reset(self) -> None:
        """
        Drops this browser's old store and outputs; keeps loaded services.
        """
        self.reset_case()
        self.reset_quick()

    def reset_case(self) -> None:
        """Starts a new case without erasing this browser's quick questions."""
        self.flow.store = SessionStore()
        self.session_id = self.flow.store.create_session()
        self.complaint = None
        self.notice = None

    def reset_quick(self) -> None:
        """Starts quick chat without erasing confirmed case information."""
        self.quick_history = []
        self.quick_sources = []
        self.notice = None

    def submit_case(self, message: str, category=None):
        """Runs actual dialogue, keeping facts intact on unexpected failure."""
        snapshot = self.case_state
        try:
            result = self.flow.handle_message(message, self.session_id,
                                              category=category)
        except Exception:
            self.flow.store.save_session(self.session_id, snapshot)
            self.notice = ("error", "Mesaj emal edilmədi. Məlumatlarınız "
                           "saxlanılıb; yenidən cəhd edin.")
            return None
        current = result.state
        if (snapshot.filled_slots != current.filled_slots
                or snapshot.dispute_category != current.dispute_category
                or snapshot.category_confirmed != current.category_confirmed
                or snapshot.retrieved_articles != current.retrieved_articles):
            self.complaint = None
        if result.status in ("retrieval_empty", "retrieval_error",
                             "invalid_input"):
            self.notice = ("warning", result.message)
        else:
            self.notice = None
        return result

    def submit_quick(self, message: str) -> None:
        """
        Runs grounded QA; failures show fixed text, never exception details.
        """
        if not isinstance(message, str) or not message.strip():
            self.notice = ("warning", "Zəhmət olmasa sualınızı yazın.")
            return
        message = message.strip()
        self.quick_history.append({"role": "user", "content": message})
        self.quick_sources = []
        try:
            qa = self.qa_fn
            if qa is None:
                from src.generation.qa_prompts import answer_question, call_llm

                def private_llm(system, user):
                    """
                    Keeps provider error details out of QA's existing logs.
                    """
                    try:
                        return call_llm(system, user)
                    except Exception:
                        raise RuntimeError(
                            "Generation service failed") from None

                result = answer_question(message, llm_fn=private_llm)
            else:
                result = qa(message)
            answer = add_legal_disclaimer(result.answer)
            if result.grounded:
                self.quick_sources = [chunk.to_dict()
                                      for chunk in result.sources]
            self.notice = None if result.grounded else (
                "warning", settings.NO_ARTICLE_FALLBACK_AZ)
        except Exception:
            answer = add_legal_disclaimer(settings.NO_ARTICLE_FALLBACK_AZ)
            self.notice = ("error", "Sual cavablandırılmadı. "
                           "Yenidən cəhd edin.")
        self.quick_history.append({"role": "assistant", "content": answer,
                                   "sources": deepcopy(self.quick_sources)})

    def generate(self):
        """Calls Phase 9 only when facts are ready; reuses unchanged drafts."""
        if not self.case_state.is_ready_for_draft():
            self.notice = ("warning", "Kateqoriyanı təsdiq edin və "
                           "tələb olunan məlumatları tamamlayın.")
            return None
        if self.complaint is not None and self.complaint.status == "drafted":
            return self.complaint
        try:
            self.complaint = self.draft_fn(self.session_id, self.flow.store)
        except Exception:
            self.notice = ("error", "Ərizə hazırlanmadı. Məlumatlarınız "
                           "saxlanılıb; yenidən cəhd edin.")
            return None
        if self.complaint.status != "drafted":
            self.notice = ("warning", self.complaint.message)
        else:
            self.notice = None
        return self.complaint
