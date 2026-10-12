"""Phase 8 dialogue orchestration and evidence-backed free-text extraction.

No complaint generation or legal advice is produced here. Dependencies are
lazy and injectable; stored facts are verbatim spans of user messages.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.config import settings
from src.dialogue.case_state import CaseState, PHASE7_MODEL_PATH
from src.dialogue.session_store import SessionStore
from src.dialogue.slot_schema import (
    DisputeCategory, SLOT_DEFS, category_label, slots_for,
)
from src.guardrails.disclaimer import add_legal_disclaimer
from src.generation.case_prompts import (
    GATHERING_PROMPT, build_gathering_prompts, gathering_question,
)


EXTRACTION_PROMPT = GATHERING_PROMPT


def _normalize(text: str) -> str:
    """Normalizes Azerbaijani case for interaction control words."""
    return text.replace("İ", "i").replace("I", "ı").lower().strip()


def _uncertain_span(value: str, message: str) -> bool:
    """Rejects uncertainty in a value or its containing message clause."""
    pattern = r"\b(bilmirəm|bəlkə|dəqiq bilmirəm)\b"
    if not value.strip(" .!?;:") or re.search(pattern, _normalize(value)):
        return True
    clauses = re.split(r"[;.!?\n]", message)
    return any(value in clause and re.search(pattern, _normalize(clause))
               for clause in clauses)


def _unique_json_object(pairs: list) -> dict:
    """Rejects duplicate JSON keys rather than silently choosing a value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate extraction JSON key.")
        result[key] = value
    return result


class SlotExtractor:
    """LLM extraction with per-slot confidence and source-span validation."""

    def __init__(self, llm_fn: Optional[Callable] = None,
                 threshold: float = 0.8) -> None:
        """Sets an injectable LLM and conservative acceptance threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1].")
        self.llm_fn = llm_fn
        self.threshold = threshold

    def extract(self, message: str, state: CaseState,
                pending_slot: Optional[str] = None) -> dict:
        """Returns validated slot records; invalid records are omitted.

        Values must be nonempty strings occurring literally in the current
        message. Dates are not normalized and numeric facts are not inferred.
        Malformed output/API failures are handled by the dialogue layer.
        """
        if not state.category_confirmed or not message.strip():
            return {}
        llm = self.llm_fn
        if llm is None:
            from src.generation.qa_prompts import call_llm
            llm = call_llm
        allowed = slots_for(state.dispute_category)
        system, user = build_gathering_prompts(message, state, pending_slot)
        raw = llm(system, user)
        data = json.loads(raw, object_pairs_hook=_unique_json_object)
        if (not isinstance(data, dict)
                or not isinstance(data.get("slots"), dict)):
            return {}
        accepted = {}
        for name, record in data["slots"].items():
            if name not in allowed or not isinstance(record, dict):
                continue
            value = record.get("value")
            confidence = record.get("confidence")
            if (not isinstance(value, str) or not value.strip()
                    or value not in message
                    or _uncertain_span(value, message)
                    or type(confidence) not in (int, float)
                    or not self.threshold <= confidence <= 1
                    or not math.isfinite(confidence)):
                continue
            accepted[name] = {
                "value": value,
                "correction": record.get("correction") is True,
            }
        return accepted


@dataclass
class DialogueResult:
    """One turn's response and detached case snapshot.

    Status is invalid_input, category_selection, category_confirmation,
    gathering, ready, retrieval_empty or retrieval_error. Ready means facts
    plus relevant sources, never permission to generate an unguarded draft.
    """

    session_id: str
    message: str
    status: str
    state: CaseState
    next_slot: Optional[str] = None


class DialogueFlow:
    """Runs confirmed, ordered slot gathering over Phase 7 sessions."""

    def __init__(self, store: Optional[SessionStore] = None,
                 classifier: Any = None, extractor: Any = None,
                 retrieve_fn: Optional[Callable] = None) -> None:
        """Injects dependencies; defaults load only when first needed."""
        self.store = store if store is not None else SessionStore()
        self.classifier = classifier
        self.extractor = (
            extractor if extractor is not None else SlotExtractor())
        self.retrieve_fn = retrieve_fn
        self._load_attempted = classifier is not None

    def _predict(self, message: str) -> Optional[DisputeCategory]:
        """Retries classification on new messages; loads the model once."""
        try:
            if not self._load_attempted:
                self._load_attempted = True
                from src.dialogue.category_classifier import CategoryClassifier
                self.classifier = CategoryClassifier.load(PHASE7_MODEL_PATH)
            if self.classifier is None:
                return None
            prediction = self.classifier.predict(message)
            if prediction is None:
                return None
            confidence = prediction.confidence
            threshold = getattr(self.classifier, "threshold", 0.6)
            if (type(confidence) not in (int, float)
                    or not math.isfinite(confidence)
                    or not threshold <= confidence <= 1):
                return None
            return DisputeCategory(prediction.category)
        except Exception:
            return None

    def _finish(self, state: CaseState, text: str, status: str,
                next_slot: Optional[str] = None) -> DialogueResult:
        """Appends a disclaimer-bearing response and saves a detached state."""
        text = add_legal_disclaimer(text)
        state.conversation_history.append({
            "role": "assistant", "content": text,
            "next_slot": next_slot, "status": status,
        })
        self.store.save_session(state.session_id, state)
        return DialogueResult(state.session_id, text, status,
                              self.store.get_session(state.session_id),
                              next_slot)

    def handle_message(self, message: str, session_id: Optional[str] = None,
                       category: Optional[DisputeCategory] = None
                       ) -> DialogueResult:
        """Handles one turn, optionally carrying explicit user category choice.

        Unknown session IDs raise KeyError rather than silently losing history.
        Category is a user selection, never an automatic classifier output.
        Exact category labels/enum values or yes to a proposal also confirm.
        Existing confirmed categories change only through explicit selection.
        """
        if session_id is None:
            session_id = self.store.create_session()
        state = self.store.get_session(session_id)
        was_confirmed = state.category_confirmed
        if not isinstance(message, str) or not message.strip():
            missing = state.missing_required_slots()
            pending = missing[0] if was_confirmed and missing else None
            return self._finish(state, "Zəhmət olmasa mesajınızı yazın.",
                                "invalid_input", pending)
        message = message.strip()
        normalized = _normalize(message)
        previous = state.conversation_history[-1:] or [{}]
        pending = previous[0].get("next_slot")
        state.conversation_history.append({"role": "user",
                                           "content": message})
        selected = category
        category_control = any(
            normalized in (_normalize(item.value),
                           _normalize(category_label(item)))
            for item in DisputeCategory
        )
        if selected is None:
            for item in DisputeCategory:
                if normalized in (_normalize(item.value),
                                  _normalize(category_label(item))):
                    selected = item
                    break
        if selected is not None:
            try:
                selected = DisputeCategory(selected)
            except (TypeError, ValueError):
                return self._finish(state, "Kateqoriya tanınmadı.",
                                    "category_selection")
            if selected != state.dispute_category:
                pending = None
            if not was_confirmed:
                # Roadmap retry after category selection is advisory: the
                # user's explicit choice wins even if the model abstains.
                self._predict(message)
            state.confirm_category(selected)
        elif not state.category_confirmed:
            if (state.dispute_category is not None
                    and normalized in ("bəli", "hə", "təsdiq edirəm")):
                state.confirm_category(state.dispute_category)
            elif normalized == "xeyr":
                state.dispute_category = None
            else:
                state.dispute_category = self._predict(message)
                if state.dispute_category is not None:
                    state.filled_slots = {
                        name: value
                        for name, value in state.filled_slots.items()
                        if name in slots_for(state.dispute_category)
                    }
        if not state.category_confirmed:
            if state.dispute_category is not None:
                label = category_label(state.dispute_category)
                return self._finish(
                    state, f"Mövzunuz '{label}'"
                    " kimi görünür. Təsdiq edirsiniz? Bəli yazın və ya "
                    "düzgün kateqoriyanı seçin.", "category_confirmation")
            choices = "; ".join(category_label(c) for c in DisputeCategory)
            return self._finish(
                state, "Hansı problemlə üzləşmisiniz? Kateqoriyanı seçin: "
                + choices + ". Aydın deyilsə, vəziyyətinizi ətraflı yazın.",
                "category_selection")
        # Extract the initial facts again after confirmation, so they are not
        # lost while the user responds to a classifier proposal.
        control = (not was_confirmed and normalized in
                   ("bəli", "hə", "təsdiq edirəm", "xeyr"))
        messages = [] if control or category_control else [message]
        if not was_confirmed:
            messages = [record["content"] for record in
                        state.conversation_history
                        if record.get("role") == "user"
                        and _normalize(record["content"]) not in
                        ("bəli", "hə", "təsdiq edirəm", "xeyr")
                        and record["content"] != message]
            if not control and not category_control:
                messages.append(message)
        for source in messages:
            try:
                records = self.extractor.extract(source, state, pending)
            except Exception:
                continue
            if not isinstance(records, dict):
                continue
            for name, record in records.items():
                if (name not in slots_for(state.dispute_category)
                        or not isinstance(record, dict)):
                    continue
                value = record.get("value")
                # Enforce provenance for injected extractors too.
                if (not isinstance(value, str) or not value.strip()
                        or value not in source
                        or _uncertain_span(value, source)):
                    continue
                if name in state.filled_slots:
                    explicit = re.search(
                        r"\b(düzəliş|düzəldirəm|səhv|əslində)\b",
                        _normalize(source),
                    )
                    if not (record.get("correction") is True and explicit):
                        continue
                before = dict(state.filled_slots)
                try:
                    state.update_slot(name, value)
                except (KeyError, TypeError, ValueError):
                    continue
                if before != state.filled_slots:
                    state.retrieved_articles = []
        missing = state.missing_required_slots()
        if missing:
            name = missing[0]
            return self._finish(state, gathering_question(name),
                                "gathering", name)
        if state.retrieved_articles:
            return self._finish(state, "Məlumatlar və mənbələr saxlanılıb.",
                                "ready")
        try:
            retrieve_fn = self.retrieve_fn
            if retrieve_fn is None:
                from src.retrieval import retrieve as retrieve_fn
            from src.guardrails.relevance import is_relevant
            from src.retrieval.case_query import enrich_query
            query = category_label(state.dispute_category)
            chunks = retrieve_fn(query, case_state=state)
            if (not chunks
                    or not is_relevant(enrich_query(query, state), chunks)):
                return self._finish(state, settings.NO_ARTICLE_FALLBACK_AZ,
                                    "retrieval_empty")
            # Preserve source metadata without inventing citations.
            state.retrieved_articles = [chunk.to_dict() for chunk in chunks]
        except Exception:
            state.retrieved_articles = []
            return self._finish(
                state, "Mənbələri axtararkən xəta baş verdi. "
                "Məlumatlarınız saxlanılıb; yenidən cəhd edin.",
                "retrieval_error")
        return self._finish(state, "Lazımi məlumatlar toplandı və "
                            "əlaqəli mənbələr tapıldı.", "ready")
