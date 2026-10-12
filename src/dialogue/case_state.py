"""Phase 7 state; question selection and extraction belong to Phase 8.

Classifier labels are proposals, not evidence that text is in scope.
Confirmation must come from the user before draft readiness is allowed.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.dialogue.slot_schema import (
    DisputeCategory, SLOT_DEFS, required_slots, slots_for,
)


PHASE7_MODEL_PATH = (
    Path(__file__).parent / "models" / "category_classifier_candidate.pkl"
)


def _copy_json(value: Any) -> Any:
    """Copies JSON data, rejecting non-finite or non-serializable values."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _validate_value(value: Any) -> None:
    """Rejects absent facts without interpreting dates or legal meaning."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("Slot value must not be empty.")
    if isinstance(value, (list, dict)) and not value:
        raise ValueError("Slot value must not be empty.")
    _copy_json(value)


@dataclass
class CaseState:
    """Serializable facts for one session, without extraction or generation.

    History and articles contain JSON dictionaries. No retrieval or legal
    validation is performed here. Dates remain exactly as supplied.
    """

    dispute_category: Optional[DisputeCategory] = None
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_articles: List[Dict[str, Any]] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid4()))
    category_confirmed: bool = False

    def __post_init__(self) -> None:
        """Validates imported state and detaches caller-owned containers."""
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if type(self.category_confirmed) is not bool:
            raise ValueError("category_confirmed must be boolean.")
        if self.dispute_category is not None:
            self.dispute_category = DisputeCategory(self.dispute_category)
        elif self.category_confirmed:
            raise ValueError("Cannot confirm an unknown category.")
        if not isinstance(self.filled_slots, dict):
            raise ValueError("filled_slots must be a dictionary.")
        for name, value in self.filled_slots.items():
            self._validate_slot(name, value)
        for records in (self.conversation_history, self.retrieved_articles):
            if not isinstance(records, list) or any(
                not isinstance(record, dict) for record in records
            ):
                raise ValueError("History/articles must be lists of dicts.")
        self.filled_slots = _copy_json(self.filled_slots)
        self.conversation_history = _copy_json(self.conversation_history)
        self.retrieved_articles = _copy_json(self.retrieved_articles)

    def _validate_slot(self, name: str, value: Any) -> None:
        """Checks membership and presence, without inventing slot values."""
        if name not in SLOT_DEFS:
            raise KeyError(name)
        if self.dispute_category is not None:
            if name not in slots_for(self.dispute_category):
                raise ValueError("Slot does not belong to this category.")
        _validate_value(value)

    def update_slot(self, name: str, value: Any) -> None:
        """Stores a validated fact or correction, preserving other facts."""
        self._validate_slot(name, value)
        self.filled_slots[name] = _copy_json(value)

    def confirm_category(self, category: DisputeCategory) -> None:
        """Records explicit selection; keeps only applicable existing facts.

        On category change, obsolete facts and retrieved context are cleared;
        conversation history remains available for explicit corrections.
        """
        category = DisputeCategory(category)
        if category != self.dispute_category:
            self.retrieved_articles = []
        self.filled_slots = {
            name: value for name, value in self.filled_slots.items()
            if name in slots_for(category)
        }
        self.dispute_category = category
        self.category_confirmed = True

    def missing_required_slots(self) -> List[str]:
        """Returns missing required facts in order; unknown gives []."""
        if self.dispute_category is None:
            return []
        missing = []
        for name in required_slots(self.dispute_category):
            try:
                _validate_value(self.filled_slots.get(name))
            except (ValueError, TypeError):
                missing.append(name)
        return missing

    def is_ready_for_draft(self) -> bool:
        """Checks confirmed category and facts, not legal grounding."""
        return bool(
            self.dispute_category is not None and self.category_confirmed
            and not self.missing_required_slots()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Returns a detached JSON-compatible snapshot."""
        return _copy_json({
            "dispute_category": (
                self.dispute_category.value if self.dispute_category else None
            ),
            "filled_slots": self.filled_slots,
            "conversation_history": self.conversation_history,
            "retrieved_articles": self.retrieved_articles,
            "session_id": self.session_id,
            "category_confirmed": self.category_confirmed,
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaseState":
        """Validates and restores a detached snapshot."""
        if not isinstance(data, dict):
            raise ValueError("State must be a dictionary.")
        return cls(**_copy_json(data))

    def init_from_user_message(
        self, message: str, classifier: Any = None,
        model_path: Path = PHASE7_MODEL_PATH,
    ) -> Any:
        """Stores an unconfirmed proposal using the explicit Phase 7 model.

        Empty input has no effect. None predictions remain unknown. Model
        failures propagate before state mutation. Existing categories cannot
        be silently replaced. Call confirm_category after user selection.
        """
        if not isinstance(message, str):
            raise TypeError("message must be a string.")
        if not message.strip():
            return None
        if self.dispute_category is not None:
            raise ValueError("Category already set; use confirm_category.")
        if classifier is None:
            from src.dialogue.category_classifier import CategoryClassifier
            classifier = CategoryClassifier.load(model_path)
        prediction = classifier.predict(message)
        category = (
            DisputeCategory(prediction.category) if prediction else None
        )
        if category is not None:
            self.filled_slots = {
                name: value for name, value in self.filled_slots.items()
                if name in slots_for(category)
            }
        self.dispute_category = category
        self.category_confirmed = False
        self.conversation_history.append({"role": "user", "content": message})
        return prediction
