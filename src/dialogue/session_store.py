"""In-memory Phase 7 sessions; restart loses data and workers are isolated.

Read/write snapshots prevent unsaved mutations leaking into stored state.
The interface can be implemented by Redis/SQLite without changing callers.
"""

from typing import Dict

from src.dialogue.case_state import CaseState


class SessionStore:
    """Process-local sessions; no disk writes or personal logging."""

    def __init__(self) -> None:
        """Creates an independent empty store."""
        self._sessions: Dict[str, dict] = {}

    def create_session(self) -> str:
        """Creates an empty case and returns its unique session ID."""
        state = CaseState()
        self._sessions[state.session_id] = state.to_dict()
        return state.session_id

    def get_session(self, session_id: str) -> CaseState:
        """Returns a detached case; raises KeyError for unknown IDs."""
        return CaseState.from_dict(self._sessions[session_id])

    def save_session(self, session_id: str, case_state: CaseState) -> None:
        """Saves a validated snapshot; unknown/mismatched IDs are rejected."""
        if session_id not in self._sessions:
            raise KeyError(session_id)
        if not isinstance(case_state, CaseState):
            raise TypeError("case_state must be CaseState.")
        if session_id != case_state.session_id:
            raise ValueError("Session ID mismatch.")
        snapshot = CaseState.from_dict(case_state.to_dict()).to_dict()
        self._sessions[session_id] = snapshot


_store = SessionStore()


def create_session() -> str:
    """Creates a session in the default process-local store."""
    return _store.create_session()


def get_session(session_id: str) -> CaseState:
    """Gets a detached state from the default store."""
    return _store.get_session(session_id)


def save_session(session_id: str, case_state: CaseState) -> None:
    """Saves a state in the default store."""
    _store.save_session(session_id, case_state)
