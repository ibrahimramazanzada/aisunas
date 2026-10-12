"""Lightweight Azerbaijani retrieval expansion from confirmed case facts."""

from src.dialogue.slot_schema import (
    DisputeCategory, category_label, slots_for,
)


def enrich_query(query: str, case_state=None) -> str:
    """Appends category and applicable facts; preserves False and zero.

    Only explicitly confirmed categories contribute case information.
    Objects without a confirmation flag leave the original query unchanged.
    No statutory article numbers or legal conclusions are added.
    """
    query = query.strip()
    if case_state is None:
        return query
    if getattr(case_state, "category_confirmed", False) is not True:
        return query
    try:
        category = DisputeCategory(case_state.dispute_category)
    except (AttributeError, TypeError, ValueError):
        return query
    terms = [query, category_label(category)]
    facts = getattr(case_state, "filled_slots", {})
    if isinstance(facts, dict):
        for name in slots_for(category):
            value = facts.get(name)
            if isinstance(value, (str, bool, int, float)) and value != "":
                terms.append(str(value))
    return " ".join(terms)
