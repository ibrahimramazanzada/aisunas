# Phase 7: CaseState and session store

Scope follows the complete `codex_prompt_full.md` roadmap: Phase 7 adds
`case_state.py` and `session_store.py`. Phase 8 owns follow-up questions,
free-text extraction and retrieval integration. No API server exists in this
repository; this phase introduces internal state interfaces only.

`CaseState` holds category, filled slots, conversation history, retrieved
article dictionaries and session ID. Required slots retain schema priority;
optional slots do not block completion. False and zero are valid facts.
Empty values and unknown/inapplicable slots are rejected. Dates and other
user-provided text are preserved verbatim, not inferred or normalized.
Corrections update one slot. Serialization and store access use detached
JSON snapshots. `is_ready_for_draft()` checks facts only: Phase 9 must still
check retrieval grounding and apply the legal disclaimer.

Explicit category selection uses `confirm_category()`. Classifier-assisted
initialization explicitly loads `category_classifier_candidate.pkl` (or a
caller-supplied path/classifier); the original classifier default is unchanged.
Loading restores threshold 0.32. Predictions populate an **unconfirmed**
category proposal. None predictions keep the category unknown. Empty input
does not load a model; model failures propagate without altering state.
Previously selected categories cannot be silently replaced by initialization.

The additional `category_confirmed` flag is the conservative fallback required
by the Phase 7 request: a classifier cannot establish that text is a labour
dispute. The candidate can accept weather/vague messages. Callers must ask the
user to explicitly select/confirm a schema category before treating a proposal
as selected or proceeding toward a draft. Phase 8 implements that interaction;
this state layer does not claim off-topic detection. No confidence is invented.
Changing category retains applicable slots, clears retrieved context and
preserves conversation history. No sensitive messages are logged.

`SessionStore` and module-level `create_session`, `get_session`, `save_session`
provide process-local memory persistence. Unknown IDs raise KeyError; saving
requires a matching ID and valid state. Callers must explicitly save updates.
Restart loses sessions; separate workers do not share them. Authentication,
durable persistence and concurrency guarantees are outside this MVP phase.

Tests mock prediction and verify real artifact loading without embeddings.
Run from the repository root with the existing Linux virtual environment
activated. If the environment is stored elsewhere, activate it first and
then navigate to this repository; no copying of project files is needed.

```sh
python -B \
  -m unittest discover -s tests -p 'test_case_state.py' -v
```

Phase 7 branch depends on the Phase 6.5 branch while PR #10 remains open.
No changes to the classifier, schema, retrieval, datasets or models are needed.
