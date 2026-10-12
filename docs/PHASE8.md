# Phase 8: Slot filling and case-aware retrieval

Scope follows Phase 8 of the original `codex_prompt_full.md` roadmap:

- [x] Unknown category: offer four schema labels, retry the classifier on
  subsequent descriptions and selection, and ask for clarification if unknown.
- [x] Require explicit user confirmation of classifier proposals.
- [x] Ask the next missing required slot in schema priority order.
- [x] Extract free-text answers with an LLM; re-ask on uncertain extraction.
- [x] Enrich both retrieval entrypoints with category and applicable facts.

Phase 9 complaint generation, its `case_prompts.py`, and Phase 10 UI are outside
this implementation. No existing API server or Streamlit application is present.

## Components and public interfaces

`src/dialogue/slot_filling.py` provides:

- `SlotExtractor(llm_fn=None, threshold=0.8)`: `extract(message, state,
  pending_slot=None)` returns validated slot records. The injectable LLM has
  the existing `(system, user) -> str` interface; the default reuses the Phase 5
  cached Gemini client, retry handling and configuration. No calls occur merely
  by constructing an extractor.
- `DialogueFlow(store=None, classifier=None, extractor=None, retrieve_fn=None)`:
  `handle_message(message, session_id=None, category=None)` returns a
  `DialogueResult` containing response, status, detached state and next slot.
  Omit session ID to create a new session; unknown supplied IDs raise `KeyError`.
  Pass `category` only for an explicit user selection, never a model proposal.
- `DialogueResult.status`: `invalid_input`, `category_selection`,
  `category_confirmation`, `gathering`, `ready`, `retrieval_empty`, or
  `retrieval_error`. Every response includes the centralized disclaimer.

Injected extractors implement `extract(message, state, pending_slot)` and
return `{slot_name: {"value": "verbatim span", "correction": false}}`.
The default extractor additionally validates each record's confidence before
returning it. Injected classifiers implement `predict(message)` and return a
Phase 6.5 `CategoryPrediction` or `None`. Model loading is lazy and attempted
once per flow; reuse the flow across turns/sessions. Load failure allows manual
selection without repeated loading; construct a new flow to retry loading.

## Lifecycle and safety

1. Save the user message, predict a proposal, then ask for explicit confirmation.
   A low-confidence/invalid prediction or classifier error offers all four
   categories. Exact Azerbaijani labels or enum values select a category; `bəli`,
   `hə`, and `təsdiq edirəm` confirm an existing proposal. `xeyr` rejects it.
   Classifier retry on selection is advisory: explicit selection always wins.
2. Once confirmed, extract earlier initial descriptions as well as the current
   answer where applicable. Category-control responses are not treated as facts.
   Ask schema questions without generating new questions or legal assertions.
3. Accept only applicable, nonempty, verbatim string spans from the current
   extraction input with finite confidence at least 0.8. Unknown/ambiguous
   values/clauses such as `bilmirəm` or `bəlkə` are omitted. Duplicate JSON keys
   are rejected. Malformed individual records do not block valid slots.
   Date expressions remain verbatim;
   neither dates nor amounts are calculated or invented. Multiple slots may
   be collected in one answer. Failed/malformed extraction re-asks the missing
   question and preserves already validated facts.
4. Existing facts change only when extraction flags a correction and the
   source contains an explicit correction marker (`düzəliş`, `düzəldirəm`,
   `səhv`, or `əslində`). A category change requires an explicit selection;
   Phase 7 retains applicable facts and history and removes obsolete context.
5. Gather every required slot before retrieval. The roadmap specifies query
   enrichment but no retrieval timing; this implementation uses Phase 7 fact
   readiness as a conservative trigger. Corrections invalidate stored sources.
   Unchanged cases reuse sources; empty/failed retrieval can be retried on a
   subsequent turn. Relevant returned chunks are serialized with all metadata.
   Weak/empty retrieval uses the fixed fallback. Failures preserve user facts.

`src/retrieval/case_query.py` supplies deterministic Azerbaijani query expansion
used by `LegalRetriever.retrieve(..., case_state=None)` and the existing
module-level `retrieve(..., case_state=None)`. Signatures remain compatible.
Unconfirmed Phase 7 categories do not enrich queries. Applicable string,
numeric and boolean facts are included in schema order, including zero/False.
No article numbers or legal conclusions are added by the expansion itself.
Objects lacking a boolean `category_confirmed=True` leave the query unchanged.
Existing single-shot callers and positional argument shapes are preserved.

`ready` means complete facts and relevant retrieved sources, not a generated
complaint. Phase 7 `is_ready_for_draft()` still checks facts only. A later Phase 9
caller must apply grounding/citation checks before producing a complaint.

## Example Azerbaijani conversations

Examples assume mocked extraction of the shown literal facts. Each assistant
response also carries the legal disclaimer.

```text
User: Maaşımı vermirlər, hazırda işləyirəm.
Assistant: Mövzunuz 'Ödənilməmiş əmək haqqı' kimi görünür.
           Təsdiq edirsiniz? Bəli yazın və ya düzgün kateqoriyanı seçin.
User: Bəli
Assistant: Əmək haqqını sonuncu dəfə nə vaxt almısınız?
User: Son maaş iyun ayında verilib; iyul üçün 1000 manat borcludurlar.
Assistant: Lazımi məlumatlar toplandı və əlaqəli mənbələr tapıldı.
User: Düzəliş: 1200 manat borcludurlar.
Assistant: Lazımi məlumatlar toplandı və əlaqəli mənbələr tapıldı.
```

The last two responses occur only when retrieval passes existing relevance
guardrails. With empty results, the fixed Azerbaijani article-not-found fallback
is returned instead. Correction triggers fresh retrieval.

```text
User: Problemim var.
Assistant: Hansı problemlə üzləşmisiniz? Kateqoriyanı seçin: ...
User: Ödənilməmiş əlavə iş haqqı
Assistant: Əmək müqaviləsinə görə iş saatlarınız necədir?
User: Bilmirəm.
Assistant: Əmək müqaviləsinə görə iş saatlarınız necədir?
```

## Validation commands

From the repository root in an environment with project dependencies:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s tests -p 'test_slot_filling.py'
python -m pycodestyle src/dialogue/slot_filling.py src/retrieval/case_query.py src/retrieval/retriever.py tests/test_slot_filling.py
git diff --check
```

Tests mock Gemini, classifier inference and retrieval backends; no production
index rebuilds, dataset writes, model retraining or external API calls occur.
Unittest discovery runs unittest classes; legacy standalone ingestion/indexing/
retrieval test functions are not executed by discovery. Phase 8 tests directly
exercise both retrieval entrypoints and both search backends using mocks.

Verified with the existing Linux Python 3.11.17 environment, running against
the Git-tracked checkout mounted in WSL: 120 discovered unit/regression tests
passed, including 32 Phase 8 tests; zero failures and zero unittest skips.
Formatting passed for all changed Python files, and `git diff --check` passed.
Repository-wide formatting reports one pre-existing E305 at
`src/ingestion/validate_parser.py:26`, left unchanged. Legacy standalone
ingestion/indexing/live retrieval functions were not run: ingestion writes
processed datasets and those model-dependent tests are outside this offline
Phase 8 verification. Production data, indexes and model artifacts were not
modified.

## Limitations

Extraction confidence is an LLM estimate, not a calibrated probability.
Verbatim provenance prevents invented strings but cannot prove semantic slot
assignment; ambiguous phrasing can still need user review. Dates and amounts
remain user text, without calendar/currency normalization. Correction markers
are deliberately limited; unsupported correction phrasing may need rewording.
No off-topic detector is claimed; manual category confirmation is essential.
Sessions remain process-local and non-durable, with no concurrent-turn locking
or multi-worker sharing. No live model/API evaluation is performed by the
offline tests. No complaint generation or application integration is included.
