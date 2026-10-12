# Phase 9: Case-building generation

The authoritative original `codex_prompt_full.md` Phase 9 section requires:

- [x] `src/generation/case_prompts.py` with GATHERING prompts used internally
  by slot filling for extraction and question generation.
- [x] DRAFT_COMPLAINT mode after `CaseState.is_ready_for_draft()`.
- [x] Structured Azerbaijani ərizə: header, collected facts, retrieved legal
  basis, and request/demand.
- [x] Reuse Phase 4 relevance, citation validation and disclaimer guardrails;
  insufficient grounding returns the fixed fallback rather than a draft.

Phase 10 UI and complaint export/submission are outside this phase.

## Interfaces and workflow

`build_gathering_prompts(message, state, pending_slot=None)` centralizes the
existing extraction instructions. `gathering_question(slot_name)` returns the
schema's deterministic Azerbaijani question. Phase 8 now uses both helpers;
`EXTRACTION_PROMPT` remains an alias for compatibility. The extraction payload
adds `mode=GATHERING`; the slot/value/confidence contract is unchanged.

`draft_complaint(state, llm_fn=None)` returns `ComplaintResult`:

- `status`: `drafted`, `unconfirmed_category`, `missing_information`,
  `invalid_case`, `insufficient_grounding`, `invalid_output`,
  `unsupported_citations`, or `provider_error`.
- `message`: Azerbaijani response with the centralized disclaimer.
- `draft`: document text only on success; `grounded`: guardrails passed,
  **not** legally verified or guaranteed correct.
- `missing_slots`: required slots in schema order when incomplete.
- `facts`, `legal_basis`, `sources`: detached validated facts, selected
  excerpts, and original source dictionaries. `flagged_articles` identifies
  references blocked by citation validation.

`draft_session(session_id, store, llm_fn=None)` reads the existing SessionStore
without changing state/history. Unknown session IDs retain `KeyError`.
Use the same store as the Phase 8 DialogueFlow. Results can be displayed later
by an explicit Phase 10 action; drafting does not implicitly submit or save a
document into conversation history.

The workflow snapshots and validates the case, requires confirmed category
and all required facts, then checks each stored source independently against
existing Phase 4 relevance thresholds. The relevance query is the category
label, preventing user-supplied article numbers from bypassing score gating.
Malformed/weak sources are excluded; conflicting chunk IDs fail closed.
No repeated retrieval, model load, index rebuild or classifier call occurs.

In DRAFT_COMPLAINT mode the existing Gemini interface selects up to three
complete, substantive source paragraphs as JSON:

```json
{"legal_basis": [{"chunk_id": "retrieved-chunk-id", "quote": "exact source paragraph"}]}
```

The application validates JSON shape, duplicate keys, chunk identities,
verbatim paragraphs and explicit citations, then renders the complaint.
Unknown fields, invented quotations, incomplete JSON, partial source-paragraph
selections, empty legal basis, unsupported references and provider failures
never expose a draft.
The renderer uses the exact stored facts in schema order, Azerbaijani labels,
source article numbers and a category-specific neutral request for investigation
and lawful resolution. Facts are labelled as the applicant's account; legal
basis is explicitly labelled as source excerpts. No unchecked LLM legal
reasoning, invented entitlement, damages calculation or free-form facts are
accepted. Source metadata, including pages/chapter/title, is returned unchanged.

Parties, recipient, filing date and signature are not collected by the current
schema. The header uses explicit fill-in placeholders; it never invents them.
Optional evidence appears only if supplied. User dates/amounts remain verbatim;
typed Phase 7 values are serialized without converting False/zero to absence.
Facts and retrieved text are untrusted prompt data; embedded instructions do
not authorize new output fields. Sensitive cases/provider exceptions are not
logged by the Phase 9 layer.

## Example Azerbaijani document shape

The citation below is a **placeholder**, not a fabricated statutory number.
Actual application output substitutes only validated retrieved source material.

```text
[Müraciət edilən orqan / işəgötürən]
Ərizəçi: [Ad, soyad və əlaqə məlumatları]
Barəsində müraciət edilən işəgötürən: [Adı]
ƏRİZƏ LAYİHƏSİ
Mövzu: Ödənilməmiş əmək haqqı

FAKTLAR (ərizəçinin bildirdiyi məlumatlar)
Hazırkı məşğulluq vəziyyəti: hazırda işləyirəm
Son əmək haqqı ödənişi: keçən ay
Ödəniş edilməmiş dövr: iyul–avqust
Bildirilmiş təxmini məbləğ: 1 234,56 manat

HÜQUQİ ƏSAS (mənbələrdən dəqiq çıxarışlar)
Maddə [mənbədəki nömrə]: «[mənbədən yoxlanılmış tam abzas]»

XAHİŞ
Bildirdiyim ödənilməmiş əmək haqqının ödənilməsi məsələsinin
araşdırılmasını və qanunvericiliyə uyğun həll edilməsini xahiş edirəm.

Tarix: [Ərizənin təqdim edildiyi tarix]
İmza: [İmza]
```

Every actual output also states that the draft is not legally verified and
includes `settings.LEGAL_DISCLAIMER_AZ`.

## Configuration and tests

Default generation reuses Phase 5 `call_llm`, its cached Gemini client and
retry behavior. Set `GEMINI_API_KEY` via the existing environment/`.env`
configuration and `LLM_MODEL_NAME` if needed. No new provider/package is added.
Inject `(system, user) -> str` to test without credentials or network calls.

Run from the repository root with the existing dependencies:

```bash
python -m unittest discover -s tests -p 'test_case_prompts.py'
python -m unittest discover -s tests -p 'test_*.py'
python -m pycodestyle src/generation/case_prompts.py src/dialogue/slot_filling.py tests/test_case_prompts.py
git diff --check
```

Unit tests mock providers and expensive models. A local artifact smoke test
loads the existing persisted BM25 index, queries real stored chunks through
LegalRetriever's existing explicit-article path, and preserves their metadata
through drafting. Dense inference and Gemini are mocked; this is a local
artifact integration test, not a real-provider or semantic-retrieval evaluation.
It skips with an explicit reason only if the index is absent. It does not
rebuild or write any index or dataset. The checked-in `articles.json` currently
contains two sample records; the smoke test uses the persisted BM25 artifact
instead. Corpus completeness/authenticity must be reviewed separately.

Validation in the existing Ubuntu WSL2 Python 3.11.17 environment ran against
the Git-tracked checkout: 143 unit/regression/integration tests passed,
including 23 Phase 9 tests, with zero failures and zero skips. The local
artifact smoke loaded 476 existing BM25 chunks and drafted from an actual
stored article 70 chunk with a mocked provider. Changed-file formatting and
Git whitespace checks passed. No real-provider test was performed; legacy
standalone dataset-writing or large-model integration tests were not run.
The existing unrelated E305 in `validate_parser.py` was left untouched.

## Limitations and Phase 10 responsibilities

Guardrails verify source selection, relevance scores and literal citations;
they cannot prove legal applicability, current law or semantic entailment.
The draft is deliberately extractive: formal structure plus exact facts and
source paragraphs, without generated legal interpretation. Existing citation
regex limitations may conservatively reject amendment/subarticle references.
Complete paragraphs are defined by newlines in the retrieved chunk; upstream
chunk boundaries may already truncate a paragraph from the original document.
Source authenticity depends on the existing retrieval pipeline, not on new
corpus auditing or internet access. This phase does not alter source data.
Phase 7 validation establishes schema membership/presence, not truth of user
facts. Prompt instructions and strict output validation limit injection but
are not a universal semantic detector of malicious source text.

No live Gemini call is included in offline validation; provider output quality,
token limits and latencies remain unevaluated. Missing or truncated provider
responses safely fail. Session storage is still process-local; drafting uses
a snapshot and does not add concurrent-turn locking. The document needs human
review and completion of placeholders before use. Phase 10 owns display,
explicit generation actions, session-state UI and presentation of disclaimers.
