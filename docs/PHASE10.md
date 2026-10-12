# Phase 10: Streamlit UI and integration

## Authoritative roadmap checklist

The supplied project roadmap assigns the Streamlit UI to Phase 10. Its five
explicit deliverables are implemented:

- Persist `session_id` and detached `CaseState` in `st.session_state`.
- Offer quick-question mode (Phase 5) and case-builder mode (Phases 8–9).
- Show conversation history and a running case summary.
- Show the complaint-generation button only when `is_ready_for_draft()` is true.
- Keep the legal disclaimer visible in both modes.

Phase 11's separate evaluation/review work remains outside this implementation.

## Architecture and safety

Entry point: `app/streamlit_app.py`. The existing Streamlit dependency is reused;
there is no separate API server or frontend build. `app/backend.py` owns one
`AppSession` per Streamlit browser session. It delegates to `DialogueFlow`,
`SessionStore`, `answer_question`, and `draft_session`; existing public backend
interfaces are unchanged. Models load lazily when their services are used.

Classifier suggestions require user confirmation. The category form also allows
explicit manual selection when classification is unavailable. Required facts
and readiness come from Phase 8; grounded document validation comes from Phase 9.
Readiness alone does not guarantee sufficient sources: empty retrieval still
produces a grounding warning rather than a complaint. Corrections and category
changes invalidate any previous draft. Repeated generation of an unchanged
successful draft does not call the provider again.

Case and quick-question histories persist independently while switching modes.
New conversation replaces the private store, clears both histories, sources,
draft and category selection, and creates a new ID. No user state is globally
cached. State is held in memory, not durable storage; browser reload/disconnection
or server restart may lose it. This is not a production authentication system.

Unexpected dialogue errors restore the prior case snapshot. Provider failures
show fixed safe messages. The adapter does not log user messages or credentials;
quick-QA provider exception details are sanitized before existing QA logging.
External providers receive the prompts required by the existing pipelines.
Source titles, excerpts, case facts, chat messages and drafts use plain-text
rendering. Article, chapter, section and page metadata are preserved from returned
sources. Search results are evidence to inspect, not a claim that every returned
article is applicable; the draft uses Phase 9's validated selected basis.

## Setup and launch

From the repository root, activate an existing Python 3.11 environment with the
repository dependencies. Windows users can run this in Ubuntu WSL2 if compiled
extensions are unavailable in their Windows Python environment.

```bash
source .venv-linux/bin/activate  # or your existing environment
python -m pip install -r requirements.txt  # only if dependencies are missing
export GEMINI_API_KEY='your-key'  # alternatively use an untracked local .env
python -m streamlit run app/streamlit_app.py
```

Open the localhost URL printed by Streamlit. Do not commit `.env` or credentials.
The existing configuration defaults to `gemini-2.5-flash` and `BAAI/bge-m3`.
`LLM_MODEL_NAME` and `EMBEDDING_MODEL_NAME` are optional environment overrides;
changing embedding models requires compatible existing artifacts.

Real answers require the existing BM25 index, compatible Chroma index and
embedding model. Automatic classification also needs the trained classifier
artifacts under `src/dialogue/models`. Missing assets or credentials do not
prevent the welcome page from opening; they prevent the corresponding live
operation. Do not rebuild indexes or retrain models merely to launch the UI.

## Example demonstration

1. Start with **Sürətli sual**, enter an employment question, and inspect the
   response and expandable source excerpts.
2. Switch to **İşin hazırlanması**. Enter `Maaşımı vermirlər`.
3. Confirm the suggested category with `bəli`, or choose the correct category
   in the form and press **Kateqoriyanı təsdiq et**.
4. Answer the follow-up questions using your actual facts. For a synthetic
   demonstration, an unpaid-wages answer can be
   `işləyirəm; son maaşımı iyunda almışam; iyul maaşı ödənilməyib; 1000 manat`.
   Extraction may ask again if an answer is ambiguous; inspect the case summary.
5. Correct a collected value using `Düzəliş: məbləğ 1200 manatdır` and verify the
   summary. Category changes use the explicit form.
6. Once required facts are complete, press **Ərizə layihəsini hazırla**. Review
   the confirmed facts and actual source excerpts. Insufficient grounding or
   provider errors must display a warning instead of an invented document.
7. Switch modes to verify continuity. **Yeni söhbət** clears the session.

These amounts and periods are demonstration input, never hardcoded production
responses. The UI is Azerbaijani with wide Streamlit layout and native sidebar,
chat, forms, expanders and loading indicators.

## Automated validation

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m pycodestyle app/__init__.py app/backend.py app/streamlit_app.py tests/test_streamlit_app.py
git diff --check
```

`tests/test_streamlit_app.py` exercises actual Streamlit AppTest widgets and
real dialogue, store, extraction validation and complaint rendering. Classifier,
extractor provider, retrieval, Gemini and quick-QA responses are mocked in these
tests. Synthetic article fixtures are confined to tests. Tests cover initialization,
confirmation, multi-slot collection, continuity, correction, category change,
draft invalidation, source metadata, empty retrieval, classifier failure, provider
timeouts, session isolation, reset, mode switching and safe text rendering.

Final regression run: **159 tests passed**, including **16 Phase 10 tests**.
Changed-file `pycodestyle` and Git whitespace checks passed. Streamlit's expected
bare-mode `ScriptRunContext` warnings did not cause failures. The unrelated
existing `validate_parser.py` formatting issue was not changed.

## Verification limits and remaining work

The local server started successfully and its health endpoint returned `ok`.
No connected browser automation surface was available: screenshot inspection,
browser end-to-end interaction and common-screen-size visual checks were not
performed. AppTest checks widget behavior and Unicode strings, not browser pixels.

No Gemini credential was configured, and the checkout had no Chroma directory.
Live Gemini and dense retrieval were therefore not tested. A cached BGE directory
and BM25 artifact existed, but no large model was downloaded, no classifier was
retrained, and no index was rebuilt. Existing regression tests include a real
read-only BM25 smoke with mocked dense inference and Gemini. This is not a claim
of fully verified live end-to-end generation.

Before a live demonstration, provide the existing compatible artifacts and
credentials, then verify real classification, extraction, hybrid retrieval and
generation with minimal synthetic input. Also complete browser desktop/mobile
visual checks and the separate Phase 11 evaluation. Calls are synchronous and
inherit existing backend timeout/retry behavior; there is no cancellation UI or
new timeout layer. Complaint templates retain Phase 9 limitations: missing party
details remain blank, facts are preserved, and the output requires human review.

Phase 10 branches from Phase 9 commit
`49e74a4edb3b5d63dd2cd85952deb9d3b9198f93`; PRs #10–#13 remained open and
unmerged at final review. The Phase 10 PR therefore targets
`feature/phase-9-complaint-generation` and depends on the earlier stacked PRs.
