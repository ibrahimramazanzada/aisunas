# az-legal-rag — Azerbaijani Labour Code AI Legal Assistant

An intelligent, multi-turn RAG-based legal assistant grounded strictly in the **Azerbaijani Labour Code (*Azərbaycan Respublikasının Əmək Məcəlləsi*)**.

Unlike single-shot legal search engines (such as e-qanun.ai), **az-legal-rag** is built as a multi-turn "case builder": it converses with users, asks clarifying questions to extract key facts regarding their labour dispute, tracks state across dialogue turns, and generates grounded legal complaint drafts (*ərizə*) citing exact statutory articles.

---

## 🚀 Key Features & Architecture

- **PDF Extraction & Layout Parsing**: Built using `PyMuPDF` with full preservation of Azerbaijani Latin diacritics (`ə, ğ, ı, ö, ş, ü, ç`). Extracts 325 legal articles (Maddə 1 to Maddə 317 + amendments) while excluding endnote amendment histories.
- **Context-Preserving Legal Chunking**: Chunks articles along clause boundaries while retaining article numbers, headings, and chapter metadata in every chunk.
- **Hybrid Retrieval (Dense + Lexical BM25)**:
  - **Dense Embeddings**: `BAAI/bge-m3` (1024-dimensional multilingual embeddings) stored in local ChromaDB.
  - **Lexical Matching**: `BM25Okapi` with customized Azerbaijani legal tokenization and IDF flooring.
  - **Fusion**: Reciprocal Rank Fusion (RRF, $k=60$) producing calibrated confidence scores in $[0.0, 1.0]$.
- **Anti-Hallucination Guardrails**: Strict article grounding and a standardized fallback message in Azerbaijani if no article meets the relevance threshold.
- **Trainable ML Component**: Dispute-category classification using Logistic Regression on top of `bge-m3` embeddings.

---

## 📂 Project Structure

```text
az-legal-rag/
├── data/
│   ├── raw/                   # Source Labour Code PDF (emek_mecelle.pdf)
│   ├── processed/             # Cleaned articles.json (325 articles) & bm25_index.pkl
│   └── training_data/         # Labeled examples for category classifier
├── src/
│   ├── config.py              # Centralized settings, thresholds & disclaimers
│   ├── ingestion/             # PDF text extraction, diacritic cleaning & parsing
│   ├── indexing/              # Chunking, bge-m3 embeddings & ChromaDB/BM25 indexing
│   ├── retrieval/             # Hybrid RRF retriever (dense + lexical BM25)
│   ├── dialogue/              # CaseState, session store & category classifier (Phase 6-8)
│   ├── generation/            # One-shot QA & draft complaint prompts (Phase 5, 9)
│   ├── guardrails/            # Grounding & citation verification (Phase 4)
│   └── utils/                 # Standardized logging
├── app/                       # Streamlit UI (Phase 10)
├── docs/                      # Phase completion walkthroughs & reports
├── tests/                     # Test suites for ingestion, indexing & retrieval
├── demo.py                    # Interactive CLI demo & automated benchmark
├── requirements.txt
└── README.md
```

---

## 🛠️ Quickstart

### 1. Clone and install dependencies

Use **Python 3.11**, the version used for regression verification. Run all
commands from the repository root. On Windows, use PowerShell; if Windows
Application Control blocks PyTorch DLLs, use Ubuntu under WSL2 with Python 3.11.

```bash
git clone https://github.com/ibrahimramazanzada/az-legal-rag.git
cd az-legal-rag
python --version
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS/WSL:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Dependencies have minimum versions rather than a lockfile, so newly resolved
versions may differ from the verified environment. Model downloads require
network access and substantial disk space; CPU inference can be slow.

### 2. Configure credentials securely

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell,
or `cp .env.example .env` on Linux/macOS). Replace the placeholder
`GEMINI_API_KEY` with your own key. `.env` is ignored by Git; never commit or
share it. Environment variables can also supply settings.

The actual setting names are `LLM_MODEL_NAME` (default `gemini-2.5-flash`),
`EMBEDDING_MODEL_NAME` (default `BAAI/bge-m3`), `CHROMA_PERSIST_DIR`,
`DATA_RAW_DIR`, and `DATA_PROCESSED_DIR`. The application loads `.env` from
the current working directory. Keep the embedding model compatible with
both indexes and the trained classifier; changing it requires coordinated
artifact preparation.

### 3. Prepare search indexes in a fresh clone

The repository includes the source PDF, parsed corpus, BM25 pickle, and
trained classifier under `src/dialogue/models/`. **ChromaDB is ignored by
Git and is absent from a fresh clone.** The completed Phase 10 corpus has
325 articles, producing 476 chunks with the default chunker; the matching
BM25 and Chroma indexes each contain 476 entries.

For a **new clone with no Chroma index**, run:

```bash
python -m src.indexing.build_index
```

This downloads/loads `BAAI/bge-m3`, creates `data/chroma_db/`, and writes
the BM25 index from `data/processed/articles.json`. If the JSON is missing,
the builder runs ingestion from the source PDF. Index preparation requires
network access for uncached models and enough memory for BGE-M3.

**Do not run this command merely to launch an existing installation:** by
default it deletes and recreates the Chroma collection and overwrites BM25.
Alternatively, obtain a trusted, matching Chroma directory from the team
(copied while its server is stopped), together with the matching corpus,
BM25 index, and classifier. Pickle artifacts must come from a trusted source.
`--no-rebuild` avoids deleting the collection but still embeds/inserts chunks
and rewrites BM25; it is not a read-only verification option.

### 4. Launch Streamlit

```bash
python -m streamlit run app/streamlit_app.py
```

Open the localhost URL printed by Streamlit. The Azerbaijani UI supports
quick questions, confirmed case classification, fact collection, source
excerpts, and validated complaint drafts. The welcome page can start without
models or credentials; real operations require the corresponding assets
and a working Gemini key. See [Phase 10 behavior and limitations](docs/PHASE10.md).

### 5. Run verification

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The completed Phase 10 regression baseline is **175 tests**. Provider and
model calls are mocked in regression tests; this does not verify live Gemini
or dense inference. Additional standalone ingestion, indexing, and retrieval
checks are available:

```bash
python -m tests.test_ingestion
python -m tests.test_indexing
python -m tests.test_retrieval
```

The indexing check uses temporary storage and a separate small model;
retrieval checks require prepared indexes and BGE-M3. For live verification,
try a Quick Question, inspect the cited article excerpts, then complete and
confirm a synthetic Case Builder case and review the generated complaint.

### Troubleshooting

- **No Streamlit application after cloning main:** ensure the reviewed
  Phases 7–10 integration has reached `main`; a PR merged into another
  feature branch does not automatically propagate into `main`.
- **Missing BM25/Chroma index or empty retrieval:** prepare matching indexes
  in the new clone as above. A successfully loaded welcome page is not a
  retrieval check.
- **Gemini errors:** check that `.env` is in the repository root, the key is
  valid, and network access/quota are available. Do not paste credentials
  into logs or issue reports.
- **Wrong setting has no effect:** use the exact names above; unknown `.env`
  entries are ignored by configuration.
- **PyTorch DLL blocked (`WinError 4551`):** use the WSL2 environment or ask
  your administrator for the approved runtime; do not disable security policy.
- **Import errors:** activate the Python 3.11 environment, run commands from
  the repository root, reinstall requirements if needed, and run `pip check`.
- **Classifier compatibility warning:** use the team's compatible Python and
  scikit-learn environment; do not retrain the classifier just to start the UI.

The CLI benchmark remains available as `python demo.py`, with interactive
mode via `python demo.py --interactive`.

---

## ⚖️ Legal Disclaimer

> **XƏBƏRDARLIQ**: Bu sistem yalnız məlumatlandırma xarakteri daşıyır və peşəkar hüquqi məsləhət hesab edilmir.
> *(DISCLAIMER: This system is for informational purposes only and does not constitute professional legal counsel.)*
