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

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/ibrahimramazanzada/az-legal-rag.git
cd az-legal-rag

python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
```

### 2. Launch the Phase 10 Application

Configure `GEMINI_API_KEY` in your environment or local `.env` file and use
the existing classifier, BM25 and Chroma artifacts. See
[Phase 10 setup, demonstration and limitations](docs/PHASE10.md).

```bash
python -m streamlit run app/streamlit_app.py
```

The Azerbaijani UI supports quick questions and confirmed case building,
with source excerpts and grounded complaint drafts.

### Interactive Mentor Demo / Benchmark
```bash
# Automated benchmark across 5 core labour topics (dismissal, wages, leave, probation, etc.)
python demo.py

# Or interactive mode to ask any custom question live
python demo.py --interactive
```

### 3. Rebuild Search Indices (Optional)
```bash
python -m src.indexing.build_index
```

### 4. Run Test Suites
```bash
python -m tests.test_ingestion
python -m tests.test_indexing
python -m tests.test_retrieval
```

---

## ⚖️ Legal Disclaimer

> **XƏBƏRDARLIQ**: Bu sistem yalnız məlumatlandırma xarakteri daşıyır və peşəkar hüquqi məsləhət hesab edilmir.
> *(DISCLAIMER: This system is for informational purposes only and does not constitute professional legal counsel.)*
