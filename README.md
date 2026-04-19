# Lexi Legal Research Agent

AI-powered precedent research agent for Indian court judgments. Built for the Lexi Backend Engineer Assessment.

## Architecture

```
User Query
    │
    ▼
[LangGraph Router]  ← LLM classifies at runtime (temperature=0.0, no hard-coded branching)
    │
    ├──▶ direct_answer      — general knowledge, no retrieval needed
    ├──▶ document_search    — single-pass hybrid RAG for factual document queries
    └──▶ precedent_research — multi-step: query expansion → hybrid retrieval (semantic + BM25)
                              → LLM classification (supporting / adverse / neutral)
                              → strategy memo generation
```

**Stack:** LangGraph · ChromaDB · sentence-transformers (`all-MiniLM-L6-v2`) · Llama 3.3 70B (Groq) via LiteLLM (fallback: Gemini 2.0 Flash) · Streamlit · BM25 (`rank-bm25`)

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd lexi-agent
pip install -r requirements.txt
```

### 2. API Keys

Create a `.env` file in the project root with both keys:

```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

- Groq (primary — Llama 3.3 70B): free key at https://console.groq.com/keys
- Google AI Studio (fallback — Gemini 2.0 Flash): free key at https://aistudio.google.com/apikey

> Both keys are read from `.env` at startup and are never exposed in the UI or logs.
> Primary and fallback live on **different providers** on purpose: when Groq's
> per-minute or per-day cap is hit, the cascade switches to Gemini's independent
> quota bucket instead of failing. Either key alone is enough to run the app —
> you'll only lose the automatic cascade if the other is missing.

### 3. Add court judgment PDFs

Download the corpus from the provided Google Drive link and place all 50 PDFs in the `docs/` folder:

```
docs/
  DOC_001.pdf
  DOC_002.pdf
  ...
  DOC_050.pdf
```

> `docs/` is git-ignored (the corpus is large and lives elsewhere). See `docs/README.md` for the corpus download link.

### 4. Ingest documents

```bash
python -m ingestion.ingest
```

Embeds all PDFs into a local ChromaDB vector store (`chroma_db/`). Only needs to run once.
Use `--reset` to wipe and re-ingest from scratch:

```bash
python -m ingestion.ingest --reset
```

> **Note:** After any change to chunking configuration (`CHUNK_SIZE`, `CHUNK_OVERLAP`, separators), always run with `--reset` to rebuild the index with the updated parameters.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

---

## Project Structure

```
lexi-agent/
├── app.py                      # Entry point — wires UI, agent, and ingestion
├── logger.py                   # Centralised logging configuration
├── agent/
│   ├── graph.py                # LangGraph agent (router + all node definitions)
│   ├── retriever.py            # Hybrid retrieval (semantic + BM25 + context expansion)
│   └── llm_client.py           # LiteLLM wrapper (Groq Llama 3.3 70B + Gemini 2.0 Flash fallback)
├── ingestion/
│   └── ingest.py               # PDF → clean → chunk → embed → ChromaDB
├── evaluation/
│   ├── eval.py                 # Evaluation framework (6 dimensions)
│   └── eval_results.json       # Results from last eval run
├── ui/
│   └── components.py           # All Streamlit rendering logic
├── docs/                       # Place PDF judgments here
├── chroma_db/                  # Auto-created on first ingest
├── ADR.md                      # Architecture Decision Record
├── requirements.txt
└── .env                        # GROQ_API_KEY + GEMINI_API_KEY — not committed
```

---

## Running Evaluations

Evaluation runs **offline** against predefined test cases. It does not execute during live queries.

```bash
# Run full 5-case suite across all 6 dimensions
python -m evaluation.eval --full-suite --output evaluation/eval_results.json

# Run a single predefined case
python -m evaluation.eval --case-id case_01_insurer_liability

# Run a custom query
python -m evaluation.eval --query "Find precedents for contributory negligence in truck accidents"
```

### Evaluation Dimensions (6 total)

| # | Dimension | Method |
|---|---|---|
| 1 | **Precision** | LLM-as-judge: of top-10 retrieved chunks, what fraction are genuinely relevant? |
| 2 | **Recall** | Concept-oracle: did the agent find all docs containing required legal concepts? |
| 3 | **Reasoning Quality** | LLM-as-judge on 4 criteria: specificity, legal logic, grounding, practical value |
| 4 | **Adverse Identification** | Checks whether unfavourable precedents were surfaced and honestly assessed |
| 5 | **Faithfulness** | Claim-level verification: does every factual claim trace back to a retrieved chunk? |
| 6 | **Citation Verification** | For each `DOC_XXX` cited, does that document's text actually support the claim? |

Run the full suite after any of the following changes:
- Chunking parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, separators)
- Retrieval parameters (`BM25_WEIGHT`, `MIN_SCORE_SECOND_CHUNK`, `n_results`)
- LLM prompt changes in `graph.py`
- Re-ingestion with new documents

---

## Key Design Choices

### Dynamic routing via LLM
The router node sends each query to the LLM and receives `{"route": "...", "rationale": "..."}` as JSON at `temperature=0.0`. The LangGraph conditional edge reads `state["route"]` to select the branch. No keyword matching or if-else branching in application code.

### Chunking — 1200 chars / 200 overlap / paragraph-aware separators
Chunk size set to 1200 chars to fit one complete Indian court paragraph per chunk (typical paragraph: 200–400 words). Separators prioritise legal structure: `\n\n` → numbered paragraphs (`\n\d+\.\s`) → section headers (`HELD`, `ORDER`, `FACTS`, `ISSUE`) → fallback to `\n` and spaces. This keeps legal reasoning units intact rather than splitting mid-sentence at a character boundary.

### Hybrid retrieval — semantic + BM25 + context expansion
Four-stage pipeline:
1. **Semantic search** — ChromaDB HNSW index (cosine similarity) via `all-MiniLM-L6-v2`, fetching 3× candidates for reranking headroom
2. **BM25 reranking** — Okapi BM25 over the candidate pool, fused at `0.7 × semantic + 0.3 × BM25`; IDF weighting boosts rare legal terms (`Section 149`, `unlicensed driver`) over common words (`court`, `held`)
3. **Top-2 deduplication per document** — retains at most 2 chunks per `doc_id`; the second only if score ≥ 0.35
4. **Context expansion** — each returned chunk is expanded with its ±1 neighbouring chunks before passing to the LLM, preserving legal holdings alongside their surrounding facts

### Deterministic LLM calls
All routing / classification / query-expansion LLM calls use `temperature=0.0` so the same query produces the same graph trace across runs. `generate_strategy` runs at `temperature=0.0` — the strategy memo benefits from natural prose variety and it is the user-facing output, not an internal decision.

### Separated UI layer
All `st.*` rendering calls live in `ui/components.py`. `app.py` is a thin orchestrator — imports and calls component functions, invokes the agent, and manages session state. No Streamlit rendering in `app.py`.

### Centralised logging
All modules import `get_logger(__name__)` from `logger.py`. A single `configure_logging()` call at startup sets format and level for the entire project. Third-party loggers (`chromadb`, `sentence_transformers`, `litellm`, `httpx`) are suppressed to WARNING.

---

## Hosted URL

> **TODO:** replace with your live Streamlit Cloud URL after deploying (see §Deployment below).
> Format: `https://<your-app-name>.streamlit.app`

---

## Deployment (free, Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Sign in to <https://share.streamlit.io> with GitHub → **New app**.
3. Pick the repo, `main` branch, main file `app.py`.
4. Under **Advanced settings → Python version** choose **3.11** (do **not** leave it on the newest option — Python 3.14 triggers deprecation-warning tracebacks from `transformers`' lazy-import system that flood the logs).
5. Under **Advanced settings → Secrets** paste both keys:
   ```
   GROQ_API_KEY = "your_free_groq_key"
   GEMINI_API_KEY = "your_free_gemini_key"
   ```
   (Groq: <https://console.groq.com/keys> · Gemini: <https://aistudio.google.com/apikey>.)
6. Either commit the prebuilt `chroma_db/` (run `python -m ingestion.ingest` locally first, then `git add -f chroma_db/`) **or** the app will build it on first load (~3 min cold start).
7. Deploy. Copy the resulting URL into the **Hosted URL** section above.

> The repo ships a `.streamlit/config.toml` that disables Streamlit's file watcher. The watcher otherwise walks every submodule of `transformers` (pulled in by `sentence-transformers`) and logs a traceback for each vision model that requires `torchvision` — which this project doesn't install, because only text embeddings are used. These tracebacks are harmless but noisy; the config turns them off.

Alternative free hosts (in priority order): Hugging Face Spaces (Streamlit SDK) → Render Free Web Service → Fly.io → Railway trial credit.

---