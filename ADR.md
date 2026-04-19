# Architecture Decision Record (ADR)
## Lexi Legal Precedent Research Agent

---

## 1. Architecture Overview

The system is a **LangGraph-based agentic RAG pipeline** with three dynamic execution paths routed by the LLM at runtime. The agent determines its own workflow based on what the query actually requires — no hard-coded steps, no if-else branching.

```
User Query
    │
    ▼
[Router Node]  ← LLM classifies at runtime (temperature=0.0)
    │
    ├──▶ direct_answer      (general knowledge, no retrieval)
    ├──▶ document_search    (single-pass retrieval, factual doc queries)
    └──▶ precedent_research (query expansion → hybrid retrieval → classification → strategy)
```

---

## 2. Key Design Decisions

### 2.1 Why LangGraph?

LangGraph represents agent execution as a **typed state machine**. Key benefits:

- **Explicit state schema** (`AgentState` TypedDict) — every node reads and writes to a shared dict. Intermediate reasoning is inspectable at every step (visible in the UI's Reasoning Steps tab).
- **Conditional edges** — routing between workflows is a first-class graph construct. No if-else branching in application code.
- **Easy extensibility** — adding a new node means one node + one edge. The rest of the graph is untouched.

LangGraph was chosen over LlamaIndex Workflows for cleaner state propagation and tighter LangChain integration.

### 2.2 Why ChromaDB + sentence-transformers?

**ChromaDB `PersistentClient`** stores the index on disk — no separate service to run. This satisfies the assessment constraint that evaluators must not run local infrastructure.

**`all-MiniLM-L6-v2`** — open-source (no API key), 22M parameters, 384-dimensional embeddings. Chosen for the speed/quality tradeoff appropriate for a 56-document corpus. Outperforms pure BM25 on legal text because legal language is highly domain-specific: "unlicensed driver voids policy" and "driver without valid licence — insurer liability" share few words but are semantically identical.

**Cosine similarity** — normalises for document length, which performs better than dot-product on legal paragraphs of varying length.

### 2.3 Chunking strategy

**Parameters:** 1200 chars / 200 overlap (increased from 800/150)

A complete Indian court paragraph making a legal point typically runs 200–400 words (~1200–2400 chars). At 800 chars, most paragraphs were being split mid-sentence, producing poor chunk embeddings. At 1200 chars, each chunk holds one complete legal reasoning unit.

**Paragraph-aware separators** — the splitter tries to break on legal structure before falling back to plain newlines:
```
\n\n  →  \n\d+\.\s (numbered paragraphs)  →  \nHELD  →  \nORDER
→  \nFACTS  →  \nISSUE  →  \nREASONS  →  \nJUDGMENT  →  \n  →  .  →  space
```
This keeps numbered paragraphs (`12. The court held...`) and section headers intact as single chunks rather than splitting mid-sentence at a character boundary.

**Text cleaning before chunking** — two regex patterns strip Indian Kanoon boilerplate present on every page of every corpus document:
1. `"Indian Kanoon - http://indiankanoon.org/doc/XXXXXXX/ N"` — URL footer
2. `"CaseName vs OtherParty on DD Month YYYY"` — repeating case title line

These are the only two regex patterns used. All other metadata (appellant, respondent, year) is extracted from the same title line pattern on the raw first page before cleaning.

### 2.4 Hybrid retrieval pipeline

Three-stage pipeline applied to every query:

**Stage 1 — Dense semantic search**
ChromaDB HNSW index queried at 3× the requested result count to give reranking headroom. Scores converted from cosine distance to similarity: `score = 1.0 - (distance / 2.0)`.

**Stage 2 — BM25 reranking with score fusion**
`rank-bm25` (Okapi BM25) computed over the candidate pool. IDF weighting means rare legal terms (`Section 149`, `unlicensed driver`) score much higher than common words (`court`, `held`). Normalised BM25 score fused with semantic score: `final = 0.7 × semantic + 0.3 × BM25`. Semantic still dominates; BM25 provides a meaningful boost for exact legal term matches.

**Stage 3 — Context window expansion (±1 neighbours)**
Each top-K chunk is expanded with its immediately preceding and following chunks from the same document. Legal reasoning almost always spans paragraph boundaries — a holding in paragraph 12 references facts in paragraph 11. Expansion gives the LLM this context without retrieving extra documents. Cost: 2 point lookups per result (~5ms total for n=10).

**Top-2 deduplication per document**
Retains at most 2 chunks per `doc_id`. The second is only kept if its score ≥ 0.35. This captures cases where a judgment has relevant content in two separate places (ratio + compensation figure) without flooding context from a single document.

### 2.5 Dynamic routing (not hard-coded)

The router node sends every query to the LLM with a structured prompt at `temperature=0.0` and receives `{"route": "...", "rationale": "..."}` as JSON. The LangGraph conditional edge reads `state["route"]` to branch. Examples:

- `"Which judgments mention contributory negligence?"` → `document_search`
- `"What is the multiplier method?"` → `direct_answer`
- `"Full case analysis — insurer denying claim"` → `precedent_research`

No keyword matching. No regex. No if-else branching in application code.

### 2.6 Deterministic LLM calls

All internal (non-user-facing) LLM calls use `temperature=0.0`:
- Routing decisions are consistent for the same query
- Classification of supporting/adverse precedents does not vary between runs
- Query expansion generates the same sub-queries for the same input
- Evaluation results are comparable across runs

`generate_strategy` runs at `temperature=0.0` — the strategy memo is the user-facing output and benefits from natural prose variety. It also carries explicit safety constraints in the prompt: *only quote monetary figures verbatim from the retrieved snippets, never fabricate case names or statute numbers, never cite a DOC_XXX not in the supporting/adverse lists*. This pins down the dangerous failure mode (hallucinated compensation figures) without forcing stilted prose on the rest of the memo.

### 2.7 Evaluation framework — 6 dimensions (offline)

Evaluation runs **offline** against 5 predefined test cases. It does not execute during live queries.

| Dimension | What it measures |
|---|---|
| **Precision** | Of top-10 retrieved chunks, what fraction are genuinely relevant? (LLM-as-judge) |
| **Recall** | Of all relevant documents, what fraction did the agent find? (concept-oracle approximation) |
| **Reasoning Quality** | Is the explanation logically coherent, specific, and grounded? (LLM-as-judge, 4 criteria) |
| **Adverse Identification** | Did the agent surface unfavourable precedents and honestly assess their risk? |
| **Faithfulness** | Does every factual claim in the answer trace back to a retrieved source chunk? (claim extraction + verification) |
| **Citation Verification** | For each `DOC_XXX` cited, does that document's text actually support the attributed principle? |

Faithfulness and citation verification were added to catch the most dangerous failure mode for legal tools: hallucinated or misattributed case citations.

### 2.8 Separated UI layer

All `st.*` rendering calls live exclusively in `ui/components.py`. `app.py` is a thin orchestration entry point — it imports and calls component functions, invokes the agent, and manages session state. This separation makes both layers independently testable and keeps the Streamlit-specific code isolated from business logic.

### 2.9 Centralised logging

All modules use `get_logger(__name__)` from `logger.py`. A single `configure_logging()` call at app startup sets format, level, and handlers for the entire project. Third-party loggers (`chromadb`, `sentence_transformers`, `litellm`, `httpx`) are suppressed to WARNING. Format: `YYYY-MM-DD HH:MM:SS | LEVEL | module.name | message`.

### 2.10 API key management

`GROQ_API_KEY` (primary) and `GEMINI_API_KEY` (fallback) are read exclusively from the `.env` file (or `st.secrets` on Streamlit Cloud) at startup. Neither is **exposed in the UI**. This prevents accidental key exposure in screenshots, screen shares, or demo recordings. If a required key is missing, the app surfaces a clear error pointing to `.env` rather than asking for it in a UI field.

### 2.11 LLM choice — Groq Llama 3.3 70B (primary) + Gemini 2.0 Flash (fallback)

The pairing is deliberately **cross-provider**: primary and fallback live on different vendors with independent quota buckets, so exhausting one provider's free-tier cap (TPM *or* TPD) does not block the cascade.

- **Llama 3.3 70B on Groq** (primary): strong instruction-following, high JSON compliance, solid legal-reasoning quality, and very high tokens/second. Handles the classification and strategy-memo nodes well.
- **Gemini 2.0 Flash on Google AI Studio** (fallback): generous free tier (1M TPM, 1500 RPD, 15 RPM), independent of Groq's org-level bucket. Used only when the Groq call fails persistently (retries exhausted, or a daily-quota hint exceeds our per-retry cap). Sub-second latency and strong JSON compliance keep downstream parsing reliable.
- **Why cross-provider** — an earlier version used `groq/llama-3.1-8b-instant` as fallback and discovered that Groq's TPD is shared across models at the organisation level, so a daily cap on the 70B cascaded straight into a cap on the 8B. Pairing Groq with Gemini keeps the two failure modes independent.
- **Provider-agnostic client** — LiteLLM routes by the `provider/model` prefix, so swapping to `openai/...`, `ollama/...`, `cerebras/...`, or any other LiteLLM-supported provider is a single constant change.

### 2.12 Rate-limit resilience — retry-with-hint before fallback

Groq's free tier enforces per-model token-per-minute ceilings (70B: 12k TPM) plus a 100k tokens-per-day cap. Back-to-back eval judge calls easily saturate both buckets. The LLM client (`agent/llm_client.py`) therefore:

1. **Retries rate-limit errors on the same model** (up to `MAX_RETRIES = 4`) before considering it failed. Sleep duration is parsed from the server hint — Groq's `"Please try again in 5.73s"` or Gemini's `retry_delay { seconds: N }` — with a 0.5s safety buffer. If no hint is present, exponential backoff (1, 2, 4, 8s) with jitter is used.
2. **Fails fast when the hint exceeds `MAX_SLEEP_PER_RETRY = 35s`** — typically Groq's tokens-per-day ceiling ("try again in 10m15s"). Spinning through retries is futile; the cascade switches to the fallback provider instead.
3. **Cascades to the fallback provider** (Gemini 2.0 Flash) once primary retries are exhausted. Because the fallback lives on a different vendor with a different API key and quota bucket, this cascade actually recovers the call instead of burning both ceilings at once.
4. **Short-circuits on authentication errors** (`401 invalid_api_key`) against the *fallback* without retrying. A primary auth failure still cascades — the fallback key is independent.
5. **Capped sleep** prevents a single pathological retry from exceeding the per-call timeout.

This trades eval wall-clock time (a slow run may take 2–3 minutes instead of 30s) for completion rate (scores land at their true value instead of 0.0 parse-error stubs).

---

## 3. Tradeoffs Made

| Decision | Chosen | Alternative | Tradeoff |
|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `text-embedding-3-small` (OpenAI) | Open-source, no API key; slightly lower quality at edge cases |
| Vector DB | ChromaDB (embedded) | Pinecone / Qdrant | No cloud dependency or extra API key; scales to ~100k chunks |
| LLM | Groq Llama 3.3 70B (+ Gemini 2.0 Flash cross-provider fallback) via LiteLLM | Same-provider fallback / GPT-4o / Claude Sonnet | Generous free tier, sub-second latency; cross-provider fallback means Groq TPM/TPD caps don't block the cascade |
| Keyword reranking | BM25 (`rank-bm25`) | Custom TF boost | BM25 IDF weighting significantly outperforms raw TF on rare legal terms |
| Chunk size | 1200 chars | 800 chars | Fits complete paragraphs; slightly larger index but higher embedding quality |
| Deduplication | Top-2 per document | Top-1 per document | Captures judgments with relevant content in two separate places |
| LLM temperature | 0.0 (all internal calls) | 0.1–0.3 | Deterministic routing and classification; slightly reduced prose variety |
| API key location | `.env` only | UI field | Eliminates key exposure in UI; requires server-side configuration |
| Eval execution | Offline only | Inline per query | No latency overhead on live queries; run when assessing system changes |

---

## 4. How the Agent Decides Query Depth

Three-way classification at runtime:

1. **General knowledge** (no corpus needed) → `direct_answer`
2. **Factual lookup** (find what is in documents) → `document_search`
3. **Legal analysis** (find supporting/adverse precedents, form strategy) → `precedent_research`

The LLM prompt provides concrete examples of each category. Decision is made at `temperature=0.0` — deterministic for the same query. No fallback heuristics.

---

## 5. Scaling to 5,000+ Documents

**Retrieval layer:**
- Replace ChromaDB embedded client with Qdrant (self-hosted) or Pinecone (managed) for horizontal scaling and concurrent writes
- Add metadata pre-filtering (`WHERE` clauses on `year`, `case_domain`, etc.) to reduce the vector search space before semantic search
- Switch embedding model to a legal-domain fine-tuned model (`legal-bert` or custom fine-tune on Indian case law) for higher domain-specific recall
- Hierarchical indexing: generate 2-sentence LLM summaries at ingest time, first-pass retrieval on summaries, second-pass on full chunks

**Ingestion:**
- Replace sequential `for pdf in pdf_files` loop with `ProcessPoolExecutor` for parallel PDF extraction and chunking
- Replace in-memory `set(collection.get()["ids"])` deduplication (pulls entire index into RAM) with per-document hash check against a Postgres/Redis store
- One LLM call per document for domain classification becomes expensive at scale — replace with a fine-tuned local DistilBERT classifier (~5ms/doc, zero API cost)

**Agent:**
- LangGraph architecture scales unchanged — the graph structure is independent of corpus size

---

## 6. What I Would Change With Another Week

1. **Legal-aware chunking** — Use spaCy + Indian legal NER to detect sentence boundaries and keep rulings, holdings, and obiter dicta as separate chunks. Would dramatically improve classification accuracy by keeping legal reasoning units semantically pure.

2. **Document-level summaries at ingest** — Generate a 2-sentence LLM summary per document. First-pass retrieval on summaries (fast, cheap) narrows candidates; second-pass on full chunks gives precision. Improves recall at scale without degrading latency.

3. **Human-annotated eval gold set** — Replace the concept-oracle recall approximation with 20–30 hand-labelled query-document pairs. Makes recall a trustworthy metric that can drive A/B comparison between system versions.

4. **Inline per-query confidence signal** — Add a lightweight heuristic confidence score to the UI (retrieval score of top chunk + citation coverage ratio) so users have a quality signal without running the full offline eval suite.

5. **Consistency check in eval** — Run 3 paraphrase variants of each eval query and measure answer overlap. Low overlap indicates unstable retrieval — small query wording changes are surfacing different documents.

---

## 7. Evaluation Framework — Where the Agent Fails

**Known weaknesses:**

- **Precision on broad queries** — queries like "find all compensation cases" cause over-retrieval. BM25 helps but is insufficient when every document contains the word "compensation". A query decomposition step (break broad query into specific sub-questions) would help.

- **Adverse identification false negatives** — the classification prompt sometimes marks a document as "neutral" when the adverse principle is buried in an obiter remark rather than the holding. Fetching the top-2 chunks per document (now implemented) partially addresses this; full document summarisation before classification would fix it completely.

- **Faithfulness on compensation figures** — when no exact figure appears in the retrieved chunks, the strategy generation prompt sometimes extrapolates. Adding an explicit instruction "only cite figures that appear verbatim in the retrieved documents" reduces this.

- **Citation verification latency** — citation verification makes one LLM call per cited `DOC_XXX`. For answers that cite 5+ documents, this adds ~5–10s to eval time. Batching multiple citation checks into a single LLM call would halve this.

**What to fix first:**

The faithfulness gap on compensation figures is the highest-priority fix because it is the most dangerous failure mode in legal practice — an incorrect compensation figure stated as fact from a case is directly harmful. An explicit "verbatim figures only" constraint in the strategy prompt is a one-line fix with significant safety impact.

---