"""
UI components for the Lexi Legal Research Agent.

All Streamlit rendering logic lives here. app.py imports and calls these
functions — it contains no st.* calls of its own.

Public API:
    apply_page_config()         — st.set_page_config (must be first st call)
    apply_styles()              — inject CSS
    render_header()             — top banner
    render_sidebar(...)          — sidebar: ingest controls, corpus stats, examples
    render_query_input()         — text area + run/clear buttons
    render_results(result)       — answer tabs, reasoning steps, retrieved docs
    get_cached_agent()           — Streamlit-cached agent (call once per session)
    get_cached_retriever(...)    — Streamlit-cached retriever (call once per session)
"""

from typing import Optional

import streamlit as st

from logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

def apply_page_config() -> None:
    st.set_page_config(
        page_title="Lexi · Legal Research Agent",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

.stApp {
    background: #F7F5F0;
}

.main-header {
    background: #1A1A2E;
    color: #E8D5A3;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    font-family: 'Playfair Display', serif;
}

.main-header h1 {
    font-size: 2.2rem;
    margin: 0;
    color: #E8D5A3 !important;
}

.main-header p {
    margin: 0.4rem 0 0;
    color: #A89B7A;
    font-size: 1rem;
}

.step-box {
    background: #FFFFFF;
    border-left: 4px solid #C9A84C;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #333;
}

.route-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.route-direct   { background: #D4EDDA; color: #155724; }
.route-search   { background: #CCE5FF; color: #004085; }
.route-research { background: #FFF3CD; color: #856404; }

.answer-container {
    background: #FFFFFF;
    border: 1px solid #DDD;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
}

.sidebar-info {
    font-size: 0.82rem;
    color: #666;
    line-height: 1.6;
}
</style>
"""

def apply_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit resource caching
# ---------------------------------------------------------------------------

@st.cache_resource
def get_cached_agent():
    """
    Build and cache the LangGraph agent.
    
    Wrapped with @st.cache_resource so the graph is compiled once per
    Streamlit session rather than on every query. Graph compilation is
    lightweight (~50ms) but caching eliminates even that overhead.
    """
    from agent.graph import build_agent
    logger.info("Building and caching LangGraph agent...")
    return build_agent()


@st.cache_resource
def get_cached_retriever(chroma_path: str):
    """
    Build and cache the RetrieverService for a given chroma_path.
    
    Wrapped with @st.cache_resource so the SentenceTransformer model
    (~80 MB, 2–5s load time) is loaded once per chroma_path per session.
    This supersedes the module-level lru_cache in retriever.py when
    running inside Streamlit.
    """
    from agent.retriever import RetrieverService
    logger.info("Initializing and caching RetrieverService for chroma_path=%s", chroma_path)
    return RetrieverService(chroma_path)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>⚖️ Lexi Legal Research Agent</h1>
            <p>AI-powered precedent research across Indian court judgments</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_EXAMPLE_QUERIES = [
    {
        "label": "List all judgments by year",
        "query": "List all the judgments in the corpus along with the year they were decided.",
    },
    {
        "label": "Judgments involving insurance companies",
        "query": "Which judgments involve a dispute with an insurance company? Summarise the outcome in each case.",
    },
    {
        "label": "Cases where the appellant won",
        "query": "In which cases did the appellant succeed? What were the key reasons given by the court?",
    },
    {
        "label": "Cases where the respondent won",
        "query": "In which cases did the respondent succeed? What arguments were accepted by the court?",
    },
    {
        "label": "Judgments by court type",
        "query": "Classify the judgments by the court that decided them — Supreme Court, High Court, Tribunal, etc.",
    },
    {
        "label": "Cases involving compensation awarded",
        "query": "Which judgments awarded monetary compensation? What were the amounts and on what grounds were they awarded?",
    },
]


def render_sidebar(ingest_fn) -> tuple[str, str]:
    """
    Render the full sidebar.

    Args:
        ingest_fn: The ingest_documents callable from ingestion.ingest.

    Returns:
        (docs_dir, chroma_path) — values entered by the user, needed by app.py
        to pass into the agent.
    """
    with st.sidebar:
        st.markdown("### 📂 Document Corpus")

        docs_dir    = st.text_input("Docs folder path", value="docs")
        chroma_path = st.text_input("ChromaDB path",    value="chroma_db")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ingest Docs", type="primary"):
                with st.spinner("Ingesting documents..."):
                    try:
                        count = ingest_fn(docs_dir=docs_dir, chroma_path=chroma_path)
                        st.success(f"✅ {count} chunks indexed")
                        logger.info("Ingestion triggered from UI: %d chunks", count)
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        logger.error("Ingestion failed from UI: %s", exc)
        with col2:
            if st.button("Re-ingest"):
                with st.spinner("Re-ingesting..."):
                    try:
                        count = ingest_fn(docs_dir=docs_dir, chroma_path=chroma_path, reset=True)
                        st.success(f"✅ {count} chunks")
                        logger.info("Re-ingestion triggered from UI: %d chunks", count)
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        logger.error("Re-ingestion failed from UI: %s", exc)

        _render_corpus_stats(chroma_path)

    return docs_dir, chroma_path


def _render_corpus_stats(chroma_path: str) -> None:
    """Show live document/chunk counts from ChromaDB using cached retriever."""
    try:
        import chromadb as _chromadb
        # Use the Streamlit-cached retriever to avoid reloading the model
        retriever = get_cached_retriever(chroma_path)

        _client = _chromadb.PersistentClient(path=chroma_path)
        _col    = _client.get_collection("judgments")
        n_chunks = _col.count()
        n_docs   = len(retriever.list_document_ids())

        st.markdown(
            f'<div class="sidebar-info">'
            f'📊 <b>Corpus status</b><br>'
            f'{n_docs} documents · {n_chunks} chunks indexed'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<div class="sidebar-info">'
            '⚠️ No corpus indexed yet. Add PDFs to the docs/ folder and click "Ingest Docs".'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

def render_query_input() -> tuple[str, bool]:
    """
    Render the query text area, action buttons, and a compact example
    queries dropdown below.

    Returns:
        (query_text, run_clicked)
    """
    query = st.text_area(
        "Enter your research query or case brief:",
        value=st.session_state.get("query_input", ""),
        height=120,
        placeholder=(
            "e.g. Find precedents on whether an insurer can deny liability to a "
            "third-party claimant on the ground that the driver was unlicensed..."
        ),
        key="query_text_area",
    )

    run_col, clear_col = st.columns([1, 5])
    with run_col:
        run_clicked = st.button("🔍 Research", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear"):
            st.session_state.pop("last_result",  None)
            st.session_state.pop("query_input",  None)
            st.rerun()

    _render_example_queries()

    return query, run_clicked


def _render_example_queries() -> None:
    """
    Render example queries inside a collapsed expander so they take up
    minimal space by default. The expander shows a single st.selectbox
    listing all query labels. Selecting one reveals the full query text
    in a st.code block with a built-in clipboard copy button.

    Layout (collapsed by default):
        ▶ 💡 Example queries
        ────────────────────────────────────────
        [Select an example query ▼            ]
        ┌──────────────────────────────────── 📋┐
        │ <full query text>                      │
        └────────────────────────────────────────┘

    The answer section renders immediately below the expander without
    a long list of code blocks taking up vertical space.
    """
    with st.expander("💡 Example queries", expanded=False):
        labels = [q["label"] for q in _EXAMPLE_QUERIES]
        query_map = {q["label"]: q["query"] for q in _EXAMPLE_QUERIES}

        selected_label = st.selectbox(
            "Select an example query",
            options=["— choose one —"] + labels,
            index=0,
            key="example_query_selector",
            label_visibility="collapsed",
        )

        if selected_label and selected_label != "— choose one —":
            selected_query = query_map[selected_label]
            st.markdown(
                "<p style='margin: 0.4rem 0 0.2rem; font-size: 0.78rem; "
                "color: #999;'>"
                "Click 📋 to copy, then paste into the chat above:</p>",
                unsafe_allow_html=True,
            )
            st.code(selected_query, language=None)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def render_results(result: dict) -> None:
    """
    Render the full results panel: route badge + three tabs.

    Args:
        result: The AgentState dict returned by run_agent().
    """
    route = result.get("route", "")
    steps = result.get("intermediate_steps", [])

    _render_route_badge(route)

    tabs = st.tabs(["📋 Answer", "🔍 Reasoning Steps", "📄 Retrieved Documents"])

    with tabs[0]:
        _render_answer(result)

    with tabs[1]:
        _render_reasoning_steps(steps)

    with tabs[2]:
        _render_retrieved_documents(result)


def _render_route_badge(route: str) -> None:
    badge_class = {
        "direct_answer":      "route-direct",
        "document_search":    "route-search",
        "precedent_research": "route-research",
    }.get(route, "route-search")
    label = route.replace("_", " ").title()
    st.markdown(
        f'<span class="route-badge {badge_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def _render_answer(result: dict) -> None:
    answer = result.get("final_answer", "")
    st.markdown('<div class="answer-container">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_reasoning_steps(steps: list) -> None:
    if steps:
        for step in steps:
            st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
    else:
        st.info("No intermediate steps recorded.")


def _render_retrieved_documents(result: dict) -> None:
    chunks         = result.get("retrieved_chunks", [])
    supporting_ids = {p["doc_id"] for p in result.get("supporting_precedents", [])}
    adverse_ids    = {p["doc_id"] for p in result.get("adverse_precedents", [])}

    if not chunks:
        st.info("No documents retrieved for this query.")
        return

    st.markdown(f"**{len(chunks)} documents retrieved**")

    for chunk in chunks:
        label = ""
        if chunk["doc_id"] in supporting_ids:
            label = "✅ Supporting"
        elif chunk["doc_id"] in adverse_ids:
            label = "❌ Adverse"

        with st.expander(
            f"{chunk['doc_id']} — score: {chunk['score']:.3f}  {label}"
        ):
            st.write(chunk.get("text_snippet", ""))
            meta = chunk.get("metadata", {})
            if meta:
                st.json({k: v for k, v in meta.items() if k not in ("doc_id", "source")})
