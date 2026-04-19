"""
LangGraph-based legal research agent.

The agent dynamically routes between:
  - "direct_answer"       : simple factual queries that don't need deep retrieval
  - "document_search"     : single-pass retrieval for straightforward document queries
  - "precedent_research"  : deep multi-step analysis for case research tasks

The routing decision is made by the LLM at runtime — no if-else branching on
keywords. This satisfies the PRD's "flexibility" requirement.
"""
from __future__ import annotations

import json
import re
from typing import Annotated, List, Dict, Any, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from agent.retriever import RetrieverService, RetrievedChunk, get_retriever
from agent.llm_client import simple_chat, chat


# ---------------------------------------------------------------------------
# JSON extraction helper
#
# LLMs frequently wrap JSON in ```json ... ``` fences and (especially Llama
# models) sometimes prefix the response with prose like "Here is the JSON:".
# `.strip("```json").strip("```")` used previously did *character-set*
# stripping, not substring stripping — it silently mangled any content
# starting/ending with {`, j, s, o, n}. This helper:
#   1. Strips code fences via regex
#   2. Tries direct json.loads
#   3. Falls back to extracting the first balanced {...} or [...] substring
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.DOTALL)


def _extract_json(raw: str) -> Any:
    """Extract JSON from an LLM response that may be fence-wrapped or prose-prefixed."""
    cleaned = _FENCE_RE.sub("", raw.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced JSON object or array in the response.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("Could not extract valid JSON", cleaned, 0)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Conversation
    user_query: str
    messages: Annotated[list, add_messages]

    # Routing
    route: str
    routing_rationale: str

    # Retrieval
    retrieved_chunks: List[Dict]   # serialised RetrievedChunk dicts
    retrieval_queries: List[str]   # sub-queries issued
    
    # Analysis
    supporting_precedents: List[Dict]
    adverse_precedents: List[Dict]
    strategy: str

    # Final
    final_answer: str
    intermediate_steps: List[str]   # visible reasoning log for the UI


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> AgentState:
    """
    Ask the LLM to classify the query and decide the research depth required.
    Returns route + rationale so the graph can branch.
    """
    query = state["user_query"]
    steps = state.get("intermediate_steps", [])

    prompt = f"""You are a routing agent for a legal research system.

Given the user's query, decide which workflow to use:

1. "direct_answer"       – The query can be answered from general knowledge without
                           searching the document corpus (e.g., "What does MVC stand for?").
2. "document_search"     – The query asks something specific about the documents but
                           does not require full case analysis (e.g., "Which judgments
                           involve commercial vehicles?", "List all cases where the claimant won").
3. "precedent_research"  – The query requires deep legal analysis: finding supporting and
                           adverse precedents, assessing legal risk, and forming a strategy
                           (e.g., full case briefs, "Find precedents for X argument", requests
                           for compensation estimates based on case facts).

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "route": "<direct_answer | document_search | precedent_research>",
  "rationale": "<one sentence explaining why>"
}}

User query: {query}"""

    raw = simple_chat(prompt, temperature=0.0)
    try:
        data = _extract_json(raw)
        route = data.get("route", "document_search")
        rationale = data.get("rationale", "")
    except Exception:
        route = "document_search"
        rationale = "Defaulted to document_search (JSON parse error)."

    steps.append(f"**Router** → `{route}`: {rationale}")
    return {
        **state,
        "route": route,
        "routing_rationale": rationale,
        "intermediate_steps": steps,
    }


def direct_answer(state: AgentState) -> AgentState:
    """Answer without retrieval — for general knowledge queries."""
    steps = state.get("intermediate_steps", [])
    steps.append("**Direct Answer** — no document retrieval needed.")

    answer = simple_chat(
        state["user_query"],
        system="You are a helpful legal research assistant. Answer concisely.",
        temperature=0.0,
    )
    return {**state, "final_answer": answer, "intermediate_steps": steps}


def document_search(state: AgentState) -> AgentState:
    """
    Single-pass retrieval for factual document queries.
    Returns a structured answer about what the documents contain.
    """
    steps = state.get("intermediate_steps", [])
    # Use cached retriever — model loads once, reuses across calls
    retriever = get_retriever()
    query = state["user_query"]

    # Generate an optimised retrieval query
    search_query = simple_chat(
        f"Rewrite this into a concise retrieval query for a legal corpus (no preamble): {query}",
        temperature=0.0,
    ).strip()
    steps.append(f"**Document Search** — retrieval query: `{search_query}`")

    chunks = retriever.retrieve(search_query, n_results=15)
    steps.append(f"Retrieved {len(chunks)} document chunks from {len(set(c.doc_id for c in chunks))} unique judgments.")

    # Format context
    context_parts = []
    for c in chunks[:10]:
        context_parts.append(
            f"[{c.doc_id}] (score: {c.score:.2f})\n{c.text[:500]}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a legal research assistant. Using ONLY the following excerpts from Indian court judgments, answer the user's query.

Be precise. Cite the document ID (e.g., DOC_012) when referencing a judgment. If the information is not in the excerpts, say so clearly.

QUERY: {query}

DOCUMENT EXCERPTS:
{context}

Answer:"""

    answer = simple_chat(prompt, temperature=0.0)
    retrieved_dicts = [
        {"doc_id": c.doc_id, "source": c.source, "score": round(c.score, 3), "text_snippet": c.text[:300]}
        for c in chunks
    ]
    return {
        **state,
        "final_answer": answer,
        "retrieved_chunks": retrieved_dicts,
        "intermediate_steps": steps,
    }


def generate_research_queries(state: AgentState) -> AgentState:
    """
    For precedent research: break the case query into targeted sub-queries
    to maximise recall across different legal dimensions.
    """
    steps = state.get("intermediate_steps", [])
    query = state["user_query"]

    prompt = f"""You are a senior legal researcher preparing a search strategy.

Given this research task, generate 4–6 targeted retrieval queries that together cover:
- The core legal issue (e.g., insurer liability for unlicensed drivers)
- Compensation principles (e.g., multiplier method, future earnings, dependency)
- Defences the opposing side might raise
- Procedural/factual parallels (e.g., commercial vehicle accidents, death of breadwinner)

Return ONLY a JSON array of query strings. No markdown, no explanation.

Research task: {query}"""

    raw = simple_chat(prompt, temperature=0.0)
    try:
        queries = _extract_json(raw)
        if not isinstance(queries, list):
            queries = [query]
    except Exception:
        queries = [query]

    queries = [q.strip() for q in queries if q.strip()][:6]
    steps.append(f"**Research Query Expansion** — generated {len(queries)} sub-queries:\n" + "\n".join(f"  - {q}" for q in queries))
    return {**state, "retrieval_queries": queries, "intermediate_steps": steps}


def retrieve_precedents(state: AgentState) -> AgentState:
    """Execute all sub-queries and merge results."""
    steps = state.get("intermediate_steps", [])
    # Use cached retriever — model loads once, reuses across calls
    retriever = get_retriever()
    queries = state.get("retrieval_queries", [state["user_query"]])

    all_chunks: Dict[str, RetrievedChunk] = {}
    for q in queries:
        hits = retriever.retrieve(q, n_results=15)
        for chunk in hits:
            if chunk.doc_id not in all_chunks or chunk.score > all_chunks[chunk.doc_id].score:
                all_chunks[chunk.doc_id] = chunk

    # Sort by score descending
    sorted_chunks = sorted(all_chunks.values(), key=lambda c: c.score, reverse=True)
    steps.append(
        f"**Retrieval Complete** — found {len(sorted_chunks)} unique documents after merging {len(queries)} sub-queries.\n"
        + "Top documents: " + ", ".join(c.doc_id for c in sorted_chunks[:8])
    )

    retrieved_dicts = [
        {
            "doc_id": c.doc_id,
            "source": c.source,
            "score": round(c.score, 3),
            "text_snippet": c.text[:400],
            "metadata": c.metadata,
        }
        for c in sorted_chunks
    ]
    return {**state, "retrieved_chunks": retrieved_dicts, "intermediate_steps": steps}


def classify_precedents(state: AgentState) -> AgentState:
    """
    For each retrieved document, ask the LLM to classify it as supporting,
    adverse, or neutral relative to the user's query/case.
    """
    steps = state.get("intermediate_steps", [])
    chunks = state.get("retrieved_chunks", [])
    query = state["user_query"]

    if not chunks:
        steps.append("**Classification** — no chunks to classify.")
        return {**state, "supporting_precedents": [], "adverse_precedents": [], "intermediate_steps": steps}

    # Build a batch classification prompt for top-15 docs
    top_chunks = chunks[:15]
    docs_block = "\n\n".join(
        f"DOC: {c['doc_id']}\n{c['text_snippet']}" for c in top_chunks
    )

    prompt = f"""You are a legal analyst. Analyse each judgment excerpt below relative to this research task and classify each as:
- "supporting": the judgment's legal principles / facts favour the client's position
- "adverse": the judgment's legal principles / facts favour the opposing party
- "neutral": the judgment is not directly relevant to this task

For every document, write a one-sentence key_principle and a one-sentence
relevance_reason.

For ADVERSE documents specifically, you MUST also populate two extra fields:
- "distinguishing_argument": one or two sentences stating how the client can
  distinguish this case on its facts (different fact pattern, different
  statute, different procedural posture, obiter vs ratio, etc.). Be concrete.
- "counter_strategy": one sentence naming the argument the client should
  proactively raise to blunt this precedent's impact.

For supporting / neutral documents, return these two fields as empty strings.

Return ONLY a valid JSON array:
[
  {{
    "doc_id": "DOC_XXX",
    "classification": "supporting | adverse | neutral",
    "key_principle": "<one sentence>",
    "relevance_reason": "<one sentence explaining why it supports/opposes/is neutral>",
    "distinguishing_argument": "<adverse only — how to distinguish on facts or law>",
    "counter_strategy": "<adverse only — one-sentence counter-argument>"
  }},
  ...
]

RESEARCH TASK:
{query}

DOCUMENTS:
{docs_block}"""

    raw = simple_chat(prompt, temperature=0.0, max_tokens=3000)
    try:
        classified = _extract_json(raw)
        if not isinstance(classified, list):
            classified = []
    except Exception:
        classified = []

    supporting = [c for c in classified if c.get("classification") == "supporting"]
    adverse = [c for c in classified if c.get("classification") == "adverse"]

    steps.append(
        f"**Precedent Classification** — {len(supporting)} supporting, {len(adverse)} adverse, "
        f"{len(classified) - len(supporting) - len(adverse)} neutral."
    )
    steps.append("Supporting: " + ", ".join(c["doc_id"] for c in supporting))
    steps.append("Adverse: " + ", ".join(c["doc_id"] for c in adverse))

    return {
        **state,
        "supporting_precedents": supporting,
        "adverse_precedents": adverse,
        "intermediate_steps": steps,
    }


def generate_strategy(state: AgentState) -> AgentState:
    """
    Synthesise supporting + adverse precedents into a strategy recommendation.
    """
    steps = state.get("intermediate_steps", [])
    query = state["user_query"]
    supporting = state.get("supporting_precedents", [])
    adverse = state.get("adverse_precedents", [])
    retrieved = state.get("retrieved_chunks", [])

    # Build snippet lookup
    snippets = {c["doc_id"]: c["text_snippet"] for c in retrieved}

    supporting_block = "\n".join(
        f"- {p['doc_id']}: {p.get('key_principle','')}\n  Snippet: {snippets.get(p['doc_id'],'')[:300]}"
        for p in supporting
    ) or "None identified."

    adverse_block = "\n".join(
        f"- {p['doc_id']}: {p.get('key_principle','')}\n  Snippet: {snippets.get(p['doc_id'],'')[:300]}"
        for p in adverse
    ) or "None identified."

    prompt = f"""You are a senior advocate preparing a legal strategy memo.

Based on the precedent analysis below, write a comprehensive strategy recommendation covering:

1. **Summary of Legal Position** – Overall strength of the client's case
2. **Key Arguments to Prioritise** – Specific legal principles from supporting precedents to lead with
3. **How to Counter Adverse Precedents** – Distinguish facts or argue exceptions for each adverse case
4. **Realistic Compensation Range** – If compensation is relevant, estimate range with reasoning
5. **Key Risks** – What the client should be prepared for

Be specific. Cite document IDs (e.g., DOC_012) when referencing precedents.

SAFETY CONSTRAINTS (critical — this memo may be relied on in legal practice):
- Only quote monetary figures (compensation amounts, multipliers, income) that
  appear verbatim in the snippets provided below. Do NOT extrapolate numbers.
- If the provided snippets do not contain a specific figure required for
  compensation estimation, say so explicitly — do not invent one.
- Do NOT cite a DOC_XXX that is not in the SUPPORTING or ADVERSE lists below.
- Do NOT fabricate case names, statutes, or section numbers not present in
  the snippets.

RESEARCH TASK / CASE CONTEXT:
{query}

SUPPORTING PRECEDENTS:
{supporting_block}

ADVERSE PRECEDENTS:
{adverse_block}"""

    # Temperature 0.0 — strategy memos benefit from natural prose variety,
    # while routing / classification upstream remain deterministic at 0.0.
    strategy = simple_chat(prompt, temperature=0.0, max_tokens=2000)
    steps.append("**Strategy Generated** ✓")
    return {**state, "strategy": strategy, "intermediate_steps": steps}


def compile_final_answer(state: AgentState) -> AgentState:
    """
    Format the full precedent research report for display.
    """
    supporting = state.get("supporting_precedents", [])
    adverse = state.get("adverse_precedents", [])
    strategy = state.get("strategy", "")
    route = state.get("route", "")

    if route == "direct_answer":
        return state

    if route == "document_search":
        return state

    # Precedent research — build structured report
    def format_precedent_list(plist, adverse: bool = False):
        if not plist:
            return "_None identified._"
        lines = []
        for p in plist:
            block = (
                f"**{p['doc_id']}**\n"
                f"- Key principle: {p.get('key_principle', 'N/A')}\n"
                f"- Reason: {p.get('relevance_reason', 'N/A')}"
            )
            if adverse:
                distinguishing = p.get("distinguishing_argument", "").strip()
                counter = p.get("counter_strategy", "").strip()
                if distinguishing:
                    block += f"\n- How to distinguish: {distinguishing}"
                if counter:
                    block += f"\n- Counter-strategy: {counter}"
            lines.append(block)
        return "\n\n".join(lines)

    report = f"""## Legal Precedent Research Report

### Supporting Precedents
{format_precedent_list(supporting)}

---

### Adverse Precedents
{format_precedent_list(adverse, adverse=True)}

---

### Strategy Recommendation
{strategy}
"""
    return {**state, "final_answer": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_agent() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("route_query", route_query)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("document_search", document_search)
    graph.add_node("generate_research_queries", generate_research_queries)
    graph.add_node("retrieve_precedents", retrieve_precedents)
    graph.add_node("classify_precedents", classify_precedents)
    graph.add_node("generate_strategy", generate_strategy)
    graph.add_node("compile_final_answer", compile_final_answer)

    graph.set_entry_point("route_query")

    def pick_branch(state: AgentState) -> str:
        return state.get("route", "document_search")

    graph.add_conditional_edges(
        "route_query",
        pick_branch,
        {
            "direct_answer": "direct_answer",
            "document_search": "document_search",
            "precedent_research": "generate_research_queries",
        },
    )

    graph.add_edge("direct_answer", END)
    graph.add_edge("document_search", END)
    graph.add_edge("generate_research_queries", "retrieve_precedents")
    graph.add_edge("retrieve_precedents", "classify_precedents")
    graph.add_edge("classify_precedents", "generate_strategy")
    graph.add_edge("generate_strategy", "compile_final_answer")
    graph.add_edge("compile_final_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_agent(query: str) -> AgentState:
    # NOTE: In Streamlit, wrap this with @st.cache_resource instead:
    #   @st.cache_resource
    #   def get_cached_agent():
    #       return build_agent()
    #   agent = get_cached_agent()
    agent = build_agent()
    initial_state: AgentState = {
        "user_query": query,
        "messages": [],
        "route": "",
        "routing_rationale": "",
        "retrieved_chunks": [],
        "retrieval_queries": [],
        "supporting_precedents": [],
        "adverse_precedents": [],
        "strategy": "",
        "final_answer": "",
        "intermediate_steps": [],
    }
    result = agent.invoke(initial_state)
    return result