"""
Automated Evaluation Framework
Measures agent performance on 6 dimensions:
  1. Precision              – of retrieved docs, what % are actually relevant?
  2. Recall                 – of truly relevant docs, what % did the agent find?
  3. Reasoning Quality      – is the explanation logically coherent and grounded?
  4. Adverse Identification – did the agent surface unfavourable precedents?
  5. Faithfulness           – does every claim in the answer trace back to a source chunk?
  6. Citation Verification  – do cited doc IDs actually support the attributed principles?

Run:
    python -m evaluation.eval --query "..."
    python -m evaluation.eval --case-id case_01_insurer_liability
    python -m evaluation.eval --full-suite --output evaluation/eval_results.json
"""
from __future__ import annotations

import re
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.graph import run_agent
from agent.llm_client import simple_chat
from agent.retriever import RetrieverService
from logger import get_logger, configure_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction helpers
#
# LLM responses occasionally come back with unescaped quotes inside a
# reasoning string or get truncated by max_tokens mid-string.  Plain
# json.loads fails with "Unterminated string" / "Expecting ',' delimiter"
# and the entire eval case scores 0.0.  These helpers try progressively
# more lenient parses before giving up.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.DOTALL)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw.strip())


def _safe_json_loads(raw: str):
    """
    Parse a JSON response that may be fence-wrapped, truncated, or contain
    trailing garbage. Tries:
      1. direct parse after fence strip
      2. parse just the first balanced {...} or [...] substring
      3. raise
    """
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try to extract the first balanced object or array
    for opener, closer in [("{", "}"), ("[", "]")]:
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


def _extract_score_regex(raw: str, key: str, default: float = 0.0) -> float:
    """
    Last-resort extractor: pull a numeric score out of a broken JSON string
    by regex. Used when JSON parsing fails entirely so the case still gets
    a representative score instead of a hard 0.0.
    """
    m = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

EVAL_CASES = [
    {
        "id": "case_01_insurer_liability",
        "query": (
            "Find precedents supporting the argument that an insurer cannot escape liability "
            "simply because the truck driver was unlicensed, in a commercial vehicle accident resulting in death."
        ),
        "required_concepts": [
            "insurer liability",
            "unlicensed driver",
            "commercial vehicle",
            "third party",
            "compensation",
        ],
        "adverse_concepts": ["policy void", "policy breach", "insurer not liable"],
        "description": "Core insurer liability argument for unlicensed driver case",
    },
    {
        "id": "case_02_compensation",
        "query": (
            "What compensation was awarded in cases involving death of a breadwinner aged 40-45 "
            "with monthly income around ₹35,000 and two minor dependents?"
        ),
        "required_concepts": ["multiplier", "dependency", "compensation", "income", "deduction"],
        "adverse_concepts": [],
        "description": "Compensation quantum calculation",
    },
    {
        "id": "case_03_adverse_identification",
        "query": (
            "Find ALL precedents — both for and against — regarding insurer liability when the "
            "driver lacks a valid driving licence in India. I need both supporting and adverse cases."
        ),
        "required_concepts": ["insurer", "licence", "liability"],
        "adverse_concepts": [],
        "description": "Adverse precedent identification test",
        "must_have_adverse": True,
    },
    {
        "id": "case_04_commercial_vehicle_filter",
        "query": "Which of these judgments involve commercial vehicles such as trucks, buses, or lorries?",
        "required_concepts": ["commercial vehicle"],
        "adverse_concepts": [],
        "description": "Factual document search",
    },
    {
        "id": "case_05_full_case_brief",
        "query": (
            "Research precedents for this case: Mrs. Lakshmi Devi's husband was killed when struck by a "
            "commercial truck driven by an unlicensed driver. National Insurance Co. is denying the claim "
            "saying the policy is void. Deceased was 42 years old, monthly income ₹35,000, with wife and two "
            "minor children as dependents. What are the supporting precedents, adverse precedents, and strategy?"
        ),
        "required_concepts": ["insurer", "compensation", "dependent"],
        "adverse_concepts": [],
        "description": "Full case brief — end-to-end test",
        "must_have_adverse": True,
    },
]


# ---------------------------------------------------------------------------
# Supporting dataclasses for faithfulness and citation verification
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerdict:
    """Result of verifying a single claim extracted from the answer."""
    claim: str
    verdict: str        # "supported" | "partially_supported" | "unsupported"
    source_chunk_id: str = ""
    reasoning: str = ""


@dataclass
class CitationVerdict:
    """Result of verifying a single DOC_XXX citation in the answer."""
    doc_id: str
    attributed_principle: str
    verified: bool
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Main result dataclass — now includes faithfulness and citation verification
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    case_id: str
    description: str
    route: str
    n_retrieved: int
    n_supporting: int
    n_adverse: int

    # Dimension 1: Precision
    precision_score: float
    precision_reasoning: str

    # Dimension 2: Recall
    recall_score: float
    recall_reasoning: str

    # Dimension 3: Reasoning quality
    reasoning_quality_score: float
    reasoning_quality_feedback: str

    # Dimension 4: Adverse identification
    adverse_id_score: float
    adverse_id_reasoning: str

    # Dimension 5: Faithfulness
    faithfulness_score: float
    faithfulness_reasoning: str
    claim_verdicts: List[Dict] = field(default_factory=list)

    # Dimension 6: Citation verification
    citation_score: float = 0.0
    citation_reasoning: str = ""
    citation_verdicts: List[Dict] = field(default_factory=list)

    # Aggregate — average of all 6 dimensions
    overall_score: float = 0.0
    latency_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Dimension 1 — Precision
# ---------------------------------------------------------------------------

def score_precision(
    retrieved_chunks: List[Dict], query: str, case: Dict
) -> tuple[float, str]:
    """
    LLM-as-judge: of the top-15 retrieved chunks, what fraction are
    genuinely relevant to the query?
    """
    if not retrieved_chunks:
        return 0.0, "No documents retrieved."

    top_chunks = retrieved_chunks[:15]
    docs_block = "\n\n".join(
        f"DOC {i+1} ({c['doc_id']}): {c.get('text_snippet', '')[:400]}"
        for i, c in enumerate(top_chunks)
    )

    prompt = f"""You are evaluating the precision of a legal research agent.

Query: {query}

For EACH document excerpt below, judge whether it is genuinely relevant to the query.
Keep every "reason" field under 15 words. Use single quotes inside reasons, never double quotes.
Do NOT truncate — the final "precision_score" field must appear in your output.

Return ONLY valid JSON:
{{
  "precision_score": <0.0-1.0, fraction that are relevant>,
  "overall_reasoning": "<one sentence>",
  "judgments": [{{"doc_id": "DOC_XXX", "relevant": true, "reason": "..."}}]
}}

DOCUMENTS:
{docs_block}"""

    raw = ""
    try:
        raw = simple_chat(prompt, temperature=0.0, max_tokens=4000)
        data = _safe_json_loads(raw)
        score = float(data.get("precision_score", 0.0))
        return min(1.0, max(0.0, score)), data.get("overall_reasoning", "")
    except Exception as exc:
        # Fall back to regex extraction — the score may be recoverable even
        # if the full JSON is malformed.
        fallback = _extract_score_regex(raw, "precision_score", default=-1.0)
        if fallback >= 0:
            return min(1.0, max(0.0, fallback)), f"Recovered via regex (JSON malformed): {exc}"
        return 0.0, f"Parse error: {exc}"


# ---------------------------------------------------------------------------
# Dimension 2 — Recall
# ---------------------------------------------------------------------------

def score_recall(
    retrieved_chunks: List[Dict],
    query: str,
    required_concepts: List[str],
    retriever: RetrieverService,
) -> tuple[float, str]:
    """
    Concept-oracle recall: retrieve docs for each required concept independently,
    union them as the oracle relevant set, then measure overlap with agent results.
    """
    if not required_concepts:
        return 1.0, "No required concepts defined — recall not applicable."

    concept_doc_sets = []
    for concept in required_concepts:
        hits = retriever.retrieve(concept, n_results=5)
        concept_doc_sets.append({c.doc_id for c in hits})

    oracle_docs = set().union(*concept_doc_sets) if concept_doc_sets else set()
    agent_docs  = {c["doc_id"] for c in retrieved_chunks}

    if not oracle_docs:
        return 1.0, "Oracle set is empty."

    found  = oracle_docs & agent_docs
    recall = len(found) / len(oracle_docs)
    reasoning = (
        f"Oracle: {sorted(oracle_docs)} | "
        f"Agent: {sorted(agent_docs)} | "
        f"Overlap: {sorted(found)} | "
        f"Recall = {len(found)}/{len(oracle_docs)} = {recall:.2f}"
    )
    return recall, reasoning


# ---------------------------------------------------------------------------
# Dimension 3 — Reasoning quality
# ---------------------------------------------------------------------------

def score_reasoning_quality(
    final_answer: str,
    query: str,
    supporting: List[Dict],
    adverse: List[Dict],
) -> tuple[float, str]:
    """
    LLM-as-judge scoring on 4 sub-criteria: specificity, legal logic,
    document grounding, and practical value.
    """
    if not final_answer.strip():
        return 0.0, "Empty answer."

    cited_docs = ", ".join(
        p["doc_id"] for p in (supporting + adverse)[:5]
    ) or "none cited"

    prompt = f"""You are evaluating the reasoning quality of a legal research agent's response.

Score 1–5 for each criterion, then give an overall score 0.0–1.0:
1. Specificity      — cites specific doc IDs and explains what each holds
2. Legal Logic      — arguments are logically coherent and well-structured
3. Doc Grounding    — grounded in retrieved documents, not hallucinated
4. Practical Value  — gives actionable guidance

Query: {query}
Agent response (first 2000 chars): {final_answer[:2000]}
Docs cited by agent: {cited_docs}

Keep "feedback" under 60 words. Use single quotes inside the feedback string, never double quotes.

Return ONLY valid JSON (put "overall_score" first so it survives any truncation):
{{
  "overall_score": <0.0-1.0>,
  "specificity": <1-5>,
  "legal_logic": <1-5>,
  "document_grounding": <1-5>,
  "practical_value": <1-5>,
  "feedback": "..."
}}"""

    raw = ""
    try:
        raw = simple_chat(prompt, temperature=0.0, max_tokens=2000)
        data = _safe_json_loads(raw)
        score = float(data.get("overall_score", 0.0))
        return min(1.0, max(0.0, score)), data.get("feedback", "")
    except Exception as exc:
        fallback = _extract_score_regex(raw, "overall_score", default=-1.0)
        if fallback >= 0:
            return min(1.0, max(0.0, fallback)), f"Recovered via regex (JSON malformed): {exc}"
        return 0.0, f"Parse error: {exc}"


# ---------------------------------------------------------------------------
# Dimension 4 — Adverse identification
# ---------------------------------------------------------------------------

def score_adverse_identification(
    adverse_precedents: List[Dict], must_have_adverse: bool, query: str
) -> tuple[float, str]:
    """
    Checks whether the agent surfaced genuinely adverse precedents and
    assessed their risk honestly.
    """
    if must_have_adverse and not adverse_precedents:
        return 0.0, "FAIL: No adverse precedents found despite being required."

    if not adverse_precedents:
        return 0.8, "Not required for this query type."

    def _fmt_adverse(p: Dict) -> str:
        parts = [f"{p['doc_id']}: {p.get('key_principle', '')}"]
        reason = (p.get("relevance_reason") or "").strip()
        if reason:
            parts.append(f"  Why adverse: {reason}")
        distinguishing = (p.get("distinguishing_argument") or "").strip()
        if distinguishing:
            parts.append(f"  How to distinguish: {distinguishing}")
        counter = (p.get("counter_strategy") or "").strip()
        if counter:
            parts.append(f"  Counter-strategy: {counter}")
        return "\n".join(parts)

    adverse_text = "\n\n".join(_fmt_adverse(p) for p in adverse_precedents)

    prompt = f"""Evaluate the quality of adverse precedent identification.

Good adverse identification means:
- Cases genuinely oppose the client's position
- WHY each case is adverse is explained
- How to distinguish/counter each is suggested

Query: {query}
Adverse precedents: {adverse_text}

Keep reasoning under 60 words, use single quotes inside the string.
Return ONLY valid JSON: {{"score": <0.0-1.0>, "reasoning": "..."}}"""

    raw = ""
    try:
        raw = simple_chat(prompt, temperature=0.0, max_tokens=1000)
        data = _safe_json_loads(raw)
        score = float(data.get("score", 0.5))
        return min(1.0, max(0.0, score)), data.get("reasoning", "")
    except Exception as exc:
        fallback = _extract_score_regex(raw, "score", default=-1.0)
        if fallback >= 0:
            return min(1.0, max(0.0, fallback)), f"Recovered via regex (JSON malformed): {exc}"
        return 0.5, f"Parse error: {exc}"


# ---------------------------------------------------------------------------
# Dimension 5 — Faithfulness
# ---------------------------------------------------------------------------

def score_faithfulness(
    final_answer: str,
    retrieved_chunks: List[Dict],
) -> tuple[float, str, List[ClaimVerdict]]:
    """
    Faithfulness measures whether every factual claim in the answer can be
    traced back to a specific retrieved chunk.

    Process:
      Step A — Extract claims: ask the LLM to break the answer into discrete
               factual claims (case citations, legal principles, compensation
               figures, statutes cited).
      Step B — Verify each claim: for each claim, check whether any retrieved
               chunk contains evidence supporting it.

    Score = supported_claims / total_claims

    A claim is "supported" if it is explicitly stated in the sources.
    "Partially supported" counts as 0.5.
    "Unsupported" counts as 0 and signals a hallucination.
    """
    if not final_answer.strip():
        return 0.0, "Empty answer — nothing to verify.", []

    if not retrieved_chunks:
        return 0.0, "No retrieved chunks to verify against.", []

    # ── Step A: Extract discrete factual claims from the answer ──────────────
    extract_prompt = f"""Extract every discrete factual claim from this legal research answer.

A "claim" is any specific assertion that could be verified against a source document, such as:
- A case cited (e.g. "DOC_040 holds that the insurer cannot avoid third-party liability")
- A legal principle stated (e.g. "Section 149 creates a statutory obligation on the insurer")
- A compensation figure or calculation mentioned
- A court's specific ruling or holding attributed to a document

Return ONLY a JSON array of claim strings. No markdown, no explanation.
Limit to the 10 most specific and verifiable claims.
Keep each claim under 25 words.
Inside a claim string, use single quotes ('...') instead of double quotes to avoid breaking JSON.

ANSWER:
{final_answer[:3000]}"""

    try:
        raw = simple_chat(extract_prompt, temperature=0.0, max_tokens=2000)
        claims_raw = _safe_json_loads(raw)
        if not isinstance(claims_raw, list):
            claims_raw = []
        claims: List[str] = [str(c).strip() for c in claims_raw if str(c).strip()][:10]
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        return 0.0, f"Claim extraction failed: {exc}", []

    if not claims:
        return 1.0, "No verifiable claims found in the answer.", []

    logger.info("  Faithfulness: verifying %d extracted claims", len(claims))

    # ── Step B: Verify each claim against the retrieved chunks ───────────────
    sources_block = "\n\n---\n\n".join(
        f"[{c['doc_id']}] chunk {c.get('metadata', {}).get('chunk_index', '?')}:\n"
        f"{c.get('text_snippet', '')[:500]}"
        for c in retrieved_chunks[:15]
    )

    claims_block = "\n".join(f"{i+1}. {claim}" for i, claim in enumerate(claims))

    verify_prompt = f"""You are a legal fact-checker. For each claim below, determine whether
it is supported by the source documents provided.

Verdict options:
- "supported"           : the claim is explicitly stated in the sources
- "partially_supported" : the claim is implied or partially present in the sources
- "unsupported"         : the claim is NOT found in the sources (potential hallucination)

Return ONLY a valid JSON array. Put "verdict" first in every object so the
score survives even if the response is truncated. Keep each "reasoning" under
20 words. Inside string values, use single quotes ('...') rather than double
quotes so the JSON is never broken by unescaped quotes.

[
  {{
    "verdict": "<supported|partially_supported|unsupported>",
    "claim": "<exact claim text>",
    "source_chunk_id": "<doc_id of the supporting chunk, or empty string>",
    "reasoning": "<one short sentence>"
  }}
]

CLAIMS TO VERIFY:
{claims_block}

SOURCE DOCUMENTS:
{sources_block}"""

    try:
        raw = simple_chat(verify_prompt, temperature=0.0, max_tokens=4000)
        verdicts_raw = _safe_json_loads(raw)
        if not isinstance(verdicts_raw, list):
            verdicts_raw = []
    except Exception as exc:
        logger.warning("Claim verification failed: %s", exc)
        return 0.0, f"Verification parse error: {exc}", []

    verdicts: List[ClaimVerdict] = []
    for v in verdicts_raw:
        verdicts.append(ClaimVerdict(
            claim=v.get("claim", ""),
            verdict=v.get("verdict", "unsupported"),
            source_chunk_id=v.get("source_chunk_id", ""),
            reasoning=v.get("reasoning", ""),
        ))

    # Score: supported=1.0, partially_supported=0.5, unsupported=0.0
    if not verdicts:
        return 0.0, "No verdicts returned.", []

    score_map = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}
    total_score = sum(score_map.get(v.verdict, 0.0) for v in verdicts)
    faithfulness = total_score / len(verdicts)

    n_supported   = sum(1 for v in verdicts if v.verdict == "supported")
    n_partial     = sum(1 for v in verdicts if v.verdict == "partially_supported")
    n_unsupported = sum(1 for v in verdicts if v.verdict == "unsupported")

    reasoning = (
        f"{n_supported} supported, {n_partial} partially supported, "
        f"{n_unsupported} unsupported out of {len(verdicts)} claims. "
        f"Score = {faithfulness:.2f}"
    )

    logger.info(
        "  Faithfulness: %d supported, %d partial, %d unsupported → %.2f",
        n_supported, n_partial, n_unsupported, faithfulness,
    )

    return round(faithfulness, 3), reasoning, verdicts


# ---------------------------------------------------------------------------
# Dimension 6 — Citation verification
# ---------------------------------------------------------------------------

def _extract_doc_ids_from_answer(answer: str) -> List[str]:
    """
    Extract all DOC_XXX references from the answer text.
    Matches patterns like DOC_001, DOC_040, doc_050 (case-insensitive).
    """
    matches = re.findall(r"\bDOC_\d+\b", answer, re.IGNORECASE)
    # Normalise to upper case and deduplicate while preserving order
    seen = set()
    unique = []
    for m in matches:
        key = m.upper()
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _extract_principle_for_doc(answer: str, doc_id: str) -> str:
    """
    Extract what the answer specifically attributes to a given doc_id.
    Returns a short phrase describing the principle claimed.

    Skips lines that are obvious structural/metadata noise (e.g.
    "- DOC_027 (score: 0.76)") and sentences that mention multiple
    doc_ids — those summarise several cases at once and don't represent
    a claim attributable to any single one.
    """
    # Find the sentence(s) in the answer that mention this doc_id.
    pattern = re.compile(
        rf"[^.!?\n]*\b{re.escape(doc_id)}\b[^.!?\n]*[.!?\n]?",
        re.IGNORECASE,
    )
    matches = pattern.findall(answer)

    # Filter out retrieval-result header noise.
    header_re = re.compile(
        rf"^[-*\s]*\b{re.escape(doc_id)}\b\s*\(score:", re.IGNORECASE,
    )
    doc_mention_re = re.compile(r"\b[Dd][Oo][Cc]_\d+\b")

    best: Optional[str] = None
    for candidate in matches:
        cand = candidate.strip()
        if not cand:
            continue
        if header_re.match(cand):
            continue  # "- DOC_027 (score: 0.76)" etc.
        # Prefer sentences that mention only this one doc — multi-doc
        # summary sentences cannot be attributed to a single citation.
        mentions = {m.upper() for m in doc_mention_re.findall(cand)}
        if mentions == {doc_id.upper()}:
            return cand[:400]
        # Otherwise remember as fallback if we find no single-mention sentence.
        if best is None:
            best = cand

    if best:
        return best[:400]
    return f"Cited in answer but context not extractable for {doc_id}"


def _build_principle_map(
    supporting: List[Dict], adverse: List[Dict]
) -> Dict[str, str]:
    """
    Build an authoritative {DOC_ID (uppercase): key_principle} map from the
    agent's own classification output. This is what the answer *intends* to
    attribute to each document, uncontaminated by the surrounding prose.
    """
    mapping: Dict[str, str] = {}
    for p in list(supporting) + list(adverse):
        doc_id = str(p.get("doc_id", "")).upper()
        if not doc_id:
            continue
        parts = []
        kp = (p.get("key_principle") or "").strip()
        if kp:
            parts.append(kp)
        reason = (p.get("relevance_reason") or "").strip()
        if reason:
            parts.append(reason)
        distinguishing = (p.get("distinguishing_argument") or "").strip()
        if distinguishing:
            parts.append(f"Distinguishing argument: {distinguishing}")
        if parts:
            mapping[doc_id] = " ".join(parts)[:600]
    return mapping


def score_citation_verification(
    final_answer: str,
    retriever: RetrieverService,
    supporting: Optional[List[Dict]] = None,
    adverse: Optional[List[Dict]] = None,
) -> tuple[float, str, List[CitationVerdict]]:
    """
    Citation verification checks that every DOC_XXX cited in the answer
    actually supports the principle attributed to it.

    Process:
      1. Extract all DOC_XXX references from the answer text using regex.
      2. For each cited doc_id, retrieve its full text from ChromaDB.
      3. Ask the LLM: does this document's text support the specific claim
         the answer attributes to it?

    Score = verified_citations / total_citations

    This directly catches the most dangerous RAG failure mode for legal
    tools: fabricated or misattributed case citations.
    """
    if not final_answer.strip():
        return 0.0, "Empty answer — no citations to verify.", []

    cited_ids = _extract_doc_ids_from_answer(final_answer)

    if not cited_ids:
        return 1.0, "No DOC_XXX citations found in the answer.", []

    logger.info("  Citation verification: found %d cited doc IDs: %s",
                len(cited_ids), cited_ids)

    # Authoritative per-doc principle mapping from the agent's own classifier.
    # Only populated on the precedent_research route; document_search queries
    # fall through to regex extraction below.
    principle_map = _build_principle_map(supporting or [], adverse or [])

    verdicts: List[CitationVerdict] = []

    for doc_id in cited_ids:
        # Prefer the classifier's key_principle for this specific doc;
        # fall back to regex extraction from the prose when absent.
        attributed_principle = principle_map.get(doc_id.upper()) \
            or _extract_principle_for_doc(final_answer, doc_id)

        # Fetch the document's actual chunks from ChromaDB.
        # ChromaDB `where` is case-sensitive, but the corpus may be stored
        # with lowercase doc_ids (e.g. "doc_029") while the answer cites the
        # uppercase form (e.g. "DOC_029"). Try both cases — whichever matches.
        doc_chunks = retriever.get_full_document_chunks(doc_id)
        if not doc_chunks:
            doc_chunks = retriever.get_full_document_chunks(doc_id.lower())
        if not doc_chunks:
            doc_chunks = retriever.get_full_document_chunks(doc_id.upper())

        if not doc_chunks:
            verdicts.append(CitationVerdict(
                doc_id=doc_id,
                attributed_principle=attributed_principle,
                verified=False,
                reasoning=f"Document {doc_id} not found in the corpus.",
            ))
            logger.warning("  Citation %s: NOT IN CORPUS", doc_id)
            continue

        # Concatenate up to first 8 chunks for verification context
        doc_text = "\n\n".join(c.text for c in doc_chunks[:8])[:4000]

        verify_prompt = f"""You are verifying a legal citation.

The answer attributes the following claim to document {doc_id}:
"{attributed_principle}"

Below is the actual text from document {doc_id}. Does this text support the claim above?

Answer YES or NO, then explain in one sentence.

Return ONLY valid JSON:
{{"verified": <true|false>, "reasoning": "<one sentence>"}}

DOCUMENT TEXT ({doc_id}):
{doc_text}"""

        try:
            raw = simple_chat(verify_prompt, temperature=0.0, max_tokens=400)
            data = _safe_json_loads(raw)
            if not isinstance(data, dict):
                data = {}
            verified  = bool(data.get("verified", False))
            reasoning = data.get("reasoning", "")
        except Exception as exc:
            verified  = False
            reasoning = f"Verification parse error: {exc}"

        verdicts.append(CitationVerdict(
            doc_id=doc_id,
            attributed_principle=attributed_principle,
            verified=verified,
            reasoning=reasoning,
        ))
        logger.info(
            "  Citation %s: %s — %s",
            doc_id,
            "VERIFIED" if verified else "FAILED",
            reasoning[:80],
        )

    if not verdicts:
        return 1.0, "No citations to verify.", []

    n_verified = sum(1 for v in verdicts if v.verified)
    score      = n_verified / len(verdicts)
    reasoning  = (
        f"{n_verified}/{len(verdicts)} citations verified. "
        f"Failed: {[v.doc_id for v in verdicts if not v.verified]}"
    )

    return round(score, 3), reasoning, verdicts


# ---------------------------------------------------------------------------
# Main evaluator — now runs all 6 dimensions
# ---------------------------------------------------------------------------

def evaluate_case(case: Dict, retriever: RetrieverService) -> EvalResult:
    logger.info("Running case: %s — %s", case["id"], case["description"])
    start = time.time()

    result = run_agent(case["query"])

    elapsed = time.time() - start
    logger.info("Agent completed in %.1fs", elapsed)

    route        = result.get("route", "")
    retrieved    = result.get("retrieved_chunks", [])
    supporting   = result.get("supporting_precedents", [])
    adverse      = result.get("adverse_precedents", [])
    final_answer = result.get("final_answer", "")

    logger.info(
        "Route: %s | Retrieved: %d | Supporting: %d | Adverse: %d",
        route, len(retrieved), len(supporting), len(adverse),
    )

    # Dim 1: Precision
    prec_score, prec_reason = score_precision(retrieved, case["query"], case)
    logger.info("Precision:              %.2f", prec_score)

    # Dim 2: Recall
    rec_score, rec_reason = score_recall(
        retrieved, case["query"], case.get("required_concepts", []), retriever
    )
    logger.info("Recall:                 %.2f", rec_score)

    # Dim 3: Reasoning quality
    rq_score, rq_feedback = score_reasoning_quality(
        final_answer, case["query"], supporting, adverse
    )
    logger.info("Reasoning quality:      %.2f", rq_score)

    # Dim 4: Adverse identification
    adv_score, adv_reason = score_adverse_identification(
        adverse, case.get("must_have_adverse", False), case["query"]
    )
    logger.info("Adverse ID:             %.2f", adv_score)

    # Dim 5: Faithfulness
    faith_score, faith_reason, claim_verdicts = score_faithfulness(
        final_answer, retrieved
    )
    logger.info("Faithfulness:           %.2f", faith_score)

    # Dim 6: Citation verification — pass classified precedents so the
    # verifier tests each doc against its own key_principle, not the
    # surrounding multi-doc summary sentence.
    cite_score, cite_reason, cite_verdicts = score_citation_verification(
        final_answer, retriever, supporting, adverse,
    )
    logger.info("Citation verification:  %.2f", cite_score)

    # Overall = equal-weight average across all 6 dimensions
    overall = (
        prec_score + rec_score + rq_score + adv_score + faith_score + cite_score
    ) / 6

    return EvalResult(
        case_id=case["id"],
        description=case["description"],
        route=route,
        n_retrieved=len(retrieved),
        n_supporting=len(supporting),
        n_adverse=len(adverse),
        precision_score=round(prec_score, 3),
        precision_reasoning=prec_reason,
        recall_score=round(rec_score, 3),
        recall_reasoning=rec_reason,
        reasoning_quality_score=round(rq_score, 3),
        reasoning_quality_feedback=rq_feedback,
        adverse_id_score=round(adv_score, 3),
        adverse_id_reasoning=adv_reason,
        faithfulness_score=round(faith_score, 3),
        faithfulness_reasoning=faith_reason,
        claim_verdicts=[asdict(v) for v in claim_verdicts],
        citation_score=round(cite_score, 3),
        citation_reasoning=cite_reason,
        citation_verdicts=[asdict(v) for v in cite_verdicts],
        overall_score=round(overall, 3),
        latency_seconds=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Full suite runner
# ---------------------------------------------------------------------------

def run_full_suite(output_path: str = "evaluation/eval_results.json"):
    retriever = RetrieverService()
    results   = []

    for case in EVAL_CASES:
        try:
            result = evaluate_case(case, retriever)
            results.append(asdict(result))
        except Exception as exc:
            logger.error("Case %s failed: %s", case["id"], exc)
            results.append({"case_id": case["id"], "error": str(exc)})

    valid = [r for r in results if "error" not in r]

    if valid:
        def avg(key: str) -> float:
            return round(sum(r[key] for r in valid) / len(valid), 3)

        summary = {
            "n_cases":                    len(EVAL_CASES),
            "n_successful":               len(valid),
            "avg_precision":              avg("precision_score"),
            "avg_recall":                 avg("recall_score"),
            "avg_reasoning_quality":      avg("reasoning_quality_score"),
            "avg_adverse_identification": avg("adverse_id_score"),
            "avg_faithfulness":           avg("faithfulness_score"),
            "avg_citation_verification":  avg("citation_score"),
            "avg_overall_score":          avg("overall_score"),
            "avg_latency_seconds":        avg("latency_seconds"),
        }

        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        for k, v in summary.items():
            logger.info("  %-35s: %s", k, v)
        logger.info("=" * 60)
    else:
        summary = {"error": "All cases failed"}

    output = {"summary": summary, "results": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Lexi agent evaluation")
    parser.add_argument("--query",      type=str,  help="Single custom query")
    parser.add_argument("--case-id",    type=str,  help="Run a specific case by ID")
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--output",     default="evaluation/eval_results.json")
    args = parser.parse_args()

    if args.query:
        retriever = RetrieverService()
        custom_case = {
            "id": "custom",
            "description": "Custom query",
            "query": args.query,
            "required_concepts": [],
            "adverse_concepts": [],
        }
        result = evaluate_case(custom_case, retriever)
        logger.info(json.dumps(asdict(result), indent=2))

    elif args.case_id:
        retriever = RetrieverService()
        case = next((c for c in EVAL_CASES if c["id"] == args.case_id), None)
        if not case:
            logger.error(
                "Case '%s' not found. Available: %s",
                args.case_id, [c["id"] for c in EVAL_CASES],
            )
        else:
            result = evaluate_case(case, retriever)
            logger.info(json.dumps(asdict(result), indent=2))

    else:
        run_full_suite(args.output)