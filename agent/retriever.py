"""
Retrieval module.
Provides hybrid retrieval over the ChromaDB collection with the following
improvements over the original implementation:
1. BM25 reranking replaces naive TF keyword boost.
   BM25 (Okapi BM25) is the industry standard for keyword relevance. Unlike
   raw term frequency, it normalises for document length and applies IDF —
   rare legal terms like "Section 149" or "unlicensed driver" score higher
   than common words like "court" or "held". This significantly improves
   precision on specific legal queries.
2. Context window expansion (±1 neighbouring chunks).
   When a chunk is identified as relevant, its immediate neighbours
   (chunk_index - 1 and chunk_index + 1) are fetched and prepended/appended
   to the matched text before returning. Legal reasoning almost always spans
   multiple sentences — a holding in paragraph 12 references facts in
   paragraph 11. Expanding the context window gives the LLM the full picture
   without retrieving extra documents.
   Latency cost: 2 point lookups per top result (~5ms total for n=10).
3. Top-2 chunks per document instead of single-chunk deduplication.
   The original code kept only the single highest-scoring chunk per doc_id.
   For a 15-page judgment, the relevant holding and the relevant compensation
   figure may be in different places. Keeping the top 2 chunks per document
   (above a minimum score threshold) captures both without flooding the
   context with every chunk from a document.
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from logger import get_logger

logger = get_logger(__name__)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "judgments"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Weight given to BM25 score vs semantic score during hybrid fusion.
# 0.3 means: final_score = 0.7 * semantic + 0.3 * bm25_normalised
# Semantic still dominates; BM25 provides a meaningful boost for exact terms.
BM25_WEIGHT = 0.3

# Minimum score a chunk must have to be included in the second slot
# when keeping top-2 chunks per document.
MIN_SCORE_SECOND_CHUNK = 0.35

# Stopwords excluded from BM25 tokenisation
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "in", "of", "to", "and", "or", "for",
    "that", "with", "on", "at", "by", "it", "as", "be", "was", "are",
    "were", "has", "have", "had", "this", "its", "from", "which", "not",
})


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    score: float
    chunk_index: int = 0
    metadata: Dict = field(default_factory=dict)


@lru_cache(maxsize=4)  # Cache up to 4 different chroma_path values
def _get_retriever_service(chroma_path: str) -> "RetrieverService":
    """
    Module-level cached factory for RetrieverService.
    
    The SentenceTransformer model is loaded once per chroma_path and reused
    across all subsequent calls. maxsize=4 accommodates multiple corpus
    configurations without unbounded memory growth.
    """
    logger.info("Initializing RetrieverService for chroma_path=%s (model loading occurs here)", chroma_path)
    return RetrieverService(chroma_path)


def get_retriever(chroma_path: Optional[str] = None) -> "RetrieverService":
    """
    Get a cached RetrieverService instance. Resolution order:
      1. explicit chroma_path argument
      2. CHROMA_PATH env var (set by the Streamlit sidebar in app.py)
      3. default "chroma_db"

    Use this instead of directly instantiating RetrieverService() in graph
    nodes or UI components — the sentence-transformers model is only loaded
    once per path.
    """
    resolved = chroma_path or os.environ.get("CHROMA_PATH") or CHROMA_PATH
    return _get_retriever_service(resolved)


class RetrieverService:
    def __init__(self, chroma_path: str = CHROMA_PATH):
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self._collection = self._client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )
        self._chroma_path = chroma_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """
        Hybrid retrieval pipeline:
          1. Dense semantic search (ChromaDB / sentence-transformers)
          2. BM25 reranking with score fusion
          3. Top-2 deduplication per document
          4. Context window expansion (±1 neighbouring chunks)

        Returns top-n chunks sorted by combined score, each expanded with
        neighbouring chunk text for richer LLM context.
        """
        # Step 1: semantic search — fetch 3× candidates for reranking headroom
        candidates = self._semantic_search(
            query, n_results=n_results * 3, where=where
        )
        if not candidates:
            return []

        # Step 2: BM25 reranking with score fusion
        reranked = self._bm25_rerank(query, candidates)
        
        # Step 3: top-2 deduplication per document
        deduped = self._dedup_top2(reranked, n_results=n_results)

        # Step 4: context window expansion
        expanded = self._expand_context(deduped)

        logger.debug(
            "retrieve(): query=%r candidates=%d deduped=%d expanded=%d",
            query[:60], len(candidates), len(deduped), len(expanded),
        )
        return expanded

    def get_full_document_chunks(self, doc_id: str) -> List[RetrievedChunk]:
        """Return all chunks belonging to a specific doc_id, in order."""
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        chunks = []
        for cid, doc, meta in zip(
            results["ids"], results["documents"], results["metadatas"]
        ):
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                text=doc,
                score=1.0,
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=meta,
            ))
        return sorted(chunks, key=lambda c: c.chunk_index)

    def collection_size(self) -> int:
        return self._collection.count()

    def list_document_ids(self) -> List[str]:
        """Return unique doc_ids in the collection."""
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return list({m.get("doc_id", "") for m in all_meta if m.get("doc_id")})

    # ------------------------------------------------------------------
    # Step 1 — Dense semantic search
    # ------------------------------------------------------------------

    def _semantic_search(
        self,
        query: str,
        n_results: int,
        where: Optional[Dict],
    ) -> List[RetrievedChunk]:
        total = self._collection.count()
        if total == 0:
            return []

        kwargs: Dict = dict(
            query_texts=[query],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        chunks = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite → [0, 1]
            score = max(0.0, 1.0 - (dist / 2.0))
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                text=doc,
                score=score,
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=meta,
            ))
        return chunks

    # ------------------------------------------------------------------
    # Step 2 — BM25 reranking with score fusion
    # ------------------------------------------------------------------

    def _bm25_rerank(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        Compute BM25 scores over the candidate chunks and fuse with semantic
        scores using a weighted sum.
        BM25 advantages over raw TF:
        - IDF component: rare query terms (e.g. "Section 149") score much
          higher than common terms (e.g. "court", "held").
        - Length normalisation: long chunks are not unfairly boosted just
          because they contain more words.
        """
        # Tokenise — lowercase, split on non-word chars, remove stopwords
        def tokenise(text: str) -> List[str]:
            tokens = re.findall(r"\b\w+\b", text.lower())
            return [t for t in tokens if t not in _STOPWORDS]

        corpus_tokens = [tokenise(c.text) for c in chunks]
        query_tokens  = tokenise(query)

        bm25 = BM25Okapi(corpus_tokens)
        raw_scores = bm25.get_scores(query_tokens)

        # Normalise BM25 scores to [0, 1]
        max_bm25 = max(raw_scores) if max(raw_scores) > 0 else 1.0
        norm_scores = [s / max_bm25 for s in raw_scores]

        # Weighted fusion: semantic dominates, BM25 boosts exact-term matches
        for chunk, bm25_norm in zip(chunks, norm_scores):
            chunk.score = (
                (1 - BM25_WEIGHT) * chunk.score +
                BM25_WEIGHT * bm25_norm
            )

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    # ------------------------------------------------------------------
    # Step 3 — Top-2 deduplication per document
    # ------------------------------------------------------------------

    def _dedup_top2(
        self, chunks: List[RetrievedChunk], n_results: int
    ) -> List[RetrievedChunk]:
        """
        Keep at most 2 chunks per document.
        The second chunk is only kept if its score exceeds MIN_SCORE_SECOND_CHUNK,
        preventing low-quality chunks from a highly-ranked document from
        crowding out higher-quality chunks from other documents.
        """
        seen: Dict[str, int] = {}
        kept: List[RetrievedChunk] = []

        for chunk in chunks:
            count = seen.get(chunk.doc_id, 0)
            if count == 0:
                kept.append(chunk)
                seen[chunk.doc_id] = 1
            elif count == 1 and chunk.score >= MIN_SCORE_SECOND_CHUNK:
                kept.append(chunk)
                seen[chunk.doc_id] = 2
            # 3rd+ chunks from same doc are always skipped
        return sorted(kept, key=lambda c: c.score, reverse=True)[:n_results]

    # ------------------------------------------------------------------
    # Step 4 — Context window expansion (±1 neighbouring chunks)
    # ------------------------------------------------------------------

    def _expand_context(
        self, chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """
        For each retrieved chunk, prepend the previous chunk's text and
        append the next chunk's text (if they exist in the same document).
        This gives the LLM the sentence before and after the matched passage —
        critical for legal reasoning where a holding references preceding facts.
        The chunk's score and metadata remain unchanged; only the text is
        expanded. The original matched text is kept as-is in the middle.
        """
        expanded = []
        for chunk in chunks:
            prev_chunk = self._get_chunk_by_index(
                chunk.doc_id, chunk.chunk_index - 1
            )
            next_chunk = self._get_chunk_by_index(
                chunk.doc_id, chunk.chunk_index + 1
            )

            parts = []
            if prev_chunk:
                parts.append(prev_chunk.text)
            parts.append(chunk.text)
            if next_chunk:
                parts.append(next_chunk.text)

            expanded_text = "\n\n".join(parts)

            expanded.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                text=expanded_text,
                score=chunk.score,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            ))

        return expanded

    def _get_chunk_by_index(
        self, doc_id: str, chunk_index: int
    ) -> Optional[RetrievedChunk]:
        """
        Fetch a single chunk by doc_id + chunk_index.
        Returns None if the chunk does not exist (e.g. index out of range).
        These are fast point lookups — no embedding computation involved.
        """
        if chunk_index < 0:
            return None
        try:
            results = self._collection.get(
                where={
                    "$and": [
                        {"doc_id":      {"$eq": doc_id}},
                        {"chunk_index": {"$eq": chunk_index}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            if not results["ids"]:
                return None
            meta = results["metadatas"][0]
            return RetrievedChunk(
                chunk_id=results["ids"][0],
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                text=results["documents"][0],
                score=0.0,
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata=meta,
            )
        except Exception:
            return None