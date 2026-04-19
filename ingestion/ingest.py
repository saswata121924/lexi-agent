"""
Document ingestion pipeline.
Loads PDF judgments from docs/, chunks them, embeds with sentence-transformers,
and stores in a persistent ChromaDB collection.

TEXT CLEANING (regex — patterns confirmed universal across ALL corpus docs):
  All corpus PDFs are sourced from Indian Kanoon. Every page of every document
  carries two boilerplate lines that add zero legal content but pollute chunk
  embeddings by appearing repeatedly across all documents:

  1. Repeating case title line (page header on every page):
       "A.V. Janaki Amma And Ors. vs Union Of India (Uoi) And Ors. on 23 October, 2003"
     Pattern: <appellant> vs <respondent> on <DD Month, YYYY>

  2. Indian Kanoon URL footer (page footer on every page):
       "Indian Kanoon - http://indiankanoon.org/doc/1631272/ 3"

  Both are stripped before chunking.

METADATA EXTRACTION (regex):
  The title line pattern yields three metadata fields reliably across all
  document types: appellant, respondent, year.

CHUNKING IMPROVEMENTS:
  - Chunk size increased from 800 → 1200 chars.
    A complete Indian court paragraph making a legal point typically runs
    200–400 words (~1200–2400 chars). At 800 chars most paragraphs were
    being cut in half — the embedding of half a legal argument is a poor
    representation. 1200 chars fits one complete legal reasoning unit.

  - Paragraph-boundary-aware separators.
    Indian court judgments consistently use numbered paragraphs and section
    headers (HELD, ORDER, FACTS, ISSUE). The separator list now prioritises
    these boundaries so numbered paragraphs stay intact as single chunks
    rather than being cut mid-sentence by a fixed character limit.

  - Overlap kept at 200 chars.
    Sufficient to carry over the trailing sentence of the previous paragraph
    into the next chunk for context continuity.
"""

import re
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Tuple

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from logger import get_logger

logger = get_logger(__name__)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "judgments"
DOCS_DIR = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
# NOTE: sentence-transformers downloads ~80MB model on first run (cached after that).

# Increased from 800 → 1200 to fit a complete legal paragraph per chunk.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Boilerplate pattern 1 — Indian Kanoon URL footer
_KANOON_URL_RE = re.compile(
    r"Indian Kanoon\s*[-\u2013]\s*https?://\S+\s*\d*",
    re.IGNORECASE,
)

# Boilerplate pattern 2 — Repeating case title line
# Used for BOTH cleaning (strip every occurrence) and metadata (capture first).
_CASE_TITLE_RE = re.compile(
    r"^(?P<appellant>.{5,120}?)\s+vs?\.?\s+(?P<respondent>.{3,120}?)\s+on\s+"
    r"(?P<day>\d{1,2})\s+(?P<month>\w+),?\s+(?P<year>\d{4})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# ---------------------------------------------------------------------------
# Paragraph-aware splitter
# Separators ordered from highest to lowest priority.
# Legal section headers (HELD, ORDER, FACTS, ISSUE) and numbered paragraphs
# are tried before falling back to generic newlines and spaces.
# ---------------------------------------------------------------------------
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",          # blank line between paragraphs — highest priority
        r"\n\d+\.\s",    # numbered paragraphs: "12. The court held..."
        "\nHELD",        # holding section header
        "\nORDER",       # order section header
        "\nFACTS",       # facts section header
        "\nISSUE",       # issue section header
        "\nREASONS",     # reasons section header
        "\nJUDGMENT",    # judgment section header
        "\n",
        ". ",
        " ",
        "",
    ],
    is_separator_regex=True,
)


# ---------------------------------------------------------------------------
# Text extraction & cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Strip the two universal Indian Kanoon boilerplate patterns from a page
    and collapse the resulting blank lines.
    """
    text = _KANOON_URL_RE.sub("", text)
    text = _CASE_TITLE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str]:
    """
    Extract text from all pages of a PDF.

    Returns:
        (raw_first_page, cleaned_full_text)
        raw_first_page: used for metadata extraction before boilerplate removal.
        cleaned_full_text: used for chunking.
    """
    reader = PdfReader(pdf_path)
    raw_first_page = ""
    cleaned_pages = []

    for i, page in enumerate(reader.pages):
        raw = page.extract_text()
        if not raw:
            continue
        if i == 0:
            raw_first_page = raw
        cleaned_pages.append(_clean_text(raw))

    return raw_first_page, "\n\n".join(cleaned_pages)


# ---------------------------------------------------------------------------
# Regex-based metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(raw_first_page: str, filename: str) -> Dict:
    """
    Extract document metadata from the raw (uncleaned) first page.

    Fields extracted:
        doc_id      — filename stem, always present
        source      — original filename, always present
        appellant   — first named party, from title line
        respondent  — second named party, from title line
        year        — 4-digit judgment year, from title line
    """
    meta: Dict = {
        "source": filename,
        "doc_id": Path(filename).stem,
    }

    match = _CASE_TITLE_RE.search(raw_first_page)
    if match:
        meta["appellant"]  = match.group("appellant").strip()
        meta["respondent"] = match.group("respondent").strip()
        meta["year"]       = match.group("year")
    else:
        logger.warning("Title line pattern not found in first page of %s", filename)

    return meta


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_documents(
    docs_dir: str = DOCS_DIR,
    chroma_path: str = CHROMA_PATH,
    reset: bool = False,
):
    """
    Ingest all PDFs from docs_dir into ChromaDB.

    Args:
        docs_dir:    Folder containing PDF files.
        chroma_path: ChromaDB storage path.
        reset:       Wipe the collection before ingesting.
    """
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        logger.error(
            "No PDF files found in '%s/'. Please add your court judgment PDFs there.",
            docs_dir,
        )
        return 0

    logger.info("Found %d PDF files.", len(pdf_files))

    client   = chromadb.PersistentClient(path=chroma_path)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get()["ids"])
    logger.info("Collection has %d existing chunks.", len(existing_ids))

    total_chunks = 0

    for pdf_path in pdf_files:
        logger.info("Processing %s ...", pdf_path.name)
        try:
            raw_first_page, cleaned_text = extract_text_from_pdf(str(pdf_path))
        except Exception as exc:
            logger.warning("Could not read %s: %s", pdf_path.name, exc)
            continue

        if not cleaned_text.strip():
            logger.warning("Empty text extracted from %s", pdf_path.name)
            continue

        metadata = extract_metadata(raw_first_page, pdf_path.name)
        logger.info("  Metadata for %s:", pdf_path.name)
        for key, value in metadata.items():
            logger.info("    %-15s: %s", key, value)

        chunks = _SPLITTER.split_text(cleaned_text)
        ids, documents, metadatas = [], [], []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{pdf_path.name}_{i}".encode()).hexdigest()
            if chunk_id in existing_ids:
                continue
            ids.append(chunk_id)
            documents.append(chunk)
            chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            # ChromaDB requires str / int / float / bool values only
            chunk_meta = {
                k: (str(v) if not isinstance(v, (int, float, bool)) else v)
                for k, v in chunk_meta.items()
            }
            metadatas.append(chunk_meta)

        if ids:
            batch_size = 100
            for start in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[start:start + batch_size],
                    documents=documents[start:start + batch_size],
                    metadatas=metadatas[start:start + batch_size],
                )
            total_chunks += len(ids)
            logger.info("  Added %d chunks from %s", len(ids), pdf_path.name)
        else:
            logger.info("  All chunks already indexed for %s", pdf_path.name)

    logger.info(
        "Ingestion complete. Added %d new chunks. Total in collection: %d",
        total_chunks,
        collection.count(),
    )
    return collection.count()


if __name__ == "__main__":
    import logging
    from logger import configure_logging
    configure_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Ingest court judgment PDFs into ChromaDB"
    )
    parser.add_argument("--docs",   default=DOCS_DIR,    help="Folder with PDFs")
    parser.add_argument("--chroma", default=CHROMA_PATH, help="ChromaDB path")
    parser.add_argument("--reset",  action="store_true", help="Wipe before ingesting")
    args = parser.parse_args()

    ingest_documents(
        docs_dir=args.docs,
        chroma_path=args.chroma,
        reset=args.reset,
    )