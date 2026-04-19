# Corpus folder

This folder holds the 50 Indian court-judgment PDFs the agent retrieves over.
The PDFs are **checked into the repository** so the deployed Streamlit Cloud
app ships with a working corpus and the sidebar **Ingest Docs** button is
functional for users clicking through the demo.

> If you fork this repo into a public space, confirm that redistributing the
> corpus PDFs is permitted under whatever terms you received them. The
> Lexi-provided assessment corpus is intended for evaluation use.

## Replacing the corpus

1. Drop your PDFs into this directory (either `DOC_001.pdf` or `doc_001.pdf`
   style — the ingester is case-insensitive).
2. From the repo root, re-run ingestion so `chroma_db/` matches the new
   corpus:

   ```bash
   python -m ingestion.ingest --reset
   ```

3. Commit both `docs/` and `chroma_db/` so the deployed app picks them up.

## Notes

- The ingest script (`ingestion/ingest.py`) accepts either case on filenames —
  `DOC_001.pdf` and `doc_001.pdf` are both picked up.
- `doc_id` metadata is derived from the filename stem, so the case used here
  will appear in citations (`DOC_012` vs `doc_012`).
- After any change to chunking parameters, re-run with `--reset`.
