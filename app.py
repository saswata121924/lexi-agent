"""
Lexi Legal Research Agent — entry point.

Responsibilities of this file:
  - Bootstrap logging
  - Configure the Streamlit page
  - Wire together the agent, ingestion, and UI components

All st.* rendering calls live in ui/components.py.
All agent logic lives in agent/.
All ingestion logic lives in ingestion/.
"""

import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from logger import configure_logging, get_logger
configure_logging(level=logging.INFO)
logger = get_logger(__name__)

from ui.components import (
    apply_page_config,
    apply_styles,
    render_header,
    render_sidebar,
    render_query_input,
    render_results,
)

from agent.graph import run_agent
from ingestion.ingest import ingest_documents

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

apply_page_config()

# ---------------------------------------------------------------------------
# Secrets bridge — Streamlit Cloud stores keys in st.secrets, not env vars.
# Bridge them to os.environ so the rest of the app (LiteLLM) sees them.
# ---------------------------------------------------------------------------

try:
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
except Exception:
    # st.secrets raises if no secrets.toml exists locally — safe to ignore.
    pass

# ---------------------------------------------------------------------------
# Styles & static UI
# ---------------------------------------------------------------------------

apply_styles()
render_header()

# ---------------------------------------------------------------------------
# Sidebar — returns user-configured paths
# ---------------------------------------------------------------------------

docs_dir, chroma_path = render_sidebar(ingest_fn=ingest_documents)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

query, run_clicked = render_query_input()

# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------

if run_clicked and query.strip():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not set. Add it to your .env file and restart the app.")
        st.stop()

    os.environ["CHROMA_PATH"] = chroma_path

    with st.spinner("Agent is thinking..."):
        try:
            result = run_agent(query.strip())
            st.session_state["last_result"] = result
            logger.info("Agent run complete. Route: %s", result.get("route", ""))
        except Exception as exc:
            logger.error("Agent run failed: %s", exc)
            st.error(f"Agent error: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "last_result" in st.session_state:
    render_results(st.session_state["last_result"])