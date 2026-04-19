"""
Centralised logging configuration for the Lexi Legal Research Agent.

All modules import `get_logger(__name__)` from here.
A single call to `configure_logging()` at startup sets the format,
level, and handlers for every logger in the project.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)

    logger.info("Collection has %d chunks", count)
    logger.warning("Empty text from %s", filename)
    logger.error("Could not read %s: %s", path, exc)
    logger.debug("Chunk %d/%d processed", i, total)
"""

import logging
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO, logfile: Optional[str] = None) -> None:
    """
    Configure the root logger once at application startup.

    Args:
        level:   Logging level (default INFO). Use logging.DEBUG for verbose output.
        logfile: Optional path to write logs to a file in addition to stdout.
    """
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers already attached (e.g. from a previous configure call)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers",
                  "transformers", "urllib3", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        logging.Logger instance namespaced under 'name'.
    """
    return logging.getLogger(name)