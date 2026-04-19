"""
LLM client using LiteLLM gateway.
Primary:  groq/llama-3.3-70b-versatile   (Groq free tier — fast, but 100k TPD)
Fallback: gemini/gemini-2.0-flash        (Google AI Studio free tier — 1M TPM,
                                          1500 RPD, separate bucket from Groq)

Using two *different providers* for primary and fallback is deliberate: when
Groq's per-minute or per-day cap is hit, the fallback needs to live in an
independent bucket. Earlier versions used groq/llama-3.1-8b-instant as the
fallback and quickly discovered that org-level TPD is shared across Groq
models — a daily cap on the 70B cascaded straight into a cap on the 8B.

This client:

1. Retries on rate-limit errors (429 / RateLimitError), sleeping for the
   duration the server hints in the error body ("try again in X.XXs" for
   Groq, "retry_delay { seconds: N }" for Gemini).
2. When primary's retries are exhausted (or a TPD hint exceeds our per-retry
   cap), cascades to the fallback provider.
3. Honours authentication errors (401) immediately without retrying.
"""
from __future__ import annotations

import os
import re
import time
import random
from typing import Optional, List, Dict

import litellm
from litellm import completion

from logger import get_logger

logger = get_logger(__name__)

litellm.suppress_debug_info = True

# Primary: Groq Llama 3.3 70B — fast inference, strong JSON compliance.
# Fallback: Gemini 2.0 Flash — independent quota bucket, generous RPD/TPM.
PRIMARY_MODEL  = "groq/llama-3.3-70b-versatile"
FALLBACK_MODEL = "gemini/gemini-2.0-flash"

# Per-call timeout (seconds). Prevents a hung request from stalling Streamlit.
LLM_TIMEOUT = 60

# Retry policy for transient errors (rate limits, 5xx).
MAX_RETRIES = 4              # total attempts per model = MAX_RETRIES + 1
MAX_SLEEP_PER_RETRY = 35.0   # cap a single server-hinted sleep (seconds)
DEFAULT_BACKOFF_BASE = 2.0   # exponential fallback when no hint is present


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------

def _resolve_api_key(model: str) -> Optional[str]:
    """Return the credential LiteLLM needs for the given model's provider."""
    if model.startswith("groq/"):
        return os.environ.get("GROQ_API_KEY")
    if model.startswith("gemini/"):
        # LiteLLM accepts either GEMINI_API_KEY or GOOGLE_API_KEY.
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return None


def _missing_key_error(model: str) -> EnvironmentError:
    if model.startswith("gemini/"):
        return EnvironmentError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )
    return EnvironmentError(
        "GROQ_API_KEY not set. Please add it to your .env file."
    )


# ---------------------------------------------------------------------------
# Error classification + retry hint parsing
# ---------------------------------------------------------------------------

# Matches Groq's "Please try again in 5.73s" / "150ms" / "4.144999999s" hints.
_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*(ms|s)\b", re.IGNORECASE)

# Matches Gemini's "retry_delay { seconds: 42 }" (or "retryDelay") hint.
_GEMINI_RETRY_RE = re.compile(
    r"retry[_ ]?delay[^\d]*(\d+)(?:\s*\.\s*(\d+))?\s*(?:seconds?|s)?",
    re.IGNORECASE,
)


def _is_rate_limit_error(err: Exception) -> bool:
    name = type(err).__name__.lower()
    msg = str(err).lower()
    return (
        "ratelimit" in name
        or "rate_limit_exceeded" in msg
        or "rate limit" in msg
        or '"code":"rate_limit_exceeded"' in msg
        or " 429 " in f" {msg} "
    )


def _is_auth_error(err: Exception) -> bool:
    """Authentication/permission errors should never be retried."""
    msg = str(err).lower()
    return (
        "invalid_api_key" in msg
        or "invalid api key" in msg
        or "unauthorized" in msg
        or "401" in msg
        or "authenticationerror" in type(err).__name__.lower()
    )


def _parse_server_retry_hint(err_msg: str) -> Optional[float]:
    """
    Extract the server's hinted retry delay from a 429 error body.
    Handles Groq's "Please try again in X.Xs" / "XXXms" form and Gemini's
    "retry_delay { seconds: N }" form.
    """
    match = _RETRY_AFTER_RE.search(err_msg)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        seconds = value / 1000.0 if unit == "ms" else value
        return max(seconds, 0.0)

    gm = _GEMINI_RETRY_RE.search(err_msg)
    if gm:
        whole = gm.group(1)
        frac = gm.group(2) or "0"
        return max(float(f"{whole}.{frac}"), 0.0)

    return None


def _is_daily_quota_error(err_msg: str) -> bool:
    """
    Detect daily-quota errors (not per-minute).
    Groq:   "tokens per day" / "TPD"
    Gemini: "PerDay" in the quota violation, or "quota exceeded" with a large
            retry delay (parsed separately via the hint cap).
    """
    lowered = err_msg.lower()
    if "tokens per day" in lowered:
        return True
    if '"type":"tokens"' in lowered and "tpd" in lowered:
        return True
    if "perday" in lowered or "per day" in lowered:
        return True
    return False


def _compute_sleep(err: Exception, attempt: int) -> Optional[float]:
    """
    Decide how long to wait before the next retry.
    Prefers the server-hinted wait; otherwise uses exponential backoff
    with a small random jitter.

    Returns None when the hint exceeds our per-retry cap — the caller should
    give up rather than spin. Typical cause: Groq's tokens-per-day (TPD)
    ceiling, which returns "try again in 10m15s" — no amount of retrying
    within a single eval run will rescue it.
    """
    hinted = _parse_server_retry_hint(str(err))
    if hinted is not None:
        # Add 0.5s safety buffer on top of Groq's hint — the counter hasn't
        # fully reset at exactly that moment in practice.
        sleep = hinted + 0.5
        # If the hint is beyond our cap, signal "don't retry" to the caller.
        if sleep > MAX_SLEEP_PER_RETRY:
            return None
        return sleep
    # No hint — fall back to exponential backoff with jitter.
    sleep = DEFAULT_BACKOFF_BASE ** attempt  # 1, 2, 4, 8, ...
    sleep += random.uniform(0, 0.5)          # jitter to avoid thundering herd
    return min(sleep, MAX_SLEEP_PER_RETRY)


# ---------------------------------------------------------------------------
# Core call with retry
# ---------------------------------------------------------------------------

def _call_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    api_key: Optional[str],
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Invoke `completion()` with retry on transient rate-limit errors.
    Auth errors short-circuit immediately. Other errors propagate after
    one retry attempt so the outer `chat()` can decide to fall back.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                timeout=LLM_TIMEOUT,
            )
            return response.choices[0].message.content or ""
        except Exception as err:
            last_err = err

            # Never retry bad credentials — it'll never succeed.
            if _is_auth_error(err):
                raise

            # For rate limits, keep retrying up to max_retries.
            if _is_rate_limit_error(err):
                if attempt >= max_retries:
                    logger.warning(
                        "Rate-limit retries exhausted for %s after %d attempts.",
                        model, attempt + 1,
                    )
                    raise
                sleep_s = _compute_sleep(err, attempt)
                # sleep_s is None when the server hint is longer than our cap
                # (e.g. daily quota = "try again in 10m15s"). Retrying is futile.
                if sleep_s is None:
                    if _is_daily_quota_error(str(err)):
                        logger.warning(
                            "%s daily token quota exhausted — skipping retries.",
                            model,
                        )
                    else:
                        logger.warning(
                            "%s rate-limit hint exceeds per-retry cap — skipping retries.",
                            model,
                        )
                    raise
                logger.info(
                    "Rate limit on %s (attempt %d/%d) — sleeping %.2fs before retry.",
                    model, attempt + 1, max_retries + 1, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            # Non-auth, non-rate-limit error: propagate so caller can fall back.
            raise

    # Defensive — should be unreachable.
    assert last_err is not None
    raise last_err


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    messages: List[Dict[str, str]],
    model: str = PRIMARY_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call LLM and return the complete text response. Retries on rate limits, then falls back."""
    api_key = _resolve_api_key(model)
    if not api_key and (model.startswith("groq/") or model.startswith("gemini/")):
        raise _missing_key_error(model)

    try:
        return _call_with_retry(model, messages, temperature, max_tokens, api_key)
    except Exception as primary_err:
        # Auth errors against the primary no longer block cascade: primary and
        # fallback now live on different providers with different keys, so a
        # bad GROQ_API_KEY is still recoverable via Gemini (and vice versa).
        # Only short-circuit if we were already on the fallback model.
        if model == FALLBACK_MODEL:
            raise

        fallback_key = _resolve_api_key(FALLBACK_MODEL)
        if not fallback_key and (
            FALLBACK_MODEL.startswith("groq/") or FALLBACK_MODEL.startswith("gemini/")
        ):
            raise _missing_key_error(FALLBACK_MODEL) from primary_err

        logger.info("Primary %s failed, trying fallback %s.", model, FALLBACK_MODEL)
        try:
            return _call_with_retry(
                FALLBACK_MODEL, messages, temperature, max_tokens, fallback_key
            )
        except Exception as fallback_err:
            raise RuntimeError(
                f"Both models failed.\nPrimary: {primary_err}\nFallback: {fallback_err}"
            )


def simple_chat(prompt: str, system: Optional[str] = None, **kwargs) -> str:
    """Convenience wrapper for single-turn prompts."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs)