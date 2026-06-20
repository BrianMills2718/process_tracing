"""Single LLM call abstraction — delegates to llm_client structured output.

The boundary is intentionally thin: it builds the system+user messages and hands
off to ``llm_client.call_llm_structured``, which owns schema enforcement (native
JSON-schema where the provider supports it, instructor fallback otherwise),
transient-error retry, and validated Pydantic parsing. We do not hand-roll schema
injection or JSON parsing here. Fail-loud: any failure raises ``LLMError``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from llm_client import call_llm_structured
from pydantic import BaseModel

# Load API keys from shared secrets, then project .env for overrides
_secrets = Path.home() / ".secrets" / "api_keys.env"
if _secrets.exists():
    load_dotenv(_secrets)
load_dotenv(override=True)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("PT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

MAX_RETRIES = 3

_SYSTEM = (
    "You are an expert political scientist and historian performing Van Evera "
    "process tracing analysis. Respond with valid JSON matching the requested "
    "schema exactly. Keep your JSON output concise—use brief descriptions, not "
    "lengthy prose."
)


class LLMError(Exception):
    """Raised when an LLM call or its structured parse fails. No fallbacks—fail fast."""


def call_llm(
    prompt: str,
    response_model: type[T],
    *,
    task: str,
    trace_id: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_budget: float = 0,
    system: str = _SYSTEM,
) -> T:
    """Call the LLM and return a validated instance of ``response_model``.

    Delegates structured-output handling (schema enforcement, instructor
    fallback, transient retry, parsing) to ``llm_client.call_llm_structured``.
    Raises ``LLMError`` on any failure (transient exhaustion, validation, etc.).
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    t0 = time.time()
    try:
        parsed, result = call_llm_structured(
            model,
            messages,
            response_model,
            num_retries=MAX_RETRIES,
            temperature=temperature,
            task=task,
            trace_id=trace_id,
            max_budget=max_budget,
        )
    except Exception as e:  # fail loud — no silent fallback
        raise LLMError(f"LLM structured call failed: {e}") from e

    elapsed = time.time() - t0
    model_used = getattr(result, "model", model)
    finish = getattr(result, "finish_reason", "?")
    print(f"  LLM call: {elapsed:.1f}s, finish={finish}, model={model_used}")
    return parsed
