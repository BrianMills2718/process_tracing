"""Single LLM call abstraction — delegates to llm_client."""

from __future__ import annotations

import json
import os
import time
from typing import TypeVar

from dotenv import load_dotenv
from llm_client import call_llm as _call_llm_raw, strip_fences
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("PT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

MAX_RETRIES = 3


class LLMError(Exception):
    """Raised when LLM call fails. No fallbacks—fail fast."""


def call_llm(
    prompt: str,
    response_model: type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    system: str = "You are an expert political scientist and historian performing Van Evera process tracing analysis. Respond with valid JSON matching the requested schema exactly. Keep your JSON output concise—use brief descriptions, not lengthy prose.",
) -> T:
    """Call LLM and parse response into a Pydantic model.

    Delegates retry on transient errors (rate limits, timeouts, empty responses),
    thinking model detection, max token defaulting, and truncation handling to
    llm_client. This wrapper adds: message building, schema injection, fence
    stripping, Pydantic validation with retry, and pipeline-specific logging.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    full_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_json}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": full_prompt},
    ]

    # Retry loop for Pydantic validation failures (bad JSON from LLM).
    # Transient errors (rate limits, timeouts, etc.) are retried inside
    # llm_client, so this loop only re-calls on parse failures.
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        try:
            result = _call_llm_raw(
                model,
                messages,
                timeout=600,
                num_retries=3,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

        elapsed = time.time() - t0
        content = strip_fences(result.content)
        print(
            f"  LLM call: {elapsed:.1f}s, {len(content)} chars, "
            f"finish={result.finish_reason}, model={result.model}"
        )

        try:
            parsed = response_model.model_validate_json(content)
            if attempt > 0:
                print(f"  Succeeded after {attempt} parse retries")
            return parsed
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  JSON parse failed (retrying): {e}")
                continue
            raise LLMError(
                f"Failed to parse LLM response: {e}\nRaw: {content[:500]}"
            ) from e

    raise LLMError(f"LLM call failed after {MAX_RETRIES + 1} attempts")
