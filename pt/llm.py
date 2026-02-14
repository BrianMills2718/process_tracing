"""Single LLM call abstraction using LiteLLM."""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import TypeVar

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("PT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

MAX_RETRIES = 3
BASE_DELAY = 1.0

_RETRYABLE_PATTERNS = [
    "invalid json",
    "json parse",
    "unterminated string",
    "malformed json",
    "expecting",
    "delimiter",
    "no json found",
    "rate limit",
    "overloaded",
    "timeout",
    "connection reset",
    "connection error",
    "service unavailable",
    "internal server error",
    "http 500",
    "http 503",
    "empty content",
    "empty response",
]


class LLMError(Exception):
    """Raised when LLM call fails. No fallbacks—fail fast."""


def _is_retryable(error: Exception) -> bool:
    """Check if an error is transient and worth retrying."""
    error_str = str(error).lower()
    return any(p in error_str for p in _RETRYABLE_PATTERNS)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 30s."""
    delay = BASE_DELAY * (2 ** attempt)
    jitter = random.uniform(0.5, 1.5)
    return min(delay * jitter, 30.0)


def call_llm(
    prompt: str,
    response_model: type[T],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 65536,
    system: str = "You are an expert political scientist and historian performing Van Evera process tracing analysis. Respond with valid JSON matching the requested schema exactly. Keep your JSON output concise—use brief descriptions, not lengthy prose.",
) -> T:
    """Call LLM and parse response into a Pydantic model.

    Retries up to MAX_RETRIES times on transient errors (JSON parse failures,
    rate limits, timeouts). Raises LLMError on non-retryable or exhausted errors.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    full_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_json}"

    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            delay = _backoff_delay(attempt - 1)
            print(f"  Retry {attempt}/{MAX_RETRIES} after {delay:.1f}s...")
            time.sleep(delay)

        t0 = time.time()
        try:
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=max_tokens,
                temperature=temperature,
                timeout=600,
            )
            # Disable reasoning for thinking models — we need output tokens, not CoT
            if "gemini-3" in model or "gemini-4" in model:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 0}
            response = litellm.completion(**kwargs)
        except Exception as e:
            last_error = e
            if _is_retryable(e) and attempt < MAX_RETRIES:
                print(f"  LLM call failed (retryable): {e}")
                continue
            raise LLMError(f"LLM call failed: {e}") from e

        elapsed = time.time() - t0
        resp_model = getattr(response, "model", "unknown")
        content = response.choices[0].message.content
        if not content:
            last_error = LLMError("LLM returned empty response")
            if attempt < MAX_RETRIES:
                print(f"  LLM returned empty response (retrying)")
                continue
            raise last_error

        # Check for truncation (finish_reason != 'stop')
        finish_reason = response.choices[0].finish_reason
        print(f"  LLM call: {elapsed:.1f}s, {len(content)} chars, finish={finish_reason}, model={resp_model}")

        if finish_reason == "length":
            raise LLMError(
                f"LLM response truncated ({len(content)} chars). "
                "Increase max_tokens or simplify the prompt."
            )

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())

        try:
            result = response_model.model_validate_json(content)
            if attempt > 0:
                print(f"  Succeeded after {attempt} retries")
            return result
        except Exception as e:
            last_error = LLMError(f"Failed to parse LLM response: {e}\nRaw: {content[:500]}")
            if attempt < MAX_RETRIES:
                print(f"  JSON parse failed (retrying): {e}")
                continue
            raise last_error from e

    # Should not reach here, but just in case
    raise LLMError(f"LLM call failed after {MAX_RETRIES + 1} attempts: {last_error}")
