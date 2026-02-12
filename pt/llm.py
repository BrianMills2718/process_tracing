"""Single LLM call abstraction using LiteLLM."""

from __future__ import annotations

import json
import os
import re
import time
from typing import TypeVar

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.getenv("PT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"


class LLMError(Exception):
    """Raised when LLM call fails. No fallbacks—fail fast."""


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

    Raises LLMError on any failure—never returns None or empty objects.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)
    full_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_json}"

    t0 = time.time()
    try:
        response = litellm.completion(
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
    except Exception as e:
        raise LLMError(f"LLM call failed: {e}") from e

    elapsed = time.time() - t0
    content = response.choices[0].message.content
    if not content:
        raise LLMError("LLM returned empty response")

    # Check for truncation (finish_reason != 'stop')
    finish_reason = response.choices[0].finish_reason
    print(f"  LLM call: {elapsed:.1f}s, {len(content)} chars, finish={finish_reason}")

    if finish_reason == "length":
        raise LLMError(
            f"LLM response truncated ({len(content)} chars). "
            "Increase max_tokens or simplify the prompt."
        )

    # Strip markdown fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content.strip())

    try:
        return response_model.model_validate_json(content)
    except Exception as e:
        raise LLMError(f"Failed to parse LLM response: {e}\nRaw: {content[:500]}") from e
