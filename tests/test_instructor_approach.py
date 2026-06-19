#!/usr/bin/env python3
"""Opt-in live structured-output smoke tests for Instructor and LiteLLM."""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from pt.llm import _schema_response_format

load_dotenv()

pytestmark = pytest.mark.skipif(
    os.getenv("PT_RUN_LIVE_LLM_TESTS") != "1",
    reason="live LLM exploratory script; set PT_RUN_LIVE_LLM_TESTS=1 to run",
)


class AssessmentSmoke(BaseModel):
    """Small live-call schema for Van Evera classification smoke tests."""

    diagnostic_type: str = Field(description="Van Evera test type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Reasoning for assessment")


def _require_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY is required for live LLM smoke tests")
    return api_key


def test_instructor_returns_pydantic_model_when_installed():
    instructor = pytest.importorskip("instructor")
    from litellm import completion

    _require_api_key()
    client = instructor.from_litellm(completion)

    response = client.chat.completions.create(
        model="gemini/gemini-2.5-flash",
        response_model=AssessmentSmoke,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this as Van Evera test: "
                    "'Evidence is necessary but not sufficient'"
                ),
            }
        ],
    )

    assert isinstance(response, AssessmentSmoke)
    assert response.diagnostic_type


def test_litellm_json_schema_fallback_parses_assessment():
    import litellm

    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify as Van Evera test. Respond with JSON: "
                    f"{AssessmentSmoke.model_json_schema()}"
                ),
            }
        ],
        response_format=_schema_response_format(AssessmentSmoke),
        api_key=_require_api_key(),
    )

    assessment = AssessmentSmoke(**json.loads(response.choices[0].message.content))
    assert assessment.diagnostic_type
