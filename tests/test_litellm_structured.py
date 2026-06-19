#!/usr/bin/env python3
"""Opt-in live LiteLLM structured-output smoke tests."""

from __future__ import annotations

import json
import os

import litellm
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


def _api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY is required for live LLM smoke tests")
    return api_key


def _complete(prompt: str):
    return litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format=_schema_response_format(AssessmentSmoke),
        api_key=_api_key(),
    )


def test_json_schema_response_format_returns_parseable_assessment():
    response = _complete(
        "Classify this as a Van Evera test type: "
        "'The evidence strongly confirms the hypothesis'"
    )

    data = json.loads(response.choices[0].message.content)
    assessment = AssessmentSmoke(**data)

    assert assessment.diagnostic_type
    assert 0.0 <= assessment.confidence <= 1.0
    assert assessment.reasoning


def test_schema_helper_response_format_returns_content():
    response = _complete(
        "Classify this as a Van Evera test type: "
        "'The evidence is necessary but not sufficient'"
    )

    assert response.choices[0].message.content


def test_schema_in_prompt_with_json_schema_response_format_parses():
    schema = AssessmentSmoke.model_json_schema()
    prompt = (
        "Classify this as a Van Evera test type: "
        "'The evidence weakly suggests the hypothesis'\n\n"
        f"Respond with valid JSON matching this schema: {schema}"
    )

    response = _complete(prompt)
    assessment = AssessmentSmoke(**json.loads(response.choices[0].message.content))

    assert assessment.diagnostic_type
