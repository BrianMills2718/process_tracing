"""Tests for the process-tracing LLM boundary contract.

The boundary delegates structured output to ``llm_client.call_llm_structured``.
These tests mock that delegate (mock-ok: the boundary's job is message assembly +
delegation + fail-loud, not provider behavior, which is covered by opt-in live
smoke tests).
"""

import os

import pytest
from pydantic import BaseModel, Field

from pt.llm import LLMError, call_llm


class _StructuredSmokeResult(BaseModel):
    """Small schema used to verify the boundary contract."""

    answer: str = Field(description="Short structured answer")


class _FakeResult:
    model = "fake-model"
    finish_reason = "stop"


def test_call_llm_delegates_to_structured(monkeypatch):
    captured = {}

    def fake_structured(model, messages, response_model, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["response_model"] = response_model
        captured["kwargs"] = kwargs
        return response_model(answer="ok"), _FakeResult()

    monkeypatch.setattr("pt.llm.call_llm_structured", fake_structured)

    out = call_llm("Why?", _StructuredSmokeResult, task="pt.test", trace_id="abc123")

    # Returns the validated model, not raw text.
    assert isinstance(out, _StructuredSmokeResult)
    assert out.answer == "ok"
    # Response model is passed through (no hand-rolled schema injection).
    assert captured["response_model"] is _StructuredSmokeResult
    # Required observability kwargs are forwarded.
    assert captured["kwargs"]["task"] == "pt.test"
    assert captured["kwargs"]["trace_id"] == "abc123"
    assert "max_budget" in captured["kwargs"]
    # System + user messages are assembled; the prompt is the user content.
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == "Why?"


def test_call_llm_fails_loud(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("pt.llm.call_llm_structured", boom)

    with pytest.raises(LLMError):
        call_llm("Why?", _StructuredSmokeResult, task="pt.test", trace_id="abc123")


@pytest.mark.skipif(
    os.getenv("PT_RUN_LIVE_LLM_TESTS") != "1",
    reason="live LLM smoke; set PT_RUN_LIVE_LLM_TESTS=1 to run",
)
def test_call_llm_structured_live_smoke():
    """Live smoke test of the real boundary against a provider (opt-in).

    Replaces the old direct-litellm exploratory tests: it exercises the actual
    pt.llm.call_llm path (llm_client.call_llm_structured), which is what the
    pipeline uses.
    """
    out = call_llm(
        "Classify this Van Evera test: 'If the suspect was in the city, we would "
        "expect hotel records.' Return diagnostic_type, confidence, reasoning.",
        _LiveAssessment,
        task="pt.test.live_smoke",
        trace_id="live-smoke",
    )
    assert isinstance(out, _LiveAssessment)
    assert out.diagnostic_type
    assert 0.0 <= out.confidence <= 1.0


class _LiveAssessment(BaseModel):
    diagnostic_type: str = Field(description="Van Evera test type")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Reasoning for the assessment")
