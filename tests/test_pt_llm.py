"""Tests for the process-tracing LLM boundary contract."""

from pydantic import BaseModel, Field

from pt.llm import _schema_response_format


class _StructuredSmokeResult(BaseModel):
    """Small schema used to verify provider response-format construction."""

    answer: str = Field(description="Short structured answer")


def test_schema_response_format_uses_json_schema_contract():
    response_format = _schema_response_format(_StructuredSmokeResult)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "_StructuredSmokeResult"
    assert response_format["json_schema"]["schema"]["properties"]["answer"]["type"] == "string"
