"""Extraction quality tests for the active `pt` pipeline.

mock-ok: These tests validate extraction contract expectations with a
deterministic LLM boundary. Real extraction quality belongs in gated live-LLM
golden tests, not default unit tests.
"""

from __future__ import annotations

from unittest.mock import patch

from pt.pass_extract import run_extract
from pt.schemas import Event, Evidence, ExtractionResult


def _source_text() -> str:
    return (
        "The Stamp Act of 1765 required colonists to pay taxes on printed materials. "
        "The Boston Massacre occurred in 1770 when British soldiers fired on colonial protesters. "
        "The Boston Tea Party in 1773 was a protest against British tea taxes."
    )


def _quality_extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="British taxation and violence escalated colonial resistance before the Boston Tea Party.",
        events=[
            Event(
                id="evt_stamp_act",
                description="The Stamp Act of 1765 required colonists to pay taxes on printed materials.",
                date="1765",
            ),
            Event(
                id="evt_boston_massacre",
                description="British soldiers fired on colonial protesters during the Boston Massacre.",
                date="1770",
            ),
            Event(
                id="evt_boston_tea_party",
                description="Colonists protested British tea taxes through the Boston Tea Party.",
                date="1773",
            ),
        ],
        evidence=[
            Evidence(
                id="evi_stamp_act",
                description="Stamp Act taxes applied to printed materials.",
                source_text="The Stamp Act of 1765 required colonists to pay taxes on printed materials.",
                evidence_type="empirical",
                approximate_date="1765",
            ),
            Evidence(
                id="evi_boston_massacre",
                description="British soldiers fired on colonial protesters in 1770.",
                source_text="The Boston Massacre occurred in 1770 when British soldiers fired on colonial protesters.",
                evidence_type="empirical",
                approximate_date="1770",
            ),
        ],
    )


def test_extraction_preserves_source_quotes_and_meaningful_descriptions():
    with patch("pt.pass_extract.call_llm", return_value=_quality_extraction()):
        extraction = run_extract(_source_text(), trace_id="test-extraction")

    assert len(extraction.events) >= 3
    for event in extraction.events:
        assert event.description != "N/A"
        assert "Description_Not_Found" not in event.description
        assert len(event.description) >= 20

    for evidence in extraction.evidence:
        assert evidence.source_text
        assert evidence.source_text in _source_text()
        assert len(evidence.description) >= 20
