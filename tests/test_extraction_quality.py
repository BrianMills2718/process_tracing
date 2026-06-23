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


def test_extraction_contract_preserves_source_markers_in_prompt_and_schema():
    captured_prompts: list[str] = []

    def _capture_call(prompt, schema, **kwargs):
        captured_prompts.append(prompt)
        return _quality_extraction()

    with patch("pt.pass_extract.call_llm", side_effect=_capture_call):
        run_extract("Source A says the committee accepted the decree.", trace_id="source-marker-contract")

    prompt = captured_prompts[0].lower()
    source_text_schema = Evidence.model_json_schema()["properties"]["source_text"]["description"].lower()

    assert "source provenance is part of the evidence" in prompt
    assert "keep that marker inside `source_text`" in prompt
    assert "preserves any source marker or citation label" in prompt
    assert "preserve source markers" in source_text_schema
    assert "citation labels" in source_text_schema


def test_extraction_contract_uses_source_packet_for_marker_coverage():
    captured_prompts: list[str] = []

    def _capture_call(prompt, schema, **kwargs):
        captured_prompts.append(prompt)
        return _quality_extraction()

    source_packet_context = (
        "Case: test\n"
        "Sources:\n"
        "- Constitution of the Year VIII; id=source_c; kind=primary legal text; "
        "text_markers=Source C; observability=formal powers and institutional constraints"
    )

    with patch("pt.pass_extract.call_llm", side_effect=_capture_call):
        run_extract(
            "Source C says the First Consul promulgates laws and appoints officials.",
            source_packet_context=source_packet_context,
            trace_id="source-packet-extract-contract",
        )

    prompt = captured_prompts[0].lower()

    assert "accepted source-packet contract" in prompt
    assert "packet metadata is not itself evidence" in prompt
    assert "for every accepted source whose configured marker appears" in prompt
    assert "legal, constitutional, procedural" in prompt
    assert "source c" in prompt
    assert "constitution of the year viii" in prompt


def test_extraction_sanitizes_non_ascii_evidence_ids():
    """LLM-assigned ids with accents/whitespace are normalized to ASCII so they
    survive later LLM round-trips (the fail-loud id match in pass_test)."""
    messy = ExtractionResult(
        summary="x",
        evidence=[
            Evidence(id="evi_levée_en_masse", description="a" * 25, source_text="q"),
            Evidence(id="evi_côté", description="b" * 25, source_text="q"),
            Evidence(id="evi_cote", description="c" * 25, source_text="q"),  # distinct raw; collides after strip
        ],
    )
    with patch("pt.pass_extract.call_llm", return_value=messy):
        extraction = run_extract("text " * 50, trace_id="t")
    ids = [e.id for e in extraction.evidence]
    assert ids == ["evi_levee_en_masse", "evi_cote", "evi_cote_2"]
    for i in ids:
        assert i.isascii()
