"""Extraction quality tests for the active `pt` pipeline.

mock-ok: These tests validate extraction contract expectations with a
deterministic LLM boundary. Real extraction quality belongs in gated live-LLM
golden tests, not default unit tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pt.pass_extract import run_extract
from pt.schemas import (
    AbsenceEvaluation,
    DateConfidence,
    Event,
    Evidence,
    ExtractionResult,
    SourceGenre,
    TraceProductionRelevance,
)


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


# ===== Slice 3: Source provenance metadata fields =====


class TestEvidenceProvenanceFields:
    """Evidence schema now carries source_group, source_genre, date_confidence,
    trace_production_relevance — all Optional, all None by default for backward compat."""

    def test_new_fields_default_to_none(self):
        ev = Evidence(id="evi_x", description="a" * 25, source_text="q")
        assert ev.source_group is None
        assert ev.source_genre is None
        assert ev.date_confidence is None
        assert ev.trace_production_relevance is None

    def test_source_genre_accepts_valid_literals(self):
        for genre in (
            "overview", "primary_document", "speech", "legal_constitutional",
            "memoir", "parliamentary_record", "secondary_analysis", "news_dispatch", "other",
        ):
            ev = Evidence(id="evi_x", description="a" * 25, source_text="q", source_genre=genre)
            assert ev.source_genre == genre

    def test_source_genre_rejects_unknown_value(self):
        with pytest.raises(Exception):
            Evidence(id="evi_x", description="a" * 25, source_text="q", source_genre="gossip")  # type: ignore[arg-type]

    def test_date_confidence_accepts_valid_literals(self):
        for dc in ("high", "medium", "low"):
            ev = Evidence(id="evi_x", description="a" * 25, source_text="q", date_confidence=dc)
            assert ev.date_confidence == dc

    def test_date_confidence_rejects_unknown_value(self):
        with pytest.raises(Exception):
            Evidence(id="evi_x", description="a" * 25, source_text="q", date_confidence="uncertain")  # type: ignore[arg-type]

    def test_trace_production_relevance_accepts_valid_literals(self):
        for tpr in ("direct", "indirect", "background"):
            ev = Evidence(id="evi_x", description="a" * 25, source_text="q", trace_production_relevance=tpr)
            assert ev.trace_production_relevance == tpr

    def test_trace_production_relevance_rejects_unknown_value(self):
        with pytest.raises(Exception):
            Evidence(id="evi_x", description="a" * 25, source_text="q", trace_production_relevance="unknown")  # type: ignore[arg-type]

    def test_source_group_is_free_text(self):
        ev = Evidence(id="evi_x", description="a" * 25, source_text="q", source_group="Primary sources section")
        assert ev.source_group == "Primary sources section"

    def test_all_new_fields_populated(self):
        ev = Evidence(
            id="evi_x",
            description="a" * 25,
            source_text="q",
            source_group="Main text",
            source_genre="primary_document",
            date_confidence="high",
            trace_production_relevance="direct",
        )
        assert ev.source_group == "Main text"
        assert ev.source_genre == "primary_document"
        assert ev.date_confidence == "high"
        assert ev.trace_production_relevance == "direct"

    def test_roundtrip_json_preserves_new_fields(self):
        ev = Evidence(
            id="evi_x",
            description="a" * 25,
            source_text="q",
            source_group="Background",
            source_genre="overview",
            date_confidence="medium",
            trace_production_relevance="background",
        )
        restored = Evidence.model_validate_json(ev.model_dump_json())
        assert restored.source_group == "Background"
        assert restored.source_genre == "overview"
        assert restored.date_confidence == "medium"
        assert restored.trace_production_relevance == "background"

    def test_old_evidence_without_new_fields_loads_cleanly(self):
        """Backward compat: result.json from before Slice 3 has no new fields."""
        old_json = '{"id": "evi_old", "description": "' + "a" * 25 + '", "source_text": "q"}'
        ev = Evidence.model_validate_json(old_json)
        assert ev.source_group is None
        assert ev.source_genre is None
        assert ev.date_confidence is None
        assert ev.trace_production_relevance is None

    def test_extraction_prompt_includes_source_provenance_metadata_section(self):
        captured: list[str] = []

        def _capture(prompt, schema, **kwargs):
            captured.append(prompt)
            return _quality_extraction()

        with patch("pt.pass_extract.call_llm", side_effect=_capture):
            run_extract(_source_text(), trace_id="slice3-prompt-contract")

        prompt = captured[0].lower()
        assert "source provenance metadata" in prompt
        assert "source_genre" in prompt
        assert "trace_production_relevance" in prompt
        assert "date_confidence" in prompt
        assert "source_group" in prompt

    def test_extraction_prompt_explains_trace_production_relevance_values(self):
        captured: list[str] = []

        def _capture(prompt, schema, **kwargs):
            captured.append(prompt)
            return _quality_extraction()

        with patch("pt.pass_extract.call_llm", side_effect=_capture):
            run_extract(_source_text(), trace_id="slice3-trace-values")

        prompt = captured[0].lower()
        assert "'direct'" in prompt or "direct" in prompt
        assert "'indirect'" in prompt or "indirect" in prompt
        assert "'background'" in prompt or "background" in prompt


class TestSourceGenreType:
    def test_all_genre_values_are_valid_literals(self):
        valid: list[SourceGenre] = [
            "overview", "primary_document", "speech", "legal_constitutional",
            "memoir", "parliamentary_record", "secondary_analysis", "news_dispatch", "other",
        ]
        for genre in valid:
            ev = Evidence(id="evi_g", description="a" * 25, source_text="q", source_genre=genre)
            assert ev.source_genre == genre


class TestDateConfidenceType:
    def test_all_confidence_values(self):
        for val in ("high", "medium", "low"):
            ev = Evidence(id="evi_d", description="a" * 25, source_text="q", date_confidence=val)  # type: ignore[arg-type]
            assert ev.date_confidence == val


class TestTraceProductionRelevanceType:
    def test_all_relevance_values(self):
        for val in ("direct", "indirect", "background"):
            ev = Evidence(id="evi_t", description="a" * 25, source_text="q", trace_production_relevance=val)  # type: ignore[arg-type]
            assert ev.trace_production_relevance == val


# ── Slice 6: Source-Design / Observability-Weighted Absence ──────────

def _absence_eval(**kwargs) -> AbsenceEvaluation:
    return AbsenceEvaluation(  # type: ignore[call-overload]
        hypothesis_id=kwargs.get("hypothesis_id", "h1"),
        prediction_id=kwargs.get("prediction_id", "pred_h1_1"),
        missing_evidence=kwargs.get("missing_evidence", "No evidence of X found"),
        reasoning=kwargs.get("reasoning", "The text covers this period comprehensively"),
        severity=kwargs.get("severity", "notable"),
        would_be_extractable=kwargs.get("would_be_extractable", True),
        expected_source_genre=kwargs.get("expected_source_genre", None),
        expected_source_location=kwargs.get("expected_source_location", None),
    )


class TestAbsenceAcquisitionFields:
    """New fields on AbsenceEvaluation: expected_source_genre + expected_source_location."""

    def test_defaults_to_none(self):
        ae = _absence_eval()
        assert ae.expected_source_genre is None
        assert ae.expected_source_location is None

    def test_expected_source_genre_accepts_all_valid_literals(self):
        valid: list[SourceGenre] = [
            "overview", "primary_document", "speech", "legal_constitutional",
            "memoir", "parliamentary_record", "secondary_analysis", "news_dispatch", "other",
        ]
        for genre in valid:
            ae = _absence_eval(expected_source_genre=genre)
            assert ae.expected_source_genre == genre

    def test_expected_source_genre_rejects_invalid(self):
        with pytest.raises(Exception):
            _absence_eval(expected_source_genre="blog_post")  # type: ignore[arg-type]

    def test_expected_source_location_stores_string(self):
        ae = _absence_eval(
            expected_source_genre="parliamentary_record",
            expected_source_location="Minutes of the Conseil des Cinq-Cents, 1799",
        )
        assert ae.expected_source_location == "Minutes of the Conseil des Cinq-Cents, 1799"

    def test_roundtrip_preserves_both_fields(self):
        ae = _absence_eval(
            expected_source_genre="primary_document",
            expected_source_location="Napoleon's private correspondence, Archives nationales",
        )
        restored = AbsenceEvaluation.model_validate_json(ae.model_dump_json())
        assert restored.expected_source_genre == "primary_document"
        assert restored.expected_source_location == ae.expected_source_location

    def test_backward_compat_old_absence_without_new_fields_loads_cleanly(self):
        """Old result.json AbsenceEvaluation without new fields parses without error."""
        old_style = {
            "hypothesis_id": "h1",
            "prediction_id": "pred_h1_1",
            "missing_evidence": "No military order found",
            "reasoning": "The text would mention this",
            "severity": "damaging",
            "would_be_extractable": True,
        }
        ae = AbsenceEvaluation.model_validate(old_style)
        assert ae.expected_source_genre is None
        assert ae.expected_source_location is None

    def test_genre_independent_of_extractability(self):
        """expected_source_genre should be populated even when would_be_extractable=False."""
        ae = _absence_eval(
            would_be_extractable=False,
            expected_source_genre="primary_document",
            expected_source_location="Police surveillance dossiers, not available in this overview",
        )
        assert ae.would_be_extractable is False
        assert ae.expected_source_genre == "primary_document"

    def test_prompt_contract_includes_expected_source_genre(self):
        """Pass 3b prompt must mention expected_source_genre to instruct the LLM."""
        import os
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "pt", "prompts", "pass3b_absence.yaml"
        )
        with open(prompt_path) as f:
            content = f.read()
        assert "expected_source_genre" in content, (
            "Pass 3b prompt must instruct the LLM to populate expected_source_genre"
        )
        # Guard the critical disambiguation: expected_source_genre is the SOURCE TO ACQUIRE,
        # not the genre of the current input text. If this phrase disappears, the LLM will
        # misinterpret the field and populate it with the current corpus genre instead.
        assert "SOURCE TO ACQUIRE" in content or "NOT the current text" in content, (
            "Pass 3b prompt must clarify that expected_source_genre is the genre of the "
            "source to acquire, not the genre of the current input text. "
            "Expected phrase: 'SOURCE TO ACQUIRE' or 'NOT the current text'"
        )

    def test_prompt_contract_includes_expected_source_location(self):
        """Pass 3b prompt must mention expected_source_location to instruct the LLM."""
        import os
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "pt", "prompts", "pass3b_absence.yaml"
        )
        with open(prompt_path) as f:
            content = f.read()
        assert "expected_source_location" in content, (
            "Pass 3b prompt must instruct the LLM to populate expected_source_location"
        )
