"""Tests for source-packet coverage checks."""

from __future__ import annotations

import pytest

from pt.schemas import Evidence, ExtractionResult
from pt.source_coverage import build_source_coverage
from pt.source_packet import SourceCandidate, SourcePacket


def _source(
    *,
    source_id: str,
    title: str,
    marker: str | None,
) -> SourceCandidate:
    return SourceCandidate(
        source_id=source_id,
        title=title,
        text_markers=[marker] if marker else [],
        source_group=title,
        source_kind="primary text",
        date_coverage="1799",
        locator=None,
        provenance_note="packet source",
        reliability_note="test source",
        expected_observability="test traces",
        relevance_to_question="tests coverage",
    )


def _packet() -> SourcePacket:
    return SourcePacket(
        case_name="case",
        research_question="Why did the outcome occur?",
        focal_window="1799",
        outcome="outcome",
        source_candidates=[
            _source(source_id="source_a", title="Source A Title", marker="Source A"),
            _source(source_id="source_b", title="Source B Title", marker="Source B"),
            _source(source_id="source_c", title="Source C Title", marker=None),
        ],
    )


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="summary",
        evidence=[
            Evidence(
                id="evi_a",
                description="A trace",
                source_text="Source A says the decisive trace appeared.",
                evidence_type="empirical",
            ),
            Evidence(
                id="evi_unassigned",
                description="Unmarked trace",
                source_text="An unmarked sentence without source provenance.",
                evidence_type="empirical",
            ),
        ],
    )


@pytest.mark.plans(3)
def test_source_coverage_links_input_and_evidence_markers():
    report = build_source_coverage(
        _packet(),
        "Header. Source A says one thing. Source B says another thing.",
        _extraction(),
    )

    by_id = {item.source_id: item for item in report.items}
    assert by_id["source_a"].status == "covered"
    assert by_id["source_a"].evidence_ids == ["evi_a"]
    assert by_id["source_b"].status == "input_only"
    assert by_id["source_c"].status == "unconfigured"
    assert report.sources_with_input_markers == 2
    assert report.sources_with_evidence == 1
    assert report.assigned_evidence_count == 1
    assert report.unassigned_evidence_ids == ["evi_unassigned"]
    assert report.input_only_source_ids == ["source_b"]
    assert report.unconfigured_source_ids == ["source_c"]


@pytest.mark.plans(3)
def test_source_coverage_reports_missing_configured_source():
    packet = SourcePacket(
        case_name="case",
        research_question="Why did the outcome occur?",
        focal_window="1799",
        outcome="outcome",
        source_candidates=[
            _source(source_id="source_missing", title="Missing", marker="Source Missing"),
        ],
    )

    report = build_source_coverage(packet, "No marker here.", ExtractionResult(summary="s"))

    assert report.items[0].status == "missing"
    assert report.missing_source_ids == ["source_missing"]
