"""Tests for source-packet loading, validation, and summarization."""

from __future__ import annotations

import json

import pytest

from pt.source_packet import (
    PreSpecifiedTest,
    SourceCandidate,
    SourceGap,
    SourcePacket,
    SourcePacketError,
    load_source_packet,
)


def _packet() -> SourcePacket:
    return SourcePacket(
        case_name="18 Brumaire",
        research_question="Why did Brumaire produce the Consulate rather than a restored Directory?",
        focal_window="1799-10 to 1799-11",
        outcome="Creation of the Consulate",
        source_candidates=[
            SourceCandidate(
                title="Official proclamation",
                source_group="official public justification",
                source_kind="primary proclamation",
                date_coverage="1799-11",
                locator="https://example.test/proclamation",
                provenance_note="Official post-coup source.",
                reliability_note="Justificatory and public-facing.",
                expected_observability="Public legitimacy claims, not private planning.",
                relevance_to_question="Tests whether order claims were post-hoc legitimation.",
            ),
            SourceCandidate(
                title="Council proceedings",
                source_group="legislative record",
                source_kind="primary legislative record",
                date_coverage="1799-11",
                locator="archive://council",
                provenance_note="Proceedings from the contested legislative moment.",
                reliability_note="Institutional record with procedural blind spots.",
                expected_observability="Procedural disruption and formal resistance.",
                relevance_to_question="Tests whether legality collapsed before military coercion.",
            ),
            SourceCandidate(
                title="Critical historiography",
                source_group="rival secondary account",
                source_kind="historiography",
                date_coverage="1799",
                locator=None,
                provenance_note="Scholarly rival interpretation.",
                reliability_note="Interpretive synthesis, not direct trace evidence.",
                expected_observability="Alternative mechanism claims and source disputes.",
                relevance_to_question="Preserves rival mechanisms for discrimination.",
            ),
        ],
        known_gaps=[
            SourceGap(
                missing_source_class="Private correspondence among conspirators",
                why_it_matters="Could reveal planning sequence and agency.",
                expected_location="Correspondence collections",
                priority="high",
            )
        ],
        pre_specified_tests=[
            PreSpecifiedTest(
                test_name="Legislative coercion test",
                target_rival_pair="military coercion vs legal exhaustion",
                expected_trace="Direct military pressure before formal consent.",
                contrary_trace="Voluntary legal transfer before coercive deployment.",
                source_classes=["legislative record", "memoir"],
            )
        ],
        limitations=["Private planning sources not yet included."],
    )


@pytest.mark.plans(3)
def test_source_packet_summary_preserves_scope_metadata():
    packet = _packet()

    summary = packet.to_summary("packet.json")

    assert summary.case_name == "18 Brumaire"
    assert summary.source_count == 3
    assert summary.source_groups == [
        "legislative record",
        "official public justification",
        "rival secondary account",
    ]
    assert summary.high_priority_gap_count == 1
    assert summary.high_priority_gaps == ["Private correspondence among conspirators"]
    assert summary.pre_specified_test_count == 1
    assert summary.source_packet_path == "packet.json"


@pytest.mark.plans(3)
def test_source_packet_loader_accepts_assistant_artifact_shape(tmp_path):
    artifact_path = tmp_path / "assistant_artifact.json"
    artifact_path.write_text(
        json.dumps({"metadata": {"trace_id": "trace-1"}, "draft": _packet().model_dump()}),
        encoding="utf-8",
    )

    loaded = load_source_packet(artifact_path)

    assert loaded.case_name == "18 Brumaire"
    assert loaded.source_candidates[0].source_group == "official public justification"
    assert "Packet metadata is not itself evidence" not in loaded.to_prompt_context()
    assert "Private correspondence among conspirators" in loaded.to_prompt_context()


@pytest.mark.plans(3)
def test_source_packet_loader_fails_loud_on_missing_file(tmp_path):
    with pytest.raises(SourcePacketError, match="file not found"):
        load_source_packet(tmp_path / "missing.json")
