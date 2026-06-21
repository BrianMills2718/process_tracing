"""Tests for pt/apply_refinement.py — fail-loud delta application."""

from __future__ import annotations

import pytest

from pt.apply_refinement import apply_refinement
from pt.schemas import (
    Event,
    Evidence,
    ExtractionResult,
    Hypothesis,
    HypothesisRefinement,
    HypothesisSpace,
    NewCausalEdge,
    Prediction,
    RefinementResult,
    ReinterpretedEvidence,
    SpuriousExtraction,
)


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="s",
        events=[Event(id="evt_a", description="a"), Event(id="evt_b", description="b")],
        evidence=[
            Evidence(id="evi_1", description="d1", source_text="q1", evidence_type="empirical"),
            Evidence(id="evi_2", description="d2", source_text="q2", evidence_type="empirical"),
        ],
    )


def _space() -> HypothesisSpace:
    return HypothesisSpace(research_question="rq", hypotheses=[
        Hypothesis(id="h1", description="d", source="text", theoretical_basis="t",
                   causal_mechanism="m", observable_predictions=[]),
    ])


class TestSpuriousRemoval:
    def test_actual_count_and_removal(self):
        ref = RefinementResult(
            spurious_extractions=[SpuriousExtraction(item_id="evi_2", item_type="evidence", reason="x")],
            analyst_notes="",
        )
        ext, _ = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert {e.id for e in ext.evidence} == {"evi_1"}

    def test_unknown_spurious_evidence_fails_loud(self):
        ref = RefinementResult(
            spurious_extractions=[SpuriousExtraction(item_id="evi_ghost", item_type="evidence", reason="x")],
            analyst_notes="",
        )
        with pytest.raises(ValueError, match="spurious evidence ids not found"):
            apply_refinement(_extraction(), _space(), ref, verbose=False)


class TestReinterpretation:
    def test_unknown_target_fails_loud(self):
        ref = RefinementResult(
            reinterpreted_evidence=[ReinterpretedEvidence(
                evidence_id="evi_ghost", original_type="empirical",
                new_type="interpretive", reinterpretation="r")],
            analyst_notes="",
        )
        with pytest.raises(ValueError, match="unknown evidence id 'evi_ghost'"):
            apply_refinement(_extraction(), _space(), ref, verbose=False)

    def test_spurious_removed_target_is_noop(self):
        ref = RefinementResult(
            spurious_extractions=[SpuriousExtraction(item_id="evi_2", item_type="evidence", reason="x")],
            reinterpreted_evidence=[ReinterpretedEvidence(
                evidence_id="evi_2", original_type="empirical",
                new_type="interpretive", reinterpretation="r")],
            analyst_notes="",
        )
        ext, _ = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert "evi_2" not in {e.id for e in ext.evidence}


class TestNewCausalEdges:
    def test_unknown_endpoint_fails_loud(self):
        ref = RefinementResult(
            new_causal_edges=[NewCausalEdge(source_id="evt_a", target_id="evt_ghost",
                                            relationship="led to", source_text_support="q")],
            analyst_notes="",
        )
        with pytest.raises(ValueError, match="endpoint 'evt_ghost' is not a known node"):
            apply_refinement(_extraction(), _space(), ref, verbose=False)

    def test_valid_endpoints_accepted(self):
        ref = RefinementResult(
            new_causal_edges=[NewCausalEdge(source_id="evt_a", target_id="evt_b",
                                            relationship="led to", source_text_support="q")],
            analyst_notes="",
        )
        ext, _ = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert any(e.source_id == "evt_a" and e.target_id == "evt_b" for e in ext.causal_edges)


class TestHypothesisRefinement:
    def test_unknown_target_fails_loud(self):
        ref = RefinementResult(
            hypothesis_refinements=[HypothesisRefinement(
                hypothesis_id="h_ghost", refinement_type="sharpen_mechanism",
                description="x", updated_causal_mechanism="m2")],
            analyst_notes="",
        )
        with pytest.raises(ValueError, match="unknown hypothesis id 'h_ghost'"):
            apply_refinement(_extraction(), _space(), ref, verbose=False)

    def test_merge_suggestion_with_synthetic_id_is_advisory(self):
        ref = RefinementResult(
            hypothesis_refinements=[HypothesisRefinement(
                hypothesis_id="h1_h2_merge_suggestion",
                refinement_type="merge_suggestion",
                description="merge h1 and h2 if the mechanisms overlap",
                updated_causal_mechanism="combined mechanism",
                new_predictions=[Prediction(id="pred_merge", description="p")],
            )],
            analyst_notes="",
        )
        _, hs = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert hs.hypotheses[0].causal_mechanism == "m"
        assert hs.hypotheses[0].observable_predictions == []

    def test_add_prediction_applies(self):
        ref = RefinementResult(
            hypothesis_refinements=[HypothesisRefinement(
                hypothesis_id="h1", refinement_type="add_prediction",
                description="adds", new_predictions=[Prediction(id="pred_x", description="p")])],
            analyst_notes="",
        )
        _, hs = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert len(hs.hypotheses[0].observable_predictions) == 1
