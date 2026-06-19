"""Tests for pt/apply_refinement.py — refinement delta application.

Covers the fail-loud behavior on dangling references: a refinement that targets
an id which was removed in the same delta is a legitimate no-op, but a target
that never existed (e.g. a hallucinated id) must raise rather than be skipped.
"""

from __future__ import annotations

import pytest

from pt.apply_refinement import apply_refinement
from pt.schemas import (
    Evidence,
    Hypothesis,
    HypothesisSpace,
    ExtractionResult,
    HypothesisRefinement,
    Prediction,
    RefinementResult,
    ReinterpretedEvidence,
    SpuriousExtraction,
)


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="s",
        evidence=[
            Evidence(id="evi_1", description="d1", source_text="q1",
                     evidence_type="empirical"),
            Evidence(id="evi_2", description="d2", source_text="q2",
                     evidence_type="empirical"),
        ],
    )


def _space() -> HypothesisSpace:
    return HypothesisSpace(research_question="rq", hypotheses=[
        Hypothesis(id="h1", description="d", source="text", theoretical_basis="t",
                   causal_mechanism="m", observable_predictions=[]),
    ])


class TestReinterpretation:
    def test_existing_target_applies(self):
        ref = RefinementResult(
            reinterpreted_evidence=[ReinterpretedEvidence(
                evidence_id="evi_1", original_type="empirical",
                new_type="interpretive", reinterpretation="r")],
            analyst_notes="",
        )
        ext, _ = apply_refinement(_extraction(), _space(), ref, verbose=False)
        ev = {e.id: e for e in ext.evidence}
        assert ev["evi_1"].evidence_type == "interpretive"

    def test_spurious_removed_target_is_noop(self):
        # evi_2 is both removed (spurious) and reinterpreted — legitimate no-op.
        ref = RefinementResult(
            spurious_extractions=[SpuriousExtraction(
                item_id="evi_2", item_type="evidence", reason="x")],
            reinterpreted_evidence=[ReinterpretedEvidence(
                evidence_id="evi_2", original_type="empirical",
                new_type="interpretive", reinterpretation="r")],
            analyst_notes="",
        )
        ext, _ = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert "evi_2" not in {e.id for e in ext.evidence}  # removed, no raise

    def test_unknown_target_fails_loud(self):
        ref = RefinementResult(
            reinterpreted_evidence=[ReinterpretedEvidence(
                evidence_id="evi_ghost", original_type="empirical",
                new_type="interpretive", reinterpretation="r")],
            analyst_notes="",
        )
        with pytest.raises(ValueError, match="unknown evidence id 'evi_ghost'"):
            apply_refinement(_extraction(), _space(), ref, verbose=False)


class TestHypothesisRefinement:
    def test_existing_target_applies(self):
        ref = RefinementResult(
            hypothesis_refinements=[HypothesisRefinement(
                hypothesis_id="h1", refinement_type="add_prediction",
                description="adds", new_predictions=[
                    Prediction(id="pred_x", description="p")])],
            analyst_notes="",
        )
        _, hs = apply_refinement(_extraction(), _space(), ref, verbose=False)
        assert len(hs.hypotheses[0].observable_predictions) == 1

    def test_unknown_target_fails_loud(self):
        ref = RefinementResult(
            hypothesis_refinements=[HypothesisRefinement(
                hypothesis_id="h_ghost", refinement_type="sharpen_mechanism",
                description="x", updated_causal_mechanism="m2")],
            analyst_notes="",
        )
        with pytest.raises(ValueError, match="unknown hypothesis id 'h_ghost'"):
            apply_refinement(_extraction(), _space(), ref, verbose=False)
