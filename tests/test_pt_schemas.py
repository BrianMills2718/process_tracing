"""Tests for pt/schemas.py — Pydantic model validation."""

import pytest
from pydantic import ValidationError

from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    Evidence,
    EvidenceEvaluation,
    EvidenceUpdate,
    ExtractionResult,
    Hypothesis,
    HypothesisPosterior,
    HypothesisSpace,
    HypothesisTestResult,
    HypothesisVerdict,
    ProcessTracingResult,
    SynthesisResult,
    TestingResult,
)


class TestEvidence:
    def test_defaults(self):
        e = Evidence(id="e1", description="test", source_text="quote")
        assert e.evidence_type == "empirical"
        assert e.approximate_date is None

    def test_interpretive_type(self):
        e = Evidence(id="e1", description="test", source_text="quote", evidence_type="interpretive")
        assert e.evidence_type == "interpretive"

    def test_with_date(self):
        e = Evidence(id="e1", description="test", source_text="quote", approximate_date="1789-07")
        assert e.approximate_date == "1789-07"


class TestEvidenceEvaluation:
    def test_defaults(self):
        ee = EvidenceEvaluation(
            evidence_id="e1",
            hypothesis_id="h1",
            finding="pass",
            p_e_given_h=0.8,
            p_e_given_not_h=0.2,
            justification="test",
        )
        assert ee.relevance == 1.0
        assert ee.prediction_id is None

    def test_probability_bounds(self):
        with pytest.raises(ValidationError):
            EvidenceEvaluation(
                evidence_id="e1",
                hypothesis_id="h1",
                finding="pass",
                p_e_given_h=1.5,  # out of bounds
                p_e_given_not_h=0.2,
                justification="test",
            )

    def test_relevance_bounds(self):
        with pytest.raises(ValidationError):
            EvidenceEvaluation(
                evidence_id="e1",
                hypothesis_id="h1",
                finding="pass",
                p_e_given_h=0.8,
                p_e_given_not_h=0.2,
                justification="test",
                relevance=1.5,  # out of bounds
            )

    def test_zero_probabilities_valid(self):
        ee = EvidenceEvaluation(
            evidence_id="e1",
            hypothesis_id="h1",
            finding="fail",
            p_e_given_h=0.0,
            p_e_given_not_h=0.0,
            justification="test",
        )
        assert ee.p_e_given_h == 0.0


class TestExtractionResult:
    def test_minimal(self):
        er = ExtractionResult(summary="Test summary")
        assert er.actors == []
        assert er.evidence == []

    def test_full(self):
        er = ExtractionResult(
            summary="Test",
            evidence=[Evidence(id="e1", description="d", source_text="s")],
        )
        assert len(er.evidence) == 1


class TestHypothesisSpace:
    def test_requires_question(self):
        with pytest.raises(ValidationError):
            HypothesisSpace(hypotheses=[])

    def test_valid(self):
        hs = HypothesisSpace(
            research_question="Why did X happen?",
            hypotheses=[
                Hypothesis(
                    id="h1",
                    description="test",
                    source="text",
                    theoretical_basis="basis",
                    causal_mechanism="mechanism",
                    observable_predictions=[],
                )
            ],
        )
        assert len(hs.hypotheses) == 1


class TestProcessTracingResult:
    def test_all_fields_required(self):
        """ProcessTracingResult requires all 5 pipeline outputs."""
        with pytest.raises(ValidationError):
            ProcessTracingResult(
                extraction=ExtractionResult(summary="test"),
                # missing other fields
            )

    def test_full_construction(self):
        result = ProcessTracingResult(
            extraction=ExtractionResult(summary="test"),
            hypothesis_space=HypothesisSpace(
                research_question="Why?",
                hypotheses=[],
            ),
            testing=TestingResult(hypothesis_tests=[]),
            absence=AbsenceResult(evaluations=[]),
            bayesian=BayesianResult(posteriors=[], ranking=[]),
            synthesis=SynthesisResult(
                verdicts=[],
                comparative_analysis="analysis",
                analytical_narrative="narrative",
                limitations=["l1"],
                suggested_further_tests=["t1"],
            ),
        )
        assert result.extraction.summary == "test"


class TestSerialization:
    def test_roundtrip(self):
        """Model can serialize to dict and back."""
        result = BayesianResult(
            posteriors=[
                HypothesisPosterior(
                    hypothesis_id="h1",
                    prior=0.5,
                    updates=[
                        EvidenceUpdate(
                            evidence_id="e1",
                            likelihood_ratio=2.0,
                            prior=0.5,
                            posterior=0.667,
                        )
                    ],
                    final_posterior=0.667,
                )
            ],
            ranking=["h1"],
        )
        d = result.model_dump()
        restored = BayesianResult(**d)
        assert restored.posteriors[0].hypothesis_id == "h1"
        assert restored.posteriors[0].updates[0].likelihood_ratio == 2.0
