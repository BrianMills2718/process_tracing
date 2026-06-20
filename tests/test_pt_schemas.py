"""Tests for pt/schemas.py — Pydantic model validation."""

import pytest
from pydantic import ValidationError

from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    Evidence,
    EvidenceLikelihood,
    EvidenceUpdate,
    ExtractionResult,
    Hypothesis,
    HypothesisLikelihood,
    HypothesisPosterior,
    HypothesisSpace,
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

    def test_evidence_type_rejects_invalid(self):
        with pytest.raises(ValidationError):
            Evidence(id="e1", description="test", source_text="quote", evidence_type="anecdotal")


def _likelihood(relevance: float = 1.0) -> EvidenceLikelihood:
    return EvidenceLikelihood(
        evidence_id="e1",
        hypothesis_likelihoods=[
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=4.0, diagnostic_type="smoking_gun"),
            HypothesisLikelihood(hypothesis_id="h2", relative_likelihood=1.0, diagnostic_type="hoop"),
        ],
        relevance=relevance,
        justification="test",
    )


class TestEvidenceLikelihood:
    def test_defaults(self):
        el = _likelihood()
        assert el.relevance == 1.0
        assert len(el.hypothesis_likelihoods) == 2

    def test_relative_likelihood_must_be_positive(self):
        with pytest.raises(ValidationError):
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=0.0, diagnostic_type="hoop")

    def test_relevance_bounds(self):
        with pytest.raises(ValidationError):
            _likelihood(relevance=1.5)  # out of bounds

    def test_vector_values_preserved(self):
        el = _likelihood()
        by_id = {hl.hypothesis_id: hl.relative_likelihood for hl in el.hypothesis_likelihoods}
        assert by_id == {"h1": 4.0, "h2": 1.0}

    def test_diagnostic_type_rejects_invalid(self):
        with pytest.raises(ValidationError):
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=1.0, diagnostic_type="bogus")

    def test_relative_likelihood_rejects_non_finite(self):
        for bad in (float("inf"), float("nan")):
            with pytest.raises(ValidationError):
                HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=bad, diagnostic_type="hoop")


class TestUpstreamIdUniqueness:
    def test_extraction_rejects_duplicate_evidence_ids(self):
        with pytest.raises(ValidationError):
            ExtractionResult(
                summary="s",
                evidence=[
                    Evidence(id="e1", description="a", source_text="q"),
                    Evidence(id="e1", description="b", source_text="r"),
                ],
            )

    def test_hypothesis_space_rejects_duplicate_ids(self):
        def _h(hid):
            return Hypothesis(id=hid, description="d", source="text",
                              theoretical_basis="t", causal_mechanism="m", observable_predictions=[])
        with pytest.raises(ValidationError):
            HypothesisSpace(research_question="rq", hypotheses=[_h("h1"), _h("h1")])


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
            testing=TestingResult(evidence_likelihoods=[]),
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
