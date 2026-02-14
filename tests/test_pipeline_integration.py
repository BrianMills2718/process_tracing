"""Pipeline integration test — mocks LLM, verifies orchestration and Bayesian math.

mock-ok: This test verifies pipeline orchestration logic and Bayesian math
with deterministic data. Real LLM calls would be non-deterministic and expensive.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pt.bayesian import run_bayesian_update
from pt.pipeline import run_pipeline
from pt.schemas import (
    AbsenceEvaluation,
    AbsenceResult,
    Actor,
    CausalEdge,
    Evidence,
    EvidenceEvaluation,
    Event,
    ExtractionResult,
    Hypothesis,
    HypothesisSpace,
    HypothesisTestResult,
    HypothesisVerdict,
    Mechanism,
    Prediction,
    SynthesisResult,
    TestingResult,
    TextHypothesis,
)


# ── Deterministic fixtures ─────────────────────────────────────────


def _make_extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="Test text about a political crisis caused by fiscal collapse and elite maneuvering.",
        actors=[
            Actor(id="actor_king", name="The King", description="Head of state"),
            Actor(id="actor_assembly", name="Assembly", description="Legislative body"),
        ],
        events=[
            Event(id="evt_crisis", description="Fiscal crisis erupts", date="1789"),
            Event(id="evt_coup", description="Military coup occurs", date="1799"),
        ],
        mechanisms=[
            Mechanism(id="mech_fiscal", description="Fiscal collapse undermines state legitimacy"),
        ],
        evidence=[
            Evidence(
                id="evi_debt",
                description="National debt doubled in a decade",
                source_text="The national debt had doubled between 1780 and 1789.",
                evidence_type="empirical",
                approximate_date="1789",
            ),
            Evidence(
                id="evi_tax_revolt",
                description="Tax revolt in provinces",
                source_text="Provinces refused to collect new taxes.",
                evidence_type="empirical",
                approximate_date="1789",
            ),
            Evidence(
                id="evi_elite_plot",
                description="Elite conspirators met secretly",
                source_text="Key leaders met in secret to plan the overthrow.",
                evidence_type="empirical",
                approximate_date="1799",
            ),
            Evidence(
                id="evi_historian_claim",
                description="Historian argues ideology was the primary driver",
                source_text="Furet argues that ideological radicalization was the key factor.",
                evidence_type="interpretive",
            ),
        ],
        hypotheses_in_text=[
            TextHypothesis(
                id="th1",
                description="Fiscal crisis caused the revolution",
                source_text="The fiscal crisis was the primary cause.",
            ),
        ],
        causal_edges=[
            CausalEdge(source_id="evt_crisis", target_id="evt_coup", relationship="led to"),
        ],
    )


def _make_hypothesis_space() -> HypothesisSpace:
    return HypothesisSpace(
        research_question="Why did the political crisis resolve via military coup rather than reform?",
        hypotheses=[
            Hypothesis(
                id="h1",
                description="Fiscal collapse made the state ungovernable",
                source="text",
                theoretical_basis="Fiscal-military state theory",
                causal_mechanism="Debt → inability to fund military/bureaucracy → state collapse → power vacuum",
                observable_predictions=[
                    Prediction(id="pred_h1_01", description="We should see evidence of debt crisis"),
                    Prediction(id="pred_h1_02", description="We should see state inability to function"),
                ],
            ),
            Hypothesis(
                id="h2",
                description="Elite conspiracy orchestrated the coup",
                source="generated",
                theoretical_basis="Elite theory of revolution",
                causal_mechanism="Small group of elites planned and executed power seizure",
                observable_predictions=[
                    Prediction(id="pred_h2_01", description="Evidence of secret meetings"),
                    Prediction(id="pred_h2_02", description="Named conspirators with specific plan"),
                ],
            ),
        ],
    )


def _make_testing() -> TestingResult:
    """Deterministic testing results with known LR values for Bayesian verification."""
    return TestingResult(
        hypothesis_tests=[
            HypothesisTestResult(
                hypothesis_id="h1",
                prediction_classifications=[],
                evidence_evaluations=[
                    # evi_debt: strong for h1 (fiscal)
                    EvidenceEvaluation(
                        evidence_id="evi_debt",
                        hypothesis_id="h1",
                        finding="pass",
                        p_e_given_h=0.9,
                        p_e_given_not_h=0.3,
                        justification="Debt directly supports fiscal hypothesis",
                        relevance=0.9,
                    ),
                    # evi_tax_revolt: moderate for h1
                    EvidenceEvaluation(
                        evidence_id="evi_tax_revolt",
                        hypothesis_id="h1",
                        finding="pass",
                        p_e_given_h=0.8,
                        p_e_given_not_h=0.4,
                        justification="Tax revolt consistent with fiscal breakdown",
                        relevance=0.85,
                    ),
                    # evi_elite_plot: against h1
                    EvidenceEvaluation(
                        evidence_id="evi_elite_plot",
                        hypothesis_id="h1",
                        finding="fail",
                        p_e_given_h=0.3,
                        p_e_given_not_h=0.7,
                        justification="Elite plotting not predicted by fiscal hypothesis",
                        relevance=0.7,
                    ),
                    # evi_historian_claim: weak/neutral
                    EvidenceEvaluation(
                        evidence_id="evi_historian_claim",
                        hypothesis_id="h1",
                        finding="ambiguous",
                        p_e_given_h=0.5,
                        p_e_given_not_h=0.5,
                        justification="Interpretive claim, not diagnostic",
                        relevance=0.3,
                    ),
                ],
            ),
            HypothesisTestResult(
                hypothesis_id="h2",
                prediction_classifications=[],
                evidence_evaluations=[
                    # evi_debt: against h2
                    EvidenceEvaluation(
                        evidence_id="evi_debt",
                        hypothesis_id="h2",
                        finding="fail",
                        p_e_given_h=0.3,
                        p_e_given_not_h=0.7,
                        justification="Debt not predicted by elite conspiracy",
                        relevance=0.5,
                    ),
                    # evi_tax_revolt: neutral for h2
                    EvidenceEvaluation(
                        evidence_id="evi_tax_revolt",
                        hypothesis_id="h2",
                        finding="ambiguous",
                        p_e_given_h=0.4,
                        p_e_given_not_h=0.5,
                        justification="Tax revolt not relevant to conspiracy",
                        relevance=0.4,
                    ),
                    # evi_elite_plot: strong for h2
                    EvidenceEvaluation(
                        evidence_id="evi_elite_plot",
                        hypothesis_id="h2",
                        finding="pass",
                        p_e_given_h=0.9,
                        p_e_given_not_h=0.2,
                        justification="Secret meetings directly support conspiracy hypothesis",
                        relevance=0.95,
                    ),
                    # evi_historian_claim: neutral
                    EvidenceEvaluation(
                        evidence_id="evi_historian_claim",
                        hypothesis_id="h2",
                        finding="ambiguous",
                        p_e_given_h=0.5,
                        p_e_given_not_h=0.5,
                        justification="Interpretive claim about ideology, not conspiracy",
                        relevance=0.2,
                    ),
                ],
            ),
        ],
    )


def _make_absence() -> AbsenceResult:
    return AbsenceResult(
        evaluations=[
            AbsenceEvaluation(
                hypothesis_id="h2",
                prediction_id="pred_h2_02",
                missing_evidence="No specific names of conspirators or details of their plan",
                reasoning="A conspiracy hypothesis requires named agents with specific actions",
                severity="notable",
                would_be_extractable=True,
            ),
        ],
    )


def _make_synthesis() -> SynthesisResult:
    return SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id="h1",
                status="supported",
                key_evidence_for=["evi_debt", "evi_tax_revolt"],
                key_evidence_against=["evi_elite_plot"],
                reasoning="Fiscal evidence is strong and direct.",
                steelman="The fiscal crisis was catastrophic and undeniable.",
                posterior_robustness="moderate",
            ),
            HypothesisVerdict(
                hypothesis_id="h2",
                status="weakened",
                key_evidence_for=["evi_elite_plot"],
                key_evidence_against=["evi_debt"],
                reasoning="Only one piece of strong evidence.",
                steelman="The elite plot evidence is a smoking gun.",
                posterior_robustness="fragile",
            ),
        ],
        comparative_analysis="H1 has broader evidentiary support while H2 relies on a single item.",
        analytical_narrative="The fiscal hypothesis emerges stronger from the analysis.",
        limitations=["Small evidence base", "Only one text analyzed"],
        suggested_further_tests=["Archival records of conspirators' correspondence"],
    )


# ── Mock dispatcher ────────────────────────────────────────────────


def _mock_call_llm(prompt: str, response_model: type, **kwargs):
    """Return deterministic data based on the response model type."""
    model_name = response_model.__name__

    if model_name == "ExtractionResult":
        return _make_extraction()
    elif model_name == "HypothesisSpace":
        return _make_hypothesis_space()
    elif model_name == "HypothesisTestResult":
        # Determine which hypothesis is being tested from the prompt
        testing = _make_testing()
        if '"h2"' in prompt or "'h2'" in prompt:
            return testing.hypothesis_tests[1]
        return testing.hypothesis_tests[0]
    elif model_name == "AbsenceResult":
        return _make_absence()
    elif model_name == "SynthesisResult":
        return _make_synthesis()
    else:
        raise ValueError(f"Unexpected response_model in mock: {model_name}")


# ── Tests ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_short_text(self):
        with pytest.raises(ValueError, match="too short"):
            run_pipeline("This is too short.", verbose=False)

    def test_rejects_empty_text(self):
        with pytest.raises(ValueError, match="too short"):
            run_pipeline("", verbose=False)

    def test_accepts_300_word_text(self):
        """300 words should pass validation (LLM is mocked)."""
        text = " ".join(["word"] * 300)
        with patch("pt.pass_extract.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_hypothesize.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_test.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_absence.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_synthesize.call_llm", side_effect=_mock_call_llm):
            result = run_pipeline(text, verbose=False)
            assert result.extraction.summary is not None


class TestPipelineOrchestration:
    """Verify the pipeline calls passes in correct order with correct data flow."""

    @pytest.fixture()
    def pipeline_result(self):
        text = " ".join(["substantive"] * 400)
        with patch("pt.pass_extract.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_hypothesize.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_test.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_absence.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_synthesize.call_llm", side_effect=_mock_call_llm):
            return run_pipeline(text, verbose=False)

    def test_extraction_populated(self, pipeline_result):
        assert len(pipeline_result.extraction.evidence) == 4
        assert len(pipeline_result.extraction.actors) == 2

    def test_hypotheses_populated(self, pipeline_result):
        assert len(pipeline_result.hypothesis_space.hypotheses) == 2
        assert pipeline_result.hypothesis_space.research_question

    def test_testing_populated(self, pipeline_result):
        assert len(pipeline_result.testing.hypothesis_tests) == 2
        for ht in pipeline_result.testing.hypothesis_tests:
            assert len(ht.evidence_evaluations) == 4

    def test_absence_populated(self, pipeline_result):
        assert len(pipeline_result.absence.evaluations) == 1

    def test_bayesian_posteriors_sum_to_one(self, pipeline_result):
        total = sum(p.final_posterior for p in pipeline_result.bayesian.posteriors)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_bayesian_ranking(self, pipeline_result):
        # With our test data, h1 should rank higher (more supporting evidence)
        assert pipeline_result.bayesian.ranking[0] == "h1"

    def test_synthesis_has_verdicts(self, pipeline_result):
        assert len(pipeline_result.synthesis.verdicts) == 2

    def test_not_refined_by_default(self, pipeline_result):
        assert pipeline_result.is_refined is False
        assert pipeline_result.refinement is None


class TestBayesianMathDeterministic:
    """Verify Bayesian math produces expected posteriors from fixed test data."""

    def test_posteriors_from_fixed_testing(self):
        testing = _make_testing()
        result = run_bayesian_update(testing)

        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        h2 = next(p for p in result.posteriors if p.hypothesis_id == "h2")

        # h1 has: 0.9/0.3=3.0 for, 0.8/0.4=2.0 for, 0.3/0.7≈0.43 against, 0.5/0.5=1.0 neutral
        # h2 has: 0.3/0.7≈0.43 against, 0.4/0.5=0.8 against, 0.9/0.2=4.5 for, 0.5/0.5=1.0 neutral
        # Both have a mix of for/against — h1 should come out ahead
        assert h1.final_posterior > h2.final_posterior
        assert h1.final_posterior + h2.final_posterior == pytest.approx(1.0, abs=0.01)

        # Verify update trails exist
        assert len(h1.updates) == 4
        assert len(h2.updates) == 4

        # Verify ranking
        assert result.ranking == ["h1", "h2"]

    def test_sensitivity_populated(self):
        testing = _make_testing()
        result = run_bayesian_update(testing)
        assert len(result.sensitivity) == 2
        for s in result.sensitivity:
            assert s.posterior_low <= s.baseline_posterior <= s.posterior_high

    def test_robustness_populated(self):
        testing = _make_testing()
        result = run_bayesian_update(testing)
        for p in result.posteriors:
            assert p.robustness in ("robust", "fragile", "moderate", "unknown")
