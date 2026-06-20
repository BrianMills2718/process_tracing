"""Pipeline integration test — mocks LLM, verifies orchestration and Bayesian math.

mock-ok: This test verifies pipeline orchestration logic and Bayesian math
with deterministic data. Real LLM calls would be non-deterministic and expensive.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pt.bayesian import run_bayesian_update
from pt.pipeline import run_pipeline
from pt.report import generate_report
from pt.schemas import (
    AbsenceEvaluation,
    AbsenceResult,
    Actor,
    CausalEdge,
    Evidence,
    EvidenceLikelihood,
    Event,
    ExtractionResult,
    Hypothesis,
    HypothesisLikelihood,
    HypothesisSpace,
    HypothesisVerdict,
    Mechanism,
    Prediction,
    ProcessTracingResult,
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


def _ev_like(evidence_id: str, h1: float, h2: float, relevance: float, dtype: str = "straw_in_the_wind") -> EvidenceLikelihood:
    """One evidence item's likelihood vector across {h1, h2}."""
    return EvidenceLikelihood(
        evidence_id=evidence_id,
        hypothesis_likelihoods=[
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=h1, diagnostic_type=dtype),
            HypothesisLikelihood(hypothesis_id="h2", relative_likelihood=h2, diagnostic_type=dtype),
        ],
        relevance=relevance,
        justification="deterministic test vector",
    )


def _make_testing() -> TestingResult:
    """Deterministic likelihood vectors. Relative likelihood = P(E|H) per hypothesis;
    derived per-hypothesis LR is the value over the vector's geometric mean."""
    return TestingResult(
        evidence_likelihoods=[
            # evi_debt: favors h1 (fiscal)
            _ev_like("evi_debt", h1=0.9, h2=0.3, relevance=0.9, dtype="smoking_gun"),
            # evi_tax_revolt: favors h1
            _ev_like("evi_tax_revolt", h1=0.8, h2=0.4, relevance=0.85, dtype="straw_in_the_wind"),
            # evi_elite_plot: favors h2 (conspiracy)
            _ev_like("evi_elite_plot", h1=0.3, h2=0.9, relevance=0.9, dtype="smoking_gun"),
            # evi_historian_claim: interpretive, low relevance, uninformative
            _ev_like("evi_historian_claim", h1=0.5, h2=0.5, relevance=0.3, dtype="straw_in_the_wind"),
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


def _mock_call_llm(prompt: str, response_model: type, *, task: str = "", trace_id: str = "", **kwargs):
    """Return deterministic data based on the response model type."""
    model_name = response_model.__name__

    if model_name == "ExtractionResult":
        return _make_extraction()
    elif model_name == "HypothesisSpace":
        return _make_hypothesis_space()
    elif model_name == "TestingResult":
        return _make_testing()
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
        assert len(pipeline_result.testing.evidence_likelihoods) == 4
        for item in pipeline_result.testing.evidence_likelihoods:
            assert len(item.hypothesis_likelihoods) == 2

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
        result = run_bayesian_update(testing, ["h1", "h2"])

        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        h2 = next(p for p in result.posteriors if p.hypothesis_id == "h2")

        # Two items favor h1 (debt, tax revolt), one favors h2 (elite plot), one
        # is uninformative — net, h1 comes out ahead.
        assert h1.final_posterior > h2.final_posterior
        assert h1.final_posterior + h2.final_posterior == pytest.approx(1.0, abs=0.01)

        # One update per evidence item, per hypothesis.
        assert len(h1.updates) == 4
        assert len(h2.updates) == 4

        assert result.ranking == ["h1", "h2"]


class TestReportConsistency:
    """Report display should match Bayesian updater semantics."""

    def test_low_relevance_extreme_evidence_hidden_as_uninformative(self):
        extraction = _make_extraction()
        hypothesis_space = _make_hypothesis_space()
        # Extreme vector but relevance below the gate ⇒ forced uninformative (LR 1.0).
        testing = TestingResult(
            evidence_likelihoods=[
                _ev_like("evi_debt", h1=1.0, h2=0.001, relevance=0.39, dtype="smoking_gun"),
            ],
        )
        bayesian = run_bayesian_update(testing, ["h1", "h2"])
        result = ProcessTracingResult(
            extraction=extraction,
            hypothesis_space=hypothesis_space,
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=bayesian,
            synthesis=_make_synthesis(),
        )

        html = generate_report(result)

        # Both hypotheses' LR for the gated item is 1.0.
        assert bayesian.posteriors[0].updates[0].likelihood_ratio == pytest.approx(1.0)
        # Report counts evaluations per (hypothesis, evidence): 2 hyps × 1 item = 2 total.
        assert "0 informative / 2 total evaluations shown" in html
        assert "LR=1000.00" not in html

    def test_sensitivity_populated(self):
        testing = _make_testing()
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert len(result.sensitivity) == 2
        for s in result.sensitivity:
            assert s.posterior_low <= s.baseline_posterior <= s.posterior_high

    def test_robustness_populated(self):
        testing = _make_testing()
        result = run_bayesian_update(testing, ["h1", "h2"])
        for p in result.posteriors:
            assert p.robustness in ("robust", "fragile", "moderate", "unknown")


class TestVectorCompleteness:
    """run_test must fail loud on incomplete / malformed likelihood matrices."""

    def _run_with(self, testing):
        from pt import pass_test
        with patch.object(pass_test, "call_llm", side_effect=lambda *a, **k: testing):
            return pass_test.run_test(_make_extraction(), _make_hypothesis_space())

    def test_rejects_missing_hypothesis_in_vector(self):
        # One item's vector covers only h1, not {h1, h2}.
        good = _make_testing()
        good.evidence_likelihoods[0].hypothesis_likelihoods = [
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=1.0, diagnostic_type="hoop")
        ]
        with pytest.raises(ValueError, match="expected exactly"):
            self._run_with(good)

    def test_rejects_missing_evidence_item(self):
        good = _make_testing()
        good.evidence_likelihoods = good.evidence_likelihoods[:3]  # drop one of 4
        with pytest.raises(ValueError, match="coverage mismatch"):
            self._run_with(good)

    def test_accepts_complete_matrix(self):
        result = self._run_with(_make_testing())
        assert len(result.evidence_likelihoods) == 4


class TestExecutiveSummary:
    """Slice 4 + truth-in-labeling: the headline surfaces a support interval and
    stability flags, framed as comparative support (not absolute probability)."""

    def _result(self):
        testing = _make_testing()
        return ProcessTracingResult(
            extraction=_make_extraction(),
            hypothesis_space=_make_hypothesis_space(),
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=run_bayesian_update(testing, ["h1", "h2"]),
            synthesis=_make_synthesis(),
        )

    def test_no_absolute_probability_overclaim(self):
        html = generate_report(self._result())
        assert "Posterior probability after Bayesian updating" not in html
        assert "Support:" in html

    def test_comparative_support_caveat(self):
        html = " ".join(generate_report(self._result()).split()).lower()
        assert "comparative support" in html
        assert "not absolute probabilities of truth" in html

    def test_support_interval_and_stability_surfaced(self):
        html = generate_report(self._result())
        assert "range " in html  # support interval badge
        assert ("robust to prior" in html) or ("prior-sensitive" in html)

    def test_overconfidence_banner_on_degenerate_fragile_posterior(self):
        # Many weakly-pro-h1 items -> near-1.0 support, fragile -> warning banner.
        items = [_ev_like(f"e{i}", h1=2.0, h2=1.0, relevance=0.9) for i in range(20)]
        testing = TestingResult(evidence_likelihoods=items)
        result = ProcessTracingResult(
            extraction=_make_extraction(),
            hypothesis_space=_make_hypothesis_space(),
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=run_bayesian_update(testing, ["h1", "h2"]),
            synthesis=_make_synthesis(),
        )
        assert result.bayesian.posteriors[0].final_posterior > 0.99
        assert "Likely overconfident" in generate_report(result)

    def test_no_overconfidence_banner_on_normal_result(self):
        assert "Likely overconfident" not in generate_report(self._result())
