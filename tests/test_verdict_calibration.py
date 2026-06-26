"""Tests for deterministic synthesis verdict calibration."""

from __future__ import annotations

from pt.schemas import (
    BayesianResult,
    HypothesisPosterior,
    HypothesisVerdict,
    SynthesisResult,
)
from pt.verdict_calibration import calibrate_synthesis_verdicts


def _synthesis(status: str) -> SynthesisResult:
    return SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id="h_low",
                status=status,
                key_evidence_for=[],
                key_evidence_against=[],
                reasoning="The model overstated this verdict.",
                steelman="A fair reader could still see a weak possible case.",
                posterior_robustness="moderate",
            )
        ],
        comparative_analysis="comparison",
        analytical_narrative="narrative",
        limitations=[],
        suggested_further_tests=[],
    )


def _bayesian(final_posterior: float, prior: float = 0.20) -> BayesianResult:
    return BayesianResult(
        posteriors=[
            HypothesisPosterior(
                hypothesis_id="h_low",
                prior=prior,
                updates=[],
                final_posterior=final_posterior,
                robustness="moderate",
            )
        ],
        ranking=["h_low"],
    )


def test_calibration_downgrades_low_supported_verdicts():
    calibrated = calibrate_synthesis_verdicts(_synthesis("supported"), _bayesian(0.003))
    verdict = calibrated.verdicts[0]

    assert verdict.status == "eliminated"
    assert "calibrated from supported to eliminated" in verdict.reasoning
    assert "0.003" in verdict.reasoning


def test_calibration_preserves_adequately_supported_verdicts():
    calibrated = calibrate_synthesis_verdicts(_synthesis("supported"), _bayesian(0.30, prior=0.20))

    assert calibrated.verdicts[0].status == "supported"
    assert "calibrated from" not in calibrated.verdicts[0].reasoning


def test_calibration_overwrites_wrong_posterior_robustness():
    """HIGH-3: posterior_robustness is always overwritten with the mechanically computed value."""
    # Synthesis says "robust" but Bayesian computed "fragile"
    synthesis = _synthesis("supported")
    synthesis = synthesis.model_copy(update={"verdicts": [
        synthesis.verdicts[0].model_copy(update={"posterior_robustness": "robust"})
    ]})
    bayesian = BayesianResult(
        posteriors=[
            HypothesisPosterior(
                hypothesis_id="h_low",
                prior=0.20,
                updates=[],
                final_posterior=0.30,
                robustness="fragile",  # computed value differs from synthesis
            )
        ],
        ranking=["h_low"],
    )
    calibrated = calibrate_synthesis_verdicts(synthesis, bayesian)
    verdict = calibrated.verdicts[0]

    assert verdict.posterior_robustness == "fragile"
    assert "posterior_robustness overwritten" in verdict.reasoning


def test_calibration_does_not_touch_correct_robustness():
    """No note is added when verdict robustness already matches computed value."""
    calibrated = calibrate_synthesis_verdicts(_synthesis("supported"), _bayesian(0.30, prior=0.20))

    assert calibrated.verdicts[0].posterior_robustness == "moderate"
    assert "posterior_robustness" not in calibrated.verdicts[0].reasoning


def test_calibration_passthrough_on_missing_hypothesis_id():
    """MED-4: verdict for unknown hypothesis_id passes through uncalibrated."""
    synthesis = SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id="h_orphan",
                status="strongly_supported",
                key_evidence_for=[],
                key_evidence_against=[],
                reasoning="Orphan verdict — no Bayesian entry.",
                steelman="No counter.",
                posterior_robustness="robust",
            )
        ],
        comparative_analysis="comp",
        analytical_narrative="narrative",
        limitations=[],
        suggested_further_tests=[],
    )
    bayesian = _bayesian(0.03)  # only has h_low
    calibrated = calibrate_synthesis_verdicts(synthesis, bayesian)
    verdict = calibrated.verdicts[0]

    # Should pass through unchanged — no Bayesian entry to calibrate against
    assert verdict.status == "strongly_supported"
    assert verdict.posterior_robustness == "robust"
    assert "calibrated" not in verdict.reasoning
