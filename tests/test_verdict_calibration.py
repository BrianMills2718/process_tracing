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
