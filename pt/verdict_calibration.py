"""Deterministic calibration for synthesis verdict status labels."""

from __future__ import annotations

from pt.schemas import BayesianResult, HypothesisVerdict, SynthesisResult, VerdictStatus


LOW_SUPPORT_FLOOR = 0.10
ELIMINATION_FLOOR = 0.02
STRONG_SUPPORT_FLOOR = 0.50


def _calibrated_status(
    status: VerdictStatus,
    *,
    posterior: float,
    prior: float,
) -> VerdictStatus:
    """Return a status label that cannot overstate computed comparative support."""
    if posterior < ELIMINATION_FLOOR and status in {"supported", "strongly_supported"}:
        return "eliminated"
    if posterior < LOW_SUPPORT_FLOOR and status in {"supported", "strongly_supported"}:
        return "weakened"
    if status == "strongly_supported" and posterior < STRONG_SUPPORT_FLOOR:
        return "supported" if posterior >= prior else "weakened"
    if status == "supported" and posterior < prior:
        return "weakened"
    return status


def calibrate_synthesis_verdicts(
    synthesis: SynthesisResult,
    bayesian: BayesianResult,
) -> SynthesisResult:
    """Downgrade synthesis status labels that contradict computed posteriors."""
    posterior_by_id = {p.hypothesis_id: p for p in bayesian.posteriors}
    verdicts: list[HypothesisVerdict] = []
    for verdict in synthesis.verdicts:
        posterior = posterior_by_id.get(verdict.hypothesis_id)
        if posterior is None:
            verdicts.append(verdict)
            continue
        calibrated = _calibrated_status(
            verdict.status,
            posterior=posterior.final_posterior,
            prior=posterior.prior,
        )
        if calibrated == verdict.status:
            verdicts.append(verdict)
            continue
        note = (
            f" Status label calibrated from {verdict.status} to {calibrated} because "
            f"computed comparative support is {posterior.final_posterior:.3f} "
            f"against prior {posterior.prior:.3f}."
        )
        verdicts.append(
            verdict.model_copy(
                update={
                    "status": calibrated,
                    "reasoning": f"{verdict.reasoning.rstrip()}{note}",
                }
            )
        )
    return synthesis.model_copy(update={"verdicts": verdicts})
