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
    """Downgrade synthesis status labels that contradict computed posteriors.

    Also overwrites posterior_robustness with the mechanically computed value
    from the Bayesian result — the LLM is instructed to copy this field, but if
    it ignores the instruction the default ("robust") would be silently wrong.
    """
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

        updates: dict = {}
        notes: list[str] = []

        if calibrated != verdict.status:
            updates["status"] = calibrated
            notes.append(
                f"Status label calibrated from {verdict.status} to {calibrated} because "
                f"computed comparative support is {posterior.final_posterior:.3f} "
                f"against prior {posterior.prior:.3f}."
            )

        # Always enforce the mechanically computed robustness label.
        if verdict.posterior_robustness != posterior.robustness:
            updates["posterior_robustness"] = posterior.robustness
            notes.append(
                f"posterior_robustness overwritten from {verdict.posterior_robustness!r} "
                f"to {posterior.robustness!r} (mechanically computed)."
            )

        if not updates:
            verdicts.append(verdict)
            continue

        combined_note = " " + " ".join(notes)
        verdicts.append(
            verdict.model_copy(
                update={
                    **updates,
                    "reasoning": f"{verdict.reasoning.rstrip()}{combined_note}",
                }
            )
        )
    return synthesis.model_copy(update={"verdicts": verdicts})
