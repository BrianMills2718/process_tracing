"""Pure math: Bayesian updating in odds space."""

from __future__ import annotations

import math

from pt.schemas import (
    BayesianResult,
    EvidenceUpdate,
    HypothesisPosterior,
    TestingResult,
)

CLAMP_MIN = 0.001
CLAMP_MAX = 0.999
LR_CAP = 20.0
LR_FLOOR = 1.0 / LR_CAP  # 0.05


def _clamp(p: float) -> float:
    return max(CLAMP_MIN, min(CLAMP_MAX, p))


def _odds(p: float) -> float:
    p = _clamp(p)
    return p / (1.0 - p)


def _prob(odds: float) -> float:
    return _clamp(odds / (1.0 + odds))


def run_bayesian_update(testing: TestingResult) -> BayesianResult:
    """Compute posterior probabilities via sequential Bayes updates.

    - Uniform priors (1/n).
    - For each hypothesis, apply every evidence evaluation as a likelihood ratio update.
    - Normalize across hypotheses at the end.
    """
    n = len(testing.hypothesis_tests)
    if n == 0:
        return BayesianResult(posteriors=[], ranking=[])

    uniform_prior = 1.0 / n
    posteriors: list[HypothesisPosterior] = []

    for ht in testing.hypothesis_tests:
        current_prob = uniform_prior
        updates: list[EvidenceUpdate] = []

        for ev_eval in ht.evidence_evaluations:
            p_e_h = max(ev_eval.p_e_given_h, 0.001)
            p_e_nh = max(ev_eval.p_e_given_not_h, 0.001)
            raw_lr = p_e_h / p_e_nh

            # Cap LR to prevent single evidence items from dominating
            capped_lr = max(LR_FLOOR, min(LR_CAP, raw_lr))

            # Hard relevance gate: very low relevance evidence is uninformative
            tr = ev_eval.relevance  # 0.0=irrelevant, 1.0=fully relevant
            if tr < 0.4:
                lr = 1.0  # Force uninformative — cannot discriminate
            else:
                # Relevance discount: pull LR toward 1.0
                lr = math.exp(tr * math.log(capped_lr)) if capped_lr > 0 else capped_lr

            prior_for_update = current_prob
            current_odds = _odds(current_prob) * lr
            current_prob = _prob(current_odds)

            updates.append(EvidenceUpdate(
                evidence_id=ev_eval.evidence_id,
                prediction_id=ev_eval.prediction_id,
                likelihood_ratio=round(lr, 4),
                prior=round(prior_for_update, 6),
                posterior=round(current_prob, 6),
            ))

        posteriors.append(HypothesisPosterior(
            hypothesis_id=ht.hypothesis_id,
            prior=round(uniform_prior, 6),
            updates=updates,
            final_posterior=current_prob,
        ))

    # Normalize so posteriors sum to 1
    total = sum(p.final_posterior for p in posteriors)
    if total > 0:
        for p in posteriors:
            p.final_posterior = round(_clamp(p.final_posterior / total), 6)

    # Rank by posterior (highest first)
    ranking = sorted(posteriors, key=lambda p: p.final_posterior, reverse=True)
    ranking_ids = [p.hypothesis_id for p in ranking]

    return BayesianResult(posteriors=posteriors, ranking=ranking_ids)
