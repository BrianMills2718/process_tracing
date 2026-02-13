"""Pure math: Bayesian updating in odds space."""

from __future__ import annotations

import math

from pt.schemas import (
    BayesianResult,
    EvidenceUpdate,
    HypothesisPosterior,
    SensitivityEntry,
    TestingResult,
)

CLAMP_MIN = 0.001
CLAMP_MAX = 0.999
LR_CAP = 20.0
LR_FLOOR = 1.0 / LR_CAP  # 0.05

# Robustness thresholds
DECISIVE_LR_THRESHOLD = 1.6   # |log(LR)| > 1.6 means LR > 5.0 or < 0.2
WEAK_LR_THRESHOLD = 0.7       # |log(LR)| < 0.7 means LR between 0.5 and 2.0
DECISIVE_COUNT_FOR_ROBUST = 3  # Need at least this many decisive items

# Sensitivity
PERTURB_FACTOR = 0.5  # ±50% perturbation on log-LR
TOP_DRIVERS_COUNT = 5  # Number of most influential LRs to perturb


def _clamp(p: float) -> float:
    return max(CLAMP_MIN, min(CLAMP_MAX, p))


def _odds(p: float) -> float:
    p = _clamp(p)
    return p / (1.0 - p)


def _prob(odds: float) -> float:
    return _clamp(odds / (1.0 + odds))


def _compute_lr(ev_eval) -> float:
    """Compute the effective LR for an evidence evaluation (same logic as main update)."""
    p_e_h = max(ev_eval.p_e_given_h, 0.001)
    p_e_nh = max(ev_eval.p_e_given_not_h, 0.001)
    raw_lr = p_e_h / p_e_nh
    capped_lr = max(LR_FLOOR, min(LR_CAP, raw_lr))
    tr = ev_eval.relevance
    if tr < 0.4:
        return 1.0
    return math.exp(tr * math.log(capped_lr)) if capped_lr > 0 else capped_lr


def _compute_robustness(updates: list[EvidenceUpdate]) -> str:
    """Determine if posterior is 'robust' or 'fragile' from LR distribution.

    Robust: posterior driven by a few decisive LRs (|log(LR)| > 1.6, i.e. LR > 5 or < 0.2).
    Fragile: posterior driven by accumulation of many weak LRs (|log(LR)| < 0.7).
    """
    if not updates:
        return "unknown"

    log_lrs = [abs(math.log(u.likelihood_ratio)) for u in updates if u.likelihood_ratio > 0]
    if not log_lrs:
        return "unknown"

    n_decisive = sum(1 for ll in log_lrs if ll > DECISIVE_LR_THRESHOLD)
    n_weak = sum(1 for ll in log_lrs if ll < WEAK_LR_THRESHOLD)
    n_informative = sum(1 for ll in log_lrs if ll > 0.01)  # anything not exactly 1.0

    if n_informative == 0:
        return "unknown"

    # Total log-odds movement from decisive items vs weak items
    decisive_impact = sum(ll for ll in log_lrs if ll > DECISIVE_LR_THRESHOLD)
    total_impact = sum(log_lrs)

    if total_impact < 0.01:
        return "unknown"

    decisive_fraction = decisive_impact / total_impact

    if n_decisive >= DECISIVE_COUNT_FOR_ROBUST and decisive_fraction > 0.5:
        return "robust"
    elif n_weak > n_decisive * 3 and decisive_fraction < 0.3:
        return "fragile"
    else:
        return "moderate"


def _top_drivers(updates: list[EvidenceUpdate], n: int = 3) -> list[str]:
    """Return evidence IDs of the N most influential LR updates."""
    scored = [
        (abs(math.log(u.likelihood_ratio)) if u.likelihood_ratio > 0 else 0, u.evidence_id)
        for u in updates
    ]
    scored.sort(reverse=True)
    return [eid for _, eid in scored[:n]]


def _posterior_from_lrs(prior: float, lrs: list[float]) -> float:
    """Compute posterior from a list of LRs (used for sensitivity)."""
    current = prior
    for lr in lrs:
        current = _prob(_odds(current) * lr)
    return current


def run_bayesian_update(testing: TestingResult) -> BayesianResult:
    """Compute posterior probabilities via sequential Bayes updates.

    - Uniform priors (1/n).
    - For each hypothesis, apply every evidence evaluation as a likelihood ratio update.
    - Normalize across hypotheses at the end.
    - Compute mechanical robustness and sensitivity analysis.
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
            lr = _compute_lr(ev_eval)

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
            robustness=_compute_robustness(updates),
            top_drivers=_top_drivers(updates),
        ))

    # Normalize so posteriors sum to 1
    total = sum(p.final_posterior for p in posteriors)
    if total > 0:
        for p in posteriors:
            p.final_posterior = round(_clamp(p.final_posterior / total), 6)

    # Rank by posterior (highest first)
    ranking = sorted(posteriors, key=lambda p: p.final_posterior, reverse=True)
    ranking_ids = [p.hypothesis_id for p in ranking]

    # Sensitivity analysis: perturb top drivers ±50% on log-LR scale
    sensitivity = _run_sensitivity(testing, posteriors, uniform_prior)

    return BayesianResult(posteriors=posteriors, ranking=ranking_ids, sensitivity=sensitivity)


def _run_sensitivity(
    testing: TestingResult,
    posteriors: list[HypothesisPosterior],
    uniform_prior: float,
) -> list[SensitivityEntry]:
    """Perturb the most influential LRs and measure posterior stability."""
    # Collect all (hypothesis_index, evidence_index, |log_lr|) for ranking
    all_drivers: list[tuple[int, int, float]] = []
    for h_idx, ht in enumerate(testing.hypothesis_tests):
        for e_idx, ev_eval in enumerate(ht.evidence_evaluations):
            lr = _compute_lr(ev_eval)
            log_lr = abs(math.log(lr)) if lr > 0 else 0
            if log_lr > 0.01:  # skip uninformative
                all_drivers.append((h_idx, e_idx, log_lr))

    # Sort by influence, take top N
    all_drivers.sort(key=lambda x: x[2], reverse=True)
    top_n = all_drivers[:TOP_DRIVERS_COUNT]

    if not top_n:
        return [
            SensitivityEntry(
                hypothesis_id=p.hypothesis_id,
                baseline_posterior=p.final_posterior,
                posterior_low=p.final_posterior,
                posterior_high=p.final_posterior,
                rank_stable=True,
            )
            for p in posteriors
        ]

    # For each hypothesis, compute posterior under two scenarios:
    # 1. All top drivers perturbed to HELP this hypothesis (log_lr * 1.5 if for, * 0.5 if against)
    # 2. All top drivers perturbed to HURT this hypothesis (log_lr * 0.5 if for, * 1.5 if against)
    baseline_ranking = [p.hypothesis_id for p in sorted(posteriors, key=lambda p: p.final_posterior, reverse=True)]

    entries: list[SensitivityEntry] = []
    for target_p in posteriors:
        target_idx = next(i for i, ht in enumerate(testing.hypothesis_tests) if ht.hypothesis_id == target_p.hypothesis_id)

        best_posteriors: dict[str, float] = {}
        worst_posteriors: dict[str, float] = {}

        for h_idx, ht in enumerate(testing.hypothesis_tests):
            # Build LR list with perturbations
            best_lrs: list[float] = []
            worst_lrs: list[float] = []

            for e_idx, ev_eval in enumerate(ht.evidence_evaluations):
                lr = _compute_lr(ev_eval)
                log_lr = math.log(lr) if lr > 0 else 0.0

                # Check if this is a top driver
                is_top = any(hi == h_idx and ei == e_idx for hi, ei, _ in top_n)

                if is_top and abs(log_lr) > 0.01:
                    # "Favorable" means this LR direction helps the target hypothesis
                    favorable = (h_idx == target_idx and log_lr >= 0) or (
                        h_idx != target_idx and log_lr < 0
                    )
                    if favorable:
                        # Best: amplify favorable LR, worst: dampen it
                        best_lrs.append(math.exp(log_lr * (1 + PERTURB_FACTOR)))
                        worst_lrs.append(math.exp(log_lr * (1 - PERTURB_FACTOR)))
                    else:
                        # Best: dampen unfavorable LR, worst: amplify it
                        best_lrs.append(math.exp(log_lr * (1 - PERTURB_FACTOR)))
                        worst_lrs.append(math.exp(log_lr * (1 + PERTURB_FACTOR)))
                else:
                    best_lrs.append(lr)
                    worst_lrs.append(lr)

            h_id = ht.hypothesis_id
            best_posteriors[h_id] = _posterior_from_lrs(uniform_prior, best_lrs)
            worst_posteriors[h_id] = _posterior_from_lrs(uniform_prior, worst_lrs)

        # Normalize
        best_total = sum(best_posteriors.values())
        worst_total = sum(worst_posteriors.values())
        if best_total > 0:
            best_posteriors = {k: _clamp(v / best_total) for k, v in best_posteriors.items()}
        if worst_total > 0:
            worst_posteriors = {k: _clamp(v / worst_total) for k, v in worst_posteriors.items()}

        target_id = target_p.hypothesis_id
        best_rank = sorted(best_posteriors, key=lambda k: best_posteriors[k], reverse=True)
        worst_rank = sorted(worst_posteriors, key=lambda k: worst_posteriors[k], reverse=True)

        baseline_pos = baseline_ranking.index(target_id)
        best_pos = best_rank.index(target_id)
        worst_pos = worst_rank.index(target_id)

        entries.append(SensitivityEntry(
            hypothesis_id=target_id,
            baseline_posterior=target_p.final_posterior,
            posterior_low=round(worst_posteriors.get(target_id, target_p.final_posterior), 6),
            posterior_high=round(best_posteriors.get(target_id, target_p.final_posterior), 6),
            rank_stable=(best_pos == baseline_pos and worst_pos == baseline_pos),
        ))

    return entries
