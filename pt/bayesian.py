"""Pure math: coherent multi-hypothesis Bayesian updating in odds space.

The testing pass elicits, per evidence item, a *likelihood vector* across all
hypotheses (relative likelihoods on a shared scale). For each item we derive a
per-hypothesis likelihood ratio against the vector's geometric mean:

    LR_{m,i} = relative_likelihood_{m,i} / geomean_j(relative_likelihood_{m,j})

clamped to [LR_FLOOR, LR_CAP] and relevance-discounted. Because every LR for an
item comes from one vector, the pairwise ratios are coherent by construction
(reciprocity/transitivity hold). The posterior is the joint multinomial update
`post_i ∝ prior_i · Π_m LR_{m,i}`, normalized across hypotheses.

No LLM here — deterministic and unit-tested (per the LLM-first exception).
"""

from __future__ import annotations

import math

from pt.schemas import (
    BayesianResult,
    EvidenceLikelihood,
    EvidenceUpdate,
    HypothesisPosterior,
    PriorSensitivity,
    SensitivityEntry,
    TestingResult,
)

CLAMP_MIN = 0.001
CLAMP_MAX = 0.999
LR_CAP = 20.0
LR_FLOOR = 1.0 / LR_CAP  # 0.05

RELEVANCE_GATE = 0.4  # below this, an item is uninformative (LR forced to 1.0)

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


def _geomean(values: list[float]) -> float:
    """Geometric mean of positive values."""
    return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))


def hypothesis_ids_from_testing(testing: TestingResult) -> list[str]:
    """Union of hypothesis ids appearing in the likelihood vectors, first-seen order.

    Prefer passing the canonical hypothesis list to run_bayesian_update; this is a
    fallback when only the testing result is available.
    """
    seen: list[str] = []
    s: set[str] = set()
    for item in testing.evidence_likelihoods:
        for hl in item.hypothesis_likelihoods:
            if hl.hypothesis_id not in s:
                s.add(hl.hypothesis_id)
                seen.append(hl.hypothesis_id)
    return seen


def item_lrs(item: EvidenceLikelihood, hypothesis_ids: list[str]) -> dict[str, float]:
    """Per-hypothesis effective LR for one evidence item, derived from its vector.

    Each hypothesis's LR is its relative likelihood over the vector's geometric
    mean, clamped and relevance-discounted. A hypothesis absent from the vector,
    or an item below the relevance gate, contributes LR 1.0 (no update).
    """
    by_id = {
        hl.hypothesis_id: max(hl.relative_likelihood, 1e-9)
        for hl in item.hypothesis_likelihoods
    }
    present = [by_id[h] for h in hypothesis_ids if h in by_id]
    if not present:
        return {h: 1.0 for h in hypothesis_ids}
    geo = _geomean(present)
    rel = item.relevance
    out: dict[str, float] = {}
    for h in hypothesis_ids:
        v = by_id.get(h)
        if v is None:
            out[h] = 1.0
            continue
        lr = max(LR_FLOOR, min(LR_CAP, v / geo))
        if rel < RELEVANCE_GATE:
            lr = 1.0
        else:
            lr = math.exp(rel * math.log(lr))  # soft relevance discount on log scale
        out[h] = lr
    return out


def lr_matrix(
    testing: TestingResult, hypothesis_ids: list[str]
) -> list[tuple[str, dict[str, float]]]:
    """Effective LRs for every (evidence item, hypothesis), in evidence order."""
    return [
        (item.evidence_id, item_lrs(item, hypothesis_ids))
        for item in testing.evidence_likelihoods
    ]


def _compute_robustness(updates: list[EvidenceUpdate]) -> str:
    """Classify a posterior as 'robust', 'fragile', or 'moderate' from its LR spread.

    Robust: driven by a few decisive LRs (|log(LR)| > 1.6).
    Fragile: driven by many *informative-but-weak* LRs (0 < |log(LR)| < 0.7).
    Uninformative (LR≈1) items are excluded from the weak count.
    """
    if not updates:
        return "unknown"

    log_lrs = [abs(math.log(u.likelihood_ratio)) for u in updates if u.likelihood_ratio > 0]
    if not log_lrs:
        return "unknown"

    n_decisive = sum(1 for ll in log_lrs if ll > DECISIVE_LR_THRESHOLD)
    n_weak = sum(1 for ll in log_lrs if 0.01 < ll < WEAK_LR_THRESHOLD)
    n_informative = sum(1 for ll in log_lrs if ll > 0.01)

    if n_informative == 0:
        return "unknown"

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
    """Evidence IDs of the N most influential LR updates."""
    scored = [
        (abs(math.log(u.likelihood_ratio)) if u.likelihood_ratio > 0 else 0, u.evidence_id)
        for u in updates
    ]
    scored.sort(reverse=True)
    return [eid for _, eid in scored[:n]]


def _posterior_from_lrs(prior: float, lrs: list[float]) -> float:
    """Posterior probability from a prior and a list of LRs (used for sensitivity)."""
    current = prior
    for lr in lrs:
        current = _prob(_odds(current) * lr)
    return current


def run_bayesian_update(
    testing: TestingResult,
    hypothesis_ids: list[str] | None = None,
    priors: dict[str, float] | None = None,
) -> BayesianResult:
    """Compute posteriors via a coherent joint multinomial update.

    Args:
        testing: per-evidence likelihood vectors.
        hypothesis_ids: the canonical hypothesis universe (so hypotheses with no
            discriminating evidence still receive a posterior = their prior).
            Falls back to the union found in the vectors.
        priors: optional researcher prior per hypothesis (need not be normalized);
            defaults to uniform.
    """
    if hypothesis_ids is None:
        hypothesis_ids = hypothesis_ids_from_testing(testing)
    n = len(hypothesis_ids)
    if n == 0:
        return BayesianResult(posteriors=[], ranking=[])

    if priors:
        total_prior = sum(max(priors.get(h, 0.0), 0.0) for h in hypothesis_ids)
        if total_prior > 0:
            prior_by_id = {h: max(priors.get(h, 0.0), 0.0) / total_prior for h in hypothesis_ids}
        else:
            prior_by_id = {h: 1.0 / n for h in hypothesis_ids}
    else:
        prior_by_id = {h: 1.0 / n for h in hypothesis_ids}

    matrix = lr_matrix(testing, hypothesis_ids)

    posteriors: list[HypothesisPosterior] = []
    for h in hypothesis_ids:
        current_prob = prior_by_id[h]
        updates: list[EvidenceUpdate] = []
        for evidence_id, lrs in matrix:
            lr = lrs[h]
            prior_for_update = current_prob
            current_prob = _prob(_odds(current_prob) * lr)
            updates.append(EvidenceUpdate(
                evidence_id=evidence_id,
                prediction_id=None,
                likelihood_ratio=round(lr, 4),
                prior=round(prior_for_update, 6),
                posterior=round(current_prob, 6),
            ))
        posteriors.append(HypothesisPosterior(
            hypothesis_id=h,
            prior=round(prior_by_id[h], 6),
            updates=updates,
            final_posterior=current_prob,
            robustness=_compute_robustness(updates),
            top_drivers=_top_drivers(updates),
        ))

    # Normalize so posteriors sum to 1 (joint multinomial normalization)
    total = sum(p.final_posterior for p in posteriors)
    if total > 0:
        for p in posteriors:
            p.final_posterior = round(_clamp(p.final_posterior / total), 6)

    ranking = sorted(posteriors, key=lambda p: p.final_posterior, reverse=True)
    ranking_ids = [p.hypothesis_id for p in ranking]

    sensitivity = _run_sensitivity(matrix, hypothesis_ids, posteriors, prior_by_id)
    prior_sens = _prior_sensitivity(matrix, hypothesis_ids, prior_by_id, ranking_ids[0]) if ranking_ids else None

    return BayesianResult(
        posteriors=posteriors,
        ranking=ranking_ids,
        sensitivity=sensitivity,
        prior_sensitivity=prior_sens,
    )


def _normalized_posteriors(
    matrix: list[tuple[str, dict[str, float]]],
    hypothesis_ids: list[str],
    prior_by_id: dict[str, float],
) -> dict[str, float]:
    """Final normalized posteriors for a given prior (no update trail)."""
    raw: dict[str, float] = {}
    for h in hypothesis_ids:
        current = prior_by_id[h]
        for _, lrs in matrix:
            current = _prob(_odds(current) * lrs[h])
        raw[h] = current
    total = sum(raw.values())
    return {h: raw[h] / total for h in hypothesis_ids} if total > 0 else raw


def _prior_sensitivity(
    matrix: list[tuple[str, dict[str, float]]],
    hypothesis_ids: list[str],
    prior_by_id: dict[str, float],
    baseline_top: str,
    factor: float = 2.0,
) -> PriorSensitivity:
    """Is the top-ranked hypothesis robust to up/down-weighting each prior by `factor`?"""
    stable = True
    for h in hypothesis_ids:
        for f in (factor, 1.0 / factor):
            pert = dict(prior_by_id)
            pert[h] = pert[h] * f
            tot = sum(pert.values())
            if tot <= 0:
                continue
            pert = {k: v / tot for k, v in pert.items()}
            post = _normalized_posteriors(matrix, hypothesis_ids, pert)
            if max(post, key=lambda k: post[k]) != baseline_top:
                stable = False
    return PriorSensitivity(
        top_hypothesis_id=baseline_top,
        stable_under_prior_perturbation=stable,
        perturbation_factor=factor,
    )


def _run_sensitivity(
    matrix: list[tuple[str, dict[str, float]]],
    hypothesis_ids: list[str],
    posteriors: list[HypothesisPosterior],
    prior_by_id: dict[str, float],
) -> list[SensitivityEntry]:
    """Perturb each hypothesis's most influential LRs ±50% on log-scale and measure
    posterior stability and rank stability under perturbation."""
    n_hyp = len(hypothesis_ids)
    if n_hyp == 0:
        return []

    # all_lrs[h_idx] = list of this hypothesis's LRs across evidence items
    all_lrs: dict[str, list[float]] = {h: [lrs[h] for _, lrs in matrix] for h in hypothesis_ids}

    # per-hypothesis top driver evidence indices (by |log LR|)
    per_hyp_top: dict[str, set[int]] = {}
    for h in hypothesis_ids:
        scored = [
            (abs(math.log(lr)) if lr > 0 else 0.0, e_idx)
            for e_idx, lr in enumerate(all_lrs[h])
        ]
        scored.sort(reverse=True)
        per_hyp_top[h] = {e_idx for mag, e_idx in scored[:TOP_DRIVERS_COUNT] if mag > 0.01}

    baseline_ranking = [p.hypothesis_id for p in sorted(
        posteriors, key=lambda p: p.final_posterior, reverse=True
    )]

    entries: list[SensitivityEntry] = []
    for target in posteriors:
        target_id = target.hypothesis_id
        # perturb target's own top drivers plus every rival's top drivers
        perturb: set[tuple[str, int]] = {(target_id, e) for e in per_hyp_top[target_id]}
        for h in hypothesis_ids:
            if h != target_id:
                for e in per_hyp_top[h]:
                    perturb.add((h, e))

        best_raw: dict[str, float] = {}
        worst_raw: dict[str, float] = {}
        for h in hypothesis_ids:
            best_lrs: list[float] = []
            worst_lrs: list[float] = []
            for e_idx, lr in enumerate(all_lrs[h]):
                log_lr = math.log(lr) if lr > 0 else 0.0
                if (h, e_idx) in perturb and abs(log_lr) > 0.01:
                    favorable = (h == target_id and log_lr >= 0) or (
                        h != target_id and log_lr < 0
                    )
                    if favorable:
                        best_lrs.append(math.exp(log_lr * (1 + PERTURB_FACTOR)))
                        worst_lrs.append(math.exp(log_lr * (1 - PERTURB_FACTOR)))
                    else:
                        best_lrs.append(math.exp(log_lr * (1 - PERTURB_FACTOR)))
                        worst_lrs.append(math.exp(log_lr * (1 + PERTURB_FACTOR)))
                else:
                    best_lrs.append(lr)
                    worst_lrs.append(lr)
            best_raw[h] = _posterior_from_lrs(prior_by_id[h], best_lrs)
            worst_raw[h] = _posterior_from_lrs(prior_by_id[h], worst_lrs)

        best_total = sum(best_raw.values())
        worst_total = sum(worst_raw.values())
        best_norm = {k: _clamp(v / best_total) for k, v in best_raw.items()} if best_total > 0 else best_raw
        worst_norm = {k: _clamp(v / worst_total) for k, v in worst_raw.items()} if worst_total > 0 else worst_raw

        best_rank = sorted(best_norm, key=lambda k: best_norm[k], reverse=True)
        worst_rank = sorted(worst_norm, key=lambda k: worst_norm[k], reverse=True)
        baseline_pos = baseline_ranking.index(target_id)

        entries.append(SensitivityEntry(
            hypothesis_id=target_id,
            baseline_posterior=target.final_posterior,
            posterior_low=round(worst_norm.get(target_id, target.final_posterior), 6),
            posterior_high=round(best_norm.get(target_id, target.final_posterior), 6),
            rank_stable=(
                best_rank.index(target_id) == baseline_pos
                and worst_rank.index(target_id) == baseline_pos
            ),
        ))

    return entries
