"""Pass 3.6: Diagnostic Test Matrix — deterministic derivation from likelihood vectors.

No LLM call. Reads the testing result's effective LRs and derives, for each rival
hypothesis pair, which evidence items discriminate between them and how strongly.

A discriminator for pair (H_i, H_j) is an evidence item where the effective LR ratio
|log(LR_i / LR_j)| >= log(2) — matching the 2× threshold used throughout the pipeline.
"""

from __future__ import annotations

import math
from itertools import combinations

from pt.bayesian import item_lrs, INTERPRETIVE_LR_CAP, LR_CAP
from pt.schemas import (
    DiagnosticMatrix,
    DiagnosticType,
    DiscriminatorStrength,
    HypothesisSpace,
    RivalDiscriminator,
    RivalPairDiagnostic,
    TestingResult,
)

_LOG2 = math.log(2)   # strong threshold: |log_lr| >= log(2)  ≈ 0.693
_LOG5 = math.log(5)   # decisive threshold: |log_lr| >= log(5) ≈ 1.609


def _diagnostic_type_for(
    evidence_id: str,
    hypothesis_id: str,
    testing: TestingResult,
) -> DiagnosticType | None:
    """Look up the LLM-assigned diagnostic type for (evidence, hypothesis) from the testing result."""
    for item in testing.evidence_likelihoods:
        if item.evidence_id != evidence_id:
            continue
        for hl in item.hypothesis_likelihoods:
            if hl.hypothesis_id == hypothesis_id:
                return hl.diagnostic_type
    return None


def compute_diagnostic_matrix(
    testing: TestingResult,
    hypothesis_space: HypothesisSpace,
    interpretive_evidence_ids: set[str] | None = None,
) -> DiagnosticMatrix:
    """Derive the diagnostic test matrix from an existing TestingResult.

    For each rival pair (H_i, H_j), finds evidence items whose effective LR ratio
    |log(LR_i / LR_j)| >= log(2). Each such item becomes a RivalDiscriminator.
    Pairs with zero discriminators receive grade_capped=True.

    interpretive_evidence_ids: set of evidence IDs that get the tighter INTERPRETIVE_LR_CAP.
    """
    interpretive = interpretive_evidence_ids or set()
    hyp_ids = [h.id for h in hypothesis_space.hypotheses]

    # Build effective LR matrix: evidence_id -> {hyp_id: lr}
    lr_by_evidence: dict[str, dict[str, float]] = {}
    for item in testing.evidence_likelihoods:
        cap = INTERPRETIVE_LR_CAP if item.evidence_id in interpretive else LR_CAP
        lr_by_evidence[item.evidence_id] = item_lrs(item, hyp_ids, cap)

    pair_diagnostics: list[RivalPairDiagnostic] = []
    pairs_without: list[list[str]] = []

    for h1_id, h2_id in combinations(hyp_ids, 2):
        discriminators: list[RivalDiscriminator] = []

        for eid, lrs in lr_by_evidence.items():
            lr_h1 = lrs.get(h1_id, 1.0)
            lr_h2 = lrs.get(h2_id, 1.0)
            if lr_h2 == 0.0:
                continue
            log_lr = math.log(lr_h1) - math.log(lr_h2)
            abs_log = abs(log_lr)
            if abs_log < _LOG2:
                continue  # not discriminating
            strength: DiscriminatorStrength = "decisive" if abs_log >= _LOG5 else "strong"
            favors = "h1" if log_lr > 0 else "h2"
            discriminators.append(
                RivalDiscriminator(
                    evidence_id=eid,
                    log_lr_h1_over_h2=log_lr,
                    favors=favors,
                    strength=strength,
                    diagnostic_type_h1=_diagnostic_type_for(eid, h1_id, testing),
                    diagnostic_type_h2=_diagnostic_type_for(eid, h2_id, testing),
                )
            )

        grade_capped = len(discriminators) == 0
        if grade_capped:
            pairs_without.append([h1_id, h2_id])

        pair_diagnostics.append(
            RivalPairDiagnostic(
                h1_id=h1_id,
                h2_id=h2_id,
                discriminators=discriminators,
                discriminator_count=len(discriminators),
                grade_capped=grade_capped,
            )
        )

    return DiagnosticMatrix(
        rival_pair_diagnostics=pair_diagnostics,
        pairs_without_discriminators=pairs_without,
        grade_cap_applied=len(pairs_without) > 0,
    )
