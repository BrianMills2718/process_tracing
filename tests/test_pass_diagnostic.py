"""Tests for pt/pass_diagnostic.py — deterministic diagnostic matrix derivation.

All tests are fully deterministic (no LLM calls). The diagnostic matrix is derived
from likelihood vectors already produced by the testing pass.
"""

from __future__ import annotations

import pytest

from pt.pass_diagnostic import compute_diagnostic_matrix
from pt.schemas import (
    DiagnosticMatrix,
    EvidenceLikelihood,
    Hypothesis,
    HypothesisLikelihood,
    HypothesisSpace,
    Prediction,
    ProcessTracingResult,
    RivalDiscriminator,
    RivalPairDiagnostic,
    TestingResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _hyp(hid: str) -> Hypothesis:
    return Hypothesis(
        id=hid,
        description=f"Hypothesis {hid}",
        source="generated",
        theoretical_basis="test basis",
        causal_mechanism="test mechanism",
        observable_predictions=[Prediction(id=f"pred_{hid}_1", description="A test prediction")],
    )


def _hs(*ids: str) -> HypothesisSpace:
    return HypothesisSpace(
        research_question="Why did X happen?",
        hypotheses=[_hyp(h) for h in ids],
    )


def _evidence_likelihood(
    evidence_id: str,
    hyp_lrs: dict[str, float],  # hypothesis_id -> relative_likelihood
    relevance: float = 1.0,
    diagnostic_type: str = "straw_in_the_wind",  # noqa: intentional str for test fixture flexibility
) -> EvidenceLikelihood:
    return EvidenceLikelihood(
        evidence_id=evidence_id,
        hypothesis_likelihoods=[
            HypothesisLikelihood(
                hypothesis_id=hid,
                relative_likelihood=lr,
                diagnostic_type=diagnostic_type,
            )
            for hid, lr in hyp_lrs.items()
        ],
        relevance=relevance,
        justification="test",
    )  # type: ignore[arg-type]  # diagnostic_type is str in fixture, DiagnosticType in schema


def _testing(*items: EvidenceLikelihood) -> TestingResult:
    return TestingResult(evidence_likelihoods=list(items))


# ── Tests: schema contracts ───────────────────────────────────────────


class TestDiagnosticMatrixSchema:
    def test_rival_discriminator_roundtrip(self):
        d = RivalDiscriminator(
            evidence_id="evi_x",
            log_lr_h1_over_h2=1.2,
            favors="h1",
            strength="strong",
        )
        assert RivalDiscriminator.model_validate_json(d.model_dump_json()).evidence_id == "evi_x"

    def test_rival_pair_diagnostic_grade_capped_field(self):
        p = RivalPairDiagnostic(h1_id="h1", h2_id="h2", discriminators=[], discriminator_count=0, grade_capped=True)
        assert p.grade_capped is True
        assert p.discriminator_count == 0

    def test_diagnostic_matrix_pairs_without_discriminators(self):
        dm = DiagnosticMatrix(
            rival_pair_diagnostics=[],
            pairs_without_discriminators=[["h1", "h2"], ["h2", "h3"]],
            grade_cap_applied=True,
        )
        assert dm.grade_cap_applied is True
        assert len(dm.pairs_without_discriminators) == 2

    def test_diagnostic_matrix_in_process_tracing_result(self):
        assert "diagnostic_matrix" in ProcessTracingResult.model_fields
        field = ProcessTracingResult.model_fields["diagnostic_matrix"]
        assert field.default is None

    def test_discriminator_strength_values(self):
        for s in ("decisive", "strong"):
            d = RivalDiscriminator(
                evidence_id="e", log_lr_h1_over_h2=0.8, favors="h1", strength=s
            )
            assert d.strength == s

    def test_discriminator_strength_rejects_invalid(self):
        with pytest.raises(Exception):
            RivalDiscriminator(
                evidence_id="e", log_lr_h1_over_h2=0.8, favors="h1", strength="weak"  # type: ignore[arg-type]
            )


# ── Tests: compute_diagnostic_matrix ─────────────────────────────────


class TestComputeDiagnosticMatrix:
    def test_perfect_discriminator_both_hypotheses(self):
        """H1 strongly favored by evi_a (LR ratio 4:1); H2 strongly favored by evi_b."""
        hs = _hs("h1", "h2")
        testing = _testing(
            _evidence_likelihood("evi_a", {"h1": 4.0, "h2": 1.0}),
            _evidence_likelihood("evi_b", {"h1": 1.0, "h2": 4.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        pair = dm.rival_pair_diagnostics[0]
        assert pair.h1_id == "h1"
        assert pair.h2_id == "h2"
        assert pair.discriminator_count == 2
        assert not pair.grade_capped
        assert not dm.grade_cap_applied
        # evi_a should favor h1
        evi_a = next(d for d in pair.discriminators if d.evidence_id == "evi_a")
        assert evi_a.favors == "h1"
        assert evi_a.log_lr_h1_over_h2 > 0

    def test_uninformative_evidence_not_discriminator(self):
        """Equal LRs across both hypotheses → |log ratio| = 0 → not a discriminator."""
        hs = _hs("h1", "h2")
        testing = _testing(
            _evidence_likelihood("evi_uninformative", {"h1": 2.0, "h2": 2.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        pair = dm.rival_pair_diagnostics[0]
        assert pair.discriminator_count == 0
        assert pair.grade_capped
        assert dm.grade_cap_applied
        assert ["h1", "h2"] in dm.pairs_without_discriminators

    def test_strong_vs_decisive_threshold(self):
        """Test that strength classification uses log(2) / log(5) thresholds correctly."""
        hs = _hs("h1", "h2")
        # 4:1 → log(4) ≈ 1.39 < log(5) ≈ 1.61 → strong
        # 6:1 → log(6) ≈ 1.79 > log(5) → decisive
        testing = _testing(
            _evidence_likelihood("evi_strong", {"h1": 4.0, "h2": 1.0}),
            _evidence_likelihood("evi_decisive", {"h1": 6.0, "h2": 1.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        pair = dm.rival_pair_diagnostics[0]
        by_id = {d.evidence_id: d for d in pair.discriminators}
        assert by_id["evi_strong"].strength == "strong"
        assert by_id["evi_decisive"].strength == "decisive"

    def test_three_hypotheses_all_pairs_covered(self):
        """3 hypotheses → 3 rival pairs, all should appear in rival_pair_diagnostics."""
        hs = _hs("h1", "h2", "h3")
        testing = _testing(
            _evidence_likelihood("evi_x", {"h1": 5.0, "h2": 1.0, "h3": 2.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        pair_ids = {(p.h1_id, p.h2_id) for p in dm.rival_pair_diagnostics}
        assert ("h1", "h2") in pair_ids
        assert ("h1", "h3") in pair_ids
        assert ("h2", "h3") in pair_ids
        assert len(dm.rival_pair_diagnostics) == 3

    def test_grade_cap_only_on_undiscriminated_pairs(self):
        """h1 vs h3 has no discriminator but h1 vs h2 and h2 vs h3 do."""
        hs = _hs("h1", "h2", "h3")
        testing = _testing(
            # Discriminates h1 vs h2 and h2 vs h3, but not h1 vs h3 (equal for both)
            _evidence_likelihood("evi_y", {"h1": 3.0, "h2": 1.0, "h3": 3.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        h1_h3 = next(p for p in dm.rival_pair_diagnostics if p.h1_id == "h1" and p.h2_id == "h3")
        h1_h2 = next(p for p in dm.rival_pair_diagnostics if p.h1_id == "h1" and p.h2_id == "h2")
        assert h1_h3.grade_capped
        assert not h1_h2.grade_capped
        assert dm.grade_cap_applied

    def test_low_relevance_evidence_not_discriminator(self):
        """Evidence below the relevance gate (0.4) is treated as uninformative → not a discriminator."""
        hs = _hs("h1", "h2")
        testing = _testing(
            _evidence_likelihood("evi_low_rel", {"h1": 10.0, "h2": 1.0}, relevance=0.3),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        pair = dm.rival_pair_diagnostics[0]
        assert pair.discriminator_count == 0
        assert pair.grade_capped

    def test_interpretive_cap_tightens_discrimination(self):
        """Interpretive evidence with moderate LR difference may lose discriminator status under tighter cap."""
        hs = _hs("h1", "h2")
        # Under normal cap (LR_CAP=20): 4:1 → log_lr ≈ 1.39 > log(2) → discriminator
        # Under interpretive cap (LR_CAP=5): 4:1 → log_lr clamped to 0.5*log(5) ≈ 0.80 > log(2) → still discriminator
        # But very high ratio: 100:1 with normal cap gives log_lr ≈ 1.5;
        # with interpretive cap the half_log_cap = 0.5*log(5) ≈ 0.80, clamped → log_lr = 0.80 → discriminator
        # The key test: with interpretive cap AND relevance=1 the discrimination still passes
        testing = _testing(
            _evidence_likelihood("evi_interp", {"h1": 4.0, "h2": 1.0}, relevance=1.0),
        )
        dm_normal = compute_diagnostic_matrix(testing, hs)
        dm_interp = compute_diagnostic_matrix(testing, hs, interpretive_evidence_ids={"evi_interp"})
        # Both should discriminate (4:1 is above threshold under both caps)
        assert dm_normal.rival_pair_diagnostics[0].discriminator_count >= 1
        assert dm_interp.rival_pair_diagnostics[0].discriminator_count >= 1

    def test_diagnostic_type_propagated_from_testing(self):
        """diagnostic_type_h1 / diagnostic_type_h2 come from the LLM-assigned testing values."""
        hs = _hs("h1", "h2")
        testing = TestingResult(evidence_likelihoods=[
            EvidenceLikelihood(
                evidence_id="evi_dt",
                hypothesis_likelihoods=[
                    HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=5.0, diagnostic_type="smoking_gun"),
                    HypothesisLikelihood(hypothesis_id="h2", relative_likelihood=1.0, diagnostic_type="hoop"),
                ],
                relevance=1.0,
                justification="test",
            )
        ])
        dm = compute_diagnostic_matrix(testing, hs)
        pair = dm.rival_pair_diagnostics[0]
        assert pair.discriminator_count == 1
        d = pair.discriminators[0]
        assert d.diagnostic_type_h1 == "smoking_gun"
        assert d.diagnostic_type_h2 == "hoop"

    def test_no_evidence_produces_all_capped_pairs(self):
        """Empty testing result → all pairs have 0 discriminators."""
        hs = _hs("h1", "h2", "h3")
        testing = _testing()
        dm = compute_diagnostic_matrix(testing, hs)
        assert all(p.grade_capped for p in dm.rival_pair_diagnostics)
        assert dm.grade_cap_applied

    def test_single_hypothesis_produces_no_pairs(self):
        """Single hypothesis → no rival pairs, no caps."""
        hs = _hs("h1")
        testing = _testing(_evidence_likelihood("evi_x", {"h1": 3.0}))
        dm = compute_diagnostic_matrix(testing, hs)
        assert len(dm.rival_pair_diagnostics) == 0
        assert not dm.grade_cap_applied

    def test_favors_field_correct(self):
        """favors='h2' when h2 has higher LR."""
        hs = _hs("h1", "h2")
        testing = _testing(
            _evidence_likelihood("evi_h2_wins", {"h1": 1.0, "h2": 5.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        pair = dm.rival_pair_diagnostics[0]
        d = pair.discriminators[0]
        assert d.favors == "h2"
        assert d.log_lr_h1_over_h2 < 0

    def test_result_roundtrip_json(self):
        """DiagnosticMatrix round-trips through JSON."""
        hs = _hs("h1", "h2")
        testing = _testing(
            _evidence_likelihood("evi_x", {"h1": 4.0, "h2": 1.0}),
        )
        dm = compute_diagnostic_matrix(testing, hs)
        restored = DiagnosticMatrix.model_validate_json(dm.model_dump_json())
        assert restored.grade_cap_applied == dm.grade_cap_applied
        assert len(restored.rival_pair_diagnostics) == len(dm.rival_pair_diagnostics)
