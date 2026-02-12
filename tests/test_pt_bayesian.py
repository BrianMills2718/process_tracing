"""Tests for pt/bayesian.py — pure math, no LLM calls needed."""

import math

import pytest

from pt.bayesian import (
    CLAMP_MAX,
    CLAMP_MIN,
    LR_CAP,
    LR_FLOOR,
    _clamp,
    _odds,
    _prob,
    run_bayesian_update,
)
from pt.schemas import (
    EvidenceEvaluation,
    HypothesisTestResult,
    TestingResult,
)


# ── Helper to build test data ───────────────────────────────────────


def _make_eval(
    evidence_id: str = "e1",
    hypothesis_id: str = "h1",
    p_e_given_h: float = 0.8,
    p_e_given_not_h: float = 0.2,
    relevance: float = 1.0,
) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        evidence_id=evidence_id,
        hypothesis_id=hypothesis_id,
        finding="pass",
        p_e_given_h=p_e_given_h,
        p_e_given_not_h=p_e_given_not_h,
        justification="test",
        relevance=relevance,
    )


def _make_testing(*hypothesis_evals: tuple[str, list[EvidenceEvaluation]]) -> TestingResult:
    """Build TestingResult from (hypothesis_id, [evals]) pairs."""
    tests = []
    for h_id, evals in hypothesis_evals:
        tests.append(HypothesisTestResult(
            hypothesis_id=h_id,
            prediction_classifications=[],
            evidence_evaluations=evals,
        ))
    return TestingResult(hypothesis_tests=tests)


# ── _clamp ───────────────────────────────────────────────────────────


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_floor(self):
        assert _clamp(0.0) == CLAMP_MIN
        assert _clamp(-1.0) == CLAMP_MIN

    def test_ceiling(self):
        assert _clamp(1.0) == CLAMP_MAX
        assert _clamp(2.0) == CLAMP_MAX

    def test_at_boundaries(self):
        assert _clamp(CLAMP_MIN) == CLAMP_MIN
        assert _clamp(CLAMP_MAX) == CLAMP_MAX


# ── _odds / _prob roundtrip ─────────────────────────────────────────


class TestOddsProb:
    def test_roundtrip(self):
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert abs(_prob(_odds(p)) - p) < 1e-6

    def test_odds_of_half(self):
        assert _odds(0.5) == pytest.approx(1.0)

    def test_odds_of_high_prob(self):
        assert _odds(0.9) == pytest.approx(9.0)

    def test_prob_of_one(self):
        assert _prob(1.0) == pytest.approx(0.5)

    def test_extreme_odds_clamped(self):
        # Very large odds should clamp to CLAMP_MAX
        assert _prob(1e10) == CLAMP_MAX


# ── LR capping ───────────────────────────────────────────────────────


class TestLRCapping:
    def test_constants(self):
        assert LR_CAP == 20.0
        assert LR_FLOOR == pytest.approx(0.05)

    def test_strong_evidence_capped(self):
        """A single piece of very strong evidence shouldn't dominate."""
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.99, p_e_given_not_h=0.01)]),
            ("h2", [_make_eval(p_e_given_h=0.01, p_e_given_not_h=0.99)]),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        h2_post = next(p for p in result.posteriors if p.hypothesis_id == "h2")
        # h1 should win but not with extreme ratio due to capping
        assert h1_post.final_posterior > h2_post.final_posterior
        assert h1_post.final_posterior < 0.99  # cap prevents near-certainty from single item


# ── Relevance gating ────────────────────────────────────────────────


class TestRelevanceGating:
    def test_low_relevance_forced_uninformative(self):
        """Evidence with relevance < 0.4 should have LR forced to 1.0."""
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.9, p_e_given_not_h=0.1, relevance=0.3)]),
            ("h2", [_make_eval(p_e_given_h=0.1, p_e_given_not_h=0.9, relevance=0.3)]),
        )
        result = run_bayesian_update(testing)
        # Both should stay at uniform prior (0.5) since LR=1.0
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(0.5, abs=0.01)

    def test_high_relevance_has_effect(self):
        """Evidence with relevance >= 0.4 should shift posteriors."""
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.9, p_e_given_not_h=0.1, relevance=0.9)]),
            ("h2", [_make_eval(p_e_given_h=0.1, p_e_given_not_h=0.9, relevance=0.9)]),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        h2_post = next(p for p in result.posteriors if p.hypothesis_id == "h2")
        assert h1_post.final_posterior > 0.5
        assert h2_post.final_posterior < 0.5

    def test_relevance_discount_monotonic(self):
        """Higher relevance should produce stronger LR effect."""
        posteriors_at_relevance = {}
        for rel in [0.4, 0.6, 0.8, 1.0]:
            testing = _make_testing(
                ("h1", [_make_eval(p_e_given_h=0.9, p_e_given_not_h=0.1, relevance=rel)]),
                ("h2", [_make_eval(p_e_given_h=0.5, p_e_given_not_h=0.5, relevance=rel)]),
            )
            result = run_bayesian_update(testing)
            h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
            posteriors_at_relevance[rel] = h1_post.final_posterior

        # Each step should increase h1's posterior
        vals = [posteriors_at_relevance[r] for r in [0.4, 0.6, 0.8, 1.0]]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-6, f"Relevance {[0.4,0.6,0.8,1.0][i]} -> {[0.4,0.6,0.8,1.0][i+1]} should increase posterior"

    def test_relevance_at_boundary(self):
        """Relevance exactly at 0.4 should NOT be gated (gate is < 0.4)."""
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.9, p_e_given_not_h=0.1, relevance=0.4)]),
            ("h2", [_make_eval(p_e_given_h=0.1, p_e_given_not_h=0.9, relevance=0.4)]),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        # Should have some effect, not be neutral
        assert h1_post.final_posterior > 0.5


# ── Normalization ────────────────────────────────────────────────────


class TestNormalization:
    def test_posteriors_sum_to_one(self):
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.8, p_e_given_not_h=0.3)]),
            ("h2", [_make_eval(p_e_given_h=0.3, p_e_given_not_h=0.8)]),
            ("h3", [_make_eval(p_e_given_h=0.5, p_e_given_not_h=0.5)]),
        )
        result = run_bayesian_update(testing)
        total = sum(p.final_posterior for p in result.posteriors)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_many_hypotheses_normalize(self):
        """Even with 10 hypotheses, posteriors should sum to ~1."""
        evals_list = [
            (f"h{i}", [_make_eval(
                hypothesis_id=f"h{i}",
                p_e_given_h=0.5 + 0.04 * i,
                p_e_given_not_h=0.5 - 0.04 * i,
            )])
            for i in range(10)
        ]
        testing = _make_testing(*evals_list)
        result = run_bayesian_update(testing)
        total = sum(p.final_posterior for p in result.posteriors)
        assert total == pytest.approx(1.0, abs=0.01)


# ── Ranking ──────────────────────────────────────────────────────────


class TestRanking:
    def test_ranking_order(self):
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.9, p_e_given_not_h=0.1)]),
            ("h2", [_make_eval(p_e_given_h=0.1, p_e_given_not_h=0.9)]),
        )
        result = run_bayesian_update(testing)
        assert result.ranking == ["h1", "h2"]

    def test_ranking_with_ties(self):
        """Equal evidence should produce equal posteriors."""
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.5, p_e_given_not_h=0.5)]),
            ("h2", [_make_eval(p_e_given_h=0.5, p_e_given_not_h=0.5)]),
        )
        result = run_bayesian_update(testing)
        assert len(result.ranking) == 2
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(0.5, abs=0.01)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_testing(self):
        testing = TestingResult(hypothesis_tests=[])
        result = run_bayesian_update(testing)
        assert result.posteriors == []
        assert result.ranking == []

    def test_single_hypothesis(self):
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.9, p_e_given_not_h=0.1)]),
        )
        result = run_bayesian_update(testing)
        assert len(result.posteriors) == 1
        # Single hypothesis normalizes to ~1.0 (clamped to CLAMP_MAX)
        assert result.posteriors[0].final_posterior == pytest.approx(CLAMP_MAX, abs=0.01)

    def test_no_evidence(self):
        """Hypothesis with zero evidence items should stay at prior."""
        testing = _make_testing(
            ("h1", []),
            ("h2", []),
        )
        result = run_bayesian_update(testing)
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(0.5, abs=0.01)

    def test_compound_bias_protection(self):
        """Many weakly-anti items shouldn't crush a hypothesis if relevance is low."""
        # 20 items all slightly against h1, but all low relevance
        anti_evals = [
            _make_eval(
                evidence_id=f"e{i}",
                p_e_given_h=0.4,
                p_e_given_not_h=0.6,
                relevance=0.3,  # below gate threshold
            )
            for i in range(20)
        ]
        neutral_evals = [
            _make_eval(
                evidence_id=f"e{i}",
                p_e_given_h=0.5,
                p_e_given_not_h=0.5,
                relevance=0.3,
            )
            for i in range(20)
        ]
        testing = _make_testing(
            ("h1", anti_evals),
            ("h2", neutral_evals),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        # Should stay near 0.5 because all items are gated
        assert h1_post.final_posterior == pytest.approx(0.5, abs=0.01)

    def test_compound_bias_with_high_relevance(self):
        """Many weakly-anti items WITH high relevance should legitimately shift posterior."""
        anti_evals = [
            _make_eval(
                evidence_id=f"e{i}",
                p_e_given_h=0.4,
                p_e_given_not_h=0.6,
                relevance=0.9,
            )
            for i in range(20)
        ]
        neutral_evals = [
            _make_eval(
                evidence_id=f"e{i}",
                p_e_given_h=0.5,
                p_e_given_not_h=0.5,
                relevance=0.9,
            )
            for i in range(20)
        ]
        testing = _make_testing(
            ("h1", anti_evals),
            ("h2", neutral_evals),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        # Should be meaningfully below 0.5 since evidence is genuinely relevant
        assert h1_post.final_posterior < 0.3

    def test_update_trail_recorded(self):
        """Each evidence item should produce an EvidenceUpdate in the trail."""
        evals = [
            _make_eval(evidence_id="e1", p_e_given_h=0.8, p_e_given_not_h=0.2),
            _make_eval(evidence_id="e2", p_e_given_h=0.6, p_e_given_not_h=0.4),
        ]
        testing = _make_testing(("h1", evals), ("h2", []))
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert len(h1_post.updates) == 2
        assert h1_post.updates[0].evidence_id == "e1"
        assert h1_post.updates[1].evidence_id == "e2"
        # Prior of second update should equal posterior of first
        assert h1_post.updates[1].prior == h1_post.updates[0].posterior


# ── Relevance discount math ─────────────────────────────────────────


class TestRelevanceDiscountMath:
    def test_full_relevance_preserves_lr(self):
        """At relevance=1.0, LR should equal the capped raw LR."""
        # LR = 0.8/0.2 = 4.0, capped stays 4.0
        # At relevance=1.0: lr = exp(1.0 * log(4.0)) = 4.0
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.8, p_e_given_not_h=0.2, relevance=1.0)]),
            ("h2", [_make_eval(p_e_given_h=0.2, p_e_given_not_h=0.8, relevance=1.0)]),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        # The update LR should be 4.0
        assert h1_post.updates[0].likelihood_ratio == pytest.approx(4.0, abs=0.01)

    def test_half_relevance_reduces_lr(self):
        """At relevance=0.5, LR should be sqrt of capped LR (exp(0.5*log(lr)))."""
        # LR = 0.8/0.2 = 4.0 → at rel=0.5: exp(0.5*log(4)) = 2.0
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.8, p_e_given_not_h=0.2, relevance=0.5)]),
            ("h2", [_make_eval(p_e_given_h=0.5, p_e_given_not_h=0.5, relevance=0.5)]),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1_post.updates[0].likelihood_ratio == pytest.approx(2.0, abs=0.01)

    def test_lr_below_one_with_relevance(self):
        """Anti-hypothesis LR should also be discounted toward 1.0 by relevance."""
        # LR = 0.2/0.8 = 0.25, capped stays 0.25
        # At relevance=0.5: exp(0.5 * log(0.25)) = exp(0.5 * -1.386) = exp(-0.693) = 0.5
        testing = _make_testing(
            ("h1", [_make_eval(p_e_given_h=0.2, p_e_given_not_h=0.8, relevance=0.5)]),
            ("h2", [_make_eval(p_e_given_h=0.5, p_e_given_not_h=0.5, relevance=0.5)]),
        )
        result = run_bayesian_update(testing)
        h1_post = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1_post.updates[0].likelihood_ratio == pytest.approx(0.5, abs=0.01)
