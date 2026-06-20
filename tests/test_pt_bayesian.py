"""Tests for pt/bayesian.py — coherent multi-hypothesis math, no LLM calls.

The testing pass produces per-evidence likelihood vectors. For one evidence item
with vector {h_i: v_i}, the derived per-hypothesis LR is v_i / geomean(v), so for
two hypotheses {h1: a, h2: b} the LR for h1 is sqrt(a/b) (and h2's is its
reciprocal) — coherent by construction.
"""

import pytest

from pt.bayesian import (
    LR_CAP,
    _compute_robustness,
    _top_drivers,
    item_lrs,
    run_bayesian_update,
)
from pt.schemas import (
    EvidenceLikelihood,
    EvidenceUpdate,
    HypothesisLikelihood,
    TestingResult,
)


# ── Helpers to build vector test data ───────────────────────────────


def _vec(evidence_id: str, likelihoods: dict[str, float], relevance: float = 1.0) -> EvidenceLikelihood:
    return EvidenceLikelihood(
        evidence_id=evidence_id,
        hypothesis_likelihoods=[
            HypothesisLikelihood(
                hypothesis_id=h, relative_likelihood=v, diagnostic_type="straw_in_the_wind"
            )
            for h, v in likelihoods.items()
        ],
        relevance=relevance,
        justification="test",
    )


def _testing(*items: EvidenceLikelihood) -> TestingResult:
    return TestingResult(evidence_likelihoods=list(items))


# ── _clamp ───────────────────────────────────────────────────────────


# ── Coherence: derived pairwise ratios are consistent ───────────────


class TestCoherence:
    def test_two_hypothesis_lr_is_sqrt_ratio(self):
        lrs = item_lrs(_vec("e1", {"h1": 16.0, "h2": 1.0}), ["h1", "h2"])
        assert lrs["h1"] == pytest.approx(4.0, abs=1e-6)   # sqrt(16/1)
        assert lrs["h2"] == pytest.approx(0.25, abs=1e-6)  # reciprocal

    def test_three_hypothesis_ratios_are_transitive(self):
        lrs = item_lrs(_vec("e1", {"h1": 8.0, "h2": 2.0, "h3": 1.0}), ["h1", "h2", "h3"])
        # Pairwise ratios are derived from one vector, so transitivity holds.
        r12 = lrs["h1"] / lrs["h2"]
        r23 = lrs["h2"] / lrs["h3"]
        r13 = lrs["h1"] / lrs["h3"]
        assert r12 * r23 == pytest.approx(r13, rel=1e-9)
        assert r12 == pytest.approx(4.0, rel=1e-9)  # 8/2

    def test_flat_vector_is_uninformative(self):
        lrs = item_lrs(_vec("e1", {"h1": 0.5, "h2": 0.5, "h3": 0.5}), ["h1", "h2", "h3"])
        for h in ("h1", "h2", "h3"):
            assert lrs[h] == pytest.approx(1.0, abs=1e-9)


class TestJointUpdate:
    """The update must be a coherent joint multinomial update (softmax of summed
    log-LRs), not per-hypothesis binary odds with post-hoc normalization."""

    def test_matches_closed_form_softmax(self):
        # {h1:16, h2:1}: LRs 4 and 0.25; joint posterior = 4/(4+0.25) = 0.941,
        # NOT the binary-odds-then-normalize value of 0.8.
        result = run_bayesian_update(_testing(_vec("e1", {"h1": 16.0, "h2": 1.0})), ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior == pytest.approx(4.0 / 4.25, abs=1e-3)

    def test_order_invariant(self):
        # 5 strongly-pro-h1 items then 5 strongly-anti-h1 items must give the SAME
        # posterior regardless of order (the old binary-odds path gave 0.001 / 0.999 / 0.5).
        pro = [_vec(f"p{i}", {"h1": 100.0, "h2": 1.0}) for i in range(5)]
        anti = [_vec(f"a{i}", {"h1": 1.0, "h2": 100.0}) for i in range(5)]

        def post(items):
            r = run_bayesian_update(_testing(*items), ["h1", "h2"])
            return next(p.final_posterior for p in r.posteriors if p.hypothesis_id == "h1")

        forward = post(pro + anti)
        reverse = post(anti + pro)
        alternating = post([x for pair in zip(pro, anti) for x in pair])
        assert forward == pytest.approx(reverse, abs=1e-6)
        assert forward == pytest.approx(alternating, abs=1e-6)
        # symmetric pro/anti -> back to 0.5
        assert forward == pytest.approx(0.5, abs=1e-6)


# ── LR capping ───────────────────────────────────────────────────────


class TestLRCapping:
    def test_constants(self):
        assert LR_CAP == 20.0

    def test_strong_evidence_capped_on_pairwise_spread(self):
        # Cap bounds a single item's PAIRWISE max:min ratio to LR_CAP, not each
        # centered LR independently. {1e6, 1} -> {sqrt(CAP), 1/sqrt(CAP)}, ratio == CAP.
        import math
        lrs = item_lrs(_vec("e1", {"h1": 1e6, "h2": 1.0}), ["h1", "h2"])
        assert lrs["h1"] == pytest.approx(math.sqrt(LR_CAP))
        assert lrs["h2"] == pytest.approx(1.0 / math.sqrt(LR_CAP))
        assert lrs["h1"] / lrs["h2"] == pytest.approx(LR_CAP)

    def test_single_item_does_not_reach_certainty(self):
        testing = _testing(_vec("e1", {"h1": 1e6, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        h2 = next(p for p in result.posteriors if p.hypothesis_id == "h2")
        assert h1.final_posterior > h2.final_posterior
        assert h1.final_posterior < 0.99  # cap prevents near-certainty from one item


# ── Relevance gating ────────────────────────────────────────────────


class TestRelevanceGating:
    def test_low_relevance_forced_uninformative(self):
        testing = _testing(_vec("e1", {"h1": 9.0, "h2": 1.0}, relevance=0.3))
        result = run_bayesian_update(testing, ["h1", "h2"])
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(0.5, abs=0.01)

    def test_high_relevance_has_effect(self):
        testing = _testing(_vec("e1", {"h1": 9.0, "h2": 1.0}, relevance=0.9))
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior > 0.5

    def test_item_lrs_gate_at_below_threshold(self):
        lrs = item_lrs(_vec("e1", {"h1": 1000.0, "h2": 1.0}, relevance=0.39), ["h1", "h2"])
        assert lrs["h1"] == pytest.approx(1.0)
        assert lrs["h2"] == pytest.approx(1.0)

    def test_relevance_discount_monotonic(self):
        vals = []
        for rel in [0.4, 0.6, 0.8, 1.0]:
            testing = _testing(_vec("e1", {"h1": 0.9, "h2": 0.5}, relevance=rel))
            result = run_bayesian_update(testing, ["h1", "h2"])
            h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
            vals.append(h1.final_posterior)
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-6

    def test_relevance_at_boundary_not_gated(self):
        testing = _testing(_vec("e1", {"h1": 9.0, "h2": 1.0}, relevance=0.4))
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior > 0.5


# ── Normalization ────────────────────────────────────────────────────


class TestNormalization:
    def test_posteriors_sum_to_one(self):
        testing = _testing(_vec("e1", {"h1": 8.0, "h2": 2.0, "h3": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2", "h3"])
        total = sum(p.final_posterior for p in result.posteriors)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_many_hypotheses_normalize(self):
        ids = [f"h{i}" for i in range(10)]
        testing = _testing(_vec("e1", {h: float(i + 1) for i, h in enumerate(ids)}))
        result = run_bayesian_update(testing, ids)
        total = sum(p.final_posterior for p in result.posteriors)
        assert total == pytest.approx(1.0, abs=0.01)


# ── Ranking ──────────────────────────────────────────────────────────


class TestRanking:
    def test_ranking_order(self):
        testing = _testing(_vec("e1", {"h1": 9.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert result.ranking == ["h1", "h2"]

    def test_ranking_with_ties(self):
        testing = _testing(_vec("e1", {"h1": 1.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert len(result.ranking) == 2
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(0.5, abs=0.01)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_testing(self):
        result = run_bayesian_update(_testing(), [])
        assert result.posteriors == []
        assert result.ranking == []

    def test_no_evidence_stays_at_prior(self):
        result = run_bayesian_update(_testing(), ["h1", "h2"])
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(0.5, abs=0.01)

    def test_compound_bias_protection(self):
        # 20 items slightly against h1, all below the relevance gate -> no shift.
        items = [_vec(f"e{i}", {"h1": 0.4, "h2": 0.6}, relevance=0.3) for i in range(20)]
        result = run_bayesian_update(_testing(*items), ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior == pytest.approx(0.5, abs=0.01)

    def test_compound_bias_with_high_relevance(self):
        items = [_vec(f"e{i}", {"h1": 0.4, "h2": 0.6}, relevance=0.9) for i in range(20)]
        result = run_bayesian_update(_testing(*items), ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior < 0.3

    def test_update_trail_recorded(self):
        items = [_vec("e1", {"h1": 4.0, "h2": 1.0}), _vec("e2", {"h1": 2.0, "h2": 1.0})]
        result = run_bayesian_update(_testing(*items), ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert len(h1.updates) == 2
        assert h1.updates[0].evidence_id == "e1"
        assert h1.updates[1].evidence_id == "e2"
        assert h1.updates[1].prior == h1.updates[0].posterior


# ── Relevance discount math ─────────────────────────────────────────


class TestRelevanceDiscountMath:
    def test_full_relevance_preserves_lr(self):
        # sqrt(16/1) = 4.0 at relevance 1.0
        testing = _testing(_vec("e1", {"h1": 16.0, "h2": 1.0}, relevance=1.0))
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.updates[0].likelihood_ratio == pytest.approx(4.0, abs=0.01)

    def test_half_relevance_reduces_lr(self):
        # base LR 4.0 -> at relevance 0.5: 4**0.5 = 2.0
        testing = _testing(_vec("e1", {"h1": 16.0, "h2": 1.0}, relevance=0.5))
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.updates[0].likelihood_ratio == pytest.approx(2.0, abs=0.01)

    def test_lr_below_one_with_relevance(self):
        # base LR sqrt(1/16)=0.25 -> at relevance 0.5: 0.25**0.5 = 0.5
        testing = _testing(_vec("e1", {"h1": 1.0, "h2": 16.0}, relevance=0.5))
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.updates[0].likelihood_ratio == pytest.approx(0.5, abs=0.01)


# ── Robustness computation (operates on EvidenceUpdate, unchanged) ──


class TestRobustness:
    def _make_update(self, lr: float, eid: str = "e1") -> EvidenceUpdate:
        return EvidenceUpdate(evidence_id=eid, likelihood_ratio=lr, prior=0.5, posterior=0.5)

    def test_robust_few_decisive(self):
        updates = [
            self._make_update(10.0, "e1"),
            self._make_update(0.1, "e2"),
            self._make_update(8.0, "e3"),
            self._make_update(0.15, "e4"),
        ]
        assert _compute_robustness(updates) == "robust"

    def test_fragile_many_weak(self):
        updates = [self._make_update(0.8, f"e{i}") for i in range(15)]
        assert _compute_robustness(updates) == "fragile"

    def test_unknown_empty(self):
        assert _compute_robustness([]) == "unknown"

    def test_unknown_all_uninformative(self):
        updates = [self._make_update(1.0, f"e{i}") for i in range(10)]
        assert _compute_robustness(updates) == "unknown"

    def test_uninformative_items_do_not_force_fragile(self):
        updates = (
            [self._make_update(3.0, f"m{i}") for i in range(4)]
            + [self._make_update(1.0, f"u{i}") for i in range(10)]
        )
        assert _compute_robustness(updates) != "fragile"

    def test_robustness_populated_in_result(self):
        testing = _testing(_vec("e1", {"h1": 19.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        for p in result.posteriors:
            assert p.robustness in ("robust", "fragile", "moderate", "unknown")


# ── Top drivers ────────────────────────────────────────────────────


class TestTopDrivers:
    def test_returns_most_influential(self):
        updates = [
            EvidenceUpdate(evidence_id="e1", likelihood_ratio=1.0, prior=0.5, posterior=0.5),
            EvidenceUpdate(evidence_id="e2", likelihood_ratio=10.0, prior=0.5, posterior=0.9),
            EvidenceUpdate(evidence_id="e3", likelihood_ratio=0.1, prior=0.5, posterior=0.1),
            EvidenceUpdate(evidence_id="e4", likelihood_ratio=2.0, prior=0.5, posterior=0.7),
        ]
        drivers = _top_drivers(updates, n=2)
        assert set(drivers) == {"e2", "e3"}

    def test_top_drivers_populated_in_result(self):
        items = [_vec("e1", {"h1": 0.95, "h2": 0.05}), _vec("e2", {"h1": 0.5, "h2": 0.5})]
        result = run_bayesian_update(_testing(*items), ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert "e1" in h1.top_drivers


# ── Sensitivity analysis ──────────────────────────────────────────


class TestSensitivity:
    def test_sensitivity_populated(self):
        testing = _testing(_vec("e1", {"h1": 9.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert len(result.sensitivity) == 2
        for s in result.sensitivity:
            assert s.posterior_low <= s.baseline_posterior <= s.posterior_high

    def test_sensitivity_with_uninformative_evidence(self):
        testing = _testing(_vec("e1", {"h1": 0.5, "h2": 0.5}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        for s in result.sensitivity:
            assert s.posterior_low == pytest.approx(s.posterior_high, abs=0.01)
            assert s.rank_stable is True

    def test_sensitivity_range_wider_with_decisive_evidence(self):
        strong = _testing(_vec("e1", {"h1": 0.95, "h2": 0.05}))
        weak = _testing(_vec("e1", {"h1": 0.55, "h2": 0.45}))
        strong_result = run_bayesian_update(strong, ["h1", "h2"])
        weak_result = run_bayesian_update(weak, ["h1", "h2"])
        strong_range = strong_result.sensitivity[0].posterior_high - strong_result.sensitivity[0].posterior_low
        weak_range = weak_result.sensitivity[0].posterior_high - weak_result.sensitivity[0].posterior_low
        assert strong_range > weak_range

    def test_empty_testing_no_sensitivity(self):
        result = run_bayesian_update(_testing(), [])
        assert result.sensitivity == []

    def test_rank_stability_flag(self):
        items = [
            _vec("e1", {"h1": 0.95, "h2": 0.05}),
            _vec("e2", {"h1": 0.90, "h2": 0.10}),
            _vec("e3", {"h1": 0.85, "h2": 0.15}),
        ]
        result = run_bayesian_update(_testing(*items), ["h1", "h2"])
        h1_sens = next(s for s in result.sensitivity if s.hypothesis_id == "h1")
        assert h1_sens.rank_stable is True


# ── Researcher priors + prior sensitivity ───────────────────────────


class TestPriors:
    def test_uniform_default(self):
        # No evidence + no priors -> uniform.
        result = run_bayesian_update(_testing(), ["h1", "h2", "h3"])
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(1 / 3, abs=0.01)

    def test_non_uniform_priors_shift_posteriors(self):
        # Uninformative evidence -> posterior reflects the (normalized) prior.
        testing = _testing(_vec("e1", {"h1": 1.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"], priors={"h1": 3.0, "h2": 1.0})
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior == pytest.approx(0.75, abs=0.01)
        assert h1.prior == pytest.approx(0.75, abs=0.01)

    def test_priors_need_not_be_normalized(self):
        testing = _testing(_vec("e1", {"h1": 1.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"], priors={"h1": 30.0, "h2": 10.0})
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert h1.final_posterior == pytest.approx(0.75, abs=0.01)

    def test_prior_sensitivity_populated(self):
        testing = _testing(_vec("e1", {"h1": 9.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert result.prior_sensitivity is not None
        assert result.prior_sensitivity.top_hypothesis_id == result.ranking[0]

    def test_prior_sensitivity_stable_when_evidence_dominant(self):
        items = [_vec(f"e{i}", {"h1": 0.95, "h2": 0.05}) for i in range(4)]
        result = run_bayesian_update(_testing(*items), ["h1", "h2"])
        assert result.prior_sensitivity.stable_under_prior_perturbation is True

    def test_prior_sensitivity_unstable_when_evidence_weak(self):
        # Near-tie evidence: a 2x prior swing flips the leader.
        testing = _testing(_vec("e1", {"h1": 1.05, "h2": 1.0}, relevance=0.5))
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert result.prior_sensitivity.stable_under_prior_perturbation is False

    def test_priors_reject_unknown_hypothesis(self):
        testing = _testing(_vec("e1", {"h1": 2.0, "h2": 1.0}))
        with pytest.raises(ValueError, match="unknown"):
            run_bayesian_update(testing, ["h1", "h2"], priors={"h1": 1.0, "h2": 1.0, "h9": 1.0})

    def test_priors_reject_missing_hypothesis(self):
        testing = _testing(_vec("e1", {"h1": 2.0, "h2": 1.0}))
        with pytest.raises(ValueError, match="missing"):
            run_bayesian_update(testing, ["h1", "h2"], priors={"h1": 1.0})

    def test_priors_reject_nonpositive_weight(self):
        testing = _testing(_vec("e1", {"h1": 2.0, "h2": 1.0}))
        with pytest.raises(ValueError, match="positive"):
            run_bayesian_update(testing, ["h1", "h2"], priors={"h1": 1.0, "h2": 0.0})
