"""Tests for pt/bayesian.py — coherent multi-hypothesis math, no LLM calls.

The testing pass produces per-evidence likelihood vectors. For one evidence item
with vector {h_i: v_i}, the derived per-hypothesis LR is v_i / geomean(v), so for
two hypotheses {h1: a, h2: b} the LR for h1 is sqrt(a/b) (and h2's is its
reciprocal) — coherent by construction.
"""

import pytest

from pt.bayesian import (
    LR_CAP,
    RESIDUAL_ID,
    _compute_robustness,
    _top_drivers,
    item_lrs,
    run_bayesian_update,
)
from pt.schemas import (
    EvidenceCluster,
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


def _testing(*items: EvidenceLikelihood, clusters: list[EvidenceCluster] | None = None) -> TestingResult:
    return TestingResult(evidence_likelihoods=list(items), dependence_clusters=clusters or [])


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


class TestDependenceClustering:
    """A dependence cluster is collapsed to one effective observation, so correlated
    evidence is not double-counted (the overconfidence fix)."""

    def test_cluster_collapses_duplicates_to_one(self):
        # Five identical items, all in one cluster, must give the SAME posterior as one item.
        five = [_vec(f"d{i}", {"h1": 0.95, "h2": 0.05}) for i in range(5)]
        cluster = EvidenceCluster(evidence_ids=[f"d{i}" for i in range(5)], reason="same fact")
        clustered = run_bayesian_update(_testing(*five, clusters=[cluster]), ["h1", "h2"])
        single = run_bayesian_update(_testing(_vec("d0", {"h1": 0.95, "h2": 0.05})), ["h1", "h2"])
        c_h1 = next(p.final_posterior for p in clustered.posteriors if p.hypothesis_id == "h1")
        s_h1 = next(p.final_posterior for p in single.posteriors if p.hypothesis_id == "h1")
        assert c_h1 == pytest.approx(s_h1, abs=1e-6)

    def test_uncollapsed_duplicates_are_overconfident(self):
        # Without clustering, five duplicates pile up (the bug clustering fixes).
        five = [_vec(f"d{i}", {"h1": 0.95, "h2": 0.05}) for i in range(5)]
        no_cluster = run_bayesian_update(_testing(*five), ["h1", "h2"])
        h1 = next(p.final_posterior for p in no_cluster.posteriors if p.hypothesis_id == "h1")
        assert h1 > 0.99  # much more extreme than a single item

    def test_cluster_trail_has_one_entry_per_cluster(self):
        five = [_vec(f"d{i}", {"h1": 0.95, "h2": 0.05}) for i in range(5)]
        cluster = EvidenceCluster(evidence_ids=[f"d{i}" for i in range(5)], reason="same fact")
        result = run_bayesian_update(_testing(*five, clusters=[cluster]), ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        assert len(h1.updates) == 1  # 5 items -> 1 effective observation

    def _h1_post(self, rho):
        five = [_vec(f"d{i}", {"h1": 0.9, "h2": 0.1}) for i in range(5)]
        cluster = EvidenceCluster(evidence_ids=[f"d{i}" for i in range(5)], reason="r", dependence_strength=rho)
        r = run_bayesian_update(_testing(*five, clusters=[cluster]), ["h1", "h2"])
        return next(p.final_posterior for p in r.posteriors if p.hypothesis_id == "h1")

    def test_rho_one_is_full_collapse(self):
        single = run_bayesian_update(_testing(_vec("d0", {"h1": 0.9, "h2": 0.1})), ["h1", "h2"])
        s_h1 = next(p.final_posterior for p in single.posteriors if p.hypothesis_id == "h1")
        assert self._h1_post(1.0) == pytest.approx(s_h1, abs=1e-6)

    def test_rho_zero_is_independent(self):
        five = [_vec(f"d{i}", {"h1": 0.9, "h2": 0.1}) for i in range(5)]
        indep = run_bayesian_update(_testing(*five), ["h1", "h2"])  # no cluster
        i_h1 = next(p.final_posterior for p in indep.posteriors if p.hypothesis_id == "h1")
        assert self._h1_post(0.0) == pytest.approx(i_h1, abs=1e-6)

    def test_partial_pooling_is_between(self):
        collapse = self._h1_post(1.0)
        independent = self._h1_post(0.0)
        partial = self._h1_post(0.5)
        assert collapse < partial < independent  # 0.5 sits strictly between


class TestResidualHypothesis:
    """Opt-in residual H0 makes the partition exhaustive (estimand completeness)."""

    def test_default_excludes_residual(self):
        result = run_bayesian_update(_testing(_vec("e1", {"h1": 9.0, "h2": 1.0})), ["h1", "h2"])
        assert all(p.hypothesis_id != RESIDUAL_ID for p in result.posteriors)

    def test_residual_present_when_enabled(self):
        result = run_bayesian_update(
            _testing(_vec("e1", {"h1": 9.0, "h2": 1.0})), ["h1", "h2"], include_residual=True
        )
        ids = {p.hypothesis_id for p in result.posteriors}
        assert RESIDUAL_ID in ids
        assert RESIDUAL_ID in result.ranking

    def test_residual_uniform_with_no_evidence(self):
        # No informative evidence -> uniform over {h1, h2, H0} = 1/3 each.
        result = run_bayesian_update(_testing(), ["h1", "h2"], include_residual=True)
        for p in result.posteriors:
            assert p.final_posterior == pytest.approx(1 / 3, abs=0.01)

    def test_residual_has_flat_likelihood(self):
        result = run_bayesian_update(
            _testing(_vec("e1", {"h1": 9.0, "h2": 1.0})), ["h1", "h2"], include_residual=True
        )
        h0 = next(p for p in result.posteriors if p.hypothesis_id == RESIDUAL_ID)
        assert all(u.likelihood_ratio == pytest.approx(1.0) for u in h0.updates)

    def test_residual_with_researcher_priors(self):
        # Priors cover only the listed hypotheses; residual still gets reserve mass.
        result = run_bayesian_update(
            _testing(_vec("e1", {"h1": 1.0, "h2": 1.0})), ["h1", "h2"],
            priors={"h1": 3.0, "h2": 1.0}, include_residual=True,
        )
        ids = {p.hypothesis_id for p in result.posteriors}
        assert RESIDUAL_ID in ids
        assert sum(p.final_posterior for p in result.posteriors) == pytest.approx(1.0, abs=1e-6)


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

    def test_interpretive_cap_is_tighter(self):
        # A per-item cap of 5 bounds the pairwise ratio to 5, not 20.
        import math
        lrs = item_lrs(_vec("e1", {"h1": 1e6, "h2": 1.0}), ["h1", "h2"], cap=5.0)
        assert lrs["h1"] / lrs["h2"] == pytest.approx(5.0)
        assert lrs["h1"] == pytest.approx(math.sqrt(5.0))

    def test_caps_applied_in_update(self):
        # Same vector, but capped at 5 via caps -> the update LR reflects the tighter cap.
        testing = _testing(_vec("e1", {"h1": 1e6, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"], caps={"e1": 5.0})
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        import math
        assert h1.updates[0].likelihood_ratio == pytest.approx(math.sqrt(5.0), abs=1e-3)


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


# ── Robustness computation (operates on individual LR floats, pre-pool) ──


class TestRobustness:
    def test_robust_few_decisive(self):
        # Individual (pre-pool) LRs: 4 decisive items
        lrs = [10.0, 0.1, 8.0, 0.15]
        assert _compute_robustness(lrs) == "robust"

    def test_fragile_many_weak(self):
        lrs = [0.8] * 15
        assert _compute_robustness(lrs) == "fragile"

    def test_unknown_empty(self):
        assert _compute_robustness([]) == "unknown"

    def test_unknown_all_uninformative(self):
        lrs = [1.0] * 10
        assert _compute_robustness(lrs) == "unknown"

    def test_uninformative_items_do_not_force_fragile(self):
        # 4 moderate items + 10 uninformative — should not be fragile
        lrs = [3.0] * 4 + [1.0] * 10
        assert _compute_robustness(lrs) != "fragile"

    def test_robustness_populated_in_result(self):
        testing = _testing(_vec("e1", {"h1": 19.0, "h2": 1.0}))
        result = run_bayesian_update(testing, ["h1", "h2"])
        for p in result.posteriors:
            assert p.robustness in ("robust", "fragile", "moderate", "unknown")

    def test_robustness_uses_individual_lrs_not_pooled(self):
        """HIGH-1: Robustness must be classified from individual LRs, not cluster-pooled LRs.

        A cluster of 10 weak items (LR≈1.4) with rho=0 would have k_eff=10 and a
        combined LR of 1.4^10 ≈ 28.9 — which looks decisive. The result must be
        classified as 'fragile' (many weak items), not 'robust' (one big combined LR).
        """
        from pt.schemas import EvidenceCluster
        # 10 weakly-discriminating items for h1 vs h2
        items = [_vec(f"e{i}", {"h1": 1.4, "h2": 1.0}) for i in range(10)]
        testing = _testing(*items)
        # Put all 10 items in one fully-independent cluster (rho=0 → k_eff=10)
        # This makes the pooled LR = 1.4^10 ≈ 28.9 (decisive) but individual LRs are weak
        testing = testing.model_copy(update={
            "dependence_clusters": [
                EvidenceCluster(
                    evidence_ids=[f"e{i}" for i in range(10)],
                    reason="same source section",
                    dependence_strength=0.0,  # fully independent → k_eff=10
                )
            ]
        })
        result = run_bayesian_update(testing, ["h1", "h2"])
        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        # With pooled LRs, this would be classified as "robust" (combined LR ≈ 28.9)
        # With individual LRs (each ≈ 1.4), this must be "fragile" or "moderate"
        assert h1.robustness in ("fragile", "moderate"), (
            f"Expected 'fragile' or 'moderate' for 10 weak items in one cluster, "
            f"got {h1.robustness!r} — robustness may be computed from pooled LRs."
        )


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


# ── Slice 5: Source-Lineage Dependence Benchmark ─────────────────────

class TestSourceLineageDependence:
    """Planted fixtures demonstrating that duplicate / shared-source items produce
    lower effective evidence than independent corroboration, and that EvidenceCluster
    lineage_type carries the explanation.

    These are the canonical 'it actually works' fixtures for Slice 5.
    """

    def _support(self, testing: TestingResult, hyp_id: str = "h1") -> float:
        result = run_bayesian_update(testing, ["h1", "h2"])
        return next(p.final_posterior for p in result.posteriors if p.hypothesis_id == hyp_id)

    def test_duplicate_items_without_cluster_inflate_support(self):
        """3 identical evidence items with no cluster: support is much higher than 1 item."""
        single = _testing(_vec("e0", {"h1": 3.0, "h2": 1.0}))
        triplicate = _testing(
            _vec("e0", {"h1": 3.0, "h2": 1.0}),
            _vec("e1", {"h1": 3.0, "h2": 1.0}),
            _vec("e2", {"h1": 3.0, "h2": 1.0}),
        )
        support_one = self._support(single)
        support_three = self._support(triplicate)
        assert support_three > support_one + 0.05  # meaningfully inflated

    def test_duplicate_items_with_cluster_collapse_to_single(self):
        """Same 3 identical items with a full-redundancy cluster: support = single item."""
        single_support = self._support(_testing(_vec("e0", {"h1": 3.0, "h2": 1.0})))
        cluster = EvidenceCluster(
            evidence_ids=["e0", "e1", "e2"],
            reason="same passage, three extractions",
            lineage_type="duplicate",
            dependence_strength=1.0,
        )
        clustered = _testing(
            _vec("e0", {"h1": 3.0, "h2": 1.0}),
            _vec("e1", {"h1": 3.0, "h2": 1.0}),
            _vec("e2", {"h1": 3.0, "h2": 1.0}),
            clusters=[cluster],
        )
        assert self._support(clustered) == pytest.approx(single_support, abs=1e-6)

    def test_shared_source_cluster_partially_corrects_inflation(self):
        """Shared-source items (dependence=0.7): support is between single and independent."""
        support_single = self._support(_testing(_vec("e0", {"h1": 3.0, "h2": 1.0})))
        support_indep = self._support(_testing(
            _vec("e0", {"h1": 3.0, "h2": 1.0}),
            _vec("e1", {"h1": 3.0, "h2": 1.0}),
            _vec("e2", {"h1": 3.0, "h2": 1.0}),
        ))
        cluster = EvidenceCluster(
            evidence_ids=["e0", "e1", "e2"],
            reason="same document, different sections",
            lineage_type="shared_source",
            dependence_strength=0.7,
        )
        support_shared = self._support(_testing(
            _vec("e0", {"h1": 3.0, "h2": 1.0}),
            _vec("e1", {"h1": 3.0, "h2": 1.0}),
            _vec("e2", {"h1": 3.0, "h2": 1.0}),
            clusters=[cluster],
        ))
        # Shared-source correction must sit strictly between the two extremes
        assert support_single < support_shared < support_indep

    def test_same_event_cluster_partially_corrects_inflation(self):
        """Same-event cluster (dependence=0.6): same monotonicity as shared_source."""
        support_single = self._support(_testing(_vec("e0", {"h1": 2.5, "h2": 1.0})))
        support_indep = self._support(_testing(
            _vec("e0", {"h1": 2.5, "h2": 1.0}),
            _vec("e1", {"h1": 2.5, "h2": 1.0}),
        ))
        cluster = EvidenceCluster(
            evidence_ids=["e0", "e1"],
            reason="two accounts of the same battle",
            lineage_type="same_event",
            dependence_strength=0.6,
        )
        support_pooled = self._support(_testing(
            _vec("e0", {"h1": 2.5, "h2": 1.0}),
            _vec("e1", {"h1": 2.5, "h2": 1.0}),
            clusters=[cluster],
        ))
        assert support_single < support_pooled < support_indep

    def test_lineage_type_field_on_cluster(self):
        """lineage_type is stored and round-trips through the schema."""
        for lt in ("duplicate", "shared_source", "same_event", "same_mechanism", "other"):
            c = EvidenceCluster(evidence_ids=["e0", "e1"], reason="r", lineage_type=lt)  # type: ignore[arg-type]
            assert c.lineage_type == lt
            restored = EvidenceCluster.model_validate_json(c.model_dump_json())
            assert restored.lineage_type == lt

    def test_lineage_type_defaults_to_none(self):
        """lineage_type is Optional — old clusters without it still load cleanly."""
        c = EvidenceCluster(evidence_ids=["e0", "e1"], reason="r")
        assert c.lineage_type is None

    def test_lineage_type_rejects_invalid(self):
        with pytest.raises(Exception):
            EvidenceCluster(evidence_ids=["e0", "e1"], reason="r", lineage_type="gossip")  # type: ignore[arg-type]

    def test_inflation_delta_is_material(self):
        """The support difference between clustered and unclustered duplicates is substantial.
        This is the core 'duplicate evidence cannot materially inflate support' criterion.
        """
        items = [_vec(f"d{i}", {"h1": 4.0, "h2": 1.0}) for i in range(5)]
        cluster = EvidenceCluster(
            evidence_ids=[f"d{i}" for i in range(5)],
            reason="same paragraph, re-extracted 5 times",
            lineage_type="duplicate",
            dependence_strength=1.0,
        )
        support_clustered = self._support(_testing(*items, clusters=[cluster]))
        support_unclustered = self._support(_testing(*items))
        # The unclustered version should be meaningfully more extreme
        assert support_unclustered - support_clustered > 0.05
