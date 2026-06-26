"""Deterministic unit tests for pass_absence."""

from __future__ import annotations

from pt.bayesian import RELEVANCE_GATE
from pt.pass_absence import _build_testing_summary
from pt.schemas import EvidenceLikelihood, HypothesisLikelihood, TestingResult


def _make_item(eid: str, relevance: float) -> EvidenceLikelihood:
    return EvidenceLikelihood(
        evidence_id=eid,
        relevance=relevance,
        justification="test",
        hypothesis_likelihoods=[
            HypothesisLikelihood(
                hypothesis_id="h1",
                relative_likelihood=1.0,
                diagnostic_type="straw_in_the_wind",
            ),
        ],
    )


def _testing(*items: EvidenceLikelihood) -> TestingResult:
    return TestingResult(
        evidence_likelihoods=list(items),
        dependence_clusters=[],
    )


class TestBuildTestingSummary:
    def test_includes_items_at_relevance_gate(self):
        """Item exactly at RELEVANCE_GATE (0.4) must be included — HIGH-2 fix."""
        testing = _testing(
            _make_item("e_at_gate", RELEVANCE_GATE),
            _make_item("e_above", 0.7),
            _make_item("e_below", RELEVANCE_GATE - 0.01),
        )
        summary = _build_testing_summary(testing)
        assert "e_at_gate" in summary["substantive_evidence_ids"]
        assert "e_above" in summary["substantive_evidence_ids"]
        assert "e_below" not in summary["substantive_evidence_ids"]

    def test_does_not_include_items_below_gate(self):
        """Items with relevance strictly below RELEVANCE_GATE must be excluded."""
        testing = _testing(
            _make_item("e_low", 0.0),
            _make_item("e_just_below", 0.39),
        )
        summary = _build_testing_summary(testing)
        assert summary["substantive_evidence_ids"] == []

    def test_threshold_matches_bayesian_gate(self):
        """The substantive threshold must match the Bayesian uninformative gate exactly.

        This prevents false-positive absence findings for items that do contribute
        to the Bayesian update (relevance between gate and the old 0.6 threshold).
        """
        # An item at 0.5 — above the Bayesian gate (0.4), below the OLD threshold (0.6).
        # With the fix, this item MUST appear in the substantive list.
        testing = _testing(_make_item("e_mid", 0.5))
        summary = _build_testing_summary(testing)
        assert "e_mid" in summary["substantive_evidence_ids"], (
            "Item with relevance=0.5 contributed to Bayesian update but was excluded "
            "from absence evaluator's substantive list (threshold mismatch bug)."
        )

    def test_result_is_sorted(self):
        """Substantive evidence IDs are returned in sorted order."""
        testing = _testing(
            _make_item("z_last", 0.9),
            _make_item("a_first", 0.9),
            _make_item("m_mid", 0.9),
        )
        summary = _build_testing_summary(testing)
        assert summary["substantive_evidence_ids"] == ["a_first", "m_mid", "z_last"]
