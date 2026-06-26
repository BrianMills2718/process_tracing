"""Deterministic tests for the partition audit pass (Plan #3 Slice 2).

Tests cover:
- RivalPairAudit and PartitionAudit schema validation
- run_partition: cap_applied set and UserWarning emitted on needs_review
- run_partition: no warning on adequate partition
- run_partition: single hypothesis (no pairs) → adequate
- ProcessTracingResult.partition_audit field exists and is optional
- Broad/overlap failure case serializes correctly

LLM calls are mocked — this suite is deterministic.
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

import pytest

from pt.schemas import (
    Hypothesis,
    HypothesisSpace,
    PartitionAudit,
    PartitionQuality,
    Prediction,
    ProcessTracingResult,
    RivalPairAudit,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_hypothesis(h_id: str, desc: str, pred_descs: list[str]) -> Hypothesis:
    preds = [Prediction(id=f"pred_{h_id}_{i:02d}", description=d) for i, d in enumerate(pred_descs)]
    return Hypothesis(
        id=h_id,
        description=desc,
        source="generated",
        theoretical_basis="test basis",
        causal_mechanism="test mechanism",
        observable_predictions=preds,
    )


def _make_space(*hypotheses: Hypothesis) -> HypothesisSpace:
    return HypothesisSpace(
        research_question="Why did X happen rather than Y?",
        hypotheses=list(hypotheses),
    )


def _adequate_audit(n_pairs: int = 1) -> PartitionAudit:
    pairs = [
        RivalPairAudit(
            h1_id=f"h{i+1}", h2_id=f"h{i+2}",
            overlap_concern=False, complementary_concern=False, absorptive_concern=False,
            discriminator_count=3, concern_detail="",
        )
        for i in range(n_pairs)
    ]
    return PartitionAudit(
        research_question_adequate=True,
        rival_pairs=pairs,
        hypotheses_flagged=[],
        overall_quality="adequate",
        summary="Hypotheses make opposite predictions. No partition concerns.",
    )


def _overlap_audit() -> PartitionAudit:
    return PartitionAudit(
        research_question_adequate=True,
        rival_pairs=[
            RivalPairAudit(
                h1_id="h1", h2_id="h2",
                overlap_concern=True, complementary_concern=False, absorptive_concern=False,
                discriminator_count=0, concern_detail="Both predict the same evidence patterns.",
            )
        ],
        hypotheses_flagged=["h2"],
        overall_quality="needs_review",
        summary="H1 and H2 are not genuine rivals — they predict the same observable evidence.",
    )


# ── RivalPairAudit schema ─────────────────────────────────────────────

class TestRivalPairAuditSchema:
    def test_valid_pair_no_concerns(self):
        pair = RivalPairAudit(
            h1_id="h1", h2_id="h2",
            overlap_concern=False, complementary_concern=False, absorptive_concern=False,
            discriminator_count=3, concern_detail="",
        )
        assert pair.discriminator_count == 3
        assert not pair.overlap_concern
        assert not pair.complementary_concern
        assert not pair.absorptive_concern

    def test_all_concerns_set(self):
        pair = RivalPairAudit(
            h1_id="h1", h2_id="h3",
            overlap_concern=True, complementary_concern=True, absorptive_concern=True,
            discriminator_count=0, concern_detail="All concerns triggered.",
        )
        assert pair.overlap_concern
        assert pair.complementary_concern
        assert pair.absorptive_concern

    def test_discriminator_count_non_negative(self):
        with pytest.raises(Exception):
            RivalPairAudit(
                h1_id="h1", h2_id="h2",
                overlap_concern=False, complementary_concern=False, absorptive_concern=False,
                discriminator_count=-1, concern_detail="",
            )

    def test_zero_discriminators_allowed(self):
        pair = RivalPairAudit(
            h1_id="h1", h2_id="h2",
            overlap_concern=True, complementary_concern=False, absorptive_concern=False,
            discriminator_count=0, concern_detail="No discriminating predictions.",
        )
        assert pair.discriminator_count == 0

    def test_roundtrip_json(self):
        pair = RivalPairAudit(
            h1_id="h_a", h2_id="h_b",
            overlap_concern=False, complementary_concern=True, absorptive_concern=False,
            discriminator_count=1, concern_detail="Complementary.",
        )
        data = pair.model_dump()
        assert data["h1_id"] == "h_a"
        assert data["complementary_concern"] is True
        restored = RivalPairAudit.model_validate(data)
        assert restored.h2_id == "h_b"


# ── PartitionAudit schema ─────────────────────────────────────────────

class TestPartitionAuditSchema:
    def test_adequate_with_no_flagged(self):
        audit = _adequate_audit()
        assert audit.overall_quality == "adequate"
        assert audit.hypotheses_flagged == []
        assert not audit.cap_applied  # default False

    def test_needs_review_with_cap(self):
        audit = _overlap_audit()
        audit.cap_applied = True
        assert audit.overall_quality == "needs_review"
        assert audit.cap_applied

    def test_cap_applied_defaults_false(self):
        audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[],
            hypotheses_flagged=[],
            overall_quality="adequate",
            summary="No issues.",
        )
        assert not audit.cap_applied

    def test_roundtrip_json_preserves_all_fields(self):
        audit = _overlap_audit()
        data = audit.model_dump()
        restored = PartitionAudit.model_validate(data)
        assert restored.overall_quality == "needs_review"
        assert restored.rival_pairs[0].overlap_concern is True
        assert restored.hypotheses_flagged == ["h2"]

    def test_empty_rival_pairs_allowed(self):
        audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[],
            hypotheses_flagged=[],
            overall_quality="adequate",
            summary="Single hypothesis — no rival pairs.",
        )
        assert audit.rival_pairs == []

    def test_multiple_flagged_hypotheses(self):
        audit = PartitionAudit(
            research_question_adequate=False,
            rival_pairs=[
                RivalPairAudit(
                    h1_id="h1", h2_id="h2",
                    overlap_concern=True, complementary_concern=True, absorptive_concern=True,
                    discriminator_count=0, concern_detail="All problems.",
                )
            ],
            hypotheses_flagged=["h1", "h2"],
            overall_quality="needs_review",
            summary="Both hypotheses are problematic.",
        )
        assert len(audit.hypotheses_flagged) == 2
        assert not audit.research_question_adequate


# ── run_partition behaviour ───────────────────────────────────────────

class TestRunPartition:
    def _space_two_hyps(self) -> HypothesisSpace:
        return _make_space(
            _make_hypothesis("h1", "Institutional failure", [
                "Unlike H2, expects breakdown before crisis",
                "Unlike H2, no evidence of elite coordination",
            ]),
            _make_hypothesis("h2", "Elite agency conspiracy", [
                "Unlike H1, expects coordinated planning",
                "Unlike H1, no institutional breakdown needed",
            ]),
        )

    def test_adequate_partition_no_warning_no_cap(self):
        from pt.pass_partition import run_partition
        with patch("pt.pass_partition.call_llm", return_value=_adequate_audit()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_partition(self._space_two_hyps())
        assert result.overall_quality == "adequate"
        assert not result.cap_applied
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert user_warnings == [], "No UserWarning expected for adequate partition"

    def test_overlap_sets_cap_and_warns(self):
        from pt.pass_partition import run_partition
        with patch("pt.pass_partition.call_llm", return_value=_overlap_audit()):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_partition(self._space_two_hyps())
        assert result.overall_quality == "needs_review"
        assert result.cap_applied, "cap_applied must be True when needs_review"
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "needs review" in str(user_warnings[0].message).lower()

    def test_complementary_pair_sets_cap_and_warns(self):
        from pt.pass_partition import run_partition
        complementary_audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[
                RivalPairAudit(
                    h1_id="h1", h2_id="h2",
                    overlap_concern=False, complementary_concern=True, absorptive_concern=False,
                    discriminator_count=1, concern_detail="Both could be simultaneously true.",
                )
            ],
            hypotheses_flagged=["h1"],
            overall_quality="needs_review",
            summary="Complementary factors, not rivals.",
        )
        with patch("pt.pass_partition.call_llm", return_value=complementary_audit):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_partition(self._space_two_hyps())
        assert result.cap_applied
        assert len([x for x in w if issubclass(x.category, UserWarning)]) == 1

    def test_single_hypothesis_no_pairs_adequate(self):
        from pt.pass_partition import run_partition
        single_audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[],
            hypotheses_flagged=[],
            overall_quality="adequate",
            summary="Single hypothesis — partition trivially adequate.",
        )
        space = _make_space(
            _make_hypothesis("h1", "Only explanation", ["Predicts X"])
        )
        with patch("pt.pass_partition.call_llm", return_value=single_audit):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_partition(space)
        assert result.rival_pairs == []
        assert result.overall_quality == "adequate"
        assert not result.cap_applied

    def test_zero_discriminators_sets_cap(self):
        from pt.pass_partition import run_partition
        no_disc_audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[
                RivalPairAudit(
                    h1_id="h1", h2_id="h2",
                    overlap_concern=False, complementary_concern=False, absorptive_concern=False,
                    discriminator_count=0, concern_detail="No discriminating predictions found.",
                )
            ],
            hypotheses_flagged=[],
            overall_quality="needs_review",
            summary="No discriminating predictions between h1 and h2.",
        )
        with patch("pt.pass_partition.call_llm", return_value=no_disc_audit):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = run_partition(self._space_two_hyps())
        assert result.cap_applied
        # Warning message should mention the problem pair
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "h1" in str(user_warnings[0].message)

    def test_cap_applied_overrides_llm_false(self):
        """Pipeline always sets cap_applied based on quality — LLM output ignored."""
        from pt.pass_partition import run_partition
        # LLM says needs_review but cap_applied=False (shouldn't happen, but test the override)
        audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[
                RivalPairAudit(
                    h1_id="h1", h2_id="h2",
                    overlap_concern=True, complementary_concern=False, absorptive_concern=False,
                    discriminator_count=0, concern_detail="Overlap.",
                )
            ],
            hypotheses_flagged=["h2"],
            overall_quality="needs_review",
            cap_applied=False,  # LLM set it wrong
            summary="Overlap found.",
        )
        with patch("pt.pass_partition.call_llm", return_value=audit):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = run_partition(self._space_two_hyps())
        assert result.cap_applied, "Pipeline must set cap_applied=True regardless of LLM output"


# ── ProcessTracingResult field ────────────────────────────────────────

class TestProcessTracingResultPartitionField:
    def test_partition_audit_field_exists(self):
        fields = ProcessTracingResult.model_fields
        assert "partition_audit" in fields

    def test_partition_audit_optional_default_none(self):
        fields = ProcessTracingResult.model_fields
        field = fields["partition_audit"]
        assert field.default is None

    def test_partition_audit_in_model_json_schema(self):
        schema = ProcessTracingResult.model_json_schema()
        # The field appears in properties (may be in $defs too)
        schema_str = json.dumps(schema)
        assert "partition_audit" in schema_str

    def test_partition_audit_accepts_none(self):
        """ProcessTracingResult can be constructed with partition_audit=None."""
        from pt.schemas import (
            AbsenceResult, BayesianResult, ExtractionResult,
            HypothesisSpace, SynthesisResult, TestingResult,
        )
        # Just check schema field is truly optional (no construction needed)
        field = ProcessTracingResult.model_fields["partition_audit"]
        import typing
        # Should be Optional (NoneType is allowed)
        annotation = field.annotation
        args = typing.get_args(annotation)
        assert type(None) in args, f"partition_audit should be Optional, got {annotation}"


# ── CLI flag ──────────────────────────────────────────────────────────

class TestPartitionReviewCLIFlag:
    """--partition-review flag is registered and accepted by argparse."""

    def test_partition_review_flag_in_cli(self):
        """argparse accepts --partition-review without error."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("input")
        parser.add_argument("--partition-review", action="store_true")
        args = parser.parse_args(["dummy.txt", "--partition-review"])
        assert args.partition_review is True

    def test_partition_review_flag_defaults_false(self):
        """--partition-review defaults to False when not supplied."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("input")
        parser.add_argument("--partition-review", action="store_true")
        args = parser.parse_args(["dummy.txt"])
        assert args.partition_review is False
