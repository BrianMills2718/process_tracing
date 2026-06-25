"""Tests for CriticFinding, CriticResult schemas and pass_critic orchestration.

mock-ok: Pipeline integration tests patch LLM calls with deterministic data.
Real LLM calls would be non-deterministic and expensive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    CausalEdge,
    CriticDelta,
    CriticFinding,
    CriticFindingType,
    CriticResult,
    DiagnosticMatrix,
    Evidence,
    EvidenceCluster,
    EvidenceLikelihood,
    ExtractionResult,
    Hypothesis,
    HypothesisLikelihood,
    HypothesisPosterior,
    HypothesisSpace,
    HypothesisVerdict,
    Prediction,
    ProcessTracingResult,
    RivalPairDiagnostic,
    SynthesisResult,
    TestingResult,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_critic_finding(
    finding_type: str = "confound",
    target: str = "evi_debt",
    target_type: str = "evidence",
    severity: str = "medium",
    reasoning: str = "A third variable also explains this.",
    recommendation: str = "Control for the third variable.",
) -> CriticFinding:
    return CriticFinding(
        finding_type=finding_type,
        target=target,
        target_type=target_type,
        severity=severity,
        reasoning=reasoning,
        recommendation=recommendation,
    )


def _make_critic_result(findings: list[CriticFinding] | None = None) -> CriticResult:
    return CriticResult(
        findings=findings or [],
        summary="Structural review complete. One medium-severity confound detected.",
    )


# ── Schema: CriticFinding ──────────────────────────────────────────────────

class TestCriticFinding:
    def test_valid_all_fields(self):
        f = _make_critic_finding()
        assert f.finding_type == "confound"
        assert f.target == "evi_debt"
        assert f.target_type == "evidence"
        assert f.severity == "medium"

    def test_finding_type_accepts_all_valid_literals(self):
        valid_types: list[CriticFindingType] = [
            "confound", "missing_pathway", "void_link", "too_strong_claim", "confirmed_link"
        ]
        for ft in valid_types:
            f = _make_critic_finding(finding_type=ft)
            assert f.finding_type == ft

    def test_finding_type_rejects_invalid(self):
        with pytest.raises(ValidationError):
            CriticFinding(
                finding_type="hallucinated_type",
                target="evi_debt",
                target_type="evidence",
                severity="medium",
                reasoning="x",
                recommendation="y",
            )

    def test_target_type_accepts_all_valid_literals(self):
        for tt in ["evidence", "hypothesis", "causal_edge"]:
            f = _make_critic_finding(target_type=tt)
            assert f.target_type == tt

    def test_target_type_rejects_invalid(self):
        with pytest.raises(ValidationError):
            CriticFinding(
                finding_type="confound",
                target="evi_debt",
                target_type="unknown_type",
                severity="medium",
                reasoning="x",
                recommendation="y",
            )

    def test_severity_accepts_all_valid_literals(self):
        for sev in ["high", "medium", "low"]:
            f = _make_critic_finding(severity=sev)
            assert f.severity == sev

    def test_severity_rejects_invalid(self):
        with pytest.raises(ValidationError):
            CriticFinding(
                finding_type="confound",
                target="evi_debt",
                target_type="evidence",
                severity="critical",
                reasoning="x",
                recommendation="y",
            )


# ── Schema: CriticResult ───────────────────────────────────────────────────

class TestCriticResult:
    def test_empty_findings_defaults(self):
        cr = CriticResult(summary="All good.")
        assert cr.findings == []
        assert cr.re_elicitation_needed is False

    def test_re_elicitation_false_when_no_high_severity(self):
        cr = CriticResult(
            findings=[
                _make_critic_finding(severity="medium"),
                _make_critic_finding(severity="low"),
            ],
            summary="Medium and low only.",
        )
        assert cr.re_elicitation_needed is False

    def test_re_elicitation_true_when_any_high_severity(self):
        cr = CriticResult(
            findings=[
                _make_critic_finding(severity="medium"),
                _make_critic_finding(severity="high"),
            ],
            summary="One high-severity finding.",
        )
        assert cr.re_elicitation_needed is True

    def test_re_elicitation_true_single_high(self):
        cr = CriticResult(
            findings=[_make_critic_finding(severity="high")],
            summary="High severity.",
        )
        assert cr.re_elicitation_needed is True

    def test_re_elicitation_false_all_low(self):
        cr = CriticResult(
            findings=[
                _make_critic_finding(severity="low"),
                _make_critic_finding(severity="low"),
                _make_critic_finding(severity="low"),
            ],
            summary="All low.",
        )
        assert cr.re_elicitation_needed is False

    def test_re_elicitation_is_computed_not_user_settable(self):
        """LLM-provided re_elicitation_needed is overridden by model_validator."""
        cr = CriticResult(
            findings=[_make_critic_finding(severity="low")],
            summary="Low only.",
            re_elicitation_needed=True,  # LLM says True...
        )
        # ...but validator overwrites it based on findings
        assert cr.re_elicitation_needed is False

    def test_serializes_and_deserializes(self):
        cr = CriticResult(
            findings=[
                _make_critic_finding(finding_type="confound", severity="high"),
                _make_critic_finding(finding_type="confirmed_link", severity="low"),
            ],
            summary="Two findings.",
        )
        data = cr.model_dump()
        cr2 = CriticResult.model_validate(data)
        assert cr2.re_elicitation_needed is True
        assert len(cr2.findings) == 2
        assert cr2.findings[0].finding_type == "confound"

    def test_critic_result_has_no_lr_fields(self):
        """CriticResult must not contain likelihood ratio fields."""
        field_names = set(CriticResult.model_fields.keys())
        lr_field_names = {"likelihood_ratio", "relative_likelihood", "evidence_likelihoods", "posteriors"}
        assert field_names.isdisjoint(lr_field_names), (
            f"CriticResult must not contain LR fields: {field_names & lr_field_names}"
        )


# ── Schema: CriticDelta ────────────────────────────────────────────────────

class TestCriticDelta:
    def test_valid_delta(self):
        d = CriticDelta(
            hypothesis_id="h1",
            posterior_base=0.45,
            posterior_critic=0.38,
            delta=-0.07,
        )
        assert d.hypothesis_id == "h1"
        assert d.delta == pytest.approx(-0.07)
        assert d.top_driver_change == []
        assert d.critic_findings_count == 0

    def test_serializes_cleanly(self):
        d = CriticDelta(
            hypothesis_id="h2",
            posterior_base=0.30,
            posterior_critic=0.31,
            delta=0.01,
            top_driver_change=["added:evi_x", "removed:evi_y"],
            critic_findings_count=2,
        )
        data = d.model_dump()
        assert data["hypothesis_id"] == "h2"
        assert data["top_driver_change"] == ["added:evi_x", "removed:evi_y"]


# ── ProcessTracingResult integration ──────────────────────────────────────

class TestProcessTracingResultCriticField:
    def _minimal_result(self) -> ProcessTracingResult:
        from tests.test_pipeline_integration import (
            _make_extraction, _make_hypothesis_space, _make_testing,
            _make_absence, _make_synthesis,
        )
        from pt.bayesian import run_bayesian_update
        return ProcessTracingResult(
            extraction=_make_extraction(),
            hypothesis_space=_make_hypothesis_space(),
            testing=_make_testing(),
            absence=_make_absence(),
            bayesian=run_bayesian_update(_make_testing(), ["h1", "h2"]),
            synthesis=_make_synthesis(),
        )

    def test_critic_defaults_to_none(self):
        result = self._minimal_result()
        assert result.critic is None

    def test_critic_field_accepts_critic_result(self):
        result = self._minimal_result()
        cr = _make_critic_result([_make_critic_finding(severity="high")])
        result2 = result.model_copy(update={"critic": cr})
        assert result2.critic is not None
        assert result2.critic.re_elicitation_needed is True

    def test_critic_none_serializes_and_deserializes(self):
        result = self._minimal_result()
        data = result.model_dump()
        assert data["critic"] is None
        result2 = ProcessTracingResult.model_validate(data)
        assert result2.critic is None

    def test_critic_populated_survives_round_trip(self):
        result = self._minimal_result()
        cr = _make_critic_result([
            _make_critic_finding(finding_type="missing_pathway", severity="medium"),
        ])
        result2 = result.model_copy(update={"critic": cr})
        data = result2.model_dump()
        result3 = ProcessTracingResult.model_validate(data)
        assert result3.critic is not None
        assert result3.critic.findings[0].finding_type == "missing_pathway"
        assert result3.critic.re_elicitation_needed is False


# ── Pipeline integration: critic flag off ─────────────────────────────────

class TestCriticPipelineOff:
    """critic=False must produce result.critic=None — no LLM call to critic pass."""

    def test_critic_off_produces_no_critic_field(self, tmp_path):
        """Pipeline with critic=False leaves result.critic = None."""
        from pt.pipeline import run_pipeline
        from tests.test_pipeline_integration import (
            _make_extraction, _make_hypothesis_space, _make_testing,
            _make_absence, _make_synthesis,
        )
        from pt.bayesian import run_bayesian_update
        from pt.pass_diagnostic import compute_diagnostic_matrix

        extraction = _make_extraction()
        hs = _make_hypothesis_space()
        testing = _make_testing()
        absence = _make_absence()
        bayesian = run_bayesian_update(testing, ["h1", "h2"])
        synthesis = _make_synthesis()
        dm = compute_diagnostic_matrix(testing, hs)

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline.run_test", return_value=testing),
            patch("pt.pipeline.run_absence", return_value=absence),
            patch("pt.pipeline.run_synthesize", return_value=synthesis),
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True,
                rival_pairs=[],
                hypotheses_flagged=[],
                overall_quality="adequate",
                summary="No partition issues.",
            )
            result = run_pipeline(
                " ".join(["word"] * 350),  # 350 words, above the 300-word minimum
                output_dir=str(tmp_path),
                critic=False,
            )

        assert result.critic is None
        # Critic output files must NOT exist
        assert not (tmp_path / "result_base.json").exists()
        assert not (tmp_path / "result_critic.json").exists()
        assert not (tmp_path / "critic_delta.json").exists()
