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

class TestCriticFindingTargetValidation:
    """CriticFinding model_validator enforces target_type/target consistency."""

    def test_causal_edge_target_requires_arrow(self):
        with pytest.raises(ValidationError, match="no '->'"):
            _make_critic_finding(
                finding_type="void_link",
                target="h1",  # bare ID — not an edge
                target_type="causal_edge",
            )

    def test_hypothesis_target_rejects_compound_with_arrow(self):
        with pytest.raises(ValidationError, match="contains '->'"):
            _make_critic_finding(
                finding_type="confound",
                target="h1->h5",  # compound — must use causal_edge type
                target_type="hypothesis",
            )

    def test_valid_causal_edge_target(self):
        f = _make_critic_finding(
            finding_type="void_link",
            target="mech_fiscal_crisis->evt_storming_bastille",
            target_type="causal_edge",
        )
        assert f.target_type == "causal_edge"
        assert "->" in f.target

    def test_valid_hypothesis_target_bare_id(self):
        f = _make_critic_finding(
            finding_type="missing_pathway",
            target="h1",
            target_type="hypothesis",
        )
        assert f.target == "h1"

    def test_confound_between_hypotheses_uses_causal_edge_type(self):
        """A confound between h1 and h5 should use causal_edge target_type."""
        f = _make_critic_finding(
            finding_type="confound",
            target="h1->h5",
            target_type="causal_edge",  # correct type for compound target
        )
        assert f.target_type == "causal_edge"

    def test_causal_edge_rejects_multi_hop_chain(self):
        """target_type='causal_edge' with more than one '->' must fail validation."""
        with pytest.raises(ValidationError, match="exactly one edge"):
            _make_critic_finding(
                finding_type="confirmed_link",
                target="evt_a->evt_b->evt_c",
                target_type="causal_edge",
            )


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
        # Each target_type needs a compatible target to pass the model_validator
        valid_combinations = [
            ("evidence", "evi_debt"),
            ("hypothesis", "h1"),
            ("causal_edge", "mech_a->evt_b"),
        ]
        for tt, tgt in valid_combinations:
            f = _make_critic_finding(target_type=tt, target=tgt)
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


# ── Pipeline integration: critic flag on ─────────────────────────────────

class TestCriticPipelineOn:
    """critic=True path: synthesis reuse, double _run_core_passes, correct artifacts."""

    def _build_mocks(self):
        """Return the standard mock fixture objects needed across tests."""
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
        return extraction, hs, testing, absence, bayesian, synthesis, dm

    def _common_patches(self, extraction, hs, testing, absence, bayesian, synthesis, dm):
        """Return a dict of patch targets → return values used in all critic-on tests."""
        from pt.schemas import PartitionAudit
        partition_audit = PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[],
            hypotheses_flagged=[],
            overall_quality="adequate",
            summary="No issues.",
        )
        return {
            "pt.pipeline.run_extract": extraction,
            "pt.pipeline.run_hypothesize": hs,
            "pt.pipeline.run_partition": partition_audit,
            "pt.pipeline.run_synthesize": synthesis,
        }

    def test_synthesis_reused_when_no_re_elicitation(self, tmp_path):
        """When critic finds no high-severity issues, synthesis is NOT re-run."""
        from pt.pipeline import run_pipeline
        extraction, hs, testing, absence, bayesian, synthesis, dm = self._build_mocks()

        low_critic = _make_critic_result([_make_critic_finding(severity="low")])
        # re_elicitation_needed is False (only low-severity)
        assert low_critic.re_elicitation_needed is False

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline._run_core_passes", return_value=(testing, absence, bayesian, dm)) as mock_core,
            patch("pt.pipeline.run_critic", return_value=low_critic),
            patch("pt.pipeline.run_synthesize", return_value=synthesis) as mock_synth,
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True, rival_pairs=[],
                hypotheses_flagged=[], overall_quality="adequate", summary="ok",
            )
            run_pipeline(
                " ".join(["word"] * 350),
                output_dir=str(tmp_path),
                critic=True,
            )

        # _run_core_passes called exactly once (no re-elicitation)
        assert mock_core.call_count == 1
        # run_synthesize called exactly once (base snapshot only; critic reuses it)
        assert mock_synth.call_count == 1

    def test_core_passes_called_twice_when_re_elicitation_triggered(self, tmp_path):
        """When critic finds a high-severity issue, _run_core_passes is called twice."""
        from pt.pipeline import run_pipeline
        extraction, hs, testing, absence, bayesian, synthesis, dm = self._build_mocks()

        high_critic = _make_critic_result([_make_critic_finding(severity="high")])
        assert high_critic.re_elicitation_needed is True

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline._run_core_passes", return_value=(testing, absence, bayesian, dm)) as mock_core,
            patch("pt.pipeline.run_critic", return_value=high_critic),
            patch("pt.pipeline.run_synthesize", return_value=synthesis) as mock_synth,
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True, rival_pairs=[],
                hypotheses_flagged=[], overall_quality="adequate", summary="ok",
            )
            run_pipeline(
                " ".join(["word"] * 350),
                output_dir=str(tmp_path),
                critic=True,
            )

        # _run_core_passes called twice: once for base, once for re-elicitation
        assert mock_core.call_count == 2
        # Second call must inject the critic summary as critic_context
        second_call_kwargs = mock_core.call_args_list[1].kwargs
        assert second_call_kwargs.get("critic_context") == high_critic.summary, (
            f"Re-elicitation must pass critic_context=critic.summary; got {second_call_kwargs.get('critic_context')!r}"
        )
        # run_synthesize called twice: once for base snapshot, once post-critic
        assert mock_synth.call_count == 2

    def test_result_base_written_result_critic_not_written(self, tmp_path):
        """result_base.json must exist; result_critic.json must NOT exist."""
        from pt.pipeline import run_pipeline
        extraction, hs, testing, absence, bayesian, synthesis, dm = self._build_mocks()

        low_critic = _make_critic_result([])

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline._run_core_passes", return_value=(testing, absence, bayesian, dm)),
            patch("pt.pipeline.run_critic", return_value=low_critic),
            patch("pt.pipeline.run_synthesize", return_value=synthesis),
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True, rival_pairs=[],
                hypotheses_flagged=[], overall_quality="adequate", summary="ok",
            )
            run_pipeline(
                " ".join(["word"] * 350),
                output_dir=str(tmp_path),
                critic=True,
            )

        assert (tmp_path / "result_base.json").exists(), "result_base.json missing"
        assert (tmp_path / "critic_delta.json").exists(), "critic_delta.json missing"
        assert not (tmp_path / "result_critic.json").exists(), "result_critic.json must NOT exist"

    def test_critic_result_stored_in_pipeline_result(self, tmp_path):
        """result.critic must be populated with the CriticResult object."""
        from pt.pipeline import run_pipeline
        extraction, hs, testing, absence, bayesian, synthesis, dm = self._build_mocks()

        medium_critic = _make_critic_result([_make_critic_finding(severity="medium")])

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline._run_core_passes", return_value=(testing, absence, bayesian, dm)),
            patch("pt.pipeline.run_critic", return_value=medium_critic),
            patch("pt.pipeline.run_synthesize", return_value=synthesis),
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True, rival_pairs=[],
                hypotheses_flagged=[], overall_quality="adequate", summary="ok",
            )
            result = run_pipeline(
                " ".join(["word"] * 350),
                output_dir=str(tmp_path),
                critic=True,
            )

        assert result.critic is not None
        assert result.critic.re_elicitation_needed is False
        assert len(result.critic.findings) == 1
        assert result.critic.findings[0].severity == "medium"

    def test_critic_model_override_passed_to_run_critic(self, tmp_path):
        """critic_model overrides the main model for the critic pass only."""
        from pt.pipeline import run_pipeline
        extraction, hs, testing, absence, bayesian, synthesis, dm = self._build_mocks()

        low_critic = _make_critic_result([_make_critic_finding(severity="low")])

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline._run_core_passes", return_value=(testing, absence, bayesian, dm)),
            patch("pt.pipeline.run_critic", return_value=low_critic) as mock_critic,
            patch("pt.pipeline.run_synthesize", return_value=synthesis),
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True, rival_pairs=[],
                hypotheses_flagged=[], overall_quality="adequate", summary="ok",
            )
            run_pipeline(
                " ".join(["word"] * 350),
                output_dir=str(tmp_path),
                model="main-model",
                critic=True,
                critic_model="critic-model-override",
            )

        # run_critic must be called with critic_model, not the main model
        call_kwargs = mock_critic.call_args.kwargs
        assert call_kwargs.get("model") == "critic-model-override", (
            f"run_critic was called with model={call_kwargs.get('model')!r}; "
            "expected 'critic-model-override'"
        )

    def test_critic_model_falls_back_to_main_model_when_not_set(self, tmp_path):
        """When critic_model is None, run_critic receives the main model."""
        from pt.pipeline import run_pipeline
        extraction, hs, testing, absence, bayesian, synthesis, dm = self._build_mocks()

        low_critic = _make_critic_result([_make_critic_finding(severity="low")])

        with (
            patch("pt.pipeline.run_extract", return_value=extraction),
            patch("pt.pipeline.run_hypothesize", return_value=hs),
            patch("pt.pipeline.run_partition") as mock_partition,
            patch("pt.pipeline._run_core_passes", return_value=(testing, absence, bayesian, dm)),
            patch("pt.pipeline.run_critic", return_value=low_critic) as mock_critic,
            patch("pt.pipeline.run_synthesize", return_value=synthesis),
        ):
            from pt.schemas import PartitionAudit
            mock_partition.return_value = PartitionAudit(
                research_question_adequate=True, rival_pairs=[],
                hypotheses_flagged=[], overall_quality="adequate", summary="ok",
            )
            run_pipeline(
                " ".join(["word"] * 350),
                output_dir=str(tmp_path),
                model="main-model",
                critic=True,
                # critic_model not set → should fall back to main model
            )

        call_kwargs = mock_critic.call_args.kwargs
        assert call_kwargs.get("model") == "main-model", (
            f"run_critic was called with model={call_kwargs.get('model')!r}; "
            "expected 'main-model' fallback"
        )


# ── _compute_critic_delta finding count logic ─────────────────────────────

class TestComputeCriticDeltaFindingCounts:
    """Unit tests for the type-dispatched finding count logic in _compute_critic_delta."""

    def _make_bayesian_result(
        self,
        h_ids: list[str],
        posteriors: list[float],
        driver_override: dict[str, list[str]] | None = None,
    ) -> "BayesianResult":
        """Build a minimal BayesianResult with controlled top_drivers per hypothesis."""
        from pt.schemas import BayesianResult, HypothesisPosterior, EvidenceUpdate
        assert len(h_ids) == len(posteriors)
        n = len(h_ids)
        prior = 1.0 / n
        return BayesianResult(
            posteriors=[
                HypothesisPosterior(
                    hypothesis_id=hid,
                    prior=prior,
                    updates=[
                        EvidenceUpdate(
                            evidence_id=f"evi_{hid}_a",
                            likelihood_ratio=1.0,
                            prior=prior,
                            posterior=p,
                        )
                    ],
                    final_posterior=p,
                    robustness="moderate",
                    top_drivers=(
                        driver_override[hid]
                        if driver_override and hid in driver_override
                        else [f"evi_{hid}_a", f"evi_{hid}_b"]
                    ),
                )
                for hid, p in zip(h_ids, posteriors)
            ],
            ranking=h_ids,
        )

    def test_hypothesis_finding_counts_by_exact_id(self):
        """hypothesis-type findings count only when target == hypothesis_id exactly."""
        from pt.pipeline import _compute_critic_delta
        from pt.schemas import CriticResult

        base = self._make_bayesian_result(["h1", "h2"], [0.6, 0.4])
        critic = self._make_bayesian_result(["h1", "h2"], [0.55, 0.45])
        findings = [
            _make_critic_finding(target="h1", target_type="hypothesis", severity="medium"),
            _make_critic_finding(target="h1", target_type="hypothesis", severity="low"),
            _make_critic_finding(target="h2", target_type="hypothesis", severity="low"),
        ]
        cr = CriticResult(findings=findings, summary="test")
        deltas = _compute_critic_delta(base, critic, cr)

        h1_delta = next(d for d in deltas if d.hypothesis_id == "h1")
        h2_delta = next(d for d in deltas if d.hypothesis_id == "h2")
        assert h1_delta.critic_findings_count == 2  # two hypothesis findings for h1
        assert h2_delta.critic_findings_count == 1  # one hypothesis finding for h2

    def test_causal_edge_findings_not_attributed_to_any_hypothesis(self):
        """causal_edge findings must not count toward any hypothesis's finding count."""
        from pt.pipeline import _compute_critic_delta
        from pt.schemas import CriticResult

        base = self._make_bayesian_result(["h1", "h2"], [0.6, 0.4])
        critic = self._make_bayesian_result(["h1", "h2"], [0.55, 0.45])
        findings = [
            _make_critic_finding(target="h1->h2", target_type="causal_edge", severity="high"),
        ]
        cr = CriticResult(findings=findings, summary="test")
        deltas = _compute_critic_delta(base, critic, cr)

        for d in deltas:
            assert d.critic_findings_count == 0, (
                f"{d.hypothesis_id} got count {d.critic_findings_count} from causal_edge finding"
            )

    def test_compound_hypothesis_target_with_arrow_would_have_been_miscounted(self):
        """Verifies schema now rejects compound hypothesis targets (they'd never match)."""
        # This tests that the old bug path can't happen — the schema validator
        # prevents 'target_type=hypothesis' with '->' in target from being constructed.
        with pytest.raises(ValidationError, match="contains '->'"):
            _make_critic_finding(
                target="h1->h5",
                target_type="hypothesis",
                severity="high",
            )

    def test_evidence_finding_counts_toward_hypotheses_sharing_driver(self):
        """Evidence-type findings count toward hypotheses that have the evidence as a top driver."""
        from pt.pipeline import _compute_critic_delta
        from pt.schemas import CriticResult

        # Give h1 a unique driver (evi_shared is NOT in h1's drivers); h2 doesn't have evi_h1_only
        base = self._make_bayesian_result(
            ["h1", "h2"], [0.6, 0.4],
            driver_override={"h1": ["evi_h1_only", "evi_common"], "h2": ["evi_h2_only", "evi_common"]},
        )
        critic = self._make_bayesian_result(
            ["h1", "h2"], [0.55, 0.45],
            driver_override={"h1": ["evi_h1_only", "evi_common"], "h2": ["evi_h2_only", "evi_common"]},
        )
        findings = [
            # targets evi_h1_only — only h1 has this as a top driver
            _make_critic_finding(target="evi_h1_only", target_type="evidence", severity="medium"),
        ]
        cr = CriticResult(findings=findings, summary="test")
        deltas = _compute_critic_delta(base, critic, cr)

        h1_delta = next(d for d in deltas if d.hypothesis_id == "h1")
        h2_delta = next(d for d in deltas if d.hypothesis_id == "h2")
        assert h1_delta.critic_findings_count == 1
        assert h2_delta.critic_findings_count == 0


# ── CRIT-2: critic+refine guard fires before any LLM calls ─────────

class TestCriticRefineGuardFiresEarly:
    """CRIT-2: --critic + --refine raises ValueError before any LLM call."""

    def test_critic_and_refine_raises_before_llm(self):
        """Guard must fire before run_extract so no budget is wasted."""
        from pt.pipeline import run_pipeline

        with patch("pt.pipeline.run_extract") as mock_extract:
            with pytest.raises(ValueError, match="--critic and --refine cannot be used together"):
                run_pipeline(
                    " ".join(["word"] * 350),
                    critic=True,
                    refine=True,
                )

        # No LLM call should have been made
        mock_extract.assert_not_called()
