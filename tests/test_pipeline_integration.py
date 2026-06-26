"""Pipeline integration test — mocks LLM, verifies orchestration and Bayesian math.

mock-ok: This test verifies pipeline orchestration logic and Bayesian math
with deterministic data. Real LLM calls would be non-deterministic and expensive.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pt.bayesian import run_bayesian_update
from pt.pipeline import _source_text_sha256, run_pipeline
from pt.report import (
    _build_vis_data,
    _diagnostic_strength_summary,
    _dom_id,
    _layout_temporal_position,
    _temporal_position,
    generate_report,
)
from pt.source_coverage import build_source_coverage
from pt.schemas import (
    AbsenceEvaluation,
    AbsenceResult,
    Actor,
    CausalEdge,
    CriticFinding,
    CriticResult,
    Evidence,
    EvidenceCluster,
    EvidenceLikelihood,
    Event,
    ExtractionResult,
    Hypothesis,
    HypothesisLikelihood,
    HypothesisSpace,
    HypothesisVerdict,
    Mechanism,
    Prediction,
    ProcessTracingResult,
    SynthesisResult,
    TestingResult,
    TextHypothesis,
)
from pt.source_design import build_source_design_state
from pt.source_packet import SourceCandidate, SourceGap, SourceGapDisposition, SourcePacket


# ── Deterministic fixtures ─────────────────────────────────────────


def _make_extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="Test text about a political crisis caused by fiscal collapse and elite maneuvering.",
        actors=[
            Actor(id="actor_king", name="The King", description="Head of state"),
            Actor(id="actor_assembly", name="Assembly", description="Legislative body"),
        ],
        events=[
            Event(id="evt_crisis", description="Fiscal crisis erupts", date="1789"),
            Event(id="evt_coup", description="Military coup occurs", date="1799"),
        ],
        mechanisms=[
            Mechanism(id="mech_fiscal", description="Fiscal collapse undermines state legitimacy"),
        ],
        evidence=[
            Evidence(
                id="evi_debt",
                description="National debt doubled in a decade",
                source_text="The national debt had doubled between 1780 and 1789.",
                evidence_type="empirical",
                approximate_date="1789",
            ),
            Evidence(
                id="evi_tax_revolt",
                description="Tax revolt in provinces",
                source_text="Provinces refused to collect new taxes.",
                evidence_type="empirical",
                approximate_date="1789",
            ),
            Evidence(
                id="evi_elite_plot",
                description="Elite conspirators met secretly",
                source_text="Key leaders met in secret to plan the overthrow.",
                evidence_type="empirical",
                approximate_date="1799",
            ),
            Evidence(
                id="evi_historian_claim",
                description="Historian argues ideology was the primary driver",
                source_text="Furet argues that ideological radicalization was the key factor.",
                evidence_type="interpretive",
            ),
        ],
        hypotheses_in_text=[
            TextHypothesis(
                id="th1",
                description="Fiscal crisis caused the revolution",
                source_text="The fiscal crisis was the primary cause.",
            ),
        ],
        causal_edges=[
            CausalEdge(source_id="evt_crisis", target_id="evt_coup", relationship="led to"),
        ],
    )


def _make_hypothesis_space() -> HypothesisSpace:
    return HypothesisSpace(
        research_question="Why did the political crisis resolve via military coup rather than reform?",
        hypotheses=[
            Hypothesis(
                id="h1",
                description="Fiscal collapse made the state ungovernable",
                source="text",
                theoretical_basis="Fiscal-military state theory",
                causal_mechanism="Debt → inability to fund military/bureaucracy → state collapse → power vacuum",
                observable_predictions=[
                    Prediction(id="pred_h1_01", description="We should see evidence of debt crisis"),
                    Prediction(id="pred_h1_02", description="We should see state inability to function"),
                ],
            ),
            Hypothesis(
                id="h2",
                description="Elite conspiracy orchestrated the coup",
                source="generated",
                theoretical_basis="Elite theory of revolution",
                causal_mechanism="Small group of elites planned and executed power seizure",
                observable_predictions=[
                    Prediction(id="pred_h2_01", description="Evidence of secret meetings"),
                    Prediction(id="pred_h2_02", description="Named conspirators with specific plan"),
                ],
            ),
        ],
    )


def _ev_like(evidence_id: str, h1: float, h2: float, relevance: float, dtype: str = "straw_in_the_wind") -> EvidenceLikelihood:
    """One evidence item's likelihood vector across {h1, h2}."""
    return EvidenceLikelihood(
        evidence_id=evidence_id,
        hypothesis_likelihoods=[
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=h1, diagnostic_type=dtype),
            HypothesisLikelihood(hypothesis_id="h2", relative_likelihood=h2, diagnostic_type=dtype),
        ],
        relevance=relevance,
        justification="deterministic test vector",
    )


def _make_testing() -> TestingResult:
    """Deterministic likelihood vectors. Relative likelihood = P(E|H) per hypothesis;
    derived per-hypothesis LR is the value over the vector's geometric mean."""
    return TestingResult(
        evidence_likelihoods=[
            # evi_debt: favors h1 (fiscal)
            _ev_like("evi_debt", h1=0.9, h2=0.3, relevance=0.9, dtype="smoking_gun"),
            # evi_tax_revolt: favors h1
            _ev_like("evi_tax_revolt", h1=0.8, h2=0.4, relevance=0.85, dtype="straw_in_the_wind"),
            # evi_elite_plot: favors h2 (conspiracy)
            _ev_like("evi_elite_plot", h1=0.3, h2=0.9, relevance=0.9, dtype="smoking_gun"),
            # evi_historian_claim: interpretive, low relevance, uninformative
            _ev_like("evi_historian_claim", h1=0.5, h2=0.5, relevance=0.3, dtype="straw_in_the_wind"),
        ],
    )


def _make_absence() -> AbsenceResult:
    return AbsenceResult(
        evaluations=[
            AbsenceEvaluation(
                hypothesis_id="h2",
                prediction_id="pred_h2_02",
                missing_evidence="No specific names of conspirators or details of their plan",
                reasoning="A conspiracy hypothesis requires named agents with specific actions",
                severity="notable",
                would_be_extractable=True,
            ),
        ],
    )


def _make_synthesis() -> SynthesisResult:
    return SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id="h1",
                status="supported",
                key_evidence_for=["evi_debt", "evi_tax_revolt"],
                key_evidence_against=["evi_elite_plot"],
                reasoning="Fiscal evidence is strong and direct.",
                steelman="The fiscal crisis was catastrophic and undeniable.",
                posterior_robustness="moderate",
            ),
            HypothesisVerdict(
                hypothesis_id="h2",
                status="weakened",
                key_evidence_for=["evi_elite_plot"],
                key_evidence_against=["evi_debt"],
                reasoning="Only one piece of strong evidence.",
                steelman="The elite plot evidence is a smoking gun.",
                posterior_robustness="fragile",
            ),
        ],
        comparative_analysis="H1 has broader evidentiary support while H2 relies on a single item.",
        analytical_narrative="The fiscal hypothesis emerges stronger from the analysis.",
        limitations=["Small evidence base", "Only one text analyzed"],
        suggested_further_tests=["Archival records of conspirators' correspondence"],
    )


def _make_process_result(source_text_sha256: str | None = None) -> ProcessTracingResult:
    return ProcessTracingResult(
        source_text_sha256=source_text_sha256,
        extraction=_make_extraction(),
        hypothesis_space=_make_hypothesis_space(),
        testing=_make_testing(),
        absence=_make_absence(),
        bayesian=run_bayesian_update(_make_testing(), ["h1", "h2"]),
        synthesis=_make_synthesis(),
    )


def _make_source_packet() -> SourcePacket:
    return SourcePacket(
        case_name="18 Brumaire",
        research_question="Why did Brumaire produce the Consulate rather than a restored Directory?",
        focal_window="1799-10 to 1799-11",
        outcome="Creation of the Consulate",
        source_candidates=[
            SourceCandidate(
                source_id="source_a",
                title="Official proclamation",
                text_markers=["Source A"],
                source_group="official public justification",
                source_kind="primary proclamation",
                date_coverage="1799-11",
                locator="https://example.test/proclamation",
                provenance_note="Official post-coup source.",
                reliability_note="Justificatory and public-facing.",
                expected_observability="Public legitimacy claims, not private planning.",
                relevance_to_question="Tests whether order claims were post-hoc legitimation.",
            ),
            SourceCandidate(
                source_id="source_b",
                title="Council proceedings",
                text_markers=["Source B"],
                source_group="legislative record",
                source_kind="primary legislative record",
                date_coverage="1799-11",
                locator="archive://council",
                provenance_note="Proceedings from the contested legislative moment.",
                reliability_note="Institutional record with procedural blind spots.",
                expected_observability="Procedural disruption and formal resistance.",
                relevance_to_question="Tests whether legality collapsed before military coercion.",
            ),
            SourceCandidate(
                source_id="source_c",
                title="Critical historiography",
                text_markers=["Source C"],
                source_group="rival secondary account",
                source_kind="historiography",
                date_coverage="1799",
                locator=None,
                provenance_note="Scholarly rival interpretation.",
                reliability_note="Interpretive synthesis, not direct trace evidence.",
                expected_observability="Alternative mechanism claims and source disputes.",
                relevance_to_question="Preserves rival mechanisms for discrimination.",
            ),
        ],
        known_gaps=[
            SourceGap(
                missing_source_class="Private correspondence among conspirators",
                why_it_matters="Could reveal planning sequence and agency.",
                expected_location="Correspondence collections",
                priority="high",
            )
        ],
        source_gap_dispositions=[
            SourceGapDisposition(
                missing_source_class="Private correspondence among conspirators",
                status="partially_mitigated",
                relevant_source_ids=["source_c"],
                expected_trace="Private planning sequence.",
                claim_implications="Memoir evidence partially mitigates but does not clear the gap.",
                search_actions=["Checked public memoir source."],
                disposition_reason="Adjacent evidence exists; direct correspondence remains missing.",
            )
        ],
        limitations=[],
    )


def _make_audit_stress_result() -> ProcessTracingResult:
    """A deterministic result that should trigger every output-quality caveat."""
    extraction = _make_extraction()
    extraction.summary = (
        "The crisis culminated in a military coup in 1799 after a decade of legitimacy loss."
    )
    evidence: list[Evidence] = []
    vectors: list[EvidenceLikelihood] = []
    for idx in range(12):
        eid = f"evi_background_{idx:02d}"
        evidence.append(
            Evidence(
                id=eid,
                description=f"Background legitimacy erosion item {idx}",
                source_text=f"Background legitimacy erosion was visible in 1789 item {idx}.",
                evidence_type="empirical",
                approximate_date="1789",
            )
        )
        vectors.append(_ev_like(eid, h1=2.0, h2=1.0, relevance=0.9))
    for idx in range(2):
        eid = f"evi_proximate_{idx:02d}"
        evidence.append(
            Evidence(
                id=eid,
                description=f"Proximate coup maneuver item {idx}",
                source_text=f"Proximate coup maneuver occurred in 1799 item {idx}.",
                evidence_type="empirical",
                approximate_date="1799",
            )
        )
        vectors.append(_ev_like(eid, h1=1.4, h2=1.0, relevance=0.8))
    extraction.evidence = evidence

    hypothesis_space = _make_hypothesis_space()
    hypothesis_space.research_question = "Why did the crisis culminate in the 1799 coup rather than reform?"
    hypothesis_space.hypotheses[0].description = (
        "A legitimacy vacuum across multiple institutions enabled the coup"
    )
    hypothesis_space.hypotheses[0].causal_mechanism = (
        "A power vacuum across institutions made a decisive coup coalition feasible"
    )

    testing = TestingResult(
        evidence_likelihoods=vectors,
        dependence_clusters=[
            EvidenceCluster(
                evidence_ids=["evi_background_00", "evi_background_01", "evi_background_02"],
                reason="Same background legitimacy-loss sub-narrative",
                dependence_strength=0.8,
            )
        ],
    )
    bayesian = run_bayesian_update(testing, ["h1", "h2"])
    absence = AbsenceResult(
        evaluations=[
            AbsenceEvaluation(
                hypothesis_id="h2",
                prediction_id="pred_h2_02",
                missing_evidence="No named conspirators with a concrete operational plan",
                reasoning="A direct conspiracy account should name agents and planning details",
                severity="damaging",
                would_be_extractable=True,
            )
        ]
    )
    synthesis = SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id="h1",
                status="strongly_supported",
                key_evidence_for=["evi_background_00"],
                key_evidence_against=[],
                reasoning="Many weak items favor the broad legitimacy-vacuum account.",
                steelman="The political order had lost legitimacy before the coup.",
                posterior_robustness="fragile",
            ),
            HypothesisVerdict(
                hypothesis_id="h2",
                status="supported",
                key_evidence_for=["evi_proximate_00"],
                key_evidence_against=["evi_background_00"],
                reasoning="Some coup maneuvering evidence exists, but support is low.",
                steelman="A coup requires agency and planning.",
                posterior_robustness="fragile",
            ),
        ],
        comparative_analysis="H1 dominates H2, but the winning frame is broad.",
        analytical_narrative="The result is useful only if caveated as broad, fragile comparative support.",
        limitations=["Single source", "Background-heavy evidence"],
        suggested_further_tests=["Find named conspirator correspondence"],
    )
    return ProcessTracingResult(
        extraction=extraction,
        hypothesis_space=hypothesis_space,
        testing=testing,
        absence=absence,
        bayesian=bayesian,
        synthesis=synthesis,
    )


# ── Mock dispatcher ────────────────────────────────────────────────


def _make_adequate_partition():
    from pt.schemas import PartitionAudit, RivalPairAudit
    return PartitionAudit(
        research_question_adequate=True,
        rival_pairs=[
            RivalPairAudit(
                h1_id="h1", h2_id="h2",
                overlap_concern=False, complementary_concern=False, absorptive_concern=False,
                discriminator_count=3, concern_detail="",
            )
        ],
        hypotheses_flagged=[],
        overall_quality="adequate",
        summary="Hypotheses make opposite predictions; no partition concerns.",
    )


def _mock_call_llm(prompt: str, response_model: type, *, task: str = "", trace_id: str = "", **kwargs):
    """Return deterministic data based on the response model type.

    mock-ok: LLM boundaries are mocked to keep the suite deterministic and fast.
    """
    model_name = response_model.__name__

    if model_name == "ExtractionResult":
        return _make_extraction()
    elif model_name == "HypothesisSpace":
        return _make_hypothesis_space()
    elif model_name in ("TestingResult", "TestingResponse"):
        return _make_testing()
    elif model_name == "AbsenceResult":
        return _make_absence()
    elif model_name == "SynthesisResult":
        return _make_synthesis()
    elif model_name == "PartitionAudit":
        return _make_adequate_partition()
    else:
        raise ValueError(f"Unexpected response_model in mock: {model_name}")


# ── Tests ──────────────────────────────────────────────────────────


class TestInputValidation:
    def test_rejects_short_text(self):
        with pytest.raises(ValueError, match="too short"):
            run_pipeline("This is too short.", verbose=False)

    def test_rejects_empty_text(self):
        with pytest.raises(ValueError, match="too short"):
            run_pipeline("", verbose=False)

    def test_accepts_300_word_text(self):
        """300 words should pass validation (LLM is mocked)."""
        text = " ".join(["word"] * 300)
        with patch("pt.pass_extract.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_hypothesize.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_partition.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_test.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_absence.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_synthesize.call_llm", side_effect=_mock_call_llm):
            result = run_pipeline(text, verbose=False)
            assert result.extraction.summary is not None


class TestPipelineOrchestration:
    """Verify the pipeline calls passes in correct order with correct data flow."""

    @pytest.fixture()
    def pipeline_result(self):
        text = " ".join(["substantive"] * 400)
        with patch("pt.pass_extract.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_hypothesize.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_partition.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_test.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_absence.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_synthesize.call_llm", side_effect=_mock_call_llm):
            return run_pipeline(text, verbose=False)

    def test_extraction_populated(self, pipeline_result):
        assert len(pipeline_result.extraction.evidence) == 4
        assert len(pipeline_result.extraction.actors) == 2

    def test_hypotheses_populated(self, pipeline_result):
        assert len(pipeline_result.hypothesis_space.hypotheses) == 2
        assert pipeline_result.hypothesis_space.research_question

    def test_testing_populated(self, pipeline_result):
        assert len(pipeline_result.testing.evidence_likelihoods) == 4
        for item in pipeline_result.testing.evidence_likelihoods:
            assert len(item.hypothesis_likelihoods) == 2

    def test_absence_populated(self, pipeline_result):
        assert len(pipeline_result.absence.evaluations) == 1

    def test_bayesian_posteriors_sum_to_one(self, pipeline_result):
        total = sum(p.final_posterior for p in pipeline_result.bayesian.posteriors)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_bayesian_ranking(self, pipeline_result):
        # With our test data, h1 should rank higher (more supporting evidence)
        assert pipeline_result.bayesian.ranking[0] == "h1"

    def test_synthesis_has_verdicts(self, pipeline_result):
        assert len(pipeline_result.synthesis.verdicts) == 2

    def test_not_refined_by_default(self, pipeline_result):
        assert pipeline_result.is_refined is False
        assert pipeline_result.refinement is None

    @pytest.mark.plans(3)
    def test_source_packet_pins_question_and_reaches_hypothesis_prompt(self):
        text = " ".join(["substantive"] * 400)
        packet = _make_source_packet()
        captured: dict[str, str] = {}

        def fake_hypothesize(prompt: str, response_model: type, *, task: str = "", trace_id: str = "", **kwargs):
            captured["prompt"] = prompt
            return _mock_call_llm(prompt, response_model, task=task, trace_id=trace_id, **kwargs)

        with patch("pt.pass_extract.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_hypothesize.call_llm", side_effect=fake_hypothesize), \
             patch("pt.pass_partition.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_test.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_absence.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_synthesize.call_llm", side_effect=_mock_call_llm):
            result = run_pipeline(
                text,
                verbose=False,
                source_packet=packet,
                source_packet_path="packet.json",
            )

        assert result.hypothesis_space.research_question == packet.research_question
        assert result.source_packet is not None
        assert result.source_packet.source_count == 3
        assert result.source_packet.high_priority_gap_count == 1
        assert result.source_packet.source_packet_path == "packet.json"
        assert result.source_coverage is not None
        assert result.source_coverage.source_count == 3
        assert "Source-packet contract" in captured["prompt"]
        assert "Private correspondence among conspirators" in captured["prompt"]
        assert "Do NOT treat the packet metadata as evidence" in captured["prompt"]

    @pytest.mark.plans(3)
    def test_source_packet_rejects_conflicting_research_question(self):
        with pytest.raises(ValueError, match="conflicts"):
            run_pipeline(
                " ".join(["substantive"] * 400),
                verbose=False,
                source_packet=_make_source_packet(),
                research_question="Why did a different outcome occur?",
            )


class TestBayesianMathDeterministic:
    """Verify Bayesian math produces expected posteriors from fixed test data."""

    def test_posteriors_from_fixed_testing(self):
        testing = _make_testing()
        result = run_bayesian_update(testing, ["h1", "h2"])

        h1 = next(p for p in result.posteriors if p.hypothesis_id == "h1")
        h2 = next(p for p in result.posteriors if p.hypothesis_id == "h2")

        # Two items favor h1 (debt, tax revolt), one favors h2 (elite plot), one
        # is uninformative — net, h1 comes out ahead.
        assert h1.final_posterior > h2.final_posterior
        assert h1.final_posterior + h2.final_posterior == pytest.approx(1.0, abs=0.01)

        # One update per evidence item, per hypothesis.
        assert len(h1.updates) == 4
        assert len(h2.updates) == 4

        assert result.ranking == ["h1", "h2"]


class TestReportConsistency:
    """Report display should match Bayesian updater semantics."""

    def test_dom_id_sanitizes_model_provided_ids(self):
        dom_id = _dom_id("detail", "h 1/evil#id")
        assert dom_id.startswith("detail-h-1-evil-id-")
        assert " " not in dom_id
        assert "/" not in dom_id
        assert "#" not in dom_id

    def test_report_strips_control_characters_from_model_text(self):
        result = _make_process_result()
        result.synthesis.analytical_narrative = "Narrative with bad\x00control text."
        result.extraction.evidence[0].description = "Evidence with bad\x01control text."
        result.extraction.evidence[0].source_text = "Quote with bad\x02control text."

        html = generate_report(result)

        assert "\x00" not in html
        assert "\x01" not in html
        assert "\x02" not in html
        assert "badcontrol text" in html

    def test_low_relevance_extreme_evidence_hidden_as_uninformative(self):
        extraction = _make_extraction()
        hypothesis_space = _make_hypothesis_space()
        # Extreme vector but relevance below the gate ⇒ forced uninformative (LR 1.0).
        testing = TestingResult(
            evidence_likelihoods=[
                _ev_like("evi_debt", h1=1.0, h2=0.001, relevance=0.39, dtype="smoking_gun"),
            ],
        )
        bayesian = run_bayesian_update(testing, ["h1", "h2"])
        result = ProcessTracingResult(
            extraction=extraction,
            hypothesis_space=hypothesis_space,
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=bayesian,
            synthesis=_make_synthesis(),
        )

        html = generate_report(result)

        # Both hypotheses' LR for the gated item is 1.0.
        assert bayesian.posteriors[0].updates[0].likelihood_ratio == pytest.approx(1.0)
        # Report counts evaluations per (hypothesis, evidence): 2 hyps × 1 item = 2 total.
        assert "0 informative / 2 total evaluations shown" in html
        assert "LR=1000.00" not in html

    def test_sensitivity_populated(self):
        testing = _make_testing()
        result = run_bayesian_update(testing, ["h1", "h2"])
        assert len(result.sensitivity) == 2
        for s in result.sensitivity:
            assert s.posterior_low <= s.baseline_posterior <= s.posterior_high

    def test_robustness_populated(self):
        testing = _make_testing()
        result = run_bayesian_update(testing, ["h1", "h2"])
        for p in result.posteriors:
            assert p.robustness in ("robust", "fragile", "moderate", "unknown")

    def test_diagnostic_summary_uses_pairwise_spread_after_caps(self):
        from scripts.audit_result_quality import _diagnostic_strength_stats

        extraction = _make_extraction()
        extraction.evidence = [
            Evidence(
                id="evi_decisive",
                description="A direct discriminator",
                source_text="A direct discriminator appeared.",
                evidence_type="empirical",
                approximate_date="1799",
            )
        ]
        testing = TestingResult(
            evidence_likelihoods=[
                _ev_like(
                    "evi_decisive",
                    h1=16.0,
                    h2=1.0,
                    relevance=1.0,
                    dtype="smoking_gun",
                )
            ]
        )
        result = ProcessTracingResult(
            extraction=extraction,
            hypothesis_space=_make_hypothesis_space(),
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=run_bayesian_update(testing, ["h1", "h2"]),
            synthesis=_make_synthesis(),
        )

        assert _diagnostic_strength_summary(result)["decisive"] == 1
        assert _diagnostic_strength_stats(result)["decisive_items"] == 1

    def test_network_uses_temporal_dag_layout(self):
        result = _make_process_result()
        nodes, edges = _build_vis_data(result)
        by_id = {node["id"]: node for node in nodes}

        assert by_id["evt_crisis"]["level"] < by_id["evt_coup"]["level"]
        assert by_id["evi_debt"]["level"] < by_id["evi_elite_plot"]["level"]
        assert by_id["h1"]["level"] > by_id["evi_elite_plot"]["level"]
        assert by_id["evt_crisis"]["x"] < by_id["evt_coup"]["x"] < by_id["h1"]["x"]
        assert by_id["h1"]["fixed"] == {"x": True, "y": True}
        causal_edges = [edge for edge in edges if edge.get("group") == "causal"]
        top_driver_edges = [edge for edge in edges if edge.get("group") == "top_driver_link"]
        background_driver_edges = [edge for edge in edges if edge.get("group") == "background_driver_link"]
        additional_evidence_edges = [edge for edge in edges if edge.get("group") == "evidence_link"]
        assert causal_edges
        assert top_driver_edges
        assert all(edge.get("hidden") is False for edge in top_driver_edges)
        assert all(edge.get("hidden") is True for edge in background_driver_edges)
        assert all(edge.get("hidden") is True for edge in additional_evidence_edges)
        assert all(by_id[edge["from"]]["level"] <= by_id[edge["to"]]["level"] for edge in causal_edges)

        html = generate_report(result)
        assert "randomSeed: 42" in html
        assert "physics: {\n      enabled: false" in html
        assert 'id="toggle-top_driver_link" checked' in html
        assert 'id="toggle-background_driver_link"' in html
        assert 'id="toggle-temporal_conflict"' in html
        assert "function focusNetwork()" in html
        assert "How to Read This Network" in html
        assert "mixed-methods research ontology" in html
        assert "within-case causal inference" in html
        assert "connect to quantitative designs" in html
        assert "Evidence is not itself the same as a cause" in html
        assert "outside the neutral LR band [0.67, 1.50]" in html
        assert "not as a claim that" in html

    def test_network_preserves_backward_edges_as_temporal_conflicts(self):
        payload = _make_process_result().model_dump()
        payload["extraction"]["causal_edges"].append({
            "source_id": "evt_coup",
            "target_id": "evt_crisis",
            "relationship": "impossible backward relation",
        })
        result = ProcessTracingResult.model_validate(payload)
        nodes, edges = _build_vis_data(result)
        by_id = {node["id"]: node for node in nodes}

        conflict_edges = [edge for edge in edges if edge.get("group") == "temporal_conflict"]

        assert conflict_edges
        assert conflict_edges[0]["from"] == "evt_coup"
        assert conflict_edges[0]["to"] == "evt_crisis"
        assert conflict_edges[0]["hidden"] is True
        assert conflict_edges[0]["dashes"] is True
        assert all(
            by_id[edge["from"]]["level"] <= by_id[edge["to"]]["level"]
            for edge in edges
            if edge.get("group") == "causal"
        )

    def test_temporal_position_parses_brumaire_dates(self):
        assert _temporal_position(
            "17 Brumaire Year VIII (8 November 1799)"
        ) < _temporal_position("18 Brumaire Year VIII (9 November 1799)")
        assert _temporal_position(
            "18 Brumaire Year VIII (9 November 1799)"
        ) < _temporal_position("19 Brumaire Year VIII (10 November 1799)")
        assert _layout_temporal_position(
            "1799-11",
            "Weeks before 18 Brumaire: conspirators met to plan the coup.",
        ) < _temporal_position("1799-11-09")

    def test_script_terminator_in_graph_data_is_escaped(self):
        result = _make_process_result()
        payload = "</script><script>x</script>"
        result.extraction.events[0].description = payload

        html = generate_report(result)

        assert payload not in html
        assert "<\\/script><script>x<\\/script>" in html

    def test_output_quality_audit_surfaces_adversarial_caveats(self):
        from scripts.audit_result_quality import audit_result

        result = _make_audit_stress_result()
        html = generate_report(result)
        normalized = " ".join(html.lower().split())

        for phrase in [
            "output quality audit",
            "high support, fragile",
            "temporal evidence mix",
            "temporal causal timeline",
            "background top-driver",
            "effective evidence",
            "raw counts",
            "network coverage",
            "not discarded",
            "academic phd review",
            "source material known to the grader",
            "given-source recommendations by pipeline output",
            "phd-level critique given accepted sources",
            "claim-scope caveat",
            "recommendations by pipeline output",
            "optimality gate",
            "evidence triage",
            "proceed until optimal",
            "verdict calibration",
            "secondary mechanism",
            "broad winning hypothesis",
            "source-scope absence",
            "top driver edge",
        ]:
            assert phrase in normalized

        audit = audit_result(result, html, focal_year_override=1799)
        assert audit["grade"] == "C"
        assert audit["score"] == 76
        assert audit["base_score"] >= 90
        assert audit["academic_cap"] == 76
        assert audit["academic_caps"]
        assert audit["priority_recommendations"]
        assert audit["optimality"]["status"] == "not_optimal"
        assert audit["optimality"]["next_iteration_mode"] == "design_stronger_conditional_tests"
        assert audit["optimality"]["blocked_by_external_evidence"] is True
        assert audit["optimality"]["acceptance_criteria"]
        assert audit["categories"]["report_usability_and_safety"]["top_graph_connected"] is True

    @pytest.mark.plans(3)
    def test_source_packet_is_visible_in_report_and_audit(self):
        from scripts.audit_result_quality import audit_result

        result = _make_process_result()
        packet = _make_source_packet()
        result.source_packet = packet.to_summary("packet.json")
        result.extraction.evidence[0].source_text = "Source A says the fiscal crisis mattered."
        result.extraction.evidence[1].source_text = "Source B says proceedings were contested."
        result.source_coverage = build_source_coverage(
            packet,
            "Source A says one thing. Source B says another thing. Source C says a third thing.",
            result.extraction,
        )

        html = generate_report(result)
        normalized = " ".join(html.lower().split())
        audit = audit_result(result, html, focal_year_override=1799)

        assert "source material known to the grader" in normalized
        assert "packet source coverage" in normalized
        assert "packet metadata is not itself evidence" in normalized
        assert "given-source recommendations by pipeline output" in normalized
        assert "claim-scope caveat" in normalized
        assert "not themselves criticisms of whether the analysis is coherent given the supplied sources" in normalized
        assert "input corpus and source base" not in normalized
        assert "private correspondence among conspirators" in normalized
        assert "source gap dispositions" in normalized
        assert "partially_mitigated" in normalized
        assert audit["categories"]["source_scope_and_absence"]["source_packet_present"] is True
        assert audit["source_material_context"]["has_source_packet"] is True
        assert audit["source_material_context"]["source_count"] == 3
        assert audit["source_material_context"]["unresolved_high_priority_gap_count"] == 1
        assert audit["source_material_context"]["source_gap_dispositions"][0]["status"] == "partially_mitigated"
        assert audit["source_material_context"]["accepted_sources"]
        assert audit["conditional_grade"]
        assert audit["claim_scope_grade"]
        assert any(
            "packet-source coverage is incomplete" in cap["reason"].lower()
            for cap in audit["conditional_caps"]
        )
        assert audit["claim_scope_caps"]
        assert audit["categories"]["source_scope_and_absence"]["source_coverage_present"] is True
        assert audit["categories"]["source_scope_and_absence"]["source_count"] == 3
        assert audit["categories"]["source_scope_and_absence"]["sources_with_evidence"] == 2
        assert audit["categories"]["source_scope_and_absence"]["high_priority_gap_count"] == 1
        assert audit["categories"]["source_scope_and_absence"]["unresolved_high_priority_gap_count"] == 1
        assert any(
            "conditional on accepted sources" in cap["reason"].lower()
            for cap in audit["claim_scope_caps"]
        )

    @pytest.mark.plans(3)
    def test_source_packet_context_reaches_extraction_pass(self):
        captured_contexts: list[str | None] = []

        def _capture_extract(text, **kwargs):
            captured_contexts.append(kwargs.get("source_packet_context"))
            extraction = _make_extraction()
            extraction.evidence[0].source_text = "Source A says the fiscal crisis mattered."
            extraction.evidence[1].source_text = "Source C says institutional design constrained outcomes."
            return extraction

        text = " ".join(["Source A records one trace. Source C records another trace."] * 80)

        with (
            patch("pt.pipeline.run_extract", side_effect=_capture_extract),
            patch("pt.pipeline.run_hypothesize", return_value=_make_hypothesis_space()),
            patch("pt.pipeline.run_partition", return_value=_make_adequate_partition()),
            patch(
                "pt.pipeline._run_passes_3_plus",
                return_value=(
                    _make_testing(),
                    _make_absence(),
                    run_bayesian_update(_make_testing(), ["h1", "h2"]),
                    _make_synthesis(),
                    None,  # diagnostic_matrix (optional)
                ),
            ),
        ):
            result = run_pipeline(
                text,
                source_packet=_make_source_packet(),
                verbose=False,
                trace_id="source-packet-extraction-wiring",
            )

        assert captured_contexts
        assert captured_contexts[0] is not None
        assert "Source C" in captured_contexts[0]
        assert "Critical historiography" in captured_contexts[0]
        assert result.source_coverage is not None
        assert result.source_coverage.sources_with_evidence == 2

    @pytest.mark.plans(3)
    def test_pipeline_result_can_refresh_source_design_state(self):
        packet = _make_source_packet()
        result = _make_process_result()
        result.source_packet = packet.to_summary("packet.json")
        state = build_source_design_state(result, source_packet=packet, iteration=1)

        refreshed = state.refresh_from_result(result)

        assert refreshed.iteration == 2
        assert refreshed.limitations == result.source_packet.limitations
        assert refreshed.source_gap_dispositions[0].status == "partially_mitigated"
        assert refreshed.proposed_next_steps == result.synthesis.suggested_further_tests

    def test_network_keeps_weak_top_driver_edges_visible(self):
        result = _make_audit_stress_result()
        top_id = result.bayesian.ranking[0]
        top = next(p for p in result.bayesian.posteriors if p.hypothesis_id == top_id)

        nodes, edges = _build_vis_data(result)
        node_ids = {node["id"] for node in nodes}
        top_driver_edges = [
            edge for edge in edges
            if edge.get("to") == top_id and edge.get("from") in set(top.top_drivers)
        ]

        assert top_id in node_ids
        assert top_driver_edges
        assert any("top driver edge" in edge.get("title", "") for edge in top_driver_edges)

    @pytest.mark.plans(3)
    def test_critic_section_renders_confirmed_links_separately(self):
        """confirmed_link findings appear in Structural Anchors subsection, not defect table."""
        result = _make_process_result()
        result.critic = CriticResult(
            findings=[
                CriticFinding(
                    finding_type="confirmed_link",
                    target="evi_debt->evi_tax_revolt",
                    target_type="causal_edge",
                    severity="low",
                    reasoning="Both items converge on the same fiscal mechanism.",
                    recommendation="No action needed; this link is well-supported.",
                ),
                CriticFinding(
                    finding_type="void_link",
                    target="evi_elite_plot->evi_debt",
                    target_type="causal_edge",
                    severity="medium",
                    reasoning="No evidence item's justification mentions this mechanism.",
                    recommendation="Collect evidence of the causal pathway from elite action to debt.",
                ),
            ],
            summary="One well-supported link; one void link. No re-elicitation needed.",
        )
        html = generate_report(result)
        assert "Structural Anchors" in html, (
            "confirmed_link findings must appear in a separate Structural Anchors section"
        )
        # void_link defect must appear in the main critic table
        assert "void_link" in html or "void link" in html.lower(), (
            "void_link finding must appear in the main critic defects table"
        )

    @pytest.mark.plans(3)
    def test_absence_table_renders_acquire_from_column(self):
        """Absence table includes Acquire from column when expected_source_genre is set."""
        result = _make_process_result()
        result.absence = AbsenceResult(
            evaluations=[
                AbsenceEvaluation(
                    hypothesis_id="h1",
                    prediction_id="pred_h1_01",
                    missing_evidence="Correspondence showing fiscal negotiations",
                    reasoning="Private correspondence would document negotiating positions.",
                    severity="damaging",
                    would_be_extractable=True,
                    expected_source_genre="primary_document",
                    expected_source_location="Archives nationales, Série AF III (Executive papers)",
                ),
            ]
        )
        html = generate_report(result)
        assert "Acquire from" in html, (
            "Absence table must include Acquire from column when expected_source_genre is set"
        )
        assert "primary_document" in html or "primary document" in html.lower(), (
            "Absence table must display the expected_source_genre value"
        )


class TestFromResultProvenance:
    def test_from_result_rejects_missing_source_hash(self):
        with pytest.raises(ValueError, match="no source_text_sha256"):
            run_pipeline(
                "new text",
                from_result=_make_process_result(),
                verbose=False,
            )

    def test_from_result_rejects_mismatched_source_hash(self):
        with pytest.raises(ValueError, match="does not match"):
            run_pipeline(
                "new text",
                from_result=_make_process_result(_source_text_sha256("old text")),
                verbose=False,
            )

    @pytest.mark.plans(3)
    def test_from_result_rejects_fresh_source_packet(self):
        text = " ".join(["substantive"] * 400)
        with pytest.raises(ValueError, match="cannot be combined"):
            run_pipeline(
                text,
                from_result=_make_process_result(_source_text_sha256(text)),
                source_packet=_make_source_packet(),
                verbose=False,
            )


class TestVectorCompleteness:
    """run_test must fail loud on incomplete / malformed likelihood matrices."""

    def _run_with(self, testing):
        from pt import pass_test
        with patch.object(pass_test, "call_llm", side_effect=lambda *a, **k: testing):
            return pass_test.run_test(_make_extraction(), _make_hypothesis_space())

    def test_rejects_missing_hypothesis_in_vector(self):
        # One item's vector covers only h1, not {h1, h2}.
        good = _make_testing()
        good.evidence_likelihoods[0].hypothesis_likelihoods = [
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=1.0, diagnostic_type="hoop")
        ]
        with pytest.raises(ValueError, match="expected exactly"):
            self._run_with(good)

    def test_rejects_missing_evidence_item(self):
        good = _make_testing()
        good.evidence_likelihoods = good.evidence_likelihoods[:3]  # drop one of 4
        with pytest.raises(ValueError, match="coverage mismatch"):
            self._run_with(good)

    def test_accepts_complete_matrix(self):
        result = self._run_with(_make_testing())
        assert len(result.evidence_likelihoods) == 4

    def test_llm_schema_enumerates_exact_evidence_and_hypothesis_ids(self):
        from pt import pass_test
        captured = {}

        def fake_call_llm(prompt, response_model, **kwargs):
            captured["schema"] = response_model.model_json_schema()
            return _make_testing()

        with patch.object(pass_test, "call_llm", side_effect=fake_call_llm):
            pass_test.run_test(_make_extraction(), _make_hypothesis_space())

        schema_json = str(captured["schema"])
        assert "'enum': ['evi_debt', 'evi_tax_revolt', 'evi_elite_plot', 'evi_historian_claim']" in schema_json
        assert "'enum': ['h1', 'h2']" in schema_json

    def test_rejects_cluster_with_unknown_evidence(self):
        from pt.schemas import EvidenceCluster
        t = _make_testing()
        t.dependence_clusters = [EvidenceCluster(evidence_ids=["evi_debt", "evi_ghost"], reason="x")]
        with pytest.raises(ValueError, match="unknown evidence"):
            self._run_with(t)

    def test_rejects_cluster_with_one_member(self):
        from pt.schemas import EvidenceCluster
        t = _make_testing()
        t.dependence_clusters = [EvidenceCluster(evidence_ids=["evi_debt", "evi_debt"], reason="x")]
        with pytest.raises(ValueError, match=">=2 distinct"):
            self._run_with(t)

    def test_rejects_overlapping_clusters(self):
        from pt.schemas import EvidenceCluster
        t = _make_testing()
        t.dependence_clusters = [
            EvidenceCluster(evidence_ids=["evi_debt", "evi_tax_revolt"], reason="x"),
            EvidenceCluster(evidence_ids=["evi_tax_revolt", "evi_elite_plot"], reason="y"),
        ]
        with pytest.raises(ValueError, match="multiple dependence clusters"):
            self._run_with(t)

    def test_repairs_overlapping_clusters_once_with_validation_feedback(self):
        from pt import pass_test
        from pt.schemas import EvidenceCluster

        bad = _make_testing()
        bad.dependence_clusters = [
            EvidenceCluster(evidence_ids=["evi_debt", "evi_tax_revolt"], reason="x"),
            EvidenceCluster(evidence_ids=["evi_tax_revolt", "evi_elite_plot"], reason="y"),
        ]
        repaired = _make_testing()
        repaired.dependence_clusters = [
            EvidenceCluster(evidence_ids=["evi_debt", "evi_tax_revolt"], reason="x"),
            EvidenceCluster(evidence_ids=["evi_elite_plot", "evi_historian_claim"], reason="y"),
        ]
        prompts: list[str] = []

        def fake_call_llm(prompt, response_model, **kwargs):
            prompts.append(prompt)
            return bad if len(prompts) == 1 else repaired

        with patch.object(pass_test, "call_llm", side_effect=fake_call_llm):
            result = pass_test.run_test(_make_extraction(), _make_hypothesis_space())

        assert len(prompts) == 2
        assert "Validation repair required" in prompts[1]
        assert "evidence in multiple dependence clusters" in prompts[1]
        assert len(result.dependence_clusters) == 2

    def test_accepts_valid_cluster(self):
        from pt.schemas import EvidenceCluster
        t = _make_testing()
        t.dependence_clusters = [EvidenceCluster(evidence_ids=["evi_debt", "evi_tax_revolt"], reason="x")]
        result = self._run_with(t)
        assert len(result.dependence_clusters) == 1


class TestExecutiveSummary:
    """Slice 4 + truth-in-labeling: the headline surfaces a support interval and
    stability flags, framed as comparative support (not absolute probability)."""

    def _result(self):
        testing = _make_testing()
        return ProcessTracingResult(
            extraction=_make_extraction(),
            hypothesis_space=_make_hypothesis_space(),
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=run_bayesian_update(testing, ["h1", "h2"]),
            synthesis=_make_synthesis(),
        )

    def test_no_absolute_probability_overclaim(self):
        html = generate_report(self._result())
        assert "Posterior probability after Bayesian updating" not in html
        assert "Support:" in html

    def test_comparative_support_caveat(self):
        html = " ".join(generate_report(self._result()).split()).lower()
        assert "comparative support" in html
        assert "not absolute probabilities of truth" in html

    def test_support_interval_and_stability_surfaced(self):
        html = generate_report(self._result())
        assert "range " in html  # support interval badge
        assert ("robust to prior" in html) or ("prior-sensitive" in html)

    def test_overconfidence_banner_on_degenerate_fragile_posterior(self):
        # Many weakly-pro-h1 items -> near-1.0 support, fragile -> warning banner.
        items = [_ev_like(f"e{i}", h1=2.0, h2=1.0, relevance=0.9) for i in range(20)]
        testing = TestingResult(evidence_likelihoods=items)
        result = ProcessTracingResult(
            extraction=_make_extraction(),
            hypothesis_space=_make_hypothesis_space(),
            testing=testing,
            absence=AbsenceResult(evaluations=[]),
            bayesian=run_bayesian_update(testing, ["h1", "h2"]),
            synthesis=_make_synthesis(),
        )
        assert result.bayesian.posteriors[0].final_posterior > 0.99
        assert "Likely overconfident" in generate_report(result)

    def test_no_overconfidence_banner_on_normal_result(self):
        assert "Likely overconfident" not in generate_report(self._result())


class TestAuditSynthesisCalibration:
    """Slice 8: overclaim language check and verdict calibration threshold tests."""

    def _fragile_result(self, narrative: str) -> ProcessTracingResult:
        """Build a result where h1 wins but robustness=fragile (many weak items)."""
        items = [_ev_like(f"e{i}", h1=2.0, h2=1.0, relevance=0.9) for i in range(12)]
        testing = TestingResult(evidence_likelihoods=items)
        bayesian = run_bayesian_update(testing, ["h1", "h2"])
        synthesis = SynthesisResult(
            verdicts=[
                HypothesisVerdict(
                    hypothesis_id="h1", status="supported",
                    key_evidence_for=["e0"], key_evidence_against=[],
                    reasoning="Many weak items favor h1.",
                    steelman="Pattern is clear despite fragility.",
                    posterior_robustness="fragile",
                ),
                HypothesisVerdict(
                    hypothesis_id="h2", status="weakened",
                    key_evidence_for=[], key_evidence_against=["e0"],
                    reasoning="H2 is outweighed.", steelman="Has some merit.",
                    posterior_robustness="fragile",
                ),
            ],
            comparative_analysis="H1 leads via many weak items.",
            analytical_narrative=narrative,
            limitations=["Fragile result"], suggested_further_tests=[],
        )
        return ProcessTracingResult(
            extraction=_make_extraction(), hypothesis_space=_make_hypothesis_space(),
            testing=testing, absence=AbsenceResult(evaluations=[]),
            bayesian=bayesian, synthesis=synthesis,
        )

    def test_overclaim_flagged_when_winner_fragile_and_narrative_has_certainty_word(self):
        from scripts.audit_result_quality import _synthesis_overclaim_check
        result = self._fragile_result(
            "The evidence conclusively establishes that the fiscal crisis caused the revolution."
        )
        top = result.bayesian.posteriors[0]
        assert top.robustness == "fragile", f"Expected fragile winner; got robustness={top.robustness}"
        issues = _synthesis_overclaim_check(result)
        assert issues, "Expected overclaim issues when narrative contains 'conclusively' and winner is fragile"
        assert any("conclusive" in issue for issue in issues)

    def test_overclaim_not_flagged_when_winner_not_fragile(self):
        from scripts.audit_result_quality import _synthesis_overclaim_check
        # DECISIVE_COUNT_FOR_ROBUST=3: need ≥3 items with |log(LR)|>1.6 for "robust".
        # Use 4 decisive items (h1=5.0, h2=0.1 each → LR≈7, |log|≈1.96 > 1.6).
        testing = TestingResult(
            evidence_likelihoods=[
                _ev_like(f"e{i}", h1=5.0, h2=0.1, relevance=0.95, dtype="smoking_gun")
                for i in range(4)
            ]
        )
        bayesian = run_bayesian_update(testing, ["h1", "h2"])
        synthesis = SynthesisResult(
            verdicts=[
                HypothesisVerdict(
                    hypothesis_id="h1", status="supported",
                    key_evidence_for=["e0"], key_evidence_against=[],
                    reasoning="Decisive item.", steelman="Clear.",
                    posterior_robustness="robust",
                ),
                HypothesisVerdict(
                    hypothesis_id="h2", status="weakened",
                    key_evidence_for=[], key_evidence_against=["e0"],
                    reasoning="Outweighed.", steelman="Has merit.",
                    posterior_robustness="robust",
                ),
            ],
            comparative_analysis="H1 leads decisively.",
            analytical_narrative="This conclusively proves that fiscal collapse caused the revolution.",
            limitations=[], suggested_further_tests=[],
        )
        result = ProcessTracingResult(
            extraction=_make_extraction(), hypothesis_space=_make_hypothesis_space(),
            testing=testing, absence=AbsenceResult(evaluations=[]),
            bayesian=bayesian, synthesis=synthesis,
        )
        top = result.bayesian.posteriors[0]
        assert top.robustness != "fragile", f"Expected non-fragile winner; got robustness={top.robustness}"
        issues = _synthesis_overclaim_check(result)
        assert not issues, f"Should not flag overclaim language when winner is {top.robustness} (not fragile)"

    def test_overclaim_not_flagged_when_narrative_is_hedged(self):
        from scripts.audit_result_quality import _synthesis_overclaim_check
        result = self._fragile_result(
            "Comparative support suggests the fiscal hypothesis is the stronger explanation, "
            "but the result is fragile and sensitive to rival interpretations."
        )
        issues = _synthesis_overclaim_check(result)
        assert not issues, "Should not flag hedged language even when winner is fragile"

    def test_verdict_calibration_flags_strongly_supported_below_strong_floor(self):
        from scripts.audit_result_quality import _verdict_calibration_issues
        # Two items weakly favoring h2 → h1 posterior ~0.30 (above 0.10, below 0.50 threshold).
        testing = TestingResult(
            evidence_likelihoods=[
                _ev_like("e0", h1=0.3, h2=0.7, relevance=0.9),
                _ev_like("e1", h1=0.3, h2=0.7, relevance=0.9),
            ]
        )
        bayesian = run_bayesian_update(testing, ["h1", "h2"])
        h1_post = next(p.final_posterior for p in bayesian.posteriors if p.hypothesis_id == "h1")
        assert 0.10 <= h1_post < 0.50, f"Expected h1 posterior in (0.10, 0.50), got {h1_post:.3f}"
        synthesis = SynthesisResult(
            verdicts=[
                HypothesisVerdict(
                    hypothesis_id="h1", status="strongly_supported",  # overclaim
                    key_evidence_for=[], key_evidence_against=["e0"],
                    reasoning="Overclaiming verdict.", steelman="Has some merit.",
                    posterior_robustness="fragile",
                ),
                HypothesisVerdict(
                    hypothesis_id="h2", status="supported",
                    key_evidence_for=["e0"], key_evidence_against=[],
                    reasoning="Supported.", steelman="Clear.", posterior_robustness="fragile",
                ),
            ],
            comparative_analysis="H2 leads.", analytical_narrative="H1 is strongly supported.",
            limitations=[], suggested_further_tests=[],
        )
        result = ProcessTracingResult(
            extraction=_make_extraction(), hypothesis_space=_make_hypothesis_space(),
            testing=testing, absence=AbsenceResult(evaluations=[]),
            bayesian=bayesian, synthesis=synthesis,
        )
        issues = _verdict_calibration_issues(result)
        assert any("strongly_supported" in issue for issue in issues), (
            f"Expected strongly_supported calibration issue for h1 with posterior {h1_post:.3f}; got: {issues}"
        )

    def test_audit_overclaim_deducts_from_comparative_support_score(self):
        from scripts.audit_result_quality import audit_result
        result = self._fragile_result(
            "The evidence conclusively proves that the fiscal crisis was the decisive cause."
        )
        html = generate_report(result)
        audit = audit_result(result, html)
        cs = audit["categories"]["comparative_support_discipline"]
        assert cs["overclaim_issues"], "Expected overclaim_issues in audit output when narrative has certainty word + fragile winner"
        assert cs["points"] < 15, "Overclaim deduction should reduce comparative support discipline score"
        assert cs["recommendations"], "Overclaim should produce a recommendation"


class TestPartitionAuditReport:
    """Partition audit section visibility in HTML report and audit score effects."""

    def _needs_review_partition(self):
        from pt.schemas import PartitionAudit, RivalPairAudit
        return PartitionAudit(
            research_question_adequate=True,
            rival_pairs=[
                RivalPairAudit(
                    h1_id="h1", h2_id="h2",
                    overlap_concern=True, complementary_concern=False, absorptive_concern=False,
                    discriminator_count=0,
                    concern_detail="Both hypotheses predict the same evidence patterns.",
                )
            ],
            hypotheses_flagged=["h2"],
            overall_quality="needs_review",
            summary="H1 and H2 are not genuine rivals.",
        )

    def _result_with_partition(self, audit):
        result = _make_process_result()
        result.partition_audit = audit
        return result

    def test_partition_section_present_when_audit_exists(self):
        """Report contains partition audit card when partition_audit is set."""
        result = self._result_with_partition(_make_adequate_partition())
        html = generate_report(result)
        assert "hypothesis partition audit" in html.lower()

    def test_partition_section_absent_when_no_audit(self):
        """Report does not contain partition card when partition_audit is None."""
        result = _make_process_result()
        assert result.partition_audit is None
        html = generate_report(result)
        # Card heading is absent; '#partitionBody' appears in expand-all JS list regardless
        assert "hypothesis partition audit" not in html.lower()

    def test_partition_adequate_shows_success_badge(self):
        """Adequate partition shows green badge, no alert."""
        result = self._result_with_partition(_make_adequate_partition())
        html = generate_report(result)
        assert "Adequate" in html
        assert "Partition concerns detected" not in html

    def test_partition_needs_review_shows_danger_badge(self):
        """needs_review partition shows danger badge and alert text."""
        result = self._result_with_partition(self._needs_review_partition())
        html = generate_report(result)
        normalized = " ".join(html.lower().split())
        assert "needs review" in normalized
        assert "partition concerns detected" in normalized
        assert "--partition-review" in html

    def test_pair_table_renders_concern_badges(self):
        """Overlap concern renders an Overlap badge in the pair table."""
        result = self._result_with_partition(self._needs_review_partition())
        html = generate_report(result)
        assert "Overlap" in html

    def test_audit_score_deducts_for_needs_review_not_visible(self):
        """Audit deducts from hypothesis_discrimination when partition needs_review and not in report."""
        from scripts.audit_result_quality import audit_result

        # Build a result with needs_review partition but NO report — simulate no partition section
        result = self._result_with_partition(self._needs_review_partition())
        # Generate report normally — partition section IS in report
        html_with_partition = generate_report(result)
        audit_with = audit_result(result, html_with_partition)

        # Also test with empty HTML (partition not visible)
        audit_without = audit_result(result, "<html>no partition here</html>")

        disc_with = audit_with["categories"]["hypothesis_discrimination"]
        disc_without = audit_without["categories"]["hypothesis_discrimination"]

        assert disc_with["partition_needs_review"] is True
        assert disc_without["partition_needs_review"] is True
        assert disc_with["partition_visible"] is True, "Report has partition section — should be visible"
        assert disc_without["partition_visible"] is False, "Empty HTML — partition not visible"
        assert disc_without["points"] < disc_with["points"], (
            "Deduction should apply when partition needs_review but not visible in report"
        )

    def test_audit_score_no_deduction_for_adequate_partition(self):
        """Audit does not deduct when partition quality is adequate."""
        from scripts.audit_result_quality import audit_result

        result = self._result_with_partition(_make_adequate_partition())
        html = generate_report(result)
        audit = audit_result(result, html)

        disc = audit["categories"]["hypothesis_discrimination"]
        assert disc["partition_needs_review"] is False
        assert disc["recommendations"] == [] or not any(
            "partition" in r.lower() for r in disc["recommendations"]
        )


class TestPartitionReviewCheckpoint:
    """Verify pipeline partition_review flag triggers review_fn at the right time.

    mock-ok: LLM boundaries are mocked; only the partition review callback logic is tested.
    """

    def _run_with_partition_audit(self, partition_audit, partition_review: bool, review_fn=None):
        """Run the pipeline with a mocked partition audit and optional review fn."""
        text = " ".join(["substantive"] * 400)
        with patch("pt.pass_extract.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_hypothesize.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_partition.call_llm", return_value=partition_audit), \
             patch("pt.pass_test.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_absence.call_llm", side_effect=_mock_call_llm), \
             patch("pt.pass_synthesize.call_llm", side_effect=_mock_call_llm):
            return run_pipeline(
                text,
                verbose=False,
                partition_review=partition_review,
                partition_review_fn=review_fn,
            )

    def test_review_fn_called_when_needs_review_and_flag_set(self):
        """partition_review_fn is invoked when quality=needs_review and partition_review=True."""
        from pt.schemas import PartitionAudit, RivalPairAudit
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
            summary="Overlap found.",
        )

        called_with = {}

        def fake_review(hypothesis_space, partition_audit, output_dir):
            called_with["called"] = True
            called_with["quality"] = partition_audit.overall_quality
            return hypothesis_space  # pass through unchanged

        self._run_with_partition_audit(audit, partition_review=True, review_fn=fake_review)

        assert called_with.get("called"), "partition_review_fn must be called when quality=needs_review"
        assert called_with["quality"] == "needs_review"

    def test_review_fn_not_called_when_adequate(self):
        """partition_review_fn is NOT called when quality=adequate, even with flag set."""
        called = []

        def fake_review(hypothesis_space, partition_audit, output_dir):
            called.append(True)
            return hypothesis_space

        self._run_with_partition_audit(
            _make_adequate_partition(), partition_review=True, review_fn=fake_review
        )

        assert called == [], "partition_review_fn must NOT be called when quality=adequate"

    def test_review_fn_not_called_when_flag_false(self):
        """partition_review_fn is NOT called when partition_review=False, even if needs_review."""
        from pt.schemas import PartitionAudit, RivalPairAudit
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
            summary="Overlap found.",
        )
        called = []

        def fake_review(hypothesis_space, partition_audit, output_dir):
            called.append(True)
            return hypothesis_space

        self._run_with_partition_audit(audit, partition_review=False, review_fn=fake_review)

        assert called == [], "partition_review_fn must NOT be called when partition_review=False"
