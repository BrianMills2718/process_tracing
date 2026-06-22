"""Tests for Pass 5 refinement prompt contracts.

mock-ok: These tests replace the provider call to inspect the prompt contract
that guards live refinement behavior.
"""

from __future__ import annotations

import pytest

from pt.pass_refine import run_refine
from pt.schemas import (
    AbsenceResult,
    Actor,
    BayesianResult,
    Evidence,
    Event,
    EvidenceUpdate,
    ExtractionResult,
    Hypothesis,
    HypothesisPosterior,
    HypothesisSpace,
    HypothesisVerdict,
    Mechanism,
    Prediction,
    RefinementResult,
    SynthesisResult,
)


def _extraction() -> ExtractionResult:
    return ExtractionResult(
        summary="A coup unfolds through planning and coercion.",
        actors=[Actor(id="actor_bonaparte", name="Bonaparte", description="general")],
        events=[Event(id="evt_coup", description="Coup occurs", date="1799")],
        mechanisms=[Mechanism(id="mech_coercion", description="Military coercion")],
        evidence=[
            Evidence(
                id="evi_resistance",
                description="Deputies resisted before troops intervened",
                source_text="Deputies resisted before troops intervened.",
                evidence_type="empirical",
                approximate_date="1799",
            )
        ],
    )


def _space() -> HypothesisSpace:
    return HypothesisSpace(
        research_question="Why did the coup produce the Consulate?",
        hypotheses=[
            Hypothesis(
                id="h1",
                description="Military coercion was decisive",
                source="generated",
                theoretical_basis="Agency account",
                causal_mechanism="Troops coerced deputies",
                observable_predictions=[
                    Prediction(id="pred_h1_01", description="Deputies resist before coercion")
                ],
            )
        ],
    )


def _bayesian() -> BayesianResult:
    return BayesianResult(
        posteriors=[
            HypothesisPosterior(
                hypothesis_id="h1",
                prior=1.0,
                updates=[
                    EvidenceUpdate(
                        evidence_id="evi_resistance",
                        likelihood_ratio=4.0,
                        prior=1.0,
                        posterior=1.0,
                    )
                ],
                final_posterior=1.0,
                robustness="robust",
                top_drivers=["evi_resistance"],
            )
        ],
        ranking=["h1"],
    )


def _synthesis() -> SynthesisResult:
    return SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id="h1",
                status="supported",
                key_evidence_for=["evi_resistance"],
                key_evidence_against=[],
                reasoning="Evidence favors coercion.",
                steelman="Coercion explains the endpoint.",
            )
        ],
        comparative_analysis="H1 dominates.",
        analytical_narrative="The sequence favors coercion.",
        limitations=[],
        suggested_further_tests=[],
    )


@pytest.mark.plans(3)
def test_refine_prompt_forbids_hypothesis_ids_as_causal_edge_endpoints(monkeypatch):
    captured: dict[str, str] = {}

    def fake_call_llm(prompt: str, response_model: type, **kwargs):
        captured["prompt"] = prompt
        return RefinementResult(analyst_notes="No changes.")

    monkeypatch.setattr("pt.pass_refine.call_llm", fake_call_llm)

    run_refine(
        "Deputies resisted before troops intervened.",
        _extraction(),
        _space(),
        _bayesian(),
        AbsenceResult(evaluations=[]),
        _synthesis(),
        trace_id="trace-refine-test",
    )

    prompt = captured["prompt"]
    assert "Valid Existing Causal-Edge Endpoints" in prompt
    assert "actor_bonaparte (actor)" in prompt
    assert "evt_coup (event, date=1799)" in prompt
    assert "mech_coercion (mechanism)" in prompt
    assert "evi_resistance (evidence)" in prompt
    assert "Do NOT use hypothesis IDs" in prompt
    assert "evi_ref_01 -> h3" in prompt
    assert "diagnostic evidence-hypothesis relationships" in prompt
