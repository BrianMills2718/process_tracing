"""Tests for process-trace-driven source acquisition planning."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from pt.source_acquisition import build_acquisition_plan
from pt.source_design import ReviewDecision, build_source_design_state
from test_pipeline_integration import _make_audit_stress_result, _make_source_packet


@pytest.mark.plans(3)
def test_acquisition_plan_prioritizes_unresolved_source_gaps_and_absences():
    result = _make_audit_stress_result()
    packet = _make_source_packet()

    plan = build_acquisition_plan(result, source_packet=packet)

    assert plan.case_name == "18 Brumaire"
    assert plan.targets[0].kind == "source_gap"
    assert plan.targets[0].priority_score == 96
    assert "Private planning sequence" in plan.targets[0].evidence_need
    assert "claim-scope cap" in plan.targets[0].inferential_payoff
    assert any("18 Brumaire" in query for query in plan.targets[0].search_queries)
    assert any(target.kind == "damaging_absence" for target in plan.targets)
    assert any(
        "No named conspirators with a concrete operational plan" in target.evidence_need
        for target in plan.targets
    )


@pytest.mark.plans(3)
def test_acquisition_plan_omits_resolved_source_gap():
    result = _make_audit_stress_result()
    packet = _make_source_packet()
    packet.source_gap_dispositions[0].status = "accepted_limit"

    plan = build_acquisition_plan(result, source_packet=packet)

    assert all(target.kind != "source_gap" for target in plan.targets)
    assert any(target.kind == "damaging_absence" for target in plan.targets)


@pytest.mark.plans(3)
def test_acquisition_plan_emits_action_records_with_status():
    result = _make_audit_stress_result()
    packet = _make_source_packet()

    plan = build_acquisition_plan(result, source_packet=packet)
    actions = plan.to_action_records()

    assert actions[0].action_id == actions[0].target_id
    assert actions[0].status == "proposed"
    assert actions[0].stop_rule == plan.targets[0].stop_rule
    assert actions[0].target_source_class == plan.targets[0].target_source_class


@pytest.mark.plans(3)
def test_review_decision_updates_gap_disposition_without_admitting_evidence():
    result = _make_audit_stress_result()
    packet = _make_source_packet()
    state = build_source_design_state(result, source_packet=packet)

    updated = state.record_review_decision(
        ReviewDecision(
            candidate_id="cand_001",
            action_id=state.acquisition_actions[0].action_id,
            missing_source_class=state.acquisition_actions[0].target_source_class,
            decision="reject_as_adjacent",
            rationale="Retrospective memoir improves observability but remains adjacent.",
        )
    )

    assert updated.source_gap_dispositions[0].status == "partially_mitigated"
    assert updated.source_gap_dispositions[0].relevant_source_ids == ["source_c"]
    assert updated.source_candidates == state.source_candidates
    assert updated.review_log[-1].decision == "reject_as_adjacent"


@pytest.mark.plans(3)
def test_source_acquisition_cli_writes_json_plan(tmp_path):
    result = _make_audit_stress_result()
    packet = _make_source_packet()
    result.source_packet = packet.to_summary("packet.json")
    result_path = tmp_path / "result.json"
    packet_path = tmp_path / "packet.json"
    output_path = tmp_path / "acquisition_plan.json"
    result_path.write_text(result.model_dump_json(), encoding="utf-8")
    packet_path.write_text(packet.model_dump_json(), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/source_acquisition_plan.py",
            str(result_path),
            "--source-packet",
            str(packet_path),
            "--output",
            str(output_path),
            "--max-targets",
            "3",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "Targets:" in completed.stdout
    assert payload["plan"]["targets"][0]["kind"] == "source_gap"
    assert payload["plan"]["targets"][0]["target_source_class"] == (
        "Private correspondence among conspirators"
    )
    assert payload["design_state"]["iteration"] == 1
    assert payload["design_state"]["acquisition_actions"][0]["status"] == "proposed"
