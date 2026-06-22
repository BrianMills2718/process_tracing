"""Tests for the agentic assistant harness.

Provider calls are mocked here (mock-ok: these tests verify process_tracing's
llm_client delegation contract, artifact persistence, and dependency boundary;
live agent behavior is covered by the opt-in smoke test).
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from pt.assistant import (
    SOURCE_PACKET_PROMPT_REF,
    SOURCE_PACKET_TASK,
    AssistantError,
    SourceCandidate,
    SourcePacketDraft,
    draft_source_packet,
    main,
)


class _FakeResult:
    model = "codex"
    finish_reason = "stop"


def _draft() -> SourcePacketDraft:
    return SourcePacketDraft(
        case_name="18 Brumaire",
        research_question="Why did Brumaire produce the Consulate?",
        focal_window="1799-11",
        outcome="Creation of the Consulate after 18 Brumaire",
        source_candidates=[
            SourceCandidate(
                title="Proclamation of the Consuls",
                source_kind="primary proclamation",
                date_coverage="1799-11",
                locator="https://example.test/brumaire",
                provenance_note="Official post-coup proclamation.",
                reliability_note="Justificatory source, not neutral evidence.",
                expected_observability="Should reveal public legitimation claims, not private planning.",
                relevance_to_question="Tests whether order/legitimacy claims were post-hoc.",
            )
        ],
        rival_interpretations=[],
        observability_notes=["Official sources are strong for public justification."],
        known_gaps=[],
        proposed_next_steps=["Add independent legislative records."],
        limitations=["No private correspondence included."],
    )


@pytest.mark.plans(3)
def test_source_packet_assistant_delegates_to_llm_client(monkeypatch, tmp_path):
    captured = {}
    context = tmp_path / "context.md"
    context.write_text("# Context\nBrumaire source notes.", encoding="utf-8")
    output = tmp_path / "artifact.json"

    def fake_structured(model, messages, response_model, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["response_model"] = response_model
        captured["kwargs"] = kwargs
        return _draft(), _FakeResult()

    monkeypatch.setattr("pt.assistant.call_llm_structured", fake_structured)

    artifact = draft_source_packet(
        case_name="18 Brumaire",
        research_question="Why did Brumaire produce the Consulate?",
        context_paths=[context],
        output_path=output,
        model="codex",
        trace_id="trace-123",
        max_budget=0.25,
        cwd=tmp_path,
        timeout=120,
    )

    assert captured["model"] == "codex"
    assert captured["response_model"] is SourcePacketDraft
    assert captured["kwargs"]["execution_mode"] == "workspace_agent"
    assert captured["kwargs"]["task"] == SOURCE_PACKET_TASK
    assert captured["kwargs"]["trace_id"] == "trace-123"
    assert captured["kwargs"]["max_budget"] == 0.25
    assert captured["kwargs"]["prompt_ref"] == SOURCE_PACKET_PROMPT_REF
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
    assert captured["kwargs"]["working_directory"] == str(tmp_path.resolve())
    assert "Brumaire source notes" in captured["messages"][0]["content"]
    assert artifact.metadata.backend_model == "codex"
    assert artifact.metadata.execution_mode == "workspace_agent"
    assert artifact.metadata.output_path == str(output.resolve())

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["trace_id"] == "trace-123"
    assert payload["draft"]["source_candidates"][0]["title"] == "Proclamation of the Consuls"


@pytest.mark.plans(3)
def test_source_packet_assistant_rejects_non_agent_model(tmp_path):
    context = tmp_path / "context.md"
    context.write_text("context", encoding="utf-8")

    with pytest.raises(ValueError, match="Codex or Claude Code"):
        draft_source_packet(
            case_name="18 Brumaire",
            context_paths=[context],
            output_path=tmp_path / "out.json",
            model="gpt-4o",
        )


@pytest.mark.plans(3)
def test_source_packet_cli_reports_invalid_model_without_traceback(tmp_path, capsys):
    context = tmp_path / "context.md"
    context.write_text("context", encoding="utf-8")

    code = main([
        "source-packet",
        "--case-name",
        "18 Brumaire",
        "--context",
        str(context),
        "--output",
        str(tmp_path / "out.json"),
        "--model",
        "gpt-4o",
    ])

    captured = capsys.readouterr()
    assert code == 1
    assert "Codex or Claude Code" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.plans(3)
def test_source_packet_assistant_fails_loud_on_missing_context(tmp_path):
    with pytest.raises(AssistantError, match="context file not found"):
        draft_source_packet(
            case_name="18 Brumaire",
            context_paths=[tmp_path / "missing.md"],
            output_path=tmp_path / "out.json",
            model="codex",
        )


@pytest.mark.plans(3)
def test_assistant_harness_has_no_direct_agent_sdk_dependencies():
    forbidden_imports = {"openai_codex_sdk", "claude_agent_sdk"}
    for path in Path("pt").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            assert not (names & forbidden_imports), f"{path} imports {names & forbidden_imports}"

    assistant_tree = ast.parse(Path("pt/assistant.py").read_text(encoding="utf-8"))
    for node in ast.walk(assistant_tree):
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            assert "subprocess" not in names
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module != "subprocess"


@pytest.mark.skipif(
    os.getenv("PT_RUN_LIVE_AGENT_TESTS") != "1",
    reason="live agent smoke; set PT_RUN_LIVE_AGENT_TESTS=1 to run",
)
@pytest.mark.plans(3)
def test_source_packet_assistant_live_smoke(tmp_path):
    context = Path("docs/source_packets/18_BRUMAIRE_RESEARCH_DESIGN.md")
    artifact = draft_source_packet(
        case_name="18 Brumaire",
        context_paths=[context],
        output_path=tmp_path / "source_packet_draft.json",
        model=os.getenv("PT_ASSISTANT_MODEL", "codex"),
        trace_id="pt-assistant-live-smoke",
        max_budget=float(os.getenv("PT_ASSISTANT_MAX_BUDGET", "1.0")),
        cwd=Path("."),
        timeout=600,
    )
    assert artifact.draft.source_candidates
    assert artifact.metadata.execution_mode == "workspace_agent"
