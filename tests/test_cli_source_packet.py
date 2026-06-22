"""CLI smoke tests for source-packet wiring.

mock-ok: These tests replace the expensive LLM pipeline with a fake return
object and verify CLI argument plumbing plus JSON persistence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pt.cli import main
from pt.source_packet import SourceCandidate, SourcePacket


class _FakeResult:
    """Minimal result object used by the CLI JSON writer."""

    def __init__(self, source_packet_path: str) -> None:
        self.source_packet_path = source_packet_path

    def model_dump(self) -> dict[str, object]:
        return {
            "ok": True,
            "source_packet_path": self.source_packet_path,
        }


def _packet() -> SourcePacket:
    return SourcePacket(
        case_name="18 Brumaire",
        research_question="Why did Brumaire produce the Consulate?",
        focal_window="1799-11",
        outcome="Creation of the Consulate",
        source_candidates=[
            SourceCandidate(
                title="Official proclamation",
                source_group="official public justification",
                source_kind="primary proclamation",
                date_coverage="1799-11",
                locator="https://example.test/proclamation",
                provenance_note="Official post-coup source.",
                reliability_note="Justificatory and public-facing.",
                expected_observability="Public legitimacy claims.",
                relevance_to_question="Tests post-coup legitimation.",
            )
        ],
    )


@pytest.mark.plans(3)
def test_cli_passes_source_packet_to_pipeline(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    input_path = tmp_path / "input.txt"
    packet_path = tmp_path / "packet.json"
    output_dir = tmp_path / "out"
    input_path.write_text(" ".join(["substantive"] * 400), encoding="utf-8")
    packet_path.write_text(json.dumps(_packet().model_dump()), encoding="utf-8")

    def fake_run_pipeline(text: str, **kwargs):
        captured["text"] = text
        captured["kwargs"] = kwargs
        return _FakeResult(str(kwargs["source_packet_path"]))

    monkeypatch.setattr("pt.pipeline.run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pt",
            str(input_path),
            "--source-packet",
            str(packet_path),
            "--output-dir",
            str(output_dir),
            "--json-only",
        ],
    )

    main()

    kwargs = captured["kwargs"]
    assert isinstance(kwargs["source_packet"], SourcePacket)
    assert kwargs["source_packet"].research_question == "Why did Brumaire produce the Consulate?"
    assert kwargs["source_packet_path"] == str(packet_path)
    assert json.loads((output_dir / "result.json").read_text(encoding="utf-8")) == {
        "ok": True,
        "source_packet_path": str(packet_path),
    }
