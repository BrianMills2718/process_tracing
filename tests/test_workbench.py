"""Tests for the local process-tracing workbench server."""

from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pt import trace_host
from pt.workbench import build_app_payload, make_handler
from test_pipeline_integration import _make_audit_stress_result, _make_source_packet


def _write_artifacts(tmp_path):
    result = _make_audit_stress_result()
    packet = _make_source_packet()
    result.source_packet = packet.to_summary("packet.json")
    result_path = tmp_path / "result.json"
    packet_path = tmp_path / "packet.json"
    result_path.write_text(result.model_dump_json(), encoding="utf-8")
    packet_path.write_text(packet.model_dump_json(), encoding="utf-8")
    return result_path, packet_path


def _write_run_inputs(tmp_path):
    input_path = tmp_path / "input.txt"
    input_path.write_text(" ".join(["substantive"] * 400), encoding="utf-8")
    packet = _make_source_packet()
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(packet.model_dump_json(), encoding="utf-8")
    theories_path = tmp_path / "theories.txt"
    theories_path.write_text("Factional politics and constitutional design", encoding="utf-8")
    return input_path, packet_path, theories_path


@pytest.mark.plans(3)
def test_workbench_payload_builds_acquisition_plan(tmp_path):
    result_path, packet_path = _write_artifacts(tmp_path)

    payload = build_app_payload(
        result_path=str(result_path),
        source_packet_path=str(packet_path),
        max_targets=2,
        retrieve=False,
    )

    assert payload["plan"]["targets"][0]["kind"] == "source_gap"
    assert len(payload["plan"]["targets"]) == 2
    assert "retrieval" not in payload


@pytest.mark.plans(3)
def test_workbench_http_exposes_button_and_json_endpoint(tmp_path):
    result_path, packet_path = _write_artifacts(tmp_path)
    report_path = "output/live_plan003_source_expansion_20260623_001/report.html"
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base_url, timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Enrich Top Targets" in html
        assert "/api/enrich" in html
        assert "report-frame" in html
        assert "Load Report" in html

        with urllib.request.urlopen(
            f"{base_url}/artifact?path={urllib.parse.quote(report_path)}", timeout=5
        ) as response:
            report_html = response.read().decode("utf-8")
        assert "Process Tracing" in report_html

        body = json.dumps(
            {
                "result_path": str(result_path),
                "source_packet_path": str(packet_path),
                "max_targets": 1,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/acquisition-plan",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["plan"]["targets"][0]["target_id"] == "acq_gap_1"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.plans(3)
def test_workbench_http_supports_stage_by_stage_runs(tmp_path, monkeypatch):
    input_path, packet_path, theories_path = _write_run_inputs(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setattr(trace_host, "run_extract", lambda *a, **k: _make_audit_stress_result().extraction)
    monkeypatch.setattr(trace_host, "run_hypothesize", lambda *a, **k: _make_audit_stress_result().hypothesis_space)
    monkeypatch.setattr(trace_host, "run_test", lambda *a, **k: _make_audit_stress_result().testing)
    monkeypatch.setattr(trace_host, "run_absence", lambda *a, **k: _make_audit_stress_result().absence)
    monkeypatch.setattr(trace_host, "run_bayesian_update", lambda *a, **k: _make_audit_stress_result().bayesian)
    monkeypatch.setattr(trace_host, "run_synthesize", lambda *a, **k: _make_audit_stress_result().synthesis)

    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = json.dumps(
            {
                "input_path": str(input_path),
                "source_packet_path": str(packet_path),
                "theories_path": str(theories_path),
                "research_question": "Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup?",
                "refine": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/runs",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        run_id = payload["run"]["run_id"]
        assert payload["run"]["current_stage"] == "extract"

        for stage_id in ["extract", "hypothesize", "test", "absence", "update", "synthesize"]:
            stage_request = urllib.request.Request(
                f"{base_url}/api/runs/{run_id}/stages/{stage_id}/run",
                data=json.dumps({"force": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(stage_request, timeout=5) as response:
                stage_payload = json.loads(response.read().decode("utf-8"))
            assert stage_payload["ok"] is True

        with urllib.request.urlopen(f"{base_url}/api/runs/{run_id}", timeout=5) as response:
            run_payload = json.loads(response.read().decode("utf-8"))

        assert run_payload["ok"] is True
        assert run_payload["run"]["result_path"].endswith("result.json")
        assert run_payload["run"]["report_path"].endswith("report.html")
        assert run_payload["run"]["stages"][1]["status"] == "complete"
    finally:
        server.shutdown()
        server.server_close()
