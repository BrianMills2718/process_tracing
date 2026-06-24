"""Tests for the local process-tracing workbench server."""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

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
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base_url, timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Enrich Top Targets" in html
        assert "/api/enrich" in html

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
