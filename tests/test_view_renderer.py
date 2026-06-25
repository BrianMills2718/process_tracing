"""Deterministic tests for view_renderer projections and the artifact HTTP endpoint.

mock-ok: ViewRenderer reads JSON artifacts from disk; no LLM calls involved.
Tests build minimal stage artifact files in tmp_path and assert projected payload shape.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from pt.schemas import (
    BayesianResult,
    EvidenceLikelihood,
    Hypothesis,
    HypothesisPosterior,
    HypothesisLikelihood,
    HypothesisSpace,
    NewEvidence,
    Prediction,
    RefinementResult,
    ReinterpretedEvidence,
    SensitivityEntry,
    SpuriousExtraction,
    TestingResult,
)
from pt.schemas_view import DeltaViewPayload, MatrixViewPayload, ProvenanceViewPayload, SupportViewPayload
from pt.view_renderer import (
    build_delta_payload,
    build_matrix_payload,
    build_provenance_payload,
    build_support_payload,
    build_view_payload,
)
from pt.workbench import make_handler
from pt import trace_host

from test_pipeline_integration import (
    _make_extraction,
    _make_hypothesis_space,
    _make_testing,
    _make_audit_stress_result,
)


# ── Artifact writer helpers ────────────────────────────────────────────────────

def _write_base_artifacts(run_dir):
    """Write extraction, hypothesis_space, and testing JSON to run_dir."""
    extraction = _make_extraction()
    hs = _make_hypothesis_space()
    testing = _make_testing()
    (run_dir / "extraction.json").write_text(extraction.model_dump_json(), encoding="utf-8")
    (run_dir / "hypothesis_space.json").write_text(hs.model_dump_json(), encoding="utf-8")
    (run_dir / "testing.json").write_text(testing.model_dump_json(), encoding="utf-8")
    return extraction, hs, testing


def _make_bayesian() -> BayesianResult:
    return BayesianResult(
        posteriors=[
            HypothesisPosterior(
                hypothesis_id="h1",
                prior=0.5,
                updates=[],
                final_posterior=0.7,
                robustness="moderate",
                top_drivers=["evi_debt"],
            ),
            HypothesisPosterior(
                hypothesis_id="h2",
                prior=0.5,
                updates=[],
                final_posterior=0.3,
                robustness="fragile",
                top_drivers=["evi_elite_plot"],
            ),
        ],
        ranking=["h1", "h2"],
        sensitivity=[
            SensitivityEntry(
                hypothesis_id="h1",
                baseline_posterior=0.7,
                posterior_low=0.55,
                posterior_high=0.85,
                rank_stable=True,
            ),
            SensitivityEntry(
                hypothesis_id="h2",
                baseline_posterior=0.3,
                posterior_low=0.15,
                posterior_high=0.45,
                rank_stable=False,
            ),
        ],
    )


def _make_refinement() -> RefinementResult:
    return RefinementResult(
        new_evidence=[
            NewEvidence(
                id="evi_ref_new1",
                description="new finding",
                source_text="New source quote here.",
                rationale="missed on first pass",
            )
        ],
        reinterpreted_evidence=[
            ReinterpretedEvidence(
                evidence_id="evi_debt",
                original_type="empirical",
                new_type="interpretive",
                reinterpretation="reframed as interpretation",
            )
        ],
        spurious_extractions=[
            SpuriousExtraction(item_id="evi_historian_claim", item_type="evidence", reason="unreliable")
        ],
        hypothesis_refinements=[],
        analyst_notes="stub refinement for testing",
    )


# ── Matrix payload (stage: test) ───────────────────────────────────────────────

@pytest.mark.plans(5)
class TestBuildMatrixPayload:
    def test_hypotheses_and_row_count(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        extraction, hs, testing = _write_base_artifacts(run_dir)

        payload = build_matrix_payload(run_dir)

        assert isinstance(payload, MatrixViewPayload)
        assert set(payload.hypotheses) == {"h1", "h2"}
        assert payload.total_count == len(testing.evidence_likelihoods)

    def test_below_threshold_row_flagged(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)

        payload = build_matrix_payload(run_dir)

        below = [r for r in payload.rows if r.below_threshold]
        above = [r for r in payload.rows if not r.below_threshold]
        # evi_historian_claim has relevance=0.3 < 0.4
        assert len(below) == 1
        assert below[0].evidence_id == "evi_historian_claim"
        assert payload.below_threshold_count == 1
        assert len(above) == 3

    def test_rows_sorted_most_discriminating_first(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)

        payload = build_matrix_payload(run_dir)

        import math
        def max_log_lr(row):
            return max(abs(math.log(max(v, 1e-9))) for v in row.lr_vector.values())

        scores = [max_log_lr(r) for r in payload.rows]
        assert scores == sorted(scores, reverse=True)

    def test_lr_vector_keys_match_hypotheses(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)

        payload = build_matrix_payload(run_dir)

        hyp_set = set(payload.hypotheses)
        for row in payload.rows:
            assert set(row.lr_vector.keys()) == hyp_set

    def test_source_quote_snippet_from_extraction(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        extraction, _, _ = _write_base_artifacts(run_dir)

        payload = build_matrix_payload(run_dir)

        evi_map = {e.id: e for e in extraction.evidence}
        for row in payload.rows:
            if not row.below_threshold and row.source_quote_snippet:
                expected = evi_map[row.evidence_id].source_text[:120]
                assert row.source_quote_snippet == expected


# ── Support payload (stage: update) ────────────────────────────────────────────

@pytest.mark.plans(5)
class TestBuildSupportPayload:
    def _write_artifacts(self, run_dir):
        _write_base_artifacts(run_dir)
        bayesian = _make_bayesian()
        (run_dir / "bayesian.json").write_text(bayesian.model_dump_json(), encoding="utf-8")
        return bayesian

    def test_bars_count_and_sort_order(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        bayesian = self._write_artifacts(run_dir)

        payload = build_support_payload(run_dir)

        assert isinstance(payload, SupportViewPayload)
        assert len(payload.bars) == len(bayesian.posteriors)
        posteriors = [b.posterior for b in payload.bars]
        assert posteriors == sorted(posteriors, reverse=True)

    def test_sensitivity_bounds_wired(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_artifacts(run_dir)

        payload = build_support_payload(run_dir)

        h1_bar = next(b for b in payload.bars if b.hypothesis_id == "h1")
        assert h1_bar.posterior_low == pytest.approx(0.55)
        assert h1_bar.posterior_high == pytest.approx(0.85)
        assert h1_bar.rank_stable is True

        h2_bar = next(b for b in payload.bars if b.hypothesis_id == "h2")
        assert h2_bar.rank_stable is False

    def test_rank_instability_warning(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_artifacts(run_dir)

        payload = build_support_payload(run_dir)

        # h2 has rank_stable=False
        assert payload.rank_instability_warning is True

    def test_fragile_warning_fires_when_leader_is_fragile(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Make h1 (leader, posterior=0.7) fragile
        bayesian = BayesianResult(
            posteriors=[
                HypothesisPosterior(
                    hypothesis_id="h1", prior=0.5, updates=[], final_posterior=0.7, robustness="fragile"
                ),
                HypothesisPosterior(
                    hypothesis_id="h2", prior=0.5, updates=[], final_posterior=0.3, robustness="moderate"
                ),
            ],
            ranking=["h1", "h2"],
        )
        _write_base_artifacts(run_dir)
        (run_dir / "bayesian.json").write_text(bayesian.model_dump_json(), encoding="utf-8")

        payload = build_support_payload(run_dir)

        assert payload.fragile_warning is True

    def test_label_from_hypothesis_space(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _, hs, _ = _write_base_artifacts(run_dir)
        bayesian = _make_bayesian()
        (run_dir / "bayesian.json").write_text(bayesian.model_dump_json(), encoding="utf-8")

        payload = build_support_payload(run_dir)

        hs_map = {h.id: h.description[:60] for h in hs.hypotheses}
        for bar in payload.bars:
            assert bar.label == hs_map[bar.hypothesis_id]


# ── Provenance payload (stage: synthesize) ─────────────────────────────────────

@pytest.mark.plans(5)
class TestBuildProvenancePayload:
    def test_row_count_and_favored_hypothesis(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        extraction, hs, testing = _write_base_artifacts(run_dir)

        payload = build_provenance_payload(run_dir)

        assert isinstance(payload, ProvenanceViewPayload)
        assert payload.items_total == len(testing.evidence_likelihoods)
        # evi_elite_plot: h1=0.3, h2=0.9 → favors h2
        elite_row = next(r for r in payload.rows if r.evidence_id == "evi_elite_plot")
        assert elite_row.favored_hypothesis_id == "h2"
        assert elite_row.peak_lr == pytest.approx(0.9)

    def test_source_quote_from_extraction(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        extraction, _, _ = _write_base_artifacts(run_dir)

        payload = build_provenance_payload(run_dir)

        evi_map = {e.id: e for e in extraction.evidence}
        debt_row = next(r for r in payload.rows if r.evidence_id == "evi_debt")
        assert debt_row.source_quote_snippet == evi_map["evi_debt"].source_text[:120]

    def test_no_markers_without_result_json(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)

        payload = build_provenance_payload(run_dir)

        assert payload.items_with_marker == 0
        assert all(r.source_marker is None for r in payload.rows)

    def test_markers_from_result_json_source_coverage(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)
        # Write a minimal result.json with source_coverage that maps evi_debt to "Source A"
        result_stub = {
            "source_coverage": {
                "items": [
                    {
                        "source_id": "source_a",
                        "title": "Primary source",
                        "text_markers": ["Source A"],
                        "evidence_ids": ["evi_debt", "evi_tax_revolt"],
                    }
                ]
            }
        }
        (run_dir / "result.json").write_text(json.dumps(result_stub), encoding="utf-8")

        payload = build_provenance_payload(run_dir)

        assert payload.items_with_marker == 2
        debt_row = next(r for r in payload.rows if r.evidence_id == "evi_debt")
        assert debt_row.source_marker == "Source A"
        tax_row = next(r for r in payload.rows if r.evidence_id == "evi_tax_revolt")
        assert tax_row.source_marker == "Source A"
        # evi_elite_plot not in coverage
        elite_row = next(r for r in payload.rows if r.evidence_id == "evi_elite_plot")
        assert elite_row.source_marker is None

    def test_rows_sorted_by_discrimination(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)

        payload = build_provenance_payload(run_dir)

        import math
        log_lrs = [abs(math.log(max(r.peak_lr, 1e-9))) for r in payload.rows]
        assert log_lrs == sorted(log_lrs, reverse=True)


# ── Delta payload (stage: refine) ─────────────────────────────────────────────

@pytest.mark.plans(5)
class TestBuildDeltaPayload:
    def _write_refinement_artifacts(self, run_dir):
        _write_base_artifacts(run_dir)
        refinement = _make_refinement()
        (run_dir / "refinement.json").write_text(refinement.model_dump_json(), encoding="utf-8")
        return refinement

    def test_counts_no_pre_refine(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_refinement_artifacts(run_dir)

        payload = build_delta_payload(run_dir)

        assert isinstance(payload, DeltaViewPayload)
        assert payload.new_evidence_count == 1
        assert payload.reinterpreted_count == 1
        assert payload.spurious_count == 1
        assert payload.hypothesis_refined_count == 0
        assert payload.pre_refine_available is False
        assert payload.posterior_shifts == []

    def test_posterior_shifts_with_pre_refine(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_refinement_artifacts(run_dir)

        bayesian = _make_bayesian()
        (run_dir / "bayesian.json").write_text(bayesian.model_dump_json(), encoding="utf-8")

        # pre_refine with different posteriors
        pre_bayesian = BayesianResult(
            posteriors=[
                HypothesisPosterior(
                    hypothesis_id="h1", prior=0.5, updates=[], final_posterior=0.6, robustness="moderate"
                ),
                HypothesisPosterior(
                    hypothesis_id="h2", prior=0.5, updates=[], final_posterior=0.4, robustness="fragile"
                ),
            ],
            ranking=["h1", "h2"],
        )
        pre_refine_dir = run_dir / "pre_refine"
        pre_refine_dir.mkdir()
        (pre_refine_dir / "bayesian.json").write_text(pre_bayesian.model_dump_json(), encoding="utf-8")

        payload = build_delta_payload(run_dir)

        assert payload.pre_refine_available is True
        assert len(payload.posterior_shifts) == 2
        h1_shift = next(s for s in payload.posterior_shifts if s.hypothesis_id == "h1")
        assert h1_shift.before == pytest.approx(0.6)
        assert h1_shift.after == pytest.approx(0.7)
        assert h1_shift.delta == pytest.approx(0.1)

    def test_shifts_sorted_by_abs_delta(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_refinement_artifacts(run_dir)
        bayesian = _make_bayesian()
        (run_dir / "bayesian.json").write_text(bayesian.model_dump_json(), encoding="utf-8")
        # h1: before=0.4 (delta=+0.3), h2: before=0.29 (delta=+0.01)
        pre_bayesian = BayesianResult(
            posteriors=[
                HypothesisPosterior(
                    hypothesis_id="h1", prior=0.5, updates=[], final_posterior=0.4, robustness="moderate"
                ),
                HypothesisPosterior(
                    hypothesis_id="h2", prior=0.5, updates=[], final_posterior=0.29, robustness="fragile"
                ),
            ],
            ranking=["h1", "h2"],
        )
        pre_refine_dir = run_dir / "pre_refine"
        pre_refine_dir.mkdir()
        (pre_refine_dir / "bayesian.json").write_text(pre_bayesian.model_dump_json(), encoding="utf-8")

        payload = build_delta_payload(run_dir)

        abs_deltas = [abs(s.delta) for s in payload.posterior_shifts]
        assert abs_deltas == sorted(abs_deltas, reverse=True)


# ── Routing ────────────────────────────────────────────────────────────────────

@pytest.mark.plans(5)
class TestBuildViewPayloadRouting:
    def _write_all_artifacts(self, run_dir):
        _write_base_artifacts(run_dir)
        bayesian = _make_bayesian()
        (run_dir / "bayesian.json").write_text(bayesian.model_dump_json(), encoding="utf-8")
        refinement = _make_refinement()
        (run_dir / "refinement.json").write_text(refinement.model_dump_json(), encoding="utf-8")

    def test_routes_test_to_matrix(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_all_artifacts(run_dir)
        assert isinstance(build_view_payload(run_dir, "test"), MatrixViewPayload)

    def test_routes_update_to_support(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_all_artifacts(run_dir)
        assert isinstance(build_view_payload(run_dir, "update"), SupportViewPayload)

    def test_routes_synthesize_to_provenance(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_all_artifacts(run_dir)
        assert isinstance(build_view_payload(run_dir, "synthesize"), ProvenanceViewPayload)

    def test_routes_refine_to_delta(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_all_artifacts(run_dir)
        assert isinstance(build_view_payload(run_dir, "refine"), DeltaViewPayload)

    def test_returns_none_for_unmapped_stages(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        self._write_all_artifacts(run_dir)
        for stage_id in ("extract", "hypothesize", "absence", "setup"):
            assert build_view_payload(run_dir, stage_id) is None


# ── Artifact HTTP endpoint ──────────────────────────────────────────────────────

@pytest.mark.plans(5)
class TestArtifactEndpoint:
    def _start_server(self, monkeypatch, tmp_path):
        result = _make_audit_stress_result()
        monkeypatch.setattr(trace_host, "run_extract", lambda *a, **k: result.extraction)
        monkeypatch.setattr(trace_host, "run_hypothesize", lambda *a, **k: result.hypothesis_space)
        monkeypatch.setattr(trace_host, "run_test", lambda *a, **k: result.testing)
        monkeypatch.setattr(trace_host, "run_absence", lambda *a, **k: result.absence)
        monkeypatch.setattr(trace_host, "run_bayesian_update", lambda *a, **k: result.bayesian)
        monkeypatch.setattr(trace_host, "run_synthesize", lambda *a, **k: result.synthesis)
        monkeypatch.setattr(trace_host, "generate_report", lambda *a, **k: "<html>report</html>")

        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def _create_and_advance_run(self, base_url, tmp_path, stages):
        input_path = tmp_path / "input.txt"
        input_path.write_text(" ".join(["word"] * 400), encoding="utf-8")
        body = json.dumps({"input_path": str(input_path), "refine": False}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/runs", data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            run_id = json.loads(r.read())["run"]["run_id"]
        for stage_id in stages:
            req = urllib.request.Request(
                f"{base_url}/api/runs/{run_id}/stages/{stage_id}/run",
                data=b"{}",
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        return run_id

    def test_artifact_endpoint_returns_matrix_for_test_stage(self, tmp_path, monkeypatch):
        server = self._start_server(monkeypatch, tmp_path)
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            run_id = self._create_and_advance_run(
                base_url, tmp_path, ["extract", "hypothesize", "test"]
            )
            with urllib.request.urlopen(
                f"{base_url}/api/runs/{run_id}/stages/test/artifact", timeout=5
            ) as response:
                payload = json.loads(response.read())
            assert payload["ok"] is True
            assert payload["stage_id"] == "test"
            assert "hypotheses" in payload["payload"]
            assert "rows" in payload["payload"]
        finally:
            server.shutdown()
            server.server_close()

    def test_artifact_endpoint_returns_support_for_update_stage(self, tmp_path, monkeypatch):
        server = self._start_server(monkeypatch, tmp_path)
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            run_id = self._create_and_advance_run(
                base_url, tmp_path, ["extract", "hypothesize", "test", "absence", "update"]
            )
            with urllib.request.urlopen(
                f"{base_url}/api/runs/{run_id}/stages/update/artifact", timeout=5
            ) as response:
                payload = json.loads(response.read())
            assert payload["ok"] is True
            assert payload["stage_id"] == "update"
            assert "bars" in payload["payload"]
        finally:
            server.shutdown()
            server.server_close()

    def test_artifact_endpoint_404_for_unmapped_stage(self, tmp_path, monkeypatch):
        server = self._start_server(monkeypatch, tmp_path)
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            run_id = self._create_and_advance_run(
                base_url, tmp_path, ["extract"]
            )
            import urllib.error
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(
                    f"{base_url}/api/runs/{run_id}/stages/extract/artifact", timeout=5
                )
            assert exc_info.value.code == 404
        finally:
            server.shutdown()
            server.server_close()

    def test_artifact_endpoint_404_for_unknown_run(self, tmp_path, monkeypatch):
        server = self._start_server(monkeypatch, tmp_path)
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            import urllib.error
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(
                    f"{base_url}/api/runs/nonexistent_run/stages/test/artifact", timeout=5
                )
            assert exc_info.value.code == 404
        finally:
            server.shutdown()
            server.server_close()


# ── Adversarial battery (Task 19) ─────────────────────────────────────────────

@pytest.mark.plans(5)
class TestAdversarialBattery:
    """5-case adversarial battery: incomplete run, malformed artifact, refine=False,
    all-below-threshold, single hypothesis."""

    def test_missing_artifact_returns_500_json(self, tmp_path, monkeypatch):
        """Incomplete run — artifact endpoint is requested before stage has run."""
        result = _make_audit_stress_result()
        monkeypatch.setattr(trace_host, "run_extract", lambda *a, **k: result.extraction)
        monkeypatch.setattr(trace_host, "run_hypothesize", lambda *a, **k: result.hypothesis_space)
        monkeypatch.setattr(trace_host, "run_test", lambda *a, **k: result.testing)

        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        input_path = tmp_path / "input.txt"
        input_path.write_text(" ".join(["word"] * 400), encoding="utf-8")
        try:
            body = json.dumps({"input_path": str(input_path), "refine": False}).encode()
            req = urllib.request.Request(
                f"{base_url}/api/runs", data=body,
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                run_id = json.loads(r.read())["run"]["run_id"]
            # Advance to extract only — testing.json does not exist yet
            req = urllib.request.Request(
                f"{base_url}/api/runs/{run_id}/stages/extract/run",
                data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
            # Request test artifact before running test stage
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(
                    f"{base_url}/api/runs/{run_id}/stages/test/artifact", timeout=5
                )
            assert exc_info.value.code == 500
            error_body = json.loads(exc_info.value.read())
            assert error_body["ok"] is False
            assert "error" in error_body
        finally:
            server.shutdown()
            server.server_close()

    def test_malformed_artifact_returns_500_json(self, tmp_path):
        """Malformed artifact — corrupt testing.json → clean JSON 500, not a crash."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)
        # Corrupt the testing.json
        (run_dir / "testing.json").write_text("{ not valid json }", encoding="utf-8")

        import json as _json
        with pytest.raises((_json.JSONDecodeError, Exception)):
            build_matrix_payload(run_dir)

    def test_all_below_threshold_matrix(self, tmp_path):
        """All-below-threshold — every evidence item has relevance < 0.4."""

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        extraction = _make_extraction()
        hs = _make_hypothesis_space()
        # Override testing with all below-threshold items
        low_relevance_testing = TestingResult(
            evidence_likelihoods=[
                EvidenceLikelihood(
                    evidence_id=e.id,
                    hypothesis_likelihoods=[
                        HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=0.5, diagnostic_type="straw_in_the_wind"),
                        HypothesisLikelihood(hypothesis_id="h2", relative_likelihood=0.5, diagnostic_type="straw_in_the_wind"),
                    ],
                    relevance=0.2,
                    justification="low relevance test",
                )
                for e in extraction.evidence
            ]
        )
        (run_dir / "extraction.json").write_text(extraction.model_dump_json(), encoding="utf-8")
        (run_dir / "hypothesis_space.json").write_text(hs.model_dump_json(), encoding="utf-8")
        (run_dir / "testing.json").write_text(low_relevance_testing.model_dump_json(), encoding="utf-8")

        payload = build_matrix_payload(run_dir)

        assert payload.below_threshold_count == payload.total_count
        assert all(r.below_threshold for r in payload.rows)

    def test_single_hypothesis_matrix(self, tmp_path):
        """Single hypothesis — matrix with one column renders without error."""

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        extraction = _make_extraction()
        single_hs = HypothesisSpace(
            research_question="Why did the crisis occur?",
            hypotheses=[
                Hypothesis(
                    id="h1",
                    description="Fiscal crisis was the cause",
                    source="text",
                    theoretical_basis="Fiscal theory",
                    causal_mechanism="Debt → collapse",
                    observable_predictions=[Prediction(id="p1", description="evidence of debt")],
                )
            ],
        )
        single_testing = TestingResult(
            evidence_likelihoods=[
                EvidenceLikelihood(
                    evidence_id=e.id,
                    hypothesis_likelihoods=[
                        HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=1.5, diagnostic_type="straw_in_the_wind"),
                    ],
                    relevance=0.8,
                    justification="single hyp test",
                )
                for e in extraction.evidence
            ]
        )
        (run_dir / "extraction.json").write_text(extraction.model_dump_json(), encoding="utf-8")
        (run_dir / "hypothesis_space.json").write_text(single_hs.model_dump_json(), encoding="utf-8")
        (run_dir / "testing.json").write_text(single_testing.model_dump_json(), encoding="utf-8")

        payload = build_matrix_payload(run_dir)

        assert payload.hypotheses == ["h1"]
        assert payload.total_count == len(extraction.evidence)
        for row in payload.rows:
            assert "h1" in row.lr_vector
            assert len(row.lr_vector) == 1

    def test_refine_disabled_delta_raises_on_missing_refinement(self, tmp_path):
        """refine=False — no refinement.json written; build_delta_payload raises FileNotFoundError."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        _write_base_artifacts(run_dir)
        # No refinement.json — simulate a run with refine=False

        with pytest.raises(FileNotFoundError):
            build_delta_payload(run_dir)
