"""Tests for the frozen benchmark runner (scripts/run_benchmark.py).

Verifies that the benchmark runner correctly evaluates planted failure cases
and produces a machine-readable scorecard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestBenchmarkFixtures:
    """Benchmark fixture factories must produce valid ProcessTracingResult objects."""

    def test_adversarial_stress_fixture_loads(self):
        from scripts.run_benchmark import _build_adversarial_stress_fixture
        result = _build_adversarial_stress_fixture()
        assert result.extraction.evidence
        assert len(result.extraction.evidence) == 14  # 12 background + 2 proximate
        assert result.synthesis.verdicts

    def test_calibration_mismatch_fixture_loads(self):
        from scripts.run_benchmark import _build_calibration_mismatch_fixture
        result = _build_calibration_mismatch_fixture()
        assert result.synthesis.verdicts
        # The strongly_supported verdict should point to a hypothesis with posterior < 0.50
        ss_verdict = next(
            v for v in result.synthesis.verdicts if v.status == "strongly_supported"
        )
        target_post = next(
            p.final_posterior for p in result.bayesian.posteriors
            if p.hypothesis_id == ss_verdict.hypothesis_id
        )
        assert target_post < 0.50, (
            f"calibration_mismatch fixture: strongly_supported hypothesis has posterior "
            f"{target_post:.3f} >= 0.50 — the planted mismatch is not present"
        )


class TestBenchmarkRunner:
    """End-to-end benchmark runner using only synthetic fixtures (no disk I/O)."""

    def test_all_fixture_cases_pass(self, tmp_path):
        """The two required fixture cases must both pass."""
        from scripts.run_benchmark import run_benchmark

        config = {
            "cases": [
                {
                    "name": "adversarial_fragile_overclaim",
                    "type": "fixture",
                    "fixture": "adversarial_stress",
                    "focal_year": 1799,
                    "optional": False,
                    "expected": {
                        "min_score": 60,
                        "max_score": 85,
                        "grade": "C",
                        "must_flag": ["broad_winning_hypothesis"],
                        "must_not_flag": [],
                    },
                },
                {
                    "name": "adversarial_calibration_mismatch",
                    "type": "fixture",
                    "fixture": "calibration_mismatch",
                    "focal_year": 1799,
                    "optional": False,
                    "expected": {
                        "min_score": 40,
                        "max_score": 100,
                        "grade": None,
                        "must_flag": ["verdict_calibration_mismatch"],
                        "must_not_flag": [],
                    },
                },
            ]
        }
        config_path = tmp_path / "benchmark_config.yaml"
        import yaml
        with config_path.open("w") as f:
            yaml.dump(config, f)

        scorecard = run_benchmark(
            config_path=config_path,
            output_path=tmp_path / "scorecard.json",
        )
        assert scorecard["all_required_pass"] is True, (
            f"Required benchmark cases failed: "
            + str([r for r in scorecard["cases"] if not r["passed"] and not r["skipped"]])
        )
        assert scorecard["passed"] == 2
        assert scorecard["failed"] == 0

    def test_optional_case_skipped_when_missing(self, tmp_path):
        """Optional result_file cases are skipped (not failed) when the path is absent."""
        from scripts.run_benchmark import run_benchmark

        config = {
            "cases": [
                {
                    "name": "missing_optional",
                    "type": "result_file",
                    "result_path": "output/nonexistent_run/result.json",
                    "optional": True,
                    "expected": {"min_score": 70, "grade": "B"},
                },
            ]
        }
        config_path = tmp_path / "benchmark_config.yaml"
        import yaml
        with config_path.open("w") as f:
            yaml.dump(config, f)

        scorecard = run_benchmark(
            config_path=config_path,
            output_path=tmp_path / "scorecard.json",
        )
        assert scorecard["all_required_pass"] is True
        assert scorecard["skipped"] == 1
        assert scorecard["failed"] == 0

    def test_required_case_fails_loudly_when_missing(self, tmp_path):
        """A required result_file case that is missing counts as a failure."""
        from scripts.run_benchmark import run_benchmark

        config = {
            "cases": [
                {
                    "name": "missing_required",
                    "type": "result_file",
                    "result_path": "output/nonexistent_run/result.json",
                    "optional": False,
                    "expected": {"min_score": 70},
                },
            ]
        }
        config_path = tmp_path / "benchmark_config.yaml"
        import yaml
        with config_path.open("w") as f:
            yaml.dump(config, f)

        scorecard = run_benchmark(
            config_path=config_path,
            output_path=tmp_path / "scorecard.json",
        )
        assert scorecard["all_required_pass"] is False
        assert scorecard["failed"] == 1

    def test_scorecard_is_valid_json(self, tmp_path):
        """Scorecard output is valid JSON with expected keys."""
        from scripts.run_benchmark import run_benchmark

        config = {"cases": [
            {"name": "calibration_mismatch", "type": "fixture", "fixture": "calibration_mismatch",
             "optional": False, "expected": {"must_flag": ["verdict_calibration_mismatch"]}},
        ]}
        config_path = tmp_path / "benchmark_config.yaml"
        output_path = tmp_path / "scorecard.json"
        import yaml
        with config_path.open("w") as f:
            yaml.dump(config, f)

        run_benchmark(config_path=config_path, output_path=output_path)

        with output_path.open() as f:
            scorecard = json.load(f)

        assert "total" in scorecard
        assert "passed" in scorecard
        assert "failed" in scorecard
        assert "all_required_pass" in scorecard
        assert "cases" in scorecard
        case = scorecard["cases"][0]
        assert "name" in case
        assert "score" in case
        assert "grade" in case
        assert "flags_found" in case

    def test_result_file_case_without_report_path_does_not_crash(self, tmp_path):
        """result_file cases with no report_path must run the audit, not fail with 'Is a directory'.

        Regression: _REPO_ROOT / "" resolves to the repo root directory. When the runner
        checked `if report_path and report_path.exists()`, that directory both was truthy
        and existed, causing open() to raise IsADirectoryError. Fix: only construct
        report_path when case["report_path"] is non-empty.
        """
        from scripts.run_benchmark import run_benchmark

        # Build a minimal fixture result and write it as result.json
        from scripts.run_benchmark import _build_calibration_mismatch_fixture
        import json as _json
        result = _build_calibration_mismatch_fixture()
        result_json = tmp_path / "result.json"
        result_json.write_text(_json.dumps(result.model_dump()))

        config = {"cases": [
            {
                "name": "no_report_path",
                "type": "result_file",
                # Absolute path: Path(_REPO_ROOT) / "/absolute" == "/absolute" in Python
                "result_path": str(result_json),
                "optional": False,
                "expected": {},
            },
        ]}
        config_path = tmp_path / "cfg.yaml"
        import yaml
        with config_path.open("w") as f:
            yaml.dump(config, f)

        scorecard = run_benchmark(
            config_path=config_path,
            output_path=tmp_path / "scorecard.json",
        )
        assert scorecard["total"] == 1
        case = scorecard["cases"][0]
        # The case must reach the audit step: score is populated (not None)
        assert case["score"] is not None, (
            "result_file case without report_path never reached the audit — "
            f"failures: {case.get('failures')}"
        )
        # No loading error (which would leave score=None and include the error string)
        assert not any("Loading failed" in str(f) for f in case.get("failures", []))
