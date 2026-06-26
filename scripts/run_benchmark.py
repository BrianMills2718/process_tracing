"""Frozen benchmark suite runner for the process_tracing pipeline.

Loads benchmark cases from docs/benchmarks/benchmark_config.yaml, runs
audit_result on each, checks scores and flags against expected outcomes,
and writes a JSON scorecard to docs/benchmarks/last_scorecard.json.

Exit 0 if all required cases pass; exit 1 on failures.

Usage:
    python scripts/run_benchmark.py [--config path] [--output path] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Fixture factories (planted failure modes) ──────────────────────


def _build_adversarial_stress_fixture():
    """12 weak background items, fragile winner, broad legitimacy-vacuum hypothesis.

    Reproduces the planted failure case in tests/test_pipeline_integration.py
    (test_output_quality_audit_surfaces_adversarial_caveats). Expected: grade C.
    """
    from pt.bayesian import run_bayesian_update
    from pt.schemas import (
        AbsenceEvaluation,
        AbsenceResult,
        Actor,
        CausalEdge,
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
    )

    def _ev(eid: str, h1: float, h2: float, relevance: float = 0.9) -> EvidenceLikelihood:
        return EvidenceLikelihood(
            evidence_id=eid,
            justification="background",
            hypothesis_likelihoods=[
                HypothesisLikelihood(
                    hypothesis_id="h1",
                    relative_likelihood=h1,
                    diagnostic_type="straw_in_the_wind",
                ),
                HypothesisLikelihood(
                    hypothesis_id="h2",
                    relative_likelihood=h2,
                    diagnostic_type="straw_in_the_wind",
                ),
            ],
            relevance_to_hypothesis=relevance,
        )

    evidence: list[Evidence] = []
    vectors: list[EvidenceLikelihood] = []
    for idx in range(12):
        eid = f"evi_background_{idx:02d}"
        evidence.append(Evidence(
            id=eid,
            description=f"Background legitimacy erosion item {idx}",
            source_text=f"Background legitimacy erosion was visible in 1789 item {idx}.",
            evidence_type="empirical",
            approximate_date="1789",
        ))
        vectors.append(_ev(eid, h1=2.0, h2=1.0, relevance=0.9))
    for idx in range(2):
        eid = f"evi_proximate_{idx:02d}"
        evidence.append(Evidence(
            id=eid,
            description=f"Proximate coup maneuver item {idx}",
            source_text=f"Proximate coup maneuver occurred in 1799 item {idx}.",
            evidence_type="empirical",
            approximate_date="1799",
        ))
        vectors.append(_ev(eid, h1=1.4, h2=1.0, relevance=0.8))

    actors = [Actor(id="actor_napoleon", name="Napoleon Bonaparte",
                    description="Military general and political leader who executed the coup")]
    events = [
        Event(id="event_coup", description="Legislative coup", date="1799-11-09"),
    ]
    edges = [CausalEdge(source_id="evi_background_00", target_id="event_coup",
                        relationship="enables")]
    mechanisms = [Mechanism(id="mech_01", description="power vacuum", hypothesis_link="h1",
                            evidence_ids=["evi_background_00"])]
    extraction = ExtractionResult(
        summary="The crisis culminated in a military coup in 1799 after a decade of legitimacy loss.",
        evidence=evidence,
        actors=actors,
        events=events,
        causal_edges=edges,
        mechanisms=mechanisms,
    )

    hyp_space = HypothesisSpace(
        research_question="Why did the crisis culminate in the 1799 coup rather than reform?",
        hypotheses=[
            Hypothesis(
                id="h1",
                description="A legitimacy vacuum across multiple institutions enabled the coup",
                source="text",
                theoretical_basis="state collapse theory",
                causal_mechanism="A power vacuum across institutions made a decisive coup coalition feasible",
                observable_predictions=[
                    Prediction(id="pred_h1_01", description="Broad institutional discrediting"),
                ],
            ),
            Hypothesis(
                id="h2",
                description="A deliberate conspiracy by Bonaparte and allies seized power",
                source="text",
                theoretical_basis="elite network theory",
                causal_mechanism="A planned coordination among military and civil factions enabled the coup",
                observable_predictions=[
                    Prediction(id="pred_h2_02", description="Named conspirators with an operational plan"),
                ],
            ),
        ],
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
    absence = AbsenceResult(evaluations=[
        AbsenceEvaluation(
            hypothesis_id="h2",
            prediction_id="pred_h2_02",
            missing_evidence="No named conspirators with a concrete operational plan",
            reasoning="A direct conspiracy account should name agents and planning details",
            severity="damaging",
            would_be_extractable=True,
        )
    ])
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
        suggested_further_tests=["Look for operational planning documents"],
    )

    return ProcessTracingResult(
        extraction=extraction,
        hypothesis_space=hyp_space,
        testing=testing,
        absence=absence,
        bayesian=bayesian,
        synthesis=synthesis,
    )


def _build_calibration_mismatch_fixture():
    """Synthesis assigns strongly_supported to a hypothesis with posterior 0.30.

    Plants a STRONG_SUPPORT_FLOOR calibration mismatch: the LLM verdict is
    strongly_supported but the posterior is below 0.50. audit_result must flag
    this via _verdict_calibration_issues().
    """
    from pt.bayesian import run_bayesian_update
    from pt.schemas import (
        AbsenceResult,
        Evidence,
        EvidenceLikelihood,
        ExtractionResult,
        Hypothesis,
        HypothesisLikelihood,
        HypothesisSpace,
        HypothesisVerdict,
        Prediction,
        ProcessTracingResult,
        SynthesisResult,
        TestingResult,
    )

    def _ev(eid: str, h1: float, h2: float) -> EvidenceLikelihood:
        return EvidenceLikelihood(
            evidence_id=eid,
            justification="j",
            hypothesis_likelihoods=[
                HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=h1,
                                     diagnostic_type="straw_in_the_wind"),
                HypothesisLikelihood(hypothesis_id="h2", relative_likelihood=h2,
                                     diagnostic_type="straw_in_the_wind"),
            ],
        )

    evidence = [
        Evidence(id="evi_1", description="d1", source_text="q1"),
        Evidence(id="evi_2", description="d2", source_text="q2"),
        Evidence(id="evi_3", description="d3", source_text="q3"),
    ]
    vectors = [
        _ev("evi_1", h1=1.5, h2=1.0),
        _ev("evi_2", h1=1.2, h2=1.0),
        _ev("evi_3", h1=1.0, h2=2.0),
    ]
    extraction = ExtractionResult(summary="A contested historical episode.", evidence=evidence)
    hyp_space = HypothesisSpace(
        research_question="Why did the event occur?",
        hypotheses=[
            Hypothesis(id="h1", description="Structural explanation",
                       source="text", theoretical_basis="t1",
                       causal_mechanism="structural forces", observable_predictions=[
                           Prediction(id="p1", description="early indicators")]),
            Hypothesis(id="h2", description="Agency explanation",
                       source="text", theoretical_basis="t2",
                       causal_mechanism="deliberate choices", observable_predictions=[
                           Prediction(id="p2", description="named actor decisions")]),
        ],
    )
    testing = TestingResult(evidence_likelihoods=vectors)
    bayesian = run_bayesian_update(testing, ["h1", "h2"])
    # Verify h1 posterior is below STRONG_SUPPORT_FLOOR (0.50)
    h1_posterior = next(p.final_posterior for p in bayesian.posteriors if p.hypothesis_id == "h1")
    # If h1 is unexpectedly >= 0.50, assign strongly_supported to h2 instead
    target_id = "h1" if h1_posterior < 0.50 else "h2"
    other_id = "h2" if target_id == "h1" else "h1"
    synthesis = SynthesisResult(
        verdicts=[
            HypothesisVerdict(
                hypothesis_id=target_id,
                status="strongly_supported",  # planted mismatch: posterior < 0.50
                key_evidence_for=["evi_1"],
                key_evidence_against=[],
                reasoning="Planted mismatch: label overstates posterior.",
                steelman="Some structural factors are present.",
            ),
            HypothesisVerdict(
                hypothesis_id=other_id,
                status="weakened",
                key_evidence_for=[],
                key_evidence_against=["evi_1"],
                reasoning="Little support.",
                steelman="Agency always plays some role.",
            ),
        ],
        comparative_analysis="Structural factors dominate.",
        analytical_narrative="The structural account has stronger comparative support than the agency account.",
        limitations=["Limited source base"],
        suggested_further_tests=[],
    )
    return ProcessTracingResult(
        extraction=extraction,
        hypothesis_space=hyp_space,
        testing=testing,
        absence=AbsenceResult(),
        bayesian=bayesian,
        synthesis=synthesis,
    )


_FIXTURE_REGISTRY = {
    "adversarial_stress": _build_adversarial_stress_fixture,
    "calibration_mismatch": _build_calibration_mismatch_fixture,
}


# ── Case runner ────────────────────────────────────────────────────


def _run_case(case: dict, verbose: bool = False) -> dict[str, Any]:
    """Run a single benchmark case and return a result record."""
    from pt.report import generate_report
    from scripts.audit_result_quality import audit_result

    name = case["name"]
    case_type = case.get("type", "result_file")
    optional = case.get("optional", False)
    focal_year = case.get("focal_year")
    expected = case.get("expected", {})
    result_record: dict[str, Any] = {
        "name": name,
        "type": case_type,
        "optional": optional,
        "description": case.get("description", ""),
        "passed": False,
        "skipped": False,
        "failures": [],
        "score": None,
        "grade": None,
        "flags_found": [],
    }

    # Load result
    try:
        if case_type == "fixture":
            fixture_name = case.get("fixture")
            if fixture_name not in _FIXTURE_REGISTRY:
                result_record["failures"].append(f"Unknown fixture: {fixture_name!r}")
                return result_record
            pt_result = _FIXTURE_REGISTRY[fixture_name]()
            html = generate_report(pt_result)
        elif case_type == "result_file":
            result_path = _REPO_ROOT / case["result_path"]
            report_path = _REPO_ROOT / case.get("report_path", "")
            if not result_path.exists():
                if optional:
                    result_record["skipped"] = True
                    result_record["skip_reason"] = f"result_path not found: {result_path}"
                    return result_record
                result_record["failures"].append(f"result_path not found: {result_path}")
                return result_record
            import json as _json
            from pt.schemas import ProcessTracingResult
            with result_path.open("r", encoding="utf-8") as f:
                pt_result = ProcessTracingResult.model_validate(_json.load(f))
            html = ""
            if report_path and report_path.exists():
                with report_path.open("r", encoding="utf-8") as f:
                    html = f.read()
            else:
                html = generate_report(pt_result)
        else:
            result_record["failures"].append(f"Unknown case type: {case_type!r}")
            return result_record
    except Exception as e:
        result_record["failures"].append(f"Loading failed: {e}")
        return result_record

    # Run audit
    try:
        audit = audit_result(pt_result, html, focal_year_override=focal_year)
    except Exception as e:
        result_record["failures"].append(f"Audit failed: {e}")
        return result_record

    score = audit["score"]
    grade = audit["grade"]
    result_record["score"] = score
    result_record["grade"] = grade
    result_record["audit_summary"] = {
        "score": score,
        "grade": grade,
        "base_score": audit.get("base_score"),
        "academic_cap": audit.get("academic_cap"),
    }

    # Collect flags present
    flags_found: list[str] = []
    # Check broad_winning_hypothesis
    if audit.get("categories", {}).get("inference_depth_and_clarity", {}).get("broad_winner_risk"):
        flags_found.append("broad_winning_hypothesis")
    # Check verdict_calibration_mismatch
    if audit.get("categories", {}).get("comparative_support_discipline", {}).get("verdict_issues"):
        flags_found.append("verdict_calibration_mismatch")
    # Check overclaim
    if audit.get("categories", {}).get("comparative_support_discipline", {}).get("overclaim_issues"):
        flags_found.append("overclaim")
    result_record["flags_found"] = flags_found

    # Evaluate expectations
    failures: list[str] = []

    min_score = expected.get("min_score")
    max_score = expected.get("max_score")
    expected_grade = expected.get("grade")
    must_flag: list[str] = expected.get("must_flag", [])
    must_not_flag: list[str] = expected.get("must_not_flag", [])

    if min_score is not None and score < min_score:
        failures.append(f"score {score} < expected min {min_score}")
    if max_score is not None and score > max_score:
        failures.append(f"score {score} > expected max {max_score}")
    if expected_grade is not None and grade != expected_grade:
        failures.append(f"grade {grade!r} != expected {expected_grade!r}")
    for flag in must_flag:
        if flag not in flags_found:
            failures.append(f"expected flag {flag!r} not raised by audit")
    for flag in must_not_flag:
        if flag in flags_found:
            failures.append(f"flag {flag!r} was raised but must not be")

    result_record["failures"] = failures
    result_record["passed"] = len(failures) == 0

    if verbose:
        status = "PASS" if result_record["passed"] else "FAIL"
        print(f"  [{status}] {name}: score={score}, grade={grade}, flags={flags_found}")
        for f in failures:
            print(f"         FAIL: {f}")

    return result_record


def run_benchmark(
    config_path: Path | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run all benchmark cases and return a scorecard dict."""
    if config_path is None:
        config_path = _REPO_ROOT / "docs" / "benchmarks" / "benchmark_config.yaml"
    if output_path is None:
        output_path = _REPO_ROOT / "docs" / "benchmarks" / "last_scorecard.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cases = config.get("cases", [])
    results: list[dict] = []
    n_pass = n_fail = n_skip = 0

    print(f"\nRunning {len(cases)} benchmark case(s)...")
    for case in cases:
        name = case.get("name", "?")
        if verbose:
            print(f"\nCase: {name}")
        record = _run_case(case, verbose=verbose)
        results.append(record)
        if record["skipped"]:
            print(f"  SKIP  {name}: {record.get('skip_reason', '')}")
            n_skip += 1
        elif record["passed"]:
            print(f"  PASS  {name}: score={record['score']}, grade={record['grade']}")
            n_pass += 1
        else:
            print(f"  FAIL  {name}: score={record['score']}, grade={record['grade']}")
            for f in record["failures"]:
                print(f"        {f}")
            n_fail += 1

    required_fail = sum(
        1 for r in results
        if not r["passed"] and not r["skipped"] and not r.get("optional", False)
    )
    scorecard: dict[str, Any] = {
        "total": len(cases),
        "passed": n_pass,
        "failed": n_fail,
        "skipped": n_skip,
        "required_failures": required_fail,
        "all_required_pass": required_fail == 0,
        "cases": results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print(f"\nBenchmark: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"Scorecard: {output_path}")
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen process-tracing benchmark suite")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to benchmark_config.yaml (default: docs/benchmarks/benchmark_config.yaml)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path for scorecard JSON (default: docs/benchmarks/last_scorecard.json)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-case audit details")
    args = parser.parse_args()

    scorecard = run_benchmark(
        config_path=args.config,
        output_path=args.output,
        verbose=args.verbose,
    )
    sys.exit(0 if scorecard["all_required_pass"] else 1)


if __name__ == "__main__":
    main()
