"""Pass 3: Diagnostic tests and evidence evaluation (heart of Van Evera)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import (
    Evidence,
    ExtractionResult,
    Hypothesis,
    HypothesisSpace,
    HypothesisTestResult,
    TestingResult,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _test_one_hypothesis(
    hypothesis: Hypothesis,
    evidence: list[Evidence],
    all_hypotheses: list[Hypothesis],
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> HypothesisTestResult:
    """Run diagnostic tests for a single hypothesis."""
    if trace_id is None:
        trace_id = uuid4().hex[:8]
    kwargs = {"model": model} if model else {}
    brief_hypotheses = [{"id": h.id, "description": h.description} for h in all_hypotheses if h.id != hypothesis.id]
    messages = render_prompt(
        PROMPTS_DIR / "pass3_test.yaml",
        hypothesis_json=json.dumps(hypothesis.model_dump(), indent=2),
        evidence_json=json.dumps([e.model_dump() for e in evidence], indent=2),
        all_hypotheses_brief=json.dumps(brief_hypotheses, indent=2),
    )
    return call_llm(
        messages[0]["content"],
        HypothesisTestResult,
        task=f"process_tracing.test.{hypothesis.id}",
        trace_id=trace_id,
        **kwargs,
    )


def run_test(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> TestingResult:
    """Run diagnostic tests for all hypotheses (one LLM call per hypothesis)."""
    if trace_id is None:
        trace_id = uuid4().hex[:8]
    results: list[HypothesisTestResult] = []
    evidence = extraction.evidence
    all_hypotheses = hypothesis_space.hypotheses

    for i, h in enumerate(all_hypotheses, 1):
        print(f"  Testing hypothesis {i}/{len(all_hypotheses)}: {h.id}")
        result = _test_one_hypothesis(h, evidence, all_hypotheses, model=model, trace_id=trace_id)
        result.hypothesis_id = h.id

        # Report coverage and balance
        n_evals = len(result.evidence_evaluations)
        n_for = sum(1 for ee in result.evidence_evaluations if ee.p_e_given_h > ee.p_e_given_not_h)
        n_against = sum(1 for ee in result.evidence_evaluations if ee.p_e_given_h < ee.p_e_given_not_h)
        n_neutral = n_evals - n_for - n_against
        print(f"    {n_evals}/{len(evidence)} evaluated | for={n_for} against={n_against} neutral={n_neutral}")
        if n_against == 0:
            print(f"    WARNING: zero disconfirming evidence — likely biased evaluation")

        results.append(result)

    return TestingResult(hypothesis_tests=results)
