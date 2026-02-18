"""Pass 3b: Absence-of-evidence evaluation (failed hoop tests)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import (
    AbsenceResult,
    ExtractionResult,
    HypothesisSpace,
    TestingResult,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _build_testing_summary(testing: TestingResult) -> list[dict]:
    """Summarize which predictions had relevant evidence evaluations.

    Deliberately omits hypothesis-level support/opposition counts to avoid
    leaking hypothesis strength to the absence evaluator (prevents confirmation bias).
    """
    summary = []
    for ht in testing.hypothesis_tests:
        pred_hits: dict[str, int] = {}
        for ev in ht.evidence_evaluations:
            if ev.prediction_id and ev.relevance >= 0.6:
                pred_hits[ev.prediction_id] = pred_hits.get(ev.prediction_id, 0) + 1
        summary.append({
            "hypothesis_id": ht.hypothesis_id,
            "predictions_with_relevant_evidence": pred_hits,
        })
    return summary


def run_absence(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    testing: TestingResult,
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> AbsenceResult:
    """Evaluate absence of evidence for all hypotheses."""
    if trace_id is None:
        trace_id = uuid4().hex[:8]
    kwargs = {"model": model} if model else {}
    testing_summary = _build_testing_summary(testing)

    messages = render_prompt(
        PROMPTS_DIR / "pass3b_absence.yaml",
        text_summary=extraction.summary,
        evidence_json=json.dumps(
            [e.model_dump() for e in extraction.evidence], indent=2
        ),
        hypotheses_json=json.dumps(
            hypothesis_space.model_dump()["hypotheses"], indent=2
        ),
        testing_summary_json=json.dumps(testing_summary, indent=2),
    )
    return call_llm(
        messages[0]["content"],
        AbsenceResult,
        task="process_tracing.absence",
        trace_id=trace_id,
        **kwargs,
    )
