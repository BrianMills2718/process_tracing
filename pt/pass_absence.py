"""Pass 3b: Absence-of-evidence evaluation (failed hoop tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
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


def _build_testing_summary(testing: TestingResult) -> dict:
    """Summarize which evidence was substantive, WITHOUT leaking hypothesis strength.

    The absence evaluator should know what evidence the testing pass found
    discriminating (so it can reason about what is genuinely missing) but not which
    hypothesis that evidence favored — that would invite confirmation bias.
    """
    substantive = sorted(
        {item.evidence_id for item in testing.evidence_likelihoods if item.relevance >= 0.6}
    )
    return {"substantive_evidence_ids": substantive}


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
    kwargs: dict[str, Any] = {"model": model} if model else {}
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
