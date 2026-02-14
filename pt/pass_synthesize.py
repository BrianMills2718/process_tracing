"""Pass 4: Written synthesis and verdicts."""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    ExtractionResult,
    HypothesisSpace,
    SynthesisResult,
    TestingResult,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_synthesize(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    testing: TestingResult,
    bayesian: BayesianResult,
    absence: AbsenceResult,
    *,
    model: str | None = None,
) -> SynthesisResult:
    """Generate final synthesis from all pipeline results."""
    kwargs = {"model": model} if model else {}
    messages = render_prompt(
        PROMPTS_DIR / "pass4_synthesize.yaml",
        extraction_json=json.dumps(extraction.model_dump(), indent=2),
        hypotheses_json=json.dumps(hypothesis_space.model_dump(), indent=2),
        testing_json=json.dumps(testing.model_dump(), indent=2),
        bayesian_json=json.dumps(bayesian.model_dump(), indent=2),
        absence_json=json.dumps(absence.model_dump(), indent=2),
    )
    return call_llm(
        messages[0]["content"],
        SynthesisResult,
        **kwargs,
    )
