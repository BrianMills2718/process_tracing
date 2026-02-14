"""Pass 2: Build hypothesis space with rivals."""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import ExtractionResult, HypothesisSpace

PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_hypothesize(
    extraction: ExtractionResult,
    *,
    model: str | None = None,
    theories: str | None = None,
) -> HypothesisSpace:
    """Build hypothesis space from extraction results.

    Args:
        theories: Optional plain-text description of theoretical frameworks
            to use as seeds for theory-derived hypotheses.
    """
    if theories:
        theories_block = (
            "## User-provided theoretical frameworks\n\n"
            "Generate at least one hypothesis from EACH of the following frameworks. "
            "These are provided by the analyst and take priority.\n\n"
            f"{theories}"
        )
    else:
        theories_block = ""

    messages = render_prompt(
        PROMPTS_DIR / "pass2_hypothesize.yaml",
        extraction_json=json.dumps(extraction.model_dump(), indent=2),
        theories_block=theories_block,
    )
    kwargs = {"model": model} if model else {}
    return call_llm(
        messages[0]["content"],
        HypothesisSpace,
        **kwargs,
    )
