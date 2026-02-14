"""Pass 1: Extract causal graph from text."""

from __future__ import annotations

from pathlib import Path

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import ExtractionResult

PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_extract(text: str, *, model: str | None = None) -> ExtractionResult:
    """Extract causal graph from text."""
    messages = render_prompt(PROMPTS_DIR / "pass1_extract.yaml", text=text)
    kwargs = {"model": model} if model else {}
    return call_llm(
        messages[0]["content"],
        ExtractionResult,
        **kwargs,
    )
