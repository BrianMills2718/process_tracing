"""Pass 2: Build hypothesis space with rivals."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import ExtractionResult, HypothesisSpace

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Internal extraction-graph id tokens (scaffolding) that occasionally leak into the
# LLM's natural-language prose. They are formatting noise, not content, so we strip
# them deterministically after parsing (the prompt also forbids them, but the model
# is unreliable about it). This is cosmetic cleanup, NOT semantic interpretation.
_SCAFFOLD_ID = re.compile(r"(?:hyp|actor|evt|evi|mech)_[a-z0-9_]+", re.IGNORECASE)


def _strip_scaffold_ids(s: str) -> str:
    """Remove leaked extraction ids from prose, including parenthetical citations,
    then tidy the whitespace/punctuation left behind."""
    if not s:
        return s
    # Drop "(evt_x)" / "(evt_x, evi_y)" parentheticals wholesale.
    s = re.sub(r"\s*\((?:[^()]*?)?" + _SCAFFOLD_ID.pattern + r"(?:[^()]*?)?\)", "", s)
    # Drop any remaining bare id tokens.
    s = _SCAFFOLD_ID.sub("", s)
    # Tidy: collapse spaces, fix space-before-punctuation and empty parens.
    s = re.sub(r"\(\s*[,;]?\s*\)", "", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def run_hypothesize(
    extraction: ExtractionResult,
    *,
    model: str | None = None,
    theories: str | None = None,
    research_question: str | None = None,
    source_packet_context: str | None = None,
    trace_id: str | None = None,
) -> HypothesisSpace:
    """Build hypothesis space from extraction results.

    Args:
        theories: Optional plain-text description of theoretical frameworks
            to use as seeds for theory-derived hypotheses.
        research_question: Optional researcher-pinned research question. When
            provided, the LLM does NOT choose the outcome to explain — it must
            generate hypotheses for THIS exact question, and the returned
            research_question is overwritten with it verbatim for reproducibility.
            When None, the LLM selects the outcome (less reproducible across runs).
        source_packet_context: Optional source-packet contract text. When present,
            use it as the source-scope and observability design for hypothesis
            generation without treating it as extracted evidence.
    """
    rq = research_question.strip() if research_question else None
    if rq:
        research_question_block = (
            "## Pinned research question (use VERBATIM)\n\n"
            "The analyst has fixed the research question. Do NOT invent or rephrase it. "
            "Set `research_question` to exactly the text below, and generate hypotheses that "
            "all explain THIS outcome via different causal paths:\n\n"
            f"{rq}"
        )
    else:
        research_question_block = ""

    if source_packet_context:
        source_packet_block = (
            "## Source-packet contract\n\n"
            "The analyst supplied this source packet before inference. Use it to "
            "preserve source-scope limits, rival interpretations, observability "
            "expectations, and pre-specified tests. Do NOT treat the packet metadata "
            "as evidence unless the extracted text contains the trace itself. Do NOT "
            "claim source coverage beyond this packet.\n\n"
            f"{source_packet_context.strip()}"
        )
    else:
        source_packet_block = ""

    if theories:
        theories_block = (
            "## User-provided theoretical frameworks\n\n"
            "Generate at least one hypothesis from EACH of the following frameworks. "
            "These are provided by the analyst and take priority.\n\n"
            f"{theories}"
        )
    else:
        theories_block = ""

    if trace_id is None:
        trace_id = uuid4().hex[:8]
    messages = render_prompt(
        PROMPTS_DIR / "pass2_hypothesize.yaml",
        extraction_json=json.dumps(extraction.model_dump(), indent=2),
        theories_block=theories_block,
        research_question_block=research_question_block,
        source_packet_block=source_packet_block,
    )
    kwargs: dict[str, Any] = {"model": model} if model else {}
    space = call_llm(
        messages[0]["content"],
        HypothesisSpace,
        task="process_tracing.hypothesize",
        trace_id=trace_id,
        **kwargs,
    )
    # Guarantee reproducibility: when the analyst pins the question, the stored
    # research_question is exactly theirs regardless of any LLM paraphrase.
    if rq:
        space.research_question = rq

    # Strip any leaked extraction-graph ids from the human-facing prose fields.
    for h in space.hypotheses:
        h.description = _strip_scaffold_ids(h.description)
        h.theoretical_basis = _strip_scaffold_ids(h.theoretical_basis)
        h.causal_mechanism = _strip_scaffold_ids(h.causal_mechanism)
        for p in h.observable_predictions:
            p.description = _strip_scaffold_ids(p.description)
    return space
