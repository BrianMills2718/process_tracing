"""Pass 1: Extract causal graph from text."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import ExtractionResult

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _ascii_id(raw: str) -> str:
    """Normalize an LLM-assigned id to an ASCII-safe identifier.

    Non-ASCII characters (e.g. 'é' in 'evi_levée_...') get mangled when ids round-trip
    through later LLM calls, which then fails the fail-loud id-matching in pass_test.
    Strip accents and restrict to [A-Za-z0-9_].
    """
    decomposed = unicodedata.normalize("NFKD", raw)
    ascii_str = decomposed.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", ascii_str).strip("_")
    return cleaned or "id"


def _sanitize_evidence_ids(extraction: ExtractionResult) -> ExtractionResult:
    """Make evidence ids ASCII-safe and unique so they survive later LLM round-trips."""
    seen: set[str] = set()
    for ev in extraction.evidence:
        new_id = _ascii_id(ev.id)
        if new_id in seen:
            suffix = 2
            while f"{new_id}_{suffix}" in seen:
                suffix += 1
            new_id = f"{new_id}_{suffix}"
        seen.add(new_id)
        ev.id = new_id
    return extraction


def run_extract(
    text: str,
    *,
    model: str | None = None,
    source_packet_context: str | None = None,
    trace_id: str | None = None,
) -> ExtractionResult:
    """Extract causal graph from text.

    Args:
        source_packet_context: Optional accepted source-packet contract text.
            When present, extraction must use it as source-scope guidance while
            still extracting evidence only from the input text itself.
    """
    if trace_id is None:
        trace_id = hashlib.sha256(text.encode()).hexdigest()[:8]
    if source_packet_context:
        source_packet_block = (
            "## Accepted source-packet contract\n\n"
            "The analyst supplied this source packet before extraction. Use it "
            "to preserve source coverage, source genre, observability limits, "
            "and exact text markers. Packet metadata is NOT evidence by itself: "
            "each extracted evidence item must still quote or closely paraphrase "
            "the input text below.\n\n"
            f"{source_packet_context.strip()}"
        )
    else:
        source_packet_block = ""
    messages = render_prompt(
        PROMPTS_DIR / "pass1_extract.yaml",
        text=text,
        source_packet_block=source_packet_block,
    )
    kwargs: dict[str, Any] = {"model": model} if model else {}
    result = call_llm(
        messages[0]["content"],
        ExtractionResult,
        task="process_tracing.extract",
        trace_id=trace_id,
        **kwargs,
    )
    return _sanitize_evidence_ids(result)
