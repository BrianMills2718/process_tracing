"""Pass 2.5: Hypothesis partition audit — flags overlap before testing.

Runs after hypothesis generation and before diagnostic testing. Checks each
rival pair for overlap, complementarity, and absorptive risk. Issues a loud
warning when partition quality is suboptimal but does not block execution —
the audit artifact is stored in result.json and partition.json for review.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import HypothesisSpace, PartitionAudit

PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_partition(
    hypothesis_space: HypothesisSpace,
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> PartitionAudit:
    """Evaluate hypothesis partition quality before Pass 3.

    Produces a PartitionAudit that records pairwise overlap/complementary/
    absorptive flags and an overall quality verdict. When quality is
    'needs_review', sets cap_applied=True and emits a UserWarning so the
    problem is visible without blocking the run.
    """
    if trace_id is None:
        trace_id = uuid4().hex[:8]

    messages = render_prompt(
        PROMPTS_DIR / "pass_partition.yaml",
        hypothesis_space_json=json.dumps(hypothesis_space.model_dump(), indent=2),
    )

    kwargs: dict = {"model": model} if model else {}
    audit = call_llm(
        messages[0]["content"],
        PartitionAudit,
        task="process_tracing.partition",
        trace_id=trace_id,
        **kwargs,
    )

    # Pipeline sets cap_applied deterministically; override whatever the LLM said.
    if audit.overall_quality == "needs_review":
        audit.cap_applied = True
        failed = [
            f"{p.h1_id}↔{p.h2_id}"
            for p in audit.rival_pairs
            if p.overlap_concern or p.complementary_concern or p.discriminator_count < 1
        ]
        detail = f" Problem pairs: {', '.join(failed)}." if failed else ""
        warnings.warn(
            f"Hypothesis partition needs review.{detail} "
            "Support scores may reflect overlap rather than genuine causal discrimination. "
            f"Summary: {audit.summary}",
            UserWarning,
            stacklevel=2,
        )

    return audit
