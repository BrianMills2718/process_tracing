"""Pass 3: coherent likelihood-vector elicitation (heart of Van Evera).

One LLM call sees all hypotheses and all evidence and returns, per evidence item,
a *vector* of relative likelihoods across the hypotheses (on a shared scale) plus
a Van Evera diagnostic label per hypothesis. Eliciting the whole vector at once —
rather than independent per-hypothesis two-way ratios — is what keeps the
likelihoods coherent (every pairwise ratio is derived from one vector).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import ExtractionResult, HypothesisSpace, TestingResult

PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_test(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> TestingResult:
    """Elicit the evidence×hypothesis likelihood matrix in a single call."""
    if trace_id is None:
        trace_id = uuid4().hex[:8]

    hyps = hypothesis_space.hypotheses
    evidence = extraction.evidence

    brief_hypotheses = [
        {
            "id": h.id,
            "description": h.description,
            "causal_mechanism": h.causal_mechanism,
        }
        for h in hyps
    ]
    evidence_json = [
        {
            "id": e.id,
            "description": e.description,
            "source_text": e.source_text,
            "evidence_type": e.evidence_type,
            "approximate_date": e.approximate_date,
        }
        for e in evidence
    ]

    messages = render_prompt(
        PROMPTS_DIR / "pass3_test.yaml",
        hypotheses_json=json.dumps(brief_hypotheses, indent=2),
        evidence_json=json.dumps(evidence_json, indent=2),
        hypothesis_ids=json.dumps([h.id for h in hyps]),
    )
    kwargs: dict[str, Any] = {"model": model} if model else {}
    result = call_llm(
        messages[0]["content"],
        TestingResult,
        task="process_tracing.test",
        trace_id=trace_id,
        **kwargs,
    )

    # Coverage + balance report (advisory; bayesian treats gaps as neutral)
    n_items = len(result.evidence_likelihoods)
    print(f"  {n_items}/{len(evidence)} evidence items vectorized across {len(hyps)} hypotheses")
    incomplete = [
        item.evidence_id
        for item in result.evidence_likelihoods
        if len(item.hypothesis_likelihoods) < len(hyps)
    ]
    if incomplete:
        print(f"    WARNING: {len(incomplete)} items missing a likelihood for some hypothesis "
              f"(treated as uninformative): {incomplete[:5]}")
    return result
