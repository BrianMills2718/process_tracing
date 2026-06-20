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
        research_question=hypothesis_space.research_question,
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

    # Fail loud: the matrix must be complete and exact — every evidence item present
    # once, and every item's vector covering exactly the hypothesis set (no missing,
    # duplicate, or unknown ids). Silently tolerating gaps would launder bad LLM
    # output as neutral evidence.
    expected_hyp_ids = {h.id for h in hyps}
    expected_ev_ids = {e.id for e in evidence}
    seen_ev_ids: list[str] = []
    for item in result.evidence_likelihoods:
        seen_ev_ids.append(item.evidence_id)
        hyp_ids = [hl.hypothesis_id for hl in item.hypothesis_likelihoods]
        if len(hyp_ids) != len(set(hyp_ids)):
            raise ValueError(
                f"testing: duplicate hypothesis ids in vector for evidence '{item.evidence_id}': {hyp_ids}"
            )
        if set(hyp_ids) != expected_hyp_ids:
            raise ValueError(
                f"testing: evidence '{item.evidence_id}' vector covers {sorted(set(hyp_ids))}, "
                f"expected exactly {sorted(expected_hyp_ids)}"
            )
    if len(seen_ev_ids) != len(set(seen_ev_ids)):
        raise ValueError("testing: duplicate evidence ids in likelihood vectors")
    if set(seen_ev_ids) != expected_ev_ids:
        missing = expected_ev_ids - set(seen_ev_ids)
        extra = set(seen_ev_ids) - expected_ev_ids
        raise ValueError(
            f"testing: evidence coverage mismatch — missing {sorted(missing)}, extra {sorted(extra)}"
        )

    # Validate dependence clusters: known evidence ids, >=2 members, no item in two clusters.
    clustered: set[str] = set()
    for cluster in result.dependence_clusters:
        ids = cluster.evidence_ids
        unknown = set(ids) - expected_ev_ids
        if unknown:
            raise ValueError(f"testing: dependence cluster references unknown evidence {sorted(unknown)}")
        if len(set(ids)) < 2:
            raise ValueError(f"testing: dependence cluster must have >=2 distinct members, got {ids}")
        overlap = clustered & set(ids)
        if overlap:
            raise ValueError(f"testing: evidence in multiple dependence clusters: {sorted(overlap)}")
        clustered |= set(ids)

    n_clusters = len(result.dependence_clusters)
    print(f"  {len(seen_ev_ids)}/{len(evidence)} evidence items vectorized across {len(hyps)} hypotheses"
          f" ({n_clusters} dependence clusters covering {len(clustered)} items)")
    return result
