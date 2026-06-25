"""Pass 3: coherent likelihood-vector elicitation (heart of Van Evera).

One LLM call sees all hypotheses and all evidence and returns, per evidence item,
a *vector* of relative likelihoods across the hypotheses (on a shared scale) plus
a Van Evera diagnostic label per hypothesis. Eliciting the whole vector at once —
rather than independent per-hypothesis two-way ratios — is what keeps the
likelihoods coherent (every pairwise ratio is derived from one vector).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from llm_client import render_prompt
from pydantic import BaseModel, Field, create_model

from pt.llm import call_llm
from pt.schemas import (
    DiagnosticType,
    ExtractionResult,
    HypothesisSpace,
    PredictionClassification,
    TestingResult,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_VALIDATION_REPAIRS = int(os.getenv("PT_TEST_VALIDATION_REPAIRS", "2"))


def _literal_enum(values: list[str]) -> Any:
    """Build a Literal type from runtime IDs for JSON-schema enum enforcement."""
    if not values:
        raise ValueError("cannot build enum schema from an empty id list")
    return Literal.__getitem__(tuple(values))


def _testing_response_model(
    *,
    evidence_ids: list[str],
    hypothesis_ids: list[str],
) -> type[BaseModel]:
    """Create an LLM-facing schema that constrains IDs to this run's contracts."""
    evidence_id_type = _literal_enum(evidence_ids)
    hypothesis_id_type = _literal_enum(hypothesis_ids)

    hypothesis_likelihood = create_model(
        "HypothesisLikelihoodResponse",
        hypothesis_id=(
            hypothesis_id_type,
            Field(description="One of the exact hypothesis ids from this run."),
        ),
        relative_likelihood=(
            float,
            Field(
                gt=0.0,
                allow_inf_nan=False,
                description="Finite positive relative likelihood on the shared evidence-item scale.",
            ),
        ),
        diagnostic_type=(
            DiagnosticType,
            Field(description="Van Evera diagnostic label for this evidence/hypothesis cell."),
        ),
    )
    hypothesis_likelihood_list: Any = list.__class_getitem__(hypothesis_likelihood)
    evidence_likelihood = create_model(
        "EvidenceLikelihoodResponse",
        evidence_id=(
            evidence_id_type,
            Field(description="One of the exact evidence ids from this extraction."),
        ),
        hypothesis_likelihoods=(
            hypothesis_likelihood_list,
            Field(description="Exactly one entry per hypothesis id from this run."),
        ),
        relevance=(
            float,
            Field(
                default=1.0,
                ge=0.0,
                le=1.0,
                description="How relevant/discriminating this evidence is.",
            ),
        ),
        justification=(
            str,
            Field(description="Why these relative likelihoods were assigned."),
        ),
    )
    evidence_id_list: Any = list.__class_getitem__(evidence_id_type)
    evidence_cluster = create_model(
        "EvidenceClusterResponse",
        evidence_ids=(
            evidence_id_list,
            Field(description="Two or more exact evidence ids from this extraction."),
        ),
        reason=(str, Field(description="Why these items are conditionally dependent.")),
        dependence_strength=(
            float,
            Field(
                default=1.0,
                ge=0.0,
                le=1.0,
                allow_inf_nan=False,
                description="How redundant the members are.",
            ),
        ),
    )
    evidence_likelihood_list: Any = list.__class_getitem__(evidence_likelihood)
    evidence_cluster_list: Any = list.__class_getitem__(evidence_cluster)
    return create_model(
        "TestingResponse",
        evidence_likelihoods=(
            evidence_likelihood_list,
            Field(description="Per-evidence likelihood vectors across all hypotheses."),
        ),
        dependence_clusters=(
            evidence_cluster_list,
            Field(
                default_factory=list,
                description="Groups of conditionally-dependent evidence items.",
            ),
        ),
        prediction_classifications=(
            list[PredictionClassification],
            Field(
                default_factory=list,
                description="Optional Van Evera classification of hypothesis predictions.",
            ),
        ),
    )


def _validate_testing_result(
    result: TestingResult,
    *,
    expected_hyp_ids: list[str],
    expected_ev_ids: list[str],
) -> set[str]:
    """Validate the deterministic matrix and dependence-cluster invariants."""

    expected_hyp_id_set = set(expected_hyp_ids)
    expected_ev_id_set = set(expected_ev_ids)
    seen_ev_ids: list[str] = []
    for item in result.evidence_likelihoods:
        seen_ev_ids.append(item.evidence_id)
        hyp_ids = [hl.hypothesis_id for hl in item.hypothesis_likelihoods]
        if len(hyp_ids) != len(set(hyp_ids)):
            raise ValueError(
                f"testing: duplicate hypothesis ids in vector for evidence '{item.evidence_id}': {hyp_ids}"
            )
        if set(hyp_ids) != expected_hyp_id_set:
            raise ValueError(
                f"testing: evidence '{item.evidence_id}' vector covers {sorted(set(hyp_ids))}, "
                f"expected exactly {sorted(expected_hyp_id_set)}"
            )
    if len(seen_ev_ids) != len(set(seen_ev_ids)):
        from collections import Counter

        duplicate_counts = {ev_id: count for ev_id, count in Counter(seen_ev_ids).items() if count > 1}
        raise ValueError(f"testing: duplicate evidence ids in likelihood vectors: {duplicate_counts}")
    if set(seen_ev_ids) != expected_ev_id_set:
        missing = expected_ev_id_set - set(seen_ev_ids)
        extra = set(seen_ev_ids) - expected_ev_id_set
        raise ValueError(
            f"testing: evidence coverage mismatch — missing {sorted(missing)}, extra {sorted(extra)}"
        )

    clustered: set[str] = set()
    for cluster in result.dependence_clusters:
        ids = cluster.evidence_ids
        unknown = set(ids) - expected_ev_id_set
        if unknown:
            raise ValueError(f"testing: dependence cluster references unknown evidence {sorted(unknown)}")
        if len(set(ids)) < 2:
            raise ValueError(f"testing: dependence cluster must have >=2 distinct members, got {ids}")
        overlap = clustered & set(ids)
        if overlap:
            raise ValueError(f"testing: evidence in multiple dependence clusters: {sorted(overlap)}")
        clustered |= set(ids)

    return clustered


def _repair_prompt(base_prompt: str, validation_error: str) -> str:
    """Append deterministic validation feedback for a complete Pass 3 retry."""

    return (
        f"{base_prompt}\n\n"
        "## Validation repair required\n\n"
        "Your previous response was valid JSON but failed deterministic "
        "process-tracing validation:\n\n"
        f"{validation_error}\n\n"
        "Return a COMPLETE corrected result for all evidence items. Do not omit "
        "any evidence likelihood vector. Dependence clusters must use only exact "
        "evidence ids from the input, each cluster must contain at least two "
        "distinct ids, and no evidence id may appear in more than one cluster. "
        "If an item could fit multiple overlapping clusters, choose the single "
        "most specific cluster for that item and leave it out of the others."
    )


def run_test(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    *,
    model: str | None = None,
    trace_id: str | None = None,
    critic_context: str | None = None,
) -> TestingResult:
    """Elicit the evidence×hypothesis likelihood matrix in a single call.

    critic_context: optional summary from Pass 3.7 structural critic to inject
    into the prompt when re-eliciting after a critic review.
    """
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
        critic_context=critic_context or "",
    )
    expected_hyp_ids = [h.id for h in hyps]
    expected_ev_ids = [e.id for e in evidence]
    response_model = _testing_response_model(
        evidence_ids=expected_ev_ids,
        hypothesis_ids=expected_hyp_ids,
    )
    kwargs: dict[str, Any] = {"model": model} if model else {}
    base_prompt = messages[0]["content"]
    prompt = base_prompt
    last_error: ValueError | None = None
    for attempt in range(MAX_VALIDATION_REPAIRS + 1):
        raw_result = call_llm(
            prompt,
            response_model,
            task="process_tracing.test",
            trace_id=trace_id if attempt == 0 else f"{trace_id}-repair{attempt}",
            **kwargs,
        )
        result = TestingResult.model_validate(raw_result.model_dump())
        try:
            clustered = _validate_testing_result(
                result,
                expected_hyp_ids=expected_hyp_ids,
                expected_ev_ids=expected_ev_ids,
            )
            break
        except ValueError as exc:
            last_error = exc
            if attempt >= MAX_VALIDATION_REPAIRS:
                raise
            print(f"  Pass 3 validation repair {attempt + 1}/{MAX_VALIDATION_REPAIRS}: {exc}")
            prompt = _repair_prompt(base_prompt, str(exc))
    else:  # pragma: no cover - loop exits by break or raise
        raise RuntimeError(f"testing validation failed without raising: {last_error}")

    n_clusters = len(result.dependence_clusters)
    print(f"  {len(result.evidence_likelihoods)}/{len(evidence)} evidence items vectorized across {len(hyps)} hypotheses"
          f" ({n_clusters} dependence clusters covering {len(clustered)} items)")
    return result
