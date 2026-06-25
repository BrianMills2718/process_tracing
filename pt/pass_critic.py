"""Pass 3.7: Structural critic — causal-graph quality review of likelihood claims.

Reads the causal extraction, hypothesis space, likelihood matrix, and diagnostic
matrix. Flags structural problems (confounds, missing pathways, void links,
too-strong claims, confirmed links) that should be corrected via re-elicitation
of Pass 3 — NOT by directly mutating likelihood values.

re_elicitation_needed is computed deterministically post-parse: any high-severity
finding triggers a Pass 3 re-run. The critic never writes to TestingResult or
BayesianResult directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import (
    CriticResult,
    DiagnosticMatrix,
    ExtractionResult,
    HypothesisSpace,
    TestingResult,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def run_critic(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    testing: TestingResult,
    diagnostic_matrix: DiagnosticMatrix,
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> CriticResult:
    """Review causal-graph structure and likelihood claims; return findings.

    The critic never modifies likelihood values. Any numeric change must be
    achieved by re-running Pass 3 with the critic summary injected as context.
    """
    if trace_id is None:
        trace_id = uuid4().hex[:8]
    kwargs: dict[str, Any] = {"model": model} if model else {}

    # Build compact likelihood matrix (justification + vectors, no raw LRs)
    lr_matrix = [
        {
            "evidence_id": el.evidence_id,
            "relevance": el.relevance,
            "justification": el.justification,
            "hypothesis_likelihoods": [
                {
                    "hypothesis_id": hl.hypothesis_id,
                    "relative_likelihood": hl.relative_likelihood,
                    "diagnostic_type": hl.diagnostic_type,
                }
                for hl in el.hypothesis_likelihoods
            ],
        }
        for el in testing.evidence_likelihoods
    ]

    # Build compact diagnostic matrix summary
    dm_summary = [
        {
            "h1_id": rpd.h1_id,
            "h2_id": rpd.h2_id,
            "discriminator_count": rpd.discriminator_count,
            "grade_capped": rpd.grade_capped,
        }
        for rpd in diagnostic_matrix.rival_pair_diagnostics
    ]

    messages = render_prompt(
        PROMPTS_DIR / "pass_critic.yaml",
        research_question=hypothesis_space.research_question,
        hypotheses_json=json.dumps(
            [
                {
                    "id": h.id,
                    "description": h.description,
                    "causal_mechanism": h.causal_mechanism,
                }
                for h in hypothesis_space.hypotheses
            ],
            indent=2,
        ),
        causal_edges_json=json.dumps(
            [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relationship": e.relationship,
                }
                for e in extraction.causal_edges
            ],
            indent=2,
        ),
        likelihood_matrix_json=json.dumps(lr_matrix, indent=2),
        diagnostic_matrix_json=json.dumps(dm_summary, indent=2),
    )

    result = call_llm(
        messages[0]["content"],
        CriticResult,
        task="process_tracing.critic",
        trace_id=trace_id,
        **kwargs,
    )
    n_high = sum(1 for f in result.findings if f.severity == "high")
    n_med = sum(1 for f in result.findings if f.severity == "medium")
    print(
        f"  Critic: {len(result.findings)} findings "
        f"({n_high} high, {n_med} medium), "
        f"re_elicitation_needed={result.re_elicitation_needed}"
    )
    return result
