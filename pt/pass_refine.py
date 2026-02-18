"""Pass 5: Analytical refinement — second reading of source text."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from llm_client import render_prompt

from pt.llm import call_llm
from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    ExtractionResult,
    HypothesisSpace,
    RefinementResult,
    SynthesisResult,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _build_bayesian_summary(
    bayesian: BayesianResult,
    hypothesis_space: HypothesisSpace,
    extraction: ExtractionResult,
) -> str:
    """Build a condensed Bayesian summary (~1.5K tokens) for the refinement prompt.

    Includes: posteriors, rankings, robustness, top drivers, evidence balance,
    and sensitivity ranges. Avoids sending the full 34K-token TestingResult.
    """
    h_map = {h.id: h for h in hypothesis_space.hypotheses}
    ev_map = {e.id: e for e in extraction.evidence}

    entries = []
    for p in bayesian.posteriors:
        h = h_map.get(p.hypothesis_id)
        h_desc = h.description[:80] if h else p.hypothesis_id

        # Count supporting vs opposing evidence
        n_supporting = sum(1 for u in p.updates if u.likelihood_ratio > 1.1)
        n_opposing = sum(1 for u in p.updates if u.likelihood_ratio < 0.9)

        # Top drivers with descriptions
        drivers = []
        for did in p.top_drivers[:3]:
            ev = ev_map.get(did)
            lr_val = next((u.likelihood_ratio for u in p.updates if u.evidence_id == did), None)
            desc = ev.description[:60] if ev else did
            lr_str = f" (LR={lr_val:.2f})" if lr_val is not None else ""
            drivers.append(f"{did}: {desc}{lr_str}")

        # Sensitivity
        sens = next((s for s in bayesian.sensitivity if s.hypothesis_id == p.hypothesis_id), None)
        sens_str = ""
        if sens:
            sens_str = (
                f"sensitivity=[{sens.posterior_low:.3f}, {sens.posterior_high:.3f}], "
                f"rank_stable={sens.rank_stable}"
            )

        entry = {
            "hypothesis_id": p.hypothesis_id,
            "description": h_desc,
            "prior": round(p.prior, 3),
            "posterior": round(p.final_posterior, 3),
            "robustness": p.robustness,
            "n_supporting": n_supporting,
            "n_opposing": n_opposing,
            "top_drivers": drivers,
        }
        if sens_str:
            entry["sensitivity"] = sens_str

        entries.append(entry)

    return json.dumps(
        {"ranking": bayesian.ranking, "posteriors": entries},
        indent=2,
    )


def run_refine(
    text: str,
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    bayesian: BayesianResult,
    absence: AbsenceResult,
    synthesis: SynthesisResult,
    *,
    model: str | None = None,
    trace_id: str | None = None,
) -> RefinementResult:
    """Re-read source text with full first-pass context, return structured delta."""
    if trace_id is None:
        trace_id = uuid4().hex[:8]
    kwargs = {"model": model} if model else {}

    bayesian_summary = _build_bayesian_summary(bayesian, hypothesis_space, extraction)

    messages = render_prompt(
        PROMPTS_DIR / "pass5_refine.yaml",
        source_text=text,
        extraction_json=json.dumps(extraction.model_dump(), indent=2),
        hypotheses_json=json.dumps(hypothesis_space.model_dump(), indent=2),
        bayesian_summary=bayesian_summary,
        absence_json=json.dumps(absence.model_dump(), indent=2),
        synthesis_json=json.dumps(synthesis.model_dump(), indent=2),
    )
    return call_llm(
        messages[0]["content"],
        RefinementResult,
        task="process_tracing.refine",
        trace_id=trace_id,
        **kwargs,
    )
