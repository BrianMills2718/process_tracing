"""Pass 5: Analytical refinement — second reading of source text."""

from __future__ import annotations

import json

from pt.llm import call_llm
from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    ExtractionResult,
    HypothesisSpace,
    RefinementResult,
    SynthesisResult,
)


PROMPT = """\
You are performing a SECOND READING of a source text for Van Evera process tracing.

A PhD analyst doesn't read a text once. After the initial analysis — extraction, hypothesis \
generation, diagnostic testing, Bayesian updating, absence evaluation, and synthesis — you go \
back to the source with fresh eyes. You now know which hypotheses emerged, which evidence was \
decisive, where sensitivity is fragile, and what absence findings revealed. This re-reading \
surfaces things the first pass couldn't because you didn't yet know what mattered.

## Your Six Tasks (priority order)

### 1. Missed Evidence
Passages that seemed unimportant initially but are now relevant given hypothesis competition \
and sensitivity fragility. You MUST quote the source text directly. Use IDs with `evi_ref_` \
prefix (e.g., `evi_ref_01`). Focus on evidence that would discriminate between the top \
hypotheses or address fragile posteriors.

### 2. Reinterpreted Evidence
Evidence items whose TYPE or DESCRIPTION should change in light of the full analysis. \
Each reinterpretation must produce a concrete change: either `new_type` differs from \
`original_type`, or `updated_description` is provided, or both. Do NOT use this for \
evidence whose diagnostic power shifted but whose type and description are still accurate — \
that's just normal Bayesian updating. Examples of genuine reinterpretations:
- Evidence coded as empirical that is actually interpretive (a historian's argument, not a fact)
- Evidence whose description misses its real significance for hypothesis discrimination
- Evidence type-coded as interpretive that contains verifiable empirical claims

### 3. New Causal Edges
Relationships the extraction missed but that the evidence pattern suggests exist. Must quote \
a supporting passage from the source text. Don't add edges that merely restate what's already \
captured.

### 4. Spurious Extractions
Evidence or causal edges the first pass extracted but that the full analysis reveals lack \
real textual support. Be CONSERVATIVE: don't remove evidence just because it's uninformative \
(LR near 1.0). Only remove items that are genuinely unsupported by the text or represent \
misreadings. For edges, use "source_id->target_id" format.

### 5. Hypothesis Refinements
Sharpen mechanisms, add predictions, or reframe hypotheses based on what the full analysis \
revealed. Types:
- `sharpen_mechanism`: Replace causal_mechanism with a more precise version
- `add_prediction`: Add new observable predictions informed by the analysis
- `reframe`: Replace both causal_mechanism and description
- `merge_suggestion`: Suggest merging two hypotheses (NOT auto-applied — advisory only)

### 6. Missing Mechanisms
Causal chains implied by the evidence pattern but not explicitly extracted. Must quote \
supporting text.

## Important Rules

- ALL new evidence must include a direct quote from the source text in `source_text`
- ALL new evidence IDs must use the `evi_ref_` prefix
- Be specific and grounded — no speculation beyond what the text supports
- Empty lists are fine — only report genuine findings, don't pad the output
- Focus your effort where it matters most: fragile posteriors, close rankings, and damaging \
absence findings

## Source Text
{source_text}

## First-Pass Extraction
{extraction_json}

## Hypothesis Space
{hypotheses_json}

## Bayesian Summary
{bayesian_summary}

## Absence-of-Evidence Findings
{absence_json}

## Synthesis
{synthesis_json}
"""


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
) -> RefinementResult:
    """Re-read source text with full first-pass context, return structured delta."""
    kwargs = {"model": model} if model else {}

    bayesian_summary = _build_bayesian_summary(bayesian, hypothesis_space, extraction)

    return call_llm(
        PROMPT.format(
            source_text=text,
            extraction_json=json.dumps(extraction.model_dump(), indent=2),
            hypotheses_json=json.dumps(hypothesis_space.model_dump(), indent=2),
            bayesian_summary=bayesian_summary,
            absence_json=json.dumps(absence.model_dump(), indent=2),
            synthesis_json=json.dumps(synthesis.model_dump(), indent=2),
        ),
        RefinementResult,
        **kwargs,
    )
