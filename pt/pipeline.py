"""Orchestrator: runs all passes sequentially."""

from __future__ import annotations

import json
import os
import time
from typing import Callable
from uuid import uuid4

from pt.apply_refinement import apply_refinement
from pt.bayesian import run_bayesian_update
from pt.pass_absence import run_absence
from pt.pass_extract import run_extract
from pt.pass_hypothesize import run_hypothesize
from pt.pass_refine import run_refine
from pt.pass_synthesize import run_synthesize
from pt.pass_test import run_test
from pt.schemas import (
    ExtractionResult,
    HypothesisSpace,
    ProcessTracingResult,
    RefinementResult,
)


def _default_review(hypothesis_space: HypothesisSpace, output_dir: str | None) -> HypothesisSpace:
    """Interactive hypothesis review: display hypotheses, let user edit JSON."""
    print("\n" + "=" * 60)
    print("HYPOTHESIS REVIEW CHECKPOINT")
    print("=" * 60)
    print(f"\nResearch question: {hypothesis_space.research_question}\n")

    for h in hypothesis_space.hypotheses:
        print(f"  {h.id}: [{h.source}] {h.description}")
        print(f"       Mechanism: {h.causal_mechanism}")
        print(f"       Predictions: {len(h.observable_predictions)}")
        print()

    # Write hypotheses JSON for editing
    hyp_path = os.path.join(output_dir, "hypotheses.json") if output_dir else "hypotheses.json"
    with open(hyp_path, "w", encoding="utf-8") as f:
        json.dump(hypothesis_space.model_dump(), f, indent=2)
    print(f"Hypotheses written to: {hyp_path}")
    print("Edit the file to merge/split/modify hypotheses, then press Enter.")
    print("Or press Enter without editing to continue as-is.")
    print("=" * 60)

    input("\nPress Enter to continue...")

    # Reload (user may have edited)
    with open(hyp_path, "r", encoding="utf-8") as f:
        edited = json.load(f)
    edited_space = HypothesisSpace.model_validate(edited)

    if len(edited_space.hypotheses) != len(hypothesis_space.hypotheses):
        print(f"  Hypotheses changed: {len(hypothesis_space.hypotheses)} → {len(edited_space.hypotheses)}")
    else:
        print("  Hypotheses unchanged.")

    return edited_space


def _default_refine_review(refinement: RefinementResult, output_dir: str | None) -> RefinementResult:
    """Interactive refinement review: display delta summary, let user edit JSON."""
    print("\n" + "=" * 60)
    print("REFINEMENT REVIEW CHECKPOINT")
    print("=" * 60)

    print(f"\n  New evidence:       {len(refinement.new_evidence)}")
    print(f"  Reinterpretations:  {len(refinement.reinterpreted_evidence)}")
    print(f"  New causal edges:   {len(refinement.new_causal_edges)}")
    print(f"  Spurious removals:  {len(refinement.spurious_extractions)}")
    print(f"  Hypothesis refine:  {len(refinement.hypothesis_refinements)}")
    print(f"  Missing mechanisms: {len(refinement.missing_mechanisms)}")

    for ne in refinement.new_evidence:
        print(f"    + {ne.id}: {ne.description[:60]}")
    for se in refinement.spurious_extractions:
        print(f"    - {se.item_id} ({se.item_type}): {se.reason[:60]}")
    for hr in refinement.hypothesis_refinements:
        print(f"    ~ {hr.hypothesis_id} [{hr.refinement_type}]: {hr.description[:60]}")

    ref_path = os.path.join(output_dir, "refinement.json") if output_dir else "refinement.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(refinement.model_dump(), f, indent=2)
    print(f"\nRefinement written to: {ref_path}")
    print("Edit the file to modify the refinement delta, then press Enter.")
    print("Or press Enter without editing to continue as-is.")
    print("=" * 60)

    input("\nPress Enter to continue...")

    with open(ref_path, "r", encoding="utf-8") as f:
        edited = json.load(f)
    return RefinementResult.model_validate(edited)


def _run_passes_3_plus(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    text: str,
    *,
    model: str | None = None,
    verbose: bool = True,
    pass_label: str = "",
    trace_id: str | None = None,
) -> tuple:
    """Run passes 3, 3b, Bayesian, and 4. Returns (testing, absence, bayesian, synthesis)."""
    prefix = f"{pass_label} " if pass_label else ""

    if verbose:
        print(f"{prefix}Pass 3: Diagnostic testing ({len(hypothesis_space.hypotheses)} hypotheses)...")
    testing = run_test(extraction, hypothesis_space, model=model, trace_id=trace_id)
    if verbose:
        total_evals = sum(len(ht.evidence_evaluations) for ht in testing.hypothesis_tests)
        print(f"  {total_evals} evidence evaluations across all hypotheses")

    if verbose:
        print(f"{prefix}Pass 3b: Evaluating absence of evidence...")
    absence = run_absence(extraction, hypothesis_space, testing, model=model, trace_id=trace_id)
    if verbose:
        n_abs = len(absence.evaluations)
        n_damaging = sum(1 for a in absence.evaluations if a.severity == "damaging")
        print(f"  {n_abs} absence findings ({n_damaging} damaging)")

    if verbose:
        print(f"{prefix}Bayesian updating...")
    bayesian = run_bayesian_update(testing)
    if verbose:
        top = bayesian.ranking[0] if bayesian.ranking else "none"
        top_post = next(
            (p.final_posterior for p in bayesian.posteriors if p.hypothesis_id == top), 0
        )
        print(f"  Top hypothesis: {top} (posterior: {top_post:.3f})")

    if verbose:
        print(f"{prefix}Pass 4: Synthesizing analysis...")
    synthesis = run_synthesize(extraction, hypothesis_space, testing, bayesian, absence, model=model, trace_id=trace_id)
    if verbose:
        print(f"  Narrative: {len(synthesis.analytical_narrative)} chars")

    return testing, absence, bayesian, synthesis


def run_pipeline(
    text: str,
    *,
    model: str | None = None,
    verbose: bool = True,
    review: bool = False,
    review_fn: Callable[[HypothesisSpace, str | None], HypothesisSpace] | None = None,
    output_dir: str | None = None,
    theories: str | None = None,
    refine: bool = False,
    from_result: ProcessTracingResult | None = None,
    trace_id: str | None = None,
) -> ProcessTracingResult:
    """Run the full process tracing pipeline.

    Pass 1: Extract → Pass 2: Hypothesize → [Review] → Pass 3: Test → 3b: Absence → Bayes → Pass 4: Synthesize
    With --refine: → Pass 5: Refine → [Review] → Apply → Re-run passes 3-4

    Args:
        review: If True, pause after hypothesis generation (and after refinement) for user review.
        review_fn: Custom review function. Defaults to interactive CLI review.
        output_dir: Directory for writing review files.
        theories: Optional plain-text theoretical frameworks for hypothesis generation.
        refine: If True, run analytical refinement after initial pipeline, then re-run passes 3+.
        from_result: Load extraction + hypothesis_space from existing result, skip passes 1-2. Implies refine.
    """
    t0 = time.time()
    if trace_id is None:
        trace_id = uuid4().hex[:8]

    # Input validation — catch garbage/trivial input before burning 9+ LLM calls
    if from_result is None:
        word_count = len(text.split())
        if word_count < 300:
            raise ValueError(
                f"Input text too short ({word_count} words). "
                f"Process tracing requires at least 300 words of substantive text "
                f"to extract meaningful evidence and hypotheses."
            )

    if from_result is not None:
        refine = True
        extraction = from_result.extraction
        hypothesis_space = from_result.hypothesis_space
        if verbose:
            print(f"Loaded from existing result: {len(extraction.evidence)} evidence, "
                  f"{len(hypothesis_space.hypotheses)} hypotheses")
    else:
        if verbose:
            print("Pass 1/4: Extracting causal graph...")
        extraction = run_extract(text, model=model, trace_id=trace_id)
        if verbose:
            print(f"  Extracted {len(extraction.events)} events, {len(extraction.evidence)} evidence, "
                  f"{len(extraction.hypotheses_in_text)} hypotheses")

        if verbose:
            extra = " (with user theories)" if theories else ""
            print(f"Pass 2/4: Building hypothesis space{extra}...")
        hypothesis_space = run_hypothesize(extraction, model=model, theories=theories, trace_id=trace_id)
        if verbose:
            print(f"  {len(hypothesis_space.hypotheses)} hypotheses "
                  f"(text + rivals), research question: {hypothesis_space.research_question[:80]}...")

        # Optional human review checkpoint
        if review:
            fn = review_fn or _default_review
            hypothesis_space = fn(hypothesis_space, output_dir)

    # Run passes 3-4 (initial)
    testing, absence, bayesian, synthesis = _run_passes_3_plus(
        extraction, hypothesis_space, text, model=model, verbose=verbose, trace_id=trace_id,
    )

    refinement_result = None

    if refine:
        if verbose:
            print("\nPass 5: Analytical refinement (second reading)...")
        refinement_result = run_refine(
            text, extraction, hypothesis_space, bayesian, absence, synthesis, model=model, trace_id=trace_id,
        )
        if verbose:
            n_new = len(refinement_result.new_evidence)
            n_reint = len(refinement_result.reinterpreted_evidence)
            n_spur = len(refinement_result.spurious_extractions)
            n_refine = len(refinement_result.hypothesis_refinements)
            print(f"  {n_new} new evidence, {n_reint} reinterpretations, "
                  f"{n_spur} removals, {n_refine} hypothesis refinements")

        # Write refinement.json audit file before applying
        if output_dir:
            ref_path = os.path.join(output_dir, "refinement.json")
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(refinement_result.model_dump(), f, indent=2)
            if verbose:
                print(f"  Refinement audit: {ref_path}")

        # Optional review of refinement delta
        if review:
            refinement_result = _default_refine_review(refinement_result, output_dir)

        # Apply delta
        if verbose:
            print("Applying refinement delta...")
        extraction, hypothesis_space = apply_refinement(
            extraction, hypothesis_space, refinement_result, verbose=verbose,
        )

        # Re-run passes 3-4 with updated data
        if verbose:
            print("\nRe-running passes 3-4 with refined data...")
        testing, absence, bayesian, synthesis = _run_passes_3_plus(
            extraction, hypothesis_space, text, model=model, verbose=verbose,
            pass_label="[Refined]", trace_id=trace_id,
        )

    elapsed = time.time() - t0
    if verbose:
        print(f"\nPipeline complete in {elapsed:.1f}s")

    return ProcessTracingResult(
        extraction=extraction,
        hypothesis_space=hypothesis_space,
        testing=testing,
        absence=absence,
        bayesian=bayesian,
        synthesis=synthesis,
        refinement=refinement_result,
        is_refined=refine,
    )
