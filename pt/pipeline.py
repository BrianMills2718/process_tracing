"""Orchestrator: runs all passes sequentially."""

from __future__ import annotations

import json
import os
import time
from typing import Callable

from pt.bayesian import run_bayesian_update
from pt.pass_absence import run_absence
from pt.pass_extract import run_extract
from pt.pass_hypothesize import run_hypothesize
from pt.pass_synthesize import run_synthesize
from pt.pass_test import run_test
from pt.schemas import HypothesisSpace, ProcessTracingResult


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


def run_pipeline(
    text: str,
    *,
    model: str | None = None,
    verbose: bool = True,
    review: bool = False,
    review_fn: Callable[[HypothesisSpace, str | None], HypothesisSpace] | None = None,
    output_dir: str | None = None,
    theories: str | None = None,
) -> ProcessTracingResult:
    """Run the full process tracing pipeline.

    Pass 1: Extract → Pass 2: Hypothesize → [Review] → Pass 3: Test → Bayes → Pass 4: Synthesize

    Args:
        review: If True, pause after hypothesis generation for user review.
        review_fn: Custom review function. Defaults to interactive CLI review.
        output_dir: Directory for writing review files.
        theories: Optional plain-text theoretical frameworks for hypothesis generation.
    """
    t0 = time.time()

    if verbose:
        print("Pass 1/4: Extracting causal graph...")
    extraction = run_extract(text, model=model)
    if verbose:
        print(f"  Extracted {len(extraction.events)} events, {len(extraction.evidence)} evidence, "
              f"{len(extraction.hypotheses_in_text)} hypotheses")

    if verbose:
        extra = " (with user theories)" if theories else ""
        print(f"Pass 2/4: Building hypothesis space{extra}...")
    hypothesis_space = run_hypothesize(extraction, model=model, theories=theories)
    if verbose:
        print(f"  {len(hypothesis_space.hypotheses)} hypotheses "
              f"(text + rivals), research question: {hypothesis_space.research_question[:80]}...")

    # Optional human review checkpoint
    if review:
        fn = review_fn or _default_review
        hypothesis_space = fn(hypothesis_space, output_dir)

    if verbose:
        print(f"Pass 3/4: Diagnostic testing ({len(hypothesis_space.hypotheses)} hypotheses)...")
    testing = run_test(extraction, hypothesis_space, model=model)
    if verbose:
        total_evals = sum(len(ht.evidence_evaluations) for ht in testing.hypothesis_tests)
        print(f"  {total_evals} evidence evaluations across all hypotheses")

    if verbose:
        print("Pass 3b: Evaluating absence of evidence...")
    absence = run_absence(extraction, hypothesis_space, testing, model=model)
    if verbose:
        n_abs = len(absence.evaluations)
        n_damaging = sum(1 for a in absence.evaluations if a.severity == "damaging")
        print(f"  {n_abs} absence findings ({n_damaging} damaging)")

    if verbose:
        print("Bayesian updating...")
    bayesian = run_bayesian_update(testing)
    if verbose:
        top = bayesian.ranking[0] if bayesian.ranking else "none"
        top_post = next(
            (p.final_posterior for p in bayesian.posteriors if p.hypothesis_id == top), 0
        )
        print(f"  Top hypothesis: {top} (posterior: {top_post:.3f})")

    if verbose:
        print("Pass 4/4: Synthesizing analysis...")
    synthesis = run_synthesize(extraction, hypothesis_space, testing, bayesian, absence, model=model)
    if verbose:
        print(f"  Narrative: {len(synthesis.analytical_narrative)} chars")

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
    )
