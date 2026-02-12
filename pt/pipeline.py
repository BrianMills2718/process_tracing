"""Orchestrator: runs all passes sequentially."""

from __future__ import annotations

import time

from pt.bayesian import run_bayesian_update
from pt.pass_extract import run_extract
from pt.pass_hypothesize import run_hypothesize
from pt.pass_synthesize import run_synthesize
from pt.pass_test import run_test
from pt.schemas import ProcessTracingResult


def run_pipeline(
    text: str,
    *,
    model: str | None = None,
    verbose: bool = True,
) -> ProcessTracingResult:
    """Run the full process tracing pipeline.

    Pass 1: Extract → Pass 2: Hypothesize → Pass 3: Test → Bayes → Pass 4: Synthesize
    """
    t0 = time.time()

    if verbose:
        print("Pass 1/4: Extracting causal graph...")
    extraction = run_extract(text, model=model)
    if verbose:
        print(f"  Extracted {len(extraction.events)} events, {len(extraction.evidence)} evidence, "
              f"{len(extraction.hypotheses_in_text)} hypotheses")

    if verbose:
        print("Pass 2/4: Building hypothesis space...")
    hypothesis_space = run_hypothesize(extraction, model=model)
    if verbose:
        print(f"  {len(hypothesis_space.hypotheses)} hypotheses "
              f"(text + rivals), research question: {hypothesis_space.research_question[:80]}...")

    if verbose:
        print(f"Pass 3/4: Diagnostic testing ({len(hypothesis_space.hypotheses)} hypotheses)...")
    testing = run_test(extraction, hypothesis_space, model=model)
    if verbose:
        total_evals = sum(len(ht.evidence_evaluations) for ht in testing.hypothesis_tests)
        print(f"  {total_evals} evidence evaluations across all hypotheses")

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
    synthesis = run_synthesize(extraction, hypothesis_space, testing, bayesian, model=model)
    if verbose:
        print(f"  Narrative: {len(synthesis.analytical_narrative)} chars")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nPipeline complete in {elapsed:.1f}s")

    return ProcessTracingResult(
        extraction=extraction,
        hypothesis_space=hypothesis_space,
        testing=testing,
        bayesian=bayesian,
        synthesis=synthesis,
    )
