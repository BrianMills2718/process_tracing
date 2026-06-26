"""Orchestrator: runs all passes sequentially."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable
from uuid import uuid4

from pt.apply_refinement import apply_refinement
from pt.bayesian import INTERPRETIVE_LR_CAP, run_bayesian_update
from pt.pass_absence import run_absence
from pt.pass_critic import run_critic
from pt.pass_extract import run_extract
from pt.pass_hypothesize import run_hypothesize
from pt.pass_diagnostic import compute_diagnostic_matrix
from pt.pass_partition import run_partition
from pt.pass_refine import run_refine
from pt.pass_synthesize import run_synthesize
from pt.pass_test import run_test
from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    CriticDelta,
    CriticResult,
    DiagnosticMatrix,
    ExtractionResult,
    HypothesisSpace,
    PartitionAudit,
    ProcessTracingResult,
    RefinementResult,
    TestingResult,
)
from pt.source_coverage import build_source_coverage
from pt.source_packet import SourcePacket


def _source_text_sha256(text: str) -> str:
    """Hash the exact source text used for the analysis."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_from_result_source(text: str, from_result: ProcessTracingResult) -> str:
    """Return current source hash or raise when cached analysis provenance differs."""
    current_hash = _source_text_sha256(text)
    cached_hash = from_result.source_text_sha256
    if cached_hash is None:
        raise ValueError(
            "--from-result file has no source_text_sha256 provenance; regenerate "
            "the result with the current code before refining from it"
        )
    if cached_hash != current_hash:
        raise ValueError(
            "--from-result source_text_sha256 does not match the input text "
            f"(result={cached_hash}, input={current_hash})"
        )
    return current_hash


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


def _run_core_passes(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    *,
    model: str | None = None,
    verbose: bool = True,
    pass_label: str = "",
    trace_id: str | None = None,
    priors: dict[str, float] | None = None,
    critic_context: str | None = None,
) -> tuple[TestingResult, AbsenceResult, BayesianResult, DiagnosticMatrix]:
    """Run passes 3, 3b, Bayesian update, and 3.6 (diagnostic matrix).

    Returns (testing, absence, bayesian, diagnostic_matrix) without synthesis.
    critic_context: optional critic summary to inject into Pass 3 re-elicitation.
    """
    prefix = f"{pass_label} " if pass_label else ""

    if verbose:
        ctx_note = " [with critic context]" if critic_context else ""
        print(f"{prefix}Pass 3: Diagnostic testing ({len(hypothesis_space.hypotheses)} hypotheses){ctx_note}...")
    testing = run_test(
        extraction, hypothesis_space, model=model, trace_id=trace_id, critic_context=critic_context
    )
    if verbose:
        print(f"  {len(testing.evidence_likelihoods)} evidence likelihood vectors "
              f"across {len(hypothesis_space.hypotheses)} hypotheses")

    if verbose:
        print(f"{prefix}Pass 3b: Evaluating absence of evidence...")
    absence = run_absence(extraction, hypothesis_space, testing, model=model, trace_id=trace_id)
    if verbose:
        n_abs = len(absence.evaluations)
        n_damaging = sum(1 for a in absence.evaluations if a.severity == "damaging")
        print(f"  {n_abs} absence findings ({n_damaging} damaging)")

    if verbose:
        print(f"{prefix}Bayesian updating...")
    # Interpretive (scholarly-claim) evidence gets a tighter pairwise cap so it
    # can't move odds as hard as direct empirical evidence (prompt Rule F, enforced).
    interpretive_caps = {
        ev.id: INTERPRETIVE_LR_CAP
        for ev in extraction.evidence
        if ev.evidence_type == "interpretive"
    }
    bayesian = run_bayesian_update(
        testing, [h.id for h in hypothesis_space.hypotheses], priors=priors,
        include_residual=True, caps=interpretive_caps,
    )
    if verbose:
        top = bayesian.ranking[0] if bayesian.ranking else "none"
        top_post = next(
            (p.final_posterior for p in bayesian.posteriors if p.hypothesis_id == top), 0
        )
        print(f"  Top hypothesis: {top} (support: {top_post:.3f})")

    interpretive_ids = {ev.id for ev in extraction.evidence if ev.evidence_type == "interpretive"}
    diagnostic_matrix = compute_diagnostic_matrix(testing, hypothesis_space, interpretive_ids)
    if verbose:
        n_pairs = len(diagnostic_matrix.rival_pair_diagnostics)
        n_capped = len(diagnostic_matrix.pairs_without_discriminators)
        cap_note = f" [{n_capped} pairs capped — no discriminators]" if n_capped else ""
        print(f"  Diagnostic matrix: {n_pairs} rival pairs{cap_note}")

    return testing, absence, bayesian, diagnostic_matrix


def _run_passes_3_plus(
    extraction: ExtractionResult,
    hypothesis_space: HypothesisSpace,
    text: str,
    *,
    model: str | None = None,
    verbose: bool = True,
    pass_label: str = "",
    trace_id: str | None = None,
    priors: dict[str, float] | None = None,
) -> tuple:
    """Run passes 3, 3b, Bayesian, 3.6, and 4. Returns (testing, absence, bayesian, synthesis, diagnostic_matrix)."""
    prefix = f"{pass_label} " if pass_label else ""

    testing, absence, bayesian, diagnostic_matrix = _run_core_passes(
        extraction, hypothesis_space,
        model=model, verbose=verbose, pass_label=pass_label, trace_id=trace_id, priors=priors,
    )

    if verbose:
        print(f"{prefix}Pass 4: Synthesizing analysis...")
    synthesis = run_synthesize(
        extraction, hypothesis_space, testing, bayesian, absence, model=model, trace_id=trace_id
    )
    if verbose:
        print(f"  Narrative: {len(synthesis.analytical_narrative)} chars")

    return testing, absence, bayesian, synthesis, diagnostic_matrix


def _compute_critic_delta(
    base_bayesian: BayesianResult,
    critic_bayesian: BayesianResult,
    critic_result: CriticResult,
) -> list[CriticDelta]:
    """Compute per-hypothesis posterior change between base and critic runs."""
    base_map = {p.hypothesis_id: p for p in base_bayesian.posteriors}
    critic_map = {p.hypothesis_id: p for p in critic_bayesian.posteriors}

    # Which evidence IDs does each critic finding target?
    ev_targets_by_hyp: dict[str, set[str]] = {}
    for f in critic_result.findings:
        if f.target_type == "evidence":
            # Evidence-level findings affect all hypotheses
            for h in base_map:
                ev_targets_by_hyp.setdefault(h, set()).add(f.target)

    deltas = []
    all_hyp_ids = sorted(set(base_map) | set(critic_map))
    for hyp_id in all_hyp_ids:
        base_p = base_map.get(hyp_id)
        critic_p = critic_map.get(hyp_id)
        post_base = base_p.final_posterior if base_p else 0.0
        post_critic = critic_p.final_posterior if critic_p else 0.0

        # Top-driver change: IDs in one set but not the other
        base_drivers = set(base_p.top_drivers) if base_p else set()
        critic_drivers = set(critic_p.top_drivers) if critic_p else set()
        driver_changes = (
            [f"added:{eid}" for eid in sorted(critic_drivers - base_drivers)]
            + [f"removed:{eid}" for eid in sorted(base_drivers - critic_drivers)]
        )

        # Count critic findings that target this hypothesis or its evidence top-drivers.
        # causal_edge findings are graph-level and not attributed to a specific hypothesis.
        hyp_finding_count = sum(
            1 for f in critic_result.findings
            if (
                (f.target_type == "hypothesis" and f.target == hyp_id)
                or (f.target_type == "evidence" and f.target in (base_drivers | critic_drivers))
            )
        )

        deltas.append(CriticDelta(
            hypothesis_id=hyp_id,
            posterior_base=round(post_base, 6),
            posterior_critic=round(post_critic, 6),
            delta=round(post_critic - post_base, 6),
            top_driver_change=driver_changes,
            critic_findings_count=hyp_finding_count,
        ))
    return deltas


def run_pipeline(
    text: str,
    *,
    model: str | None = None,
    verbose: bool = True,
    review: bool = False,
    review_fn: Callable[[HypothesisSpace, str | None], HypothesisSpace] | None = None,
    output_dir: str | None = None,
    theories: str | None = None,
    research_question: str | None = None,
    refine: bool = False,
    from_result: ProcessTracingResult | None = None,
    source_packet: SourcePacket | None = None,
    source_packet_path: str | None = None,
    trace_id: str | None = None,
    priors: dict[str, float] | None = None,
    critic: bool = False,
) -> ProcessTracingResult:
    """Run the full process tracing pipeline.

    Pass 1: Extract → Pass 2: Hypothesize → [Review] → Pass 3: Test → 3b: Absence → Bayes → 3.6 →
    [Pass 3.7: Critic] → Pass 4: Synthesize
    With --refine: → Pass 5: Refine → [Review] → Apply → Re-run passes 3-4

    Args:
        review: If True, pause after hypothesis generation (and after refinement) for user review.
        review_fn: Custom review function. Defaults to interactive CLI review.
        output_dir: Directory for writing review files.
        theories: Optional plain-text theoretical frameworks for hypothesis generation.
        research_question: Optional researcher-pinned research question. Pins the outcome to
            explain (reproducible across runs); when None the LLM selects it.
        refine: If True, run analytical refinement after initial pipeline, then re-run passes 3+.
        from_result: Load extraction + hypothesis_space from existing result, skip passes 1-2. Implies refine.
        source_packet: Optional source-packet contract that pins source scope,
            observability assumptions, and the research question before inference.
        critic: If True, run structural critic (Pass 3.7) after diagnostic matrix and before
            synthesis. Writes result_base.json, result_critic.json, and critic_delta.json to
            output_dir. Re-elicits Pass 3 when high-severity findings are present.
    """
    t0 = time.time()
    if trace_id is None:
        trace_id = uuid4().hex[:8]
    source_text_sha256 = _source_text_sha256(text)

    if from_result is not None and source_packet is not None:
        raise ValueError(
            "--source-packet cannot be combined with --from-result because "
            "--from-result reuses an existing hypothesis space"
        )

    if source_packet is not None:
        packet_rq = source_packet.research_question.strip()
        pinned_rq = research_question.strip() if research_question else None
        if pinned_rq and pinned_rq != packet_rq:
            raise ValueError(
                "--research-question conflicts with source_packet.research_question "
                f"(research_question={pinned_rq!r}, source_packet={packet_rq!r})"
            )
        research_question = packet_rq
        if verbose:
            print(
                f"Source packet: {source_packet.case_name} "
                f"({len(source_packet.source_candidates)} sources, "
                f"{len(source_packet.known_gaps)} known gaps)"
            )

    source_packet_summary = (
        source_packet.to_summary(source_packet_path)
        if source_packet is not None
        else from_result.source_packet if from_result is not None else None
    )
    source_coverage = from_result.source_coverage if from_result is not None else None

    # Input validation — catch garbage/trivial input before burning 9+ LLM calls
    if from_result is None:
        word_count = len(text.split())
        if word_count < 300:
            raise ValueError(
                f"Input text too short ({word_count} words). "
                f"Process tracing requires at least 300 words of substantive text "
                f"to extract meaningful evidence and hypotheses."
            )

    partition_audit: PartitionAudit | None = None

    if from_result is not None:
        refine = True
        source_text_sha256 = _validate_from_result_source(text, from_result)
        extraction = from_result.extraction
        hypothesis_space = from_result.hypothesis_space
        partition_audit = from_result.partition_audit
        if verbose:
            print(f"Loaded from existing result: {len(extraction.evidence)} evidence, "
                  f"{len(hypothesis_space.hypotheses)} hypotheses")
    else:
        if verbose:
            print("Pass 1/4: Extracting causal graph...")
        extraction = run_extract(
            text,
            model=model,
            source_packet_context=source_packet.to_prompt_context() if source_packet else None,
            trace_id=trace_id,
        )
        if verbose:
            print(f"  Extracted {len(extraction.events)} events, {len(extraction.evidence)} evidence, "
                  f"{len(extraction.hypotheses_in_text)} hypotheses")

        if verbose:
            extra = " (with user theories)" if theories else ""
            print(f"Pass 2/4: Building hypothesis space{extra}...")
        hypothesis_space = run_hypothesize(
            extraction, model=model, theories=theories,
            research_question=research_question,
            source_packet_context=source_packet.to_prompt_context() if source_packet else None,
            trace_id=trace_id,
        )
        if verbose:
            print(f"  {len(hypothesis_space.hypotheses)} hypotheses "
                  f"(text + rivals), research question: {hypothesis_space.research_question[:80]}...")

        # Optional human review checkpoint
        if review:
            fn = review_fn or _default_review
            hypothesis_space = fn(hypothesis_space, output_dir)

        if verbose:
            print("Pass 2.5: Hypothesis partition audit...")
        partition_audit = run_partition(hypothesis_space, model=model, trace_id=trace_id)
        if verbose:
            quality = partition_audit.overall_quality
            n_pairs = len(partition_audit.rival_pairs)
            flagged = len(partition_audit.hypotheses_flagged)
            cap = " [CAP APPLIED]" if partition_audit.cap_applied else ""
            print(f"  {n_pairs} rival pairs, quality={quality}{', '+str(flagged)+' flagged' if flagged else ''}{cap}")

        if output_dir:
            partition_path = os.path.join(output_dir, "partition.json")
            with open(partition_path, "w", encoding="utf-8") as f:
                json.dump(partition_audit.model_dump(), f, indent=2)
            if verbose:
                print(f"  Partition audit: {partition_path}")

    # Run passes 3-4 (initial)
    critic_result: CriticResult | None = None

    if critic:
        # Run core passes (3, 3b, Bayesian, 3.6) without synthesis first
        if verbose:
            print("Pass 3/4: Running core passes (critic mode — synthesis deferred)...")
        testing, absence, bayesian, diagnostic_matrix = _run_core_passes(
            extraction, hypothesis_space,
            model=model, verbose=verbose, trace_id=trace_id, priors=priors,
        )

        # Run synthesis for the base snapshot (needed for result_base.json audit)
        if verbose:
            print("Pass 4 (base): Synthesizing for base snapshot...")
        synthesis_base = run_synthesize(
            extraction, hypothesis_space, testing, bayesian, absence,
            model=model, trace_id=f"{trace_id}-base",
        )

        # Write result_base.json
        if output_dir:
            base_result = ProcessTracingResult(
                source_text_sha256=source_text_sha256,
                extraction=extraction,
                hypothesis_space=hypothesis_space,
                partition_audit=partition_audit,
                diagnostic_matrix=diagnostic_matrix,
                testing=testing,
                absence=absence,
                bayesian=bayesian,
                synthesis=synthesis_base,
                source_packet=source_packet_summary,
            )
            base_path = os.path.join(output_dir, "result_base.json")
            with open(base_path, "w", encoding="utf-8") as f:
                json.dump(base_result.model_dump(), f, indent=2)
            if verbose:
                print(f"  Base snapshot: {base_path}")

        # Run structural critic (Pass 3.7)
        if verbose:
            print("Pass 3.7: Structural critic review...")
        critic_result = run_critic(
            extraction, hypothesis_space, testing, diagnostic_matrix,
            model=model, trace_id=f"{trace_id}-critic",
        )
        base_bayesian = bayesian  # save for delta computation

        # Re-elicit Pass 3 if high-severity findings were found
        if critic_result.re_elicitation_needed:
            if verbose:
                print("  Re-eliciting Pass 3 with critic context...")
            testing, absence, bayesian, diagnostic_matrix = _run_core_passes(
                extraction, hypothesis_space,
                model=model, verbose=verbose, pass_label="[Critic re-elicit]",
                trace_id=f"{trace_id}-reelicit", priors=priors,
                critic_context=critic_result.summary,
            )

        # Final synthesis: only re-run if inputs changed via re-elicitation.
        # When re_elicitation_needed=False, testing/bayesian/absence are unchanged
        # so synthesis_base is identical — reuse it to avoid a redundant LLM call.
        if critic_result.re_elicitation_needed:
            if verbose:
                print("Pass 4 (critic): Final synthesis...")
            synthesis = run_synthesize(
                extraction, hypothesis_space, testing, bayesian, absence,
                model=model, trace_id=f"{trace_id}-critic-synth",
            )
            if verbose:
                print(f"  Narrative: {len(synthesis.analytical_narrative)} chars")
        else:
            if verbose:
                print("Pass 4 (critic): Reusing base synthesis (no re-elicitation).")
            synthesis = synthesis_base

        # Compute and write critic delta
        if output_dir:
            deltas = _compute_critic_delta(base_bayesian, bayesian, critic_result)
            delta_path = os.path.join(output_dir, "critic_delta.json")
            with open(delta_path, "w", encoding="utf-8") as f:
                json.dump([d.model_dump() for d in deltas], f, indent=2)
            if verbose:
                n_moved = sum(1 for d in deltas if abs(d.delta) > 0.001)
                print(f"  Critic delta: {delta_path} ({n_moved}/{len(deltas)} hypotheses moved)")

        # result.json (the canonical output written at the bottom) IS the post-critic result.
        # result_critic.json would be identical — skip it. The ablation pair is:
        #   result_base.json  (pre-critic) vs  result.json  (post-critic).

    else:
        # Standard flow: no critic
        testing, absence, bayesian, synthesis, diagnostic_matrix = _run_passes_3_plus(
            extraction, hypothesis_space, text, model=model, verbose=verbose, trace_id=trace_id,
            priors=priors,
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
        testing, absence, bayesian, synthesis, diagnostic_matrix = _run_passes_3_plus(
            extraction, hypothesis_space, text, model=model, verbose=verbose,
            pass_label="[Refined]", trace_id=trace_id, priors=priors,
        )

    if source_packet is not None:
        source_coverage = build_source_coverage(source_packet, text, extraction)
        if verbose:
            print(
                "Source coverage: "
                f"{source_coverage.sources_with_evidence}/{source_coverage.source_count} "
                "packet sources represented in extracted evidence"
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"\nPipeline complete in {elapsed:.1f}s")

    return ProcessTracingResult(
        source_text_sha256=source_text_sha256,
        extraction=extraction,
        hypothesis_space=hypothesis_space,
        partition_audit=partition_audit,
        diagnostic_matrix=diagnostic_matrix,
        testing=testing,
        absence=absence,
        bayesian=bayesian,
        synthesis=synthesis,
        source_packet=source_packet_summary,
        source_coverage=source_coverage,
        refinement=refinement_result,
        is_refined=refine,
        critic=critic_result,
    )
