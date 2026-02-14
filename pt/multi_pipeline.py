"""Multi-document cross-case pipeline orchestrator.

Runs the single-text pipeline on N texts with result caching,
then binarizes against a causal model and optionally bridges to CausalQueries.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from pt.schemas import ProcessTracingResult
from pt.schemas_multi import (
    BinarizationSensitivity,
    BinarizationSensitivityRun,
    CaseBinarization,
    CausalModelSpec,
    CausalQueriesResult,
    MultiDocResult,
)


def _case_id_from_path(path: str) -> str:
    """Derive a case ID from a file path: 'input_text/revolutions/french_revolution.txt' -> 'french_revolution'."""
    return os.path.splitext(os.path.basename(path))[0]


def _run_single_case(
    input_path: str,
    case_output_dir: str,
    *,
    model: str | None = None,
    review: bool = False,
    theories: str | None = None,
) -> ProcessTracingResult:
    """Run single-text pipeline for one case, with caching.

    If case_output_dir/result.json exists, loads from cache.
    Otherwise runs the full pipeline and saves result.
    """
    result_path = os.path.join(case_output_dir, "result.json")

    if os.path.isfile(result_path):
        print(f"  Cache hit: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            return ProcessTracingResult.model_validate(json.load(f))

    os.makedirs(case_output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"Input file is empty: {input_path}")

    from pt.pipeline import run_pipeline
    from pt.report import generate_report

    result = run_pipeline(
        text,
        model=model,
        review=review,
        output_dir=case_output_dir,
        theories=theories,
    )

    # Save result
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)

    # Save HTML report
    html_path = os.path.join(case_output_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(generate_report(result))

    return result


def _default_model_review(model_spec: CausalModelSpec, yaml_path: str) -> CausalModelSpec:
    """Interactive review checkpoint for proposed causal model."""
    print("\n" + "=" * 60)
    print("CAUSAL MODEL REVIEW CHECKPOINT")
    print("=" * 60)
    print(f"\nProposed model: {model_spec.name}")
    print(f"Variables ({len(model_spec.variables)}):")
    for v in model_spec.variables:
        print(f"  {v.name}: 1={v.description[:60]}")
    print(f"\nDAG: {model_spec.dagitty_statement}")
    print(f"\nYAML written to: {yaml_path}")
    print("Edit the YAML file to modify variables/edges, then press Enter.")
    print("Or press Enter without editing to continue as-is.")
    print("=" * 60)

    input("\nPress Enter to continue...")

    # Reload (user may have edited)
    edited = CausalModelSpec.from_yaml(yaml_path)
    errors = edited.validate_dag()
    if errors:
        raise ValueError(f"Edited causal model has errors: {errors}")

    if len(edited.variables) != len(model_spec.variables):
        print(f"  Variables changed: {len(model_spec.variables)} -> {len(edited.variables)}")
    else:
        print("  Model unchanged.")

    return edited


def _apply_confidence_threshold(
    binarizations: list[CaseBinarization],
    threshold: float,
) -> tuple[list[dict[str, int | None]], int]:
    """Re-binarize with a confidence threshold: codings below it become NA.

    Returns (data_frame, n_na_codings).
    """
    n_na = 0
    rows: list[dict[str, int | None]] = []
    for b in binarizations:
        row: dict[str, int | None] = {}
        for c in b.codings:
            if c.confidence < threshold:
                row[c.variable_name] = None
                if c.value is not None:
                    n_na += 1
            else:
                row[c.variable_name] = c.value
        rows.append(row)
    return rows, n_na


def _run_sensitivity(
    binarizations: list[CaseBinarization],
    causal_model: CausalModelSpec,
    *,
    skip_cq: bool = False,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> BinarizationSensitivity:
    """Run CQ at multiple confidence thresholds to assess binarization sensitivity."""
    runs: list[BinarizationSensitivityRun] = []

    for threshold in thresholds:
        df, n_na = _apply_confidence_threshold(binarizations, threshold)

        cq_result: CausalQueriesResult | None = None
        if not skip_cq:
            try:
                from pt.cq_bridge import run_causal_queries
                cq_result = run_causal_queries(causal_model, df)
            except Exception as e:
                print(f"  CQ at threshold {threshold} failed: {e}")

        runs.append(BinarizationSensitivityRun(
            confidence_threshold=threshold,
            data_frame=df,
            cq_result=cq_result,
            n_na_codings=n_na,
        ))

    # Classify estimands as stable vs fragile
    stable: list[str] = []
    fragile: list[str] = []

    # Collect all query strings that appear in any run
    all_queries: set[str] = set()
    for run in runs:
        if run.cq_result:
            for est in run.cq_result.population_estimands:
                if est.using == "posteriors":
                    all_queries.add(est.query)

    for query in sorted(all_queries):
        means: list[float] = []
        for run in runs:
            if run.cq_result:
                for est in run.cq_result.population_estimands:
                    if est.query == query and est.using == "posteriors":
                        means.append(est.mean)
        if len(means) >= 2:
            spread = max(means) - min(means)
            if spread < 0.1:
                stable.append(query)
            else:
                fragile.append(query)

    return BinarizationSensitivity(
        runs=runs,
        stable_estimands=stable,
        fragile_estimands=fragile,
    )


def run_multi_pipeline(
    input_paths: list[str],
    output_dir: str,
    *,
    causal_model: CausalModelSpec | None = None,
    model: str | None = None,
    review: bool = False,
    theories: str | None = None,
    skip_cq: bool = False,
) -> MultiDocResult:
    """Run multi-document cross-case analysis.

    Args:
        input_paths: Paths to input text files.
        output_dir: Top-level output directory.
        causal_model: Pre-specified model (theory-driven). If None, LLM proposes one (data-driven).
        model: LLM model override.
        review: Pause at review checkpoints.
        theories: Theoretical frameworks for hypothesis generation.
        skip_cq: Skip CausalQueries R bridge.
    """
    t0 = time.time()
    workflow = "theory_driven" if causal_model else "data_driven"

    # ── Step 1: Run single-text pipeline on each case ──────────────
    print(f"\n{'='*60}")
    print(f"STEP 1: Analyzing {len(input_paths)} cases")
    print(f"{'='*60}")

    case_results: dict[str, str] = {}
    case_data: dict[str, ProcessTracingResult] = {}

    for i, path in enumerate(input_paths, 1):
        case_id = _case_id_from_path(path)
        case_dir = os.path.join(output_dir, "cases", case_id)
        result_path = os.path.join(case_dir, "result.json")

        print(f"\n[{i}/{len(input_paths)}] {case_id}")
        result = _run_single_case(
            path, case_dir,
            model=model, review=review, theories=theories,
        )
        case_results[case_id] = result_path
        case_data[case_id] = result

    # ── Step 2: Get or propose causal model ────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: Causal model")
    print(f"{'='*60}")

    if causal_model:
        print(f"Using provided model: {causal_model.name}")
    else:
        print("Data-driven workflow: proposing causal model from extraction results...")
        from pt.pass_propose_model import propose_causal_model

        extractions = {cid: r.extraction for cid, r in case_data.items()}
        causal_model = propose_causal_model(extractions, model=model)

        # Write proposed model for review
        yaml_path = os.path.join(output_dir, "proposed_model.yaml")
        causal_model.to_yaml(yaml_path)
        print(f"Proposed model written to: {yaml_path}")

        if review:
            causal_model = _default_model_review(causal_model, yaml_path)

    # ── Step 3: Binarize each case ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STEP 3: Binarizing {len(case_data)} cases against '{causal_model.name}'")
    print(f"{'='*60}")

    from pt.pass_binarize import binarize_case

    binarizations: list[CaseBinarization] = []
    for case_id, result in case_data.items():
        print(f"\n  Binarizing: {case_id}")
        binarization = binarize_case(
            case_id=case_id,
            source_file=case_results[case_id],
            extraction=result.extraction,
            bayesian=result.bayesian,
            causal_model=causal_model,
            model=model,
        )
        binarizations.append(binarization)

        row = binarization.to_row()
        coded = {k: v for k, v in row.items() if v is not None}
        na = {k for k, v in row.items() if v is None}
        print(f"    Coded: {coded}")
        if na:
            print(f"    NA: {na}")

    # Build primary data frame (no threshold filtering)
    data_frame = [b.to_row() for b in binarizations]

    # ── Step 4: CausalQueries bridge ───────────────────────────────
    cq_result: CausalQueriesResult | None = None
    sensitivity: BinarizationSensitivity | None = None

    if skip_cq:
        print("\n  Skipping CausalQueries (--skip-cq)")
    else:
        print(f"\n{'='*60}")
        print("STEP 4: CausalQueries estimation")
        print(f"{'='*60}")

        try:
            from pt.cq_bridge import run_causal_queries
            cq_result = run_causal_queries(causal_model, data_frame)
            print(f"  Population estimands: {len(cq_result.population_estimands)}")
            print(f"  Case-level estimands: {len(cq_result.case_level_estimands)}")
        except Exception as e:
            print(f"  CausalQueries failed (continuing without): {e}")

        # Sensitivity analysis
        print("\n  Running binarization sensitivity analysis...")
        sensitivity = _run_sensitivity(
            binarizations, causal_model, skip_cq=skip_cq,
        )
        n_stable = len(sensitivity.stable_estimands)
        n_fragile = len(sensitivity.fragile_estimands)
        print(f"  Stable estimands: {n_stable}, Fragile: {n_fragile}")

    elapsed = time.time() - t0
    print(f"\nMulti-document pipeline complete in {elapsed:.1f}s")

    return MultiDocResult(
        causal_model=causal_model,
        case_results=case_results,
        binarizations=binarizations,
        data_frame=data_frame,
        cq_result=cq_result,
        sensitivity=sensitivity,
        workflow=workflow,
    )
