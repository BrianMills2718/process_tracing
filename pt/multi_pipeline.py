"""Multi-document cross-case pipeline orchestrator.

Runs the single-text pipeline on N texts with result caching,
then binarizes against a causal model and optionally bridges to CausalQueries.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional
from uuid import uuid4

from pt.schemas import ProcessTracingResult
from pt.schemas_multi import (
    BinarizationSensitivity,
    BinarizationSensitivityRun,
    CaseBinarization,
    CausalModelSpec,
    CausalQueriesResult,
    CrossCaseEligibility,
    MultiDocResult,
    VariableEligibility,
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

    The cache is keyed on the input text content (sha256), the resolved model
    id, and the user theory text. A cached result is reused only when all match
    — otherwise it is recomputed. Without this, re-running with a different
    --model, edited input text, or changed --theories silently returned the
    previous (wrong) extraction.
    """
    from pt.llm import DEFAULT_MODEL

    result_path = os.path.join(case_output_dir, "result.json")
    meta_path = os.path.join(case_output_dir, "cache_meta.json")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError(f"Input file is empty: {input_path}")

    effective_model = model or DEFAULT_MODEL
    cache_key = {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "model": effective_model,
        "theories_sha256": (
            hashlib.sha256(theories.encode("utf-8")).hexdigest()
            if theories is not None else None
        ),
    }

    if os.path.isfile(result_path):
        cached_key = None
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_key = json.load(f)
        if cached_key == cache_key:
            print(f"  Cache hit: {result_path}")
            with open(result_path, "r", encoding="utf-8") as f:
                return ProcessTracingResult.model_validate(json.load(f))
        reason = "model, input text, or theories changed" if cached_key else "no cache key recorded"
        print(f"  Cache stale ({reason}) — recomputing {result_path}")

    os.makedirs(case_output_dir, exist_ok=True)

    from pt.pipeline import run_pipeline
    from pt.report import generate_report

    result = run_pipeline(
        text,
        model=model,
        review=review,
        output_dir=case_output_dir,
        theories=theories,
    )

    # Save result and its cache key
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cache_key, f, indent=2)

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
    variable_order: list[str] | None = None,
) -> tuple[list[dict[str, int | None]], int]:
    """Re-binarize with a confidence threshold: codings below it become NA.

    Returns (data_frame, n_na_codings). When ``variable_order`` is given, every
    row is emitted with that fixed key order (see ``CaseBinarization.to_row``).
    """
    n_na = 0
    rows: list[dict[str, int | None]] = []
    for b in binarizations:
        coded: dict[str, int | None] = {}
        seen: set[str] = set()
        allowed = set(variable_order) if variable_order is not None else None
        for c in b.codings:
            if c.variable_name in seen:
                raise ValueError(
                    f"duplicate variable coding in {b.case_id}: {c.variable_name}"
                )
            seen.add(c.variable_name)
            if allowed is not None and c.variable_name not in allowed:
                raise ValueError(
                    f"binarization for {b.case_id} has variable not in model: {c.variable_name}"
                )
            if c.confidence < threshold:
                coded[c.variable_name] = None
                if c.value is not None:
                    n_na += 1
            else:
                coded[c.variable_name] = c.value
        if variable_order is None:
            rows.append(coded)
        else:
            rows.append({name: coded.get(name) for name in variable_order})
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
        df, n_na = _apply_confidence_threshold(
            binarizations, threshold, causal_model.variable_names
        )

        cq_result: CausalQueriesResult | None = None
        if not skip_cq:
            from pt.cq_bridge import is_r_available, run_causal_queries
            if not is_r_available():
                print(f"  CQ at threshold {threshold} skipped: R not installed")
            else:
                # R is present: a failure here is a real error, not graceful
                # degradation, so let it propagate (fail loud per project policy).
                cq_result = run_causal_queries(
                    causal_model,
                    df,
                    case_ids=[b.case_id for b in binarizations],
                )

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


def _check_cross_case_eligibility(
    binarizations: list[CaseBinarization],
    causal_model: CausalModelSpec,
) -> CrossCaseEligibility:
    """Deterministic eligibility assessment for CausalQueries estimation.

    CQ requires outcome variation and a minimum number of cases. This check
    runs after binarization and gates the CQ bridge — it never involves an LLM.
    """
    n_cases = len(binarizations)
    outcome_var = causal_model.outcome_variable
    var_names = causal_model.variable_names

    # Collect per-variable coding statistics
    var_stats: dict[str, dict] = {
        name: {"n_coded": 0, "n_na": 0, "n_zero": 0, "n_one": 0,
               "confidences": [], "warnings": []}
        for name in var_names
    }

    for b in binarizations:
        coded = {c.variable_name: c for c in b.codings}
        for name in var_names:
            s = var_stats[name]
            if name not in coded:
                s["n_na"] += 1
            else:
                c = coded[name]
                s["confidences"].append(c.confidence)
                if c.value is None:
                    s["n_na"] += 1
                elif c.value == 0:
                    s["n_coded"] += 1
                    s["n_zero"] += 1
                else:
                    s["n_coded"] += 1
                    s["n_one"] += 1

    variable_checks: list[VariableEligibility] = []
    for name in var_names:
        s = var_stats[name]
        confs = s["confidences"]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        varies = s["n_zero"] > 0 and s["n_one"] > 0
        warnings: list[str] = []
        if mean_conf < 0.5 and s["n_coded"] > 0:
            warnings.append(f"mean confidence {mean_conf:.2f} < 0.50 — codings may be unreliable")
        if s["n_na"] == n_cases:
            warnings.append("no cases have a non-NA coding")
        elif s["n_na"] > 0:
            warnings.append(f"{s['n_na']}/{n_cases} cases have NA coding")
        if not varies and s["n_coded"] > 0:
            val = "all-1" if s["n_one"] > 0 else "all-0"
            warnings.append(f"no variation ({val}) — cannot identify effect")
        variable_checks.append(VariableEligibility(
            variable_name=name,
            n_cases=n_cases,
            n_coded=s["n_coded"],
            n_na=s["n_na"],
            n_zero=s["n_zero"],
            n_one=s["n_one"],
            mean_confidence=round(mean_conf, 4),
            varies=varies,
            warnings=warnings,
        ))

    # Outcome-specific stats
    os_ = var_stats[outcome_var]
    outcome_varies = os_["n_zero"] > 0 and os_["n_one"] > 0

    n_variables_with_variation = sum(
        1 for vc in variable_checks
        if vc.variable_name != outcome_var and vc.varies
    )

    # Eligibility logic
    ineligible_reasons: list[str] = []
    warnings: list[str] = []

    if n_cases < 2:
        ineligible_reasons.append(
            f"only {n_cases} case(s) — CausalQueries requires ≥2"
        )
    if os_["n_coded"] == 0:
        ineligible_reasons.append(
            f"outcome variable '{outcome_var}' has no non-NA codings across all cases"
        )
    elif not outcome_varies:
        dominant = "1" if os_["n_one"] > 0 else "0"
        ineligible_reasons.append(
            f"outcome '{outcome_var}' does not vary: all coded cases have value={dominant}. "
            "CausalQueries cannot estimate causal effects without outcome variation."
        )

    if n_variables_with_variation == 0 and not ineligible_reasons:
        warnings.append(
            "no non-outcome variables vary across cases — effect estimation will be degenerate"
        )
    elif n_variables_with_variation < 2:
        warnings.append(
            f"only {n_variables_with_variation} non-outcome variable(s) vary — "
            "estimation power is very low"
        )

    # Check for NA-heavy variables
    na_heavy = [
        vc.variable_name for vc in variable_checks
        if vc.n_na > n_cases // 2
    ]
    if na_heavy:
        warnings.append(
            f"high NA rate (>50%) on: {', '.join(na_heavy)} — "
            "consider collecting more evidence before cross-case analysis"
        )

    eligible_for_cq = len(ineligible_reasons) == 0

    return CrossCaseEligibility(
        n_cases=n_cases,
        outcome_variable=outcome_var,
        outcome_n_coded=os_["n_coded"],
        outcome_n_zero=os_["n_zero"],
        outcome_n_one=os_["n_one"],
        outcome_varies=outcome_varies,
        variable_checks=variable_checks,
        n_variables_with_variation=n_variables_with_variation,
        eligible_for_cq=eligible_for_cq,
        ineligible_reasons=ineligible_reasons,
        warnings=warnings,
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
    trace_id = uuid4().hex[:8]
    workflow = "theory_driven" if causal_model else "data_driven"

    # ── Step 1: Run single-text pipeline on each case ──────────────
    print(f"\n{'='*60}")
    print(f"STEP 1: Analyzing {len(input_paths)} cases")
    print(f"{'='*60}")

    case_results: dict[str, str] = {}
    case_data: dict[str, ProcessTracingResult] = {}

    for i, path in enumerate(input_paths, 1):
        case_id = _case_id_from_path(path)
        if case_id in case_results:
            # Case dirs are derived from the file basename; two inputs with the
            # same basename would share a cache dir and silently overwrite each
            # other. Fail loud rather than corrupt the cross-case set.
            raise ValueError(
                f"Duplicate case id '{case_id}' from input '{path}' — two inputs "
                f"share a basename. Rename one so case ids stay unique."
            )
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
        causal_model = propose_causal_model(extractions, model=model, trace_id=trace_id)

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
            trace_id=trace_id,
        )
        binarizations.append(binarization)

        row = binarization.to_row()
        coded = {k: v for k, v in row.items() if v is not None}
        na = {k for k, v in row.items() if v is None}
        print(f"    Coded: {coded}")
        if na:
            print(f"    NA: {na}")

    # Build primary data frame (no threshold filtering). Order columns by the
    # model's canonical variable list so every row has identical key order —
    # the R bridge binds rows positionally, so inconsistent order corrupts data.
    var_order = causal_model.variable_names
    data_frame = [b.to_row(var_order) for b in binarizations]

    # ── Step 3.5: Cross-case eligibility check ─────────────────────
    print(f"\n{'='*60}")
    print("STEP 3.5: Cross-case eligibility assessment")
    print(f"{'='*60}")

    eligibility = _check_cross_case_eligibility(binarizations, causal_model)
    print(f"  Outcome '{eligibility.outcome_variable}' varies: {eligibility.outcome_varies}")
    print(f"  Non-outcome variables with variation: {eligibility.n_variables_with_variation}")
    print(f"  Eligible for CausalQueries: {eligibility.eligible_for_cq}")
    if eligibility.ineligible_reasons:
        for reason in eligibility.ineligible_reasons:
            print(f"  INELIGIBLE: {reason}")
    for w in eligibility.warnings:
        print(f"  WARNING: {w}")

    # ── Step 4: CausalQueries bridge ───────────────────────────────
    cq_result: CausalQueriesResult | None = None
    sensitivity: BinarizationSensitivity | None = None

    if skip_cq:
        print("\n  Skipping CausalQueries (--skip-cq)")
    elif not eligibility.eligible_for_cq:
        print(f"\n{'='*60}")
        print("STEP 4: CausalQueries estimation — BLOCKED by eligibility gate")
        print(f"{'='*60}")
        for reason in eligibility.ineligible_reasons:
            print(f"  {reason}")
        print("  To override, fix the eligibility issues or supply cases with outcome variation.")
    else:
        print(f"\n{'='*60}")
        print("STEP 4: CausalQueries estimation")
        print(f"{'='*60}")

        from pt.cq_bridge import is_r_available, run_causal_queries
        if not is_r_available():
            # Documented graceful degradation: the pipeline works without R
            # through binarization. R absent is the ONLY swallowed condition.
            print("  R not installed — skipping CausalQueries (use --skip-cq to "
                  "silence this). Binarization output is still produced.")
        else:
            # R is present: any failure is a real error (bad model, Stan crash,
            # malformed data) and must fail loud, not silently produce empty
            # estimands that read downstream as a successful run.
            cq_result = run_causal_queries(
                causal_model,
                data_frame,
                case_ids=[b.case_id for b in binarizations],
            )
            print(f"  Population estimands: {len(cq_result.population_estimands)}")
            print(f"  Case-level estimands: {len(cq_result.case_level_estimands)}")

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
        eligibility=eligibility,
        workflow=workflow,
    )
