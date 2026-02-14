"""CLI entry point for multi-document cross-case analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-document cross-case process tracing with CausalQueries bridge"
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Paths to input text files (one per case)",
    )
    parser.add_argument(
        "--causal-model", default=None,
        help="Path to YAML causal model specification (theory-driven workflow)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory (default: output/multi_<timestamp>)",
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="LLM model override (default: PT_MODEL env or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Pause for human review at checkpoints (model proposal, hypothesis generation)",
    )
    parser.add_argument(
        "--theories", default=None,
        help="Path to text file with theoretical frameworks for hypothesis generation",
    )
    parser.add_argument(
        "--skip-cq", action="store_true",
        help="Skip CausalQueries R bridge (just binarize, no causal estimation)",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Output JSON only, skip HTML report",
    )
    args = parser.parse_args()

    # Validate inputs
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.causal_model and not os.path.isfile(args.causal_model):
        print(f"Error: causal model file not found: {args.causal_model}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        import time
        output_dir = os.path.join("output", f"multi_{int(time.time())}")
    os.makedirs(output_dir, exist_ok=True)

    # Load theories
    theories = None
    if args.theories:
        if not os.path.isfile(args.theories):
            print(f"Error: theories file not found: {args.theories}", file=sys.stderr)
            sys.exit(1)
        with open(args.theories, "r", encoding="utf-8") as f:
            theories = f.read().strip()
        if not theories:
            print("Error: theories file is empty", file=sys.stderr)
            sys.exit(1)
        print(f"Theories: {args.theories} ({len(theories)} chars)")

    # Load causal model if provided
    from pt.schemas_multi import CausalModelSpec

    causal_model: CausalModelSpec | None = None
    if args.causal_model:
        causal_model = CausalModelSpec.from_yaml(args.causal_model)
        errors = causal_model.validate_dag()
        if errors:
            print(f"Error: invalid causal model:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Causal model: {causal_model.name} ({len(causal_model.variables)} variables, {len(causal_model.edges)} edges)")

    workflow = "theory_driven" if causal_model else "data_driven"
    print(f"Workflow: {workflow}")
    print(f"Inputs: {len(args.inputs)} texts")
    print(f"Output: {output_dir}")

    from pt.multi_pipeline import run_multi_pipeline

    result = run_multi_pipeline(
        input_paths=args.inputs,
        output_dir=output_dir,
        causal_model=causal_model,
        model=args.model,
        review=args.review,
        theories=theories,
        skip_cq=args.skip_cq,
    )

    # Write JSON result
    json_path = os.path.join(output_dir, "multi_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"\nJSON: {json_path}")

    # Write HTML report
    if not args.json_only:
        from pt.report_multi import generate_multi_report
        html_path = os.path.join(output_dir, "multi_report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(generate_multi_report(result, output_dir))
        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
