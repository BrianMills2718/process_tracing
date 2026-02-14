"""CLI entry point for process tracing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Van Evera process tracing: multi-pass LLM pipeline"
    )
    parser.add_argument("input", help="Path to input text file")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (default: output/<basename>)")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, skip HTML report")
    parser.add_argument("--model", "-m", default=None, help="LLM model override (default: PT_MODEL env or gemini-2.5-flash)")
    parser.add_argument("--review", action="store_true", help="Pause after hypothesis generation for human review")
    parser.add_argument("--theories", default=None, help="Path to text file with theoretical frameworks for hypothesis generation")
    parser.add_argument("--refine", action="store_true", help="Run analytical refinement after initial pipeline, then re-run passes 3+")
    parser.add_argument("--from-result", default=None, help="Path to existing result.json; skips passes 1-2, implies --refine")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        print("Error: input file is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {args.input} ({len(text)} chars)")

    # Determine output directory
    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or os.path.join("output", basename)
    os.makedirs(output_dir, exist_ok=True)

    from pt.pipeline import run_pipeline
    from pt.report import generate_report
    from pt.schemas import ProcessTracingResult

    # Load theories file if provided
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

    # Load existing result if --from-result
    from_result = None
    if args.from_result:
        if not os.path.isfile(args.from_result):
            print(f"Error: result file not found: {args.from_result}", file=sys.stderr)
            sys.exit(1)
        with open(args.from_result, "r", encoding="utf-8") as f:
            from_result = ProcessTracingResult.model_validate(json.load(f))
        print(f"Loaded result: {args.from_result}")

    result = run_pipeline(
        text, model=args.model, review=args.review, output_dir=output_dir,
        theories=theories, refine=args.refine, from_result=from_result,
    )

    # Write JSON
    json_path = os.path.join(output_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"JSON: {json_path}")

    # Write HTML
    if not args.json_only:
        html_path = os.path.join(output_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(generate_report(result))
        print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
