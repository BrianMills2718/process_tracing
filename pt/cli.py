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

    result = run_pipeline(text, model=args.model, review=args.review, output_dir=output_dir)

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
