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
    parser.add_argument("--research-question", default=None, help="Pin the research question (the outcome to explain). Makes runs reproducible; when omitted the LLM selects it.")
    parser.add_argument("--refine", action="store_true", help="Run analytical refinement after initial pipeline, then re-run passes 3+")
    parser.add_argument("--from-result", default=None, help="Path to existing result.json; skips passes 1-2, implies --refine")
    parser.add_argument("--source-packet", default=None, help="Path to source-packet JSON or assistant source-packet artifact. Pins research question and source-scope metadata.")
    parser.add_argument("--priors", default=None, help="Path to JSON file mapping hypothesis_id -> prior weight (need not sum to 1). Default: uniform.")
    parser.add_argument("--critic", action="store_true", default=False, help="Run structural critic pass (Pass 3.7) after diagnostic matrix. Writes result_base.json and critic_delta.json (ablation pair: result_base.json pre-critic vs result.json post-critic). Re-elicits Pass 3 when high-severity findings are present.")
    parser.add_argument("--critic-model", default=None, help="LLM model for the critic pass (Pass 3.7). Defaults to --model. Use a distinct model for independent critique (avoids same-model rationalization). Requires --critic.")
    parser.add_argument("--max-budget", type=float, default=None, help="Per-call LLM budget cap in dollars (default: PT_MAX_BUDGET or 1.0)")
    args = parser.parse_args()

    if args.critic_model and not args.critic:
        print("Error: --critic-model requires --critic", file=sys.stderr)
        sys.exit(1)

    if args.max_budget is not None:
        if args.max_budget <= 0:
            print("Error: --max-budget must be greater than 0", file=sys.stderr)
            sys.exit(1)
        os.environ["PT_MAX_BUDGET"] = str(args.max_budget)

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
    from pt.source_packet import SourcePacketError, load_source_packet

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

    source_packet = None
    if args.source_packet:
        if args.from_result:
            print("Error: --source-packet cannot be combined with --from-result", file=sys.stderr)
            sys.exit(1)
        try:
            source_packet = load_source_packet(args.source_packet)
        except SourcePacketError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Source packet: {args.source_packet} "
            f"({len(source_packet.source_candidates)} sources, "
            f"{len(source_packet.known_gaps)} known gaps)"
        )

    # Load researcher priors if provided
    priors = None
    if args.priors:
        if not os.path.isfile(args.priors):
            print(f"Error: priors file not found: {args.priors}", file=sys.stderr)
            sys.exit(1)
        with open(args.priors, "r", encoding="utf-8") as f:
            priors = json.load(f)
        if not isinstance(priors, dict) or not priors:
            print("Error: priors file must be a non-empty JSON object {hypothesis_id: weight}", file=sys.stderr)
            sys.exit(1)
        print(f"Priors: {args.priors} ({len(priors)} hypotheses)")

    result = run_pipeline(
        text, model=args.model, review=args.review, output_dir=output_dir,
        theories=theories, research_question=args.research_question,
        refine=args.refine, from_result=from_result,
        source_packet=source_packet,
        source_packet_path=args.source_packet,
        priors=priors,
        critic=args.critic,
        critic_model=args.critic_model,
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
