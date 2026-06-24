"""Build and optionally retrieve source-acquisition targets for a trace result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pt.source_acquisition import (
    available_retrieval_providers,
    build_acquisition_plan,
    load_packet_for_result,
    load_process_result,
    retrieve_for_plan,
)
from pt.source_design import build_source_design_state


def _render_text(payload: dict[str, Any]) -> str:
    plan = payload["plan"]
    lines = [
        f"Case: {plan['case_name']}",
        f"Rationale: {plan['rationale']}",
        "",
        "Targets:",
    ]
    for target in plan["targets"]:
        lines.extend(
            [
                f"- {target['target_id']} [{target['kind']}] score={target['priority_score']}",
                f"  need: {target['evidence_need']}",
                f"  payoff: {target['inferential_payoff']}",
                f"  source_class: {target['target_source_class']}",
                f"  hypotheses: {', '.join(target['related_hypotheses']) or 'none'}",
                f"  queries: {' | '.join(target['search_queries'])}",
                f"  stop_rule: {target['stop_rule']}",
            ]
        )
    retrieval = payload.get("retrieval") or []
    if retrieval:
        lines.extend(["", "Retrieval hits:"])
        for item in retrieval:
            lines.append(f"- {item['target_id']} query={item['query']!r}")
            for hit in item["hits"]:
                suffix = (
                    f" [extracted {hit['text_char_count']} chars]"
                    if hit.get("extracted")
                    else " [not extracted]"
                )
                lines.append(
                    f"  {hit['provider']} #{hit['rank']}: {hit['title']} — {hit['url']}{suffix}"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Path to process-tracing result.json")
    parser.add_argument("--source-packet", type=Path, help="Optional full source packet JSON")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--retrieve", action="store_true", help="Run live open_web_retrieval search")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--queries-per-target", type=int, default=1)
    parser.add_argument(
        "--provider",
        action="append",
        choices=["brave", "searxng", "tavily", "exa"],
        default=[],
        help="Search provider to use; may be repeated. Defaults to configured env providers.",
    )
    args = parser.parse_args()

    result = load_process_result(args.result)
    packet = load_packet_for_result(args.source_packet, result, repo_root=REPO_ROOT)
    plan = build_acquisition_plan(result, source_packet=packet, max_targets=args.max_targets)
    payload: dict[str, Any] = {"plan": plan.model_dump()}
    if packet is not None:
        payload["design_state"] = build_source_design_state(
            result,
            source_packet=packet,
            max_targets=args.max_targets,
        ).model_dump()
    if args.retrieve:
        payload["retrieval"] = retrieve_for_plan(
            plan,
            providers=available_retrieval_providers(args.provider),
            top_k=args.top_k,
            queries_per_target=args.queries_per_target,
            cache_dir=REPO_ROOT / "output" / "open_web_cache",
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(_render_text(payload))


if __name__ == "__main__":
    main()
