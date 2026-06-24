"""Build and optionally retrieve source-acquisition targets for a trace result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pt.schemas import ProcessTracingResult
from pt.source_acquisition import AcquisitionPlan, build_acquisition_plan
from pt.source_packet import SourcePacket, load_source_packet


Provider = Literal["brave", "searxng", "tavily", "exa"]


def _load_result(path: Path) -> ProcessTracingResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProcessTracingResult.model_validate(data)


def _load_packet(path: Path | None, result: ProcessTracingResult) -> SourcePacket | None:
    if path is not None:
        return load_source_packet(path)
    packet_summary = result.source_packet
    if packet_summary is None or packet_summary.source_packet_path is None:
        return None
    candidate = Path(packet_summary.source_packet_path)
    if not candidate.is_file():
        candidate = REPO_ROOT / packet_summary.source_packet_path
    if candidate.is_file():
        return load_source_packet(candidate)
    return None


def _available_providers(requested: list[str]) -> list[Provider]:
    if requested:
        return [cast(Provider, provider) for provider in requested]
    providers: list[Provider] = []
    if os.environ.get("EXA_API_KEY"):
        providers.append("exa")
    if os.environ.get("TAVILY_API_KEY"):
        providers.append("tavily")
    if os.environ.get("BRAVE_API_KEY"):
        providers.append("brave")
    if os.environ.get("SEARXNG_BASE_URL"):
        providers.append("searxng")
    return providers


def _retrieve_for_plan(
    plan: AcquisitionPlan,
    *,
    providers: list[Provider],
    top_k: int,
    queries_per_target: int,
) -> list[dict[str, Any]]:
    from open_web_retrieval.client import OpenWebRetrievalClient
    from open_web_retrieval.models import SearchQuery

    if not providers:
        raise RuntimeError(
            "no open_web_retrieval providers configured; set EXA_API_KEY, "
            "TAVILY_API_KEY, BRAVE_API_KEY, or SEARXNG_BASE_URL"
        )

    records: list[dict[str, Any]] = []
    with OpenWebRetrievalClient(
        exa_api_key=os.environ.get("EXA_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        brave_api_key=os.environ.get("BRAVE_API_KEY"),
        searxng_base_url=os.environ.get("SEARXNG_BASE_URL"),
        cache_dir=str(REPO_ROOT / "output" / "open_web_cache"),
        timeout_seconds=20.0,
    ) as client:
        for target in plan.targets:
            for query_text in target.search_queries[:queries_per_target]:
                query = SearchQuery(
                    query=query_text,
                    providers=providers,
                    top_k=top_k,
                    search_depth="advanced",
                    result_detail="summary",
                    retrieval_instruction=target.evidence_need,
                )
                batch = client.retrieve(
                    query,
                    allow_partial=True,
                    trace_id=f"source-acquisition-{target.target_id}",
                    task="process_tracing_source_acquisition",
                )
                records.append(
                    {
                        "target_id": target.target_id,
                        "query": query_text,
                        "providers": providers,
                        "hits": [
                            {
                                "provider": record.search_hit.provider,
                                "rank": record.search_hit.rank,
                                "title": record.search_hit.title,
                                "url": record.search_hit.url,
                                "snippet": record.search_hit.snippet,
                                "publisher": record.search_hit.publisher,
                                "published_at": record.search_hit.published_at.isoformat()
                                if record.search_hit.published_at is not None
                                else None,
                                "fetched": record.fetched_resource is not None,
                                "extracted": record.extracted_document is not None,
                                "extracted_title": (
                                    record.extracted_document.title
                                    if record.extracted_document is not None
                                    else None
                                ),
                                "text_char_count": (
                                    len(record.extracted_document.text)
                                    if record.extracted_document is not None
                                    else 0
                                ),
                                "error": record.provenance.get("error"),
                            }
                            for record in batch.records
                        ],
                    }
                )
    return records


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

    result = _load_result(args.result)
    packet = _load_packet(args.source_packet, result)
    plan = build_acquisition_plan(result, source_packet=packet, max_targets=args.max_targets)
    payload: dict[str, Any] = {"plan": plan.model_dump()}
    if args.retrieve:
        payload["retrieval"] = _retrieve_for_plan(
            plan,
            providers=_available_providers(args.provider),
            top_k=args.top_k,
            queries_per_target=args.queries_per_target,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(_render_text(payload))


if __name__ == "__main__":
    main()
