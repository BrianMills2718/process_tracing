"""Grade a process-tracing result/report against the output quality rubric."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pt.schemas import ProcessTracingResult


def _first_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", value)
    if not match:
        return None
    return int(match.group(1))


def _focal_year(result: ProcessTracingResult, override: int | None = None) -> int | None:
    if override is not None:
        return override
    years: list[int] = []
    for text in (result.hypothesis_space.research_question, result.extraction.summary):
        years.extend(int(y) for y in re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text))
    if years:
        return max(years)
    evidence_years = [
        year for ev in result.extraction.evidence
        if (year := _first_year(ev.approximate_date)) is not None
    ]
    return max(evidence_years) if evidence_years else None


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _report_has(report_html: str, *phrases: str) -> bool:
    lower = " ".join(report_html.lower().split())
    return all(" ".join(p.lower().split()) in lower for p in phrases)


def _verdict_calibration_issues(result: ProcessTracingResult) -> list[str]:
    posterior = {p.hypothesis_id: p.final_posterior for p in result.bayesian.posteriors}
    issues: list[str] = []
    for verdict in result.synthesis.verdicts:
        support = posterior.get(verdict.hypothesis_id)
        if support is None:
            continue
        if verdict.status in {"supported", "strongly_supported"} and support < 0.10:
            issues.append(
                f"{verdict.hypothesis_id} is labeled {verdict.status} with support {support:.3f}"
            )
    return issues


def _temporal_stats(result: ProcessTracingResult, focal_year: int | None) -> dict[str, Any]:
    if focal_year is None:
        return {
            "focal_year": None,
            "proximate": 0,
            "background": 0,
            "unknown": len(result.extraction.evidence),
            "total": len(result.extraction.evidence),
            "top_driver_background": [],
        }

    proximate = 0
    background = 0
    unknown = 0
    by_id = {ev.id: ev for ev in result.extraction.evidence}
    for ev in result.extraction.evidence:
        year = _first_year(ev.approximate_date)
        if year is None:
            unknown += 1
        elif year >= focal_year - 2:
            proximate += 1
        elif year < focal_year - 5:
            background += 1

    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    top = next((p for p in result.bayesian.posteriors if p.hypothesis_id == top_id), None)
    top_driver_background: list[str] = []
    if top:
        for evidence_id in top.top_drivers:
            ev = by_id.get(evidence_id)
            if ev is None:
                continue
            year = _first_year(ev.approximate_date)
            if year is not None and year < focal_year - 5:
                top_driver_background.append(evidence_id)

    return {
        "focal_year": focal_year,
        "proximate": proximate,
        "background": background,
        "unknown": unknown,
        "total": len(result.extraction.evidence),
        "top_driver_background": top_driver_background,
    }


def _top_fragility(result: ProcessTracingResult) -> dict[str, Any]:
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    top = next((p for p in result.bayesian.posteriors if p.hypothesis_id == top_id), None)
    sens = next((s for s in result.bayesian.sensitivity if s.hypothesis_id == top_id), None)
    if top is None:
        return {"top_id": None, "high_fragile": False, "range_width": None}
    width = None
    if sens:
        width = sens.posterior_high - sens.posterior_low
    return {
        "top_id": top_id,
        "support": top.final_posterior,
        "robustness": top.robustness,
        "high_fragile": top.final_posterior >= 0.75 and top.robustness == "fragile",
        "range_width": width,
    }


def _broad_winner_risk(result: ProcessTracingResult) -> bool:
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    top_h = next((h for h in result.hypothesis_space.hypotheses if h.id == top_id), None)
    if top_h is None:
        return False
    text = f"{top_h.description} {top_h.causal_mechanism}".lower()
    broad_terms = ["vacuum", "combined", "across", "all social", "confluence", "multiple"]
    return any(term in text for term in broad_terms)


def _network_visual_stats(result: ProcessTracingResult) -> dict[str, Any]:
    from pt.report import _build_vis_data

    nodes, edges = _build_vis_data(result)
    node_ids = {str(node["id"]) for node in nodes}
    degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = str(edge["from"])
        target = str(edge["to"])
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1

    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    top_degree = degree.get(top_id or "", 0)
    top_in_graph = top_id in node_ids if top_id else True
    top_graph_connected = top_id is None or (top_in_graph and top_degree > 0)
    isolated_count = sum(1 for node_id in node_ids if degree.get(node_id, 0) == 0)
    evidence_ids = {ev.id for ev in result.extraction.evidence}
    isolated_evidence_count = sum(
        1 for evidence_id in evidence_ids if degree.get(evidence_id, 0) == 0
    )
    isolated_share = isolated_count / len(node_ids) if node_ids else 0.0

    return {
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "isolated_count": isolated_count,
        "isolated_evidence_count": isolated_evidence_count,
        "isolated_share": round(isolated_share, 3),
        "top_id": top_id,
        "top_in_graph": top_in_graph,
        "top_degree": top_degree,
        "top_graph_connected": top_graph_connected,
    }


def audit_result(
    result: ProcessTracingResult,
    report_html: str,
    *,
    focal_year_override: int | None = None,
) -> dict[str, Any]:
    """Return rubric scoring details for a result/report pair."""
    score = 0
    categories: dict[str, dict[str, Any]] = {}

    evidence_ids = [ev.id for ev in result.extraction.evidence]
    vector_ids = [ev.evidence_id for ev in result.testing.evidence_likelihoods]
    hyp_ids = [h.id for h in result.hypothesis_space.hypotheses]
    vector_hyp_sets = [
        {hl.hypothesis_id for hl in ev.hypothesis_likelihoods}
        for ev in result.testing.evidence_likelihoods
    ]
    contract_ok = (
        len(evidence_ids) == len(set(evidence_ids))
        and len(hyp_ids) == len(set(hyp_ids))
        and set(evidence_ids) == set(vector_ids)
        and all(s == set(hyp_ids) for s in vector_hyp_sets)
    )
    categories["contract_integrity"] = {
        "points": 15 if contract_ok else 5,
        "max": 15,
        "ok": contract_ok,
    }
    score += categories["contract_integrity"]["points"]

    verdict_issues = _verdict_calibration_issues(result)
    comparative_visible = _report_has(report_html, "comparative support", "not absolute")
    calibration_visible = (not verdict_issues) or _report_has(
        report_html, "verdict calibration", "secondary mechanism"
    )
    comparative_points = 15
    if not comparative_visible:
        comparative_points -= 7
    if not calibration_visible:
        comparative_points -= 6
    categories["comparative_support_discipline"] = {
        "points": max(comparative_points, 0),
        "max": 15,
        "verdict_issues": verdict_issues,
        "comparative_visible": comparative_visible,
        "calibration_visible": calibration_visible,
    }
    score += categories["comparative_support_discipline"]["points"]

    temporal = _temporal_stats(result, _focal_year(result, focal_year_override))
    temporal_visible = _report_has(
        report_html,
        "temporal evidence mix",
        "proximate",
        "temporal causal timeline",
    )
    background_visible = (not temporal["top_driver_background"]) or _report_has(
        report_html, "background top-driver"
    )
    temporal_points = 15
    if not temporal_visible:
        temporal_points -= 8
    if not background_visible:
        temporal_points -= 5
    categories["temporal_and_causal_proximity"] = {
        "points": max(temporal_points, 0),
        "max": 15,
        **temporal,
        "temporal_visible": temporal_visible,
        "background_visible": background_visible,
    }
    score += categories["temporal_and_causal_proximity"]["points"]

    fragility = _top_fragility(result)
    fragile_warning_visible = (not fragility["high_fragile"]) or _report_has(
        report_html, "high support, fragile"
    )
    robustness_visible = _report_has(report_html, "range", "rank-stable") or _report_has(
        report_html, "rank NOT stable"
    )
    robustness_points = 15
    if not fragile_warning_visible:
        robustness_points -= 8
    if not robustness_visible:
        robustness_points -= 4
    categories["robustness_and_fragility"] = {
        "points": max(robustness_points, 0),
        "max": 15,
        **fragility,
        "fragile_warning_visible": fragile_warning_visible,
        "robustness_visible": robustness_visible,
    }
    score += categories["robustness_and_fragility"]["points"]

    clustered_ids = {eid for c in result.testing.dependence_clusters for eid in c.evidence_ids}
    dependence_visible = _report_has(report_html, "dependence", "cluster")
    weighted_visible = _report_has(report_html, "effective evidence", "raw counts")
    evidence_points = 15
    if result.testing.dependence_clusters and not dependence_visible:
        evidence_points -= 5
    if not weighted_visible:
        evidence_points -= 7
    categories["evidence_weighting_and_dependence"] = {
        "points": max(evidence_points, 0),
        "max": 15,
        "clusters": len(result.testing.dependence_clusters),
        "clustered_items": len(clustered_ids),
        "dependence_visible": dependence_visible,
        "weighted_visible": weighted_visible,
    }
    score += categories["evidence_weighting_and_dependence"]["points"]

    broad_risk = _broad_winner_risk(result)
    broad_visible = (not broad_risk) or _report_has(report_html, "broad winning hypothesis")
    discrimination_points = 10 if broad_visible else 4
    categories["hypothesis_discrimination"] = {
        "points": discrimination_points,
        "max": 10,
        "broad_winner_risk": broad_risk,
        "broad_warning_visible": broad_visible,
    }
    score += discrimination_points

    damaging_absences = [
        ev for ev in result.absence.evaluations if ev.severity == "damaging"
    ]
    source_scope_visible = (not damaging_absences) or _report_has(
        report_html, "source-scope", "absence"
    )
    source_points = 10 if source_scope_visible else 5
    categories["source_scope_and_absence"] = {
        "points": source_points,
        "max": 10,
        "damaging_absence_count": len(damaging_absences),
        "source_scope_visible": source_scope_visible,
    }
    score += source_points

    network = _network_visual_stats(result)
    network_legend_visible = _report_has(report_html, "top driver edge", "not discarded")
    safe = "</script><script>" not in report_html and 'id="detail-' in report_html
    visual_ok = network["top_graph_connected"] and network_legend_visible
    categories["report_usability_and_safety"] = {
        "points": 5 if safe and visual_ok else 2 if safe else 0,
        "max": 5,
        "safe": safe,
        "visual_ok": visual_ok,
        "network_legend_visible": network_legend_visible,
        **network,
    }
    score += categories["report_usability_and_safety"]["points"]

    return {
        "score": score,
        "grade": _grade(score),
        "categories": categories,
    }


def _render_text(audit: dict[str, Any]) -> str:
    lines = [f"Grade: {audit['grade']} ({audit['score']}/100)"]
    for name, details in audit["categories"].items():
        lines.append(f"- {name}: {details['points']}/{details['max']}")
        for key, value in details.items():
            if key in {"points", "max"}:
                continue
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="Path to result.json")
    parser.add_argument("--report", required=True, help="Path to report.html")
    parser.add_argument("--focal-year", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    result = ProcessTracingResult.model_validate_json(Path(args.result).read_text())
    report_html = Path(args.report).read_text(encoding="utf-8")
    audit = audit_result(result, report_html, focal_year_override=args.focal_year)
    if args.json:
        print(json.dumps(audit, indent=2))
    else:
        print(_render_text(audit))


if __name__ == "__main__":
    main()
