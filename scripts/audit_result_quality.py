"""Grade a process-tracing result/report against the output quality rubric."""

from __future__ import annotations

import argparse
import json
import math
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


def _diagnostic_strength_stats(result: ProcessTracingResult) -> dict[str, Any]:
    from pt.bayesian import INTERPRETIVE_LR_CAP, lr_matrix

    hypothesis_ids = [h.id for h in result.hypothesis_space.hypotheses]
    interpretive_caps = {
        ev.id: INTERPRETIVE_LR_CAP
        for ev in result.extraction.evidence
        if ev.evidence_type == "interpretive"
    }
    strengths: list[float] = []
    for _, lrs in lr_matrix(result.testing, hypothesis_ids, interpretive_caps):
        if not lrs:
            continue
        values = [max(lr, 0.01) for lr in lrs.values()]
        if len(values) < 2:
            strengths.append(0.0)
            continue
        strengths.append(math.log(max(values) / min(values)))
    decisive_threshold = math.log(5.0)
    moderate_threshold = math.log(2.0)
    return {
        "decisive_items": sum(1 for s in strengths if s >= decisive_threshold),
        "moderate_items": sum(
            1 for s in strengths if moderate_threshold <= s < decisive_threshold
        ),
        "weak_items": sum(1 for s in strengths if 0.1 < s < moderate_threshold),
        "near_neutral_items": sum(1 for s in strengths if s <= 0.1),
        "max_log_lr": round(max(strengths), 3) if strengths else 0.0,
    }


def _source_material_context(result: ProcessTracingResult) -> dict[str, Any]:
    """Summarize the source material visible to the grader."""
    packet = result.source_packet
    coverage = result.source_coverage
    accepted_sources: list[dict[str, Any]] = []
    if coverage is not None:
        accepted_sources = [
            {
                "source_id": item.source_id,
                "title": item.title,
                "source_group": item.source_group,
                "source_kind": item.source_kind,
                "status": item.status,
                "input_marker_hits": item.input_marker_hits,
                "evidence_count": item.evidence_count,
            }
            for item in coverage.items
        ]
    return {
        "has_source_packet": packet is not None,
        "case_name": packet.case_name if packet else None,
        "research_question": packet.research_question if packet else result.hypothesis_space.research_question,
        "source_count": packet.source_count if packet else 0,
        "source_groups": packet.source_groups if packet else [],
        "source_kinds": packet.source_kinds if packet else [],
        "date_coverage": packet.date_coverage if packet else [],
        "known_gap_count": packet.known_gap_count if packet else 0,
        "high_priority_gap_count": packet.high_priority_gap_count if packet else 0,
        "high_priority_gaps": packet.high_priority_gaps if packet else [],
        "packet_limitations": packet.limitations if packet else [],
        "has_source_coverage": coverage is not None,
        "sources_with_evidence": coverage.sources_with_evidence if coverage else 0,
        "assigned_evidence_count": coverage.assigned_evidence_count if coverage else 0,
        "unassigned_evidence_count": len(coverage.unassigned_evidence_ids) if coverage else 0,
        "accepted_sources": accepted_sources,
        "interpretation": (
            "Conditional/given-source critique uses the accepted source packet and extracted "
            "evidence coverage. Known gaps cap broader claim scope; they are not by themselves "
            "failures of the analysis given the accepted sources."
        ),
    }


def _academic_caps(
    result: ProcessTracingResult,
    *,
    temporal: dict[str, Any],
    fragility: dict[str, Any],
    verdict_issues: list[str],
    network: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], int, list[dict[str, Any]], list[str], list[str]]:
    """Hard caps split into given-source critique and broader claim scope."""
    conditional_caps: list[dict[str, Any]] = []
    claim_scope_caps: list[dict[str, Any]] = []
    conditional_recommendations: list[str] = []
    claim_scope_recommendations: list[str] = []

    def add(
        cap: int,
        reason: str,
        recommendation: str,
        action_type: str,
        acceptance_criteria: str,
        *,
        track: str,
    ) -> None:
        entry = {
            "cap": cap,
            "reason": reason,
            "recommendation": recommendation,
            "action_type": action_type,
            "acceptance_criteria": acceptance_criteria,
            "track": track,
        }
        if track == "conditional":
            conditional_caps.append(entry)
            conditional_recommendations.append(recommendation)
        elif track == "claim_scope":
            claim_scope_caps.append(entry)
            claim_scope_recommendations.append(recommendation)
        else:
            raise ValueError(f"unknown cap track: {track}")

    limitations = " ".join(result.synthesis.limitations).lower()
    source_packet = result.source_packet
    source_coverage = result.source_coverage
    source_scope_capped = False
    if any(term in limitations for term in ("single historical text", "single source", "single text")):
        if source_packet is None:
            add(
                78,
                "The given-source critique may be coherent, but no source packet is stored to define how far claims can travel beyond the supplied text.",
                "Keep conclusions conditional on the supplied text; add an explicit source packet only before broader publication-strength claims.",
                "claim_scope",
                "At least three independent source groups are represented, including one primary-source group and one rival/critical secondary account.",
                track="claim_scope",
            )
        elif (
            source_packet.source_count < 3
            or source_packet.high_priority_gap_count > 0
            or source_packet.limitations
        ):
            add(
                82,
                "The accepted-source critique may be coherent, but the packet still names source-scope limits that cap broader claims.",
                "Keep the critique conditional on accepted sources; repair or explicitly disposition packet limits only before broader publication-strength claims.",
                "claim_scope",
                "The packet has at least three independent source groups, no unresolved high-priority gaps, and packet limitations are either resolved or explicitly accepted.",
                track="claim_scope",
            )
        else:
            add(
                88,
                "A source packet is present, but synthesis still describes the evidence base as single-source.",
                "Regenerate or repair synthesis so source-scope limitations are based on the accepted packet rather than stale generic caveats.",
                "report_or_synthesis",
                "The synthesis and report agree on the accepted source packet and remaining source-scope limits.",
                track="conditional",
            )
        source_scope_capped = True

    if (
        source_packet is not None
        and not source_scope_capped
        and (
            source_packet.source_count < 3
            or source_packet.high_priority_gap_count > 0
            or source_packet.limitations
        )
    ):
        add(
            82,
            "The critique is conditional on accepted sources; unresolved packet gaps cap claims beyond that corpus.",
            "Do not treat source gaps as failures of the given-source critique; acquire, extend, or explicitly disposition them before publication-strength claims.",
            "claim_scope",
            "The packet has at least three independent source groups, no unresolved high-priority gaps, and packet limitations are either resolved or explicitly accepted.",
            track="claim_scope",
        )

    if source_packet is not None and source_coverage is None:
        add(
            82,
            "A source packet is present, but no packet-source coverage report is stored.",
            "Compute packet-source coverage so the grader can see which accepted source groups produced evidence.",
            "report_or_model",
            "Every accepted source-packet run stores source coverage with per-source input marker and evidence counts.",
            track="conditional",
        )
    elif source_coverage is not None and (
        source_coverage.sources_with_evidence < source_coverage.source_count
        or source_coverage.unconfigured_source_ids
    ):
        add(
            82,
            "Packet-source coverage is incomplete or not explicitly configured for every source.",
            "Repair packet source markers or assemble the corpus so every packet source is represented in extracted evidence, or explicitly accept the gap.",
            "report_or_model",
            "Every packet source has explicit text markers and at least one extracted evidence item, or the missing source is moved to known_gaps with an explicit disposition.",
            track="conditional",
        )

    diagnostic = _diagnostic_strength_stats(result)
    if diagnostic["decisive_items"] == 0 and diagnostic["moderate_items"] == 0:
        add(
            76,
            "No evidence item clears even the moderate diagnostic-strength threshold.",
            "Design a small set of pre-specified hoop, smoking-gun, and discriminating straw-in-the-wind tests before rerunning likelihood elicitation.",
            "test_design",
            "Each major rival pair has at least one pre-specified discriminator and at least one evidence item reaches moderate diagnostic strength.",
            track="conditional",
        )
    elif diagnostic["decisive_items"] == 0:
        add(
            84,
            "The result lacks decisive process-tracing tests.",
            "Seek direct traces that would be unlikely under rival hypotheses, not just background facts that weakly favor one story.",
            "test_design",
            "At least one decisive or multiple moderate diagnostic traces are found for the leading mechanism.",
            track="conditional",
        )

    total = temporal["total"] or 1
    proximate_share = temporal["proximate"] / total
    if proximate_share < 0.20:
        add(
            80,
            f"Only {temporal['proximate']}/{temporal['total']} evidence items are proximate to the focal outcome.",
            "Collect and separately score outcome-proximate evidence from the final decision window; do not let background conditions carry the main causal claim.",
            "test_design",
            "At least 20% of scored evidence is proximate to the focal outcome and one proximate item is among the leading hypothesis's top drivers.",
            track="conditional",
        )
    if temporal["top_driver_background"]:
        add(
            82,
            "One or more top drivers are background-context items rather than proximate mechanism traces.",
            "Separate background enabling conditions from mechanism evidence and require at least one proximate top driver for a publication-strength claim.",
            "model_design",
            "Top-driver summaries distinguish enabling conditions from mechanism traces, with at least one proximate mechanism trace for the winner.",
            track="conditional",
        )

    if fragility.get("high_fragile"):
        add(
            84,
            "The leading hypothesis has high comparative support but fragile robustness.",
            "Treat the winner as a provisional ranking; seek fewer, stronger discriminating traces and rerun sensitivity after dependence-cluster review.",
            "test_design",
            "The winning hypothesis is robust or moderate after adding stronger traces and reviewing dependence clusters.",
            track="conditional",
        )

    if _broad_winner_risk(result):
        add(
            84,
            "The winning hypothesis is broad enough to absorb mechanisms from rival explanations.",
            "Split the broad winner into narrower mechanisms or add explicit discriminators so overlap with rivals cannot create an artificial victory.",
            "hypothesis_design",
            "The broad hypothesis is split or narrowed, and every rival pair has named discriminating predictions.",
            track="conditional",
        )

    if verdict_issues:
        add(
            86,
            "At least one synthesis verdict overstates a low-posterior hypothesis.",
            "Calibrate verdict labels to comparative support; label low-support but plausible mechanisms as residual or secondary, not supported.",
            "report_or_synthesis",
            "No supported/strongly-supported verdict has comparative support below 0.10 unless explicitly labeled secondary.",
            track="conditional",
        )

    if network["isolated_evidence_count"] > len(result.extraction.evidence) * 0.5:
        add(
            88,
            "More than half of extracted evidence items have no displayed graph edge.",
            "Classify unlinked evidence as background, discarded, or pending-test evidence so readers can distinguish unused inventory from causal support.",
            "report_or_model",
            "Every unlinked evidence item is triaged as background, near-neutral, low-relevance, discarded, or pending-test evidence.",
            track="conditional",
        )

    conditional_cap = min([entry["cap"] for entry in conditional_caps], default=100)
    claim_scope_cap = min([entry["cap"] for entry in claim_scope_caps], default=100)
    return (
        conditional_cap,
        conditional_caps,
        claim_scope_cap,
        claim_scope_caps,
        conditional_recommendations,
        claim_scope_recommendations,
    )


def _optimality_status(
    conditional_score: int,
    conditional_caps: list[dict[str, Any]],
    claim_scope_caps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe optimality for the given-source critique and claim scope."""
    design_types = {"test_design", "hypothesis_design"}
    design_blockers = [
        cap for cap in conditional_caps
        if cap.get("action_type") in design_types
    ]
    report_blockers = [
        cap for cap in conditional_caps
        if cap.get("action_type") in {"report_or_model", "report_or_synthesis", "model_design"}
    ]
    if conditional_score >= 90 and not conditional_caps:
        return {
            "status": "optimal_given_accepted_sources",
            "next_iteration_mode": "none_for_given_source_critique",
            "blocked_by_external_evidence": False,
            "blocked_by_claim_scope": bool(claim_scope_caps),
            "acceptance_criteria": [],
            "claim_scope_acceptance_criteria": [
                cap["acceptance_criteria"] for cap in claim_scope_caps
            ],
        }
    if design_blockers:
        mode = "design_stronger_conditional_tests"
    elif report_blockers:
        mode = "repair_report_or_model"
    else:
        mode = "repair_conditional_analysis"
    return {
        "status": "not_optimal",
        "next_iteration_mode": mode,
        "blocked_by_external_evidence": bool(design_blockers),
        "blocked_by_claim_scope": bool(claim_scope_caps),
        "acceptance_criteria": [
            cap["acceptance_criteria"] for cap in conditional_caps
        ],
        "claim_scope_acceptance_criteria": [
            cap["acceptance_criteria"] for cap in claim_scope_caps
        ],
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
        "recommendations": [] if contract_ok else [
            "Repair ID/vector coverage before interpreting any substantive output."
        ],
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
        "recommendations": [] if comparative_visible and calibration_visible else [
            "State that support is comparative, not absolute; caveat low-posterior supported verdicts as secondary mechanisms."
        ],
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
        "recommendations": [] if temporal_visible and background_visible else [
            "Add a chronological timeline and separate background conditions from proximate causal mechanism traces."
        ],
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
        "recommendations": [] if fragile_warning_visible and robustness_visible else [
            "Surface high-support fragile winners, sensitivity ranges, rank stability, and prior sensitivity in the headline."
        ],
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
        "recommendations": [] if dependence_visible and weighted_visible else [
            "Show effective evidence after dependence pooling; do not let raw counts imply independent corroboration."
        ],
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
        "recommendations": [] if broad_visible else [
            "Flag broad winners and add sharper discriminators against overlapping rival hypotheses."
        ],
    }
    score += discrimination_points

    damaging_absences = [
        ev for ev in result.absence.evaluations if ev.severity == "damaging"
    ]
    packet = result.source_packet
    coverage = result.source_coverage
    source_packet_visible = packet is None or _report_has(
        report_html, "source material known to the grader"
    ) or _report_has(report_html, "source packet contract")
    source_coverage_visible = coverage is None or _report_has(report_html, "packet source coverage")
    source_scope_visible = (not damaging_absences) or _report_has(
        report_html, "source-scope", "absence"
    )
    source_scope_visible = source_scope_visible and source_packet_visible and source_coverage_visible
    source_points = 10 if source_scope_visible else 5
    categories["source_scope_and_absence"] = {
        "points": source_points,
        "max": 10,
        "damaging_absence_count": len(damaging_absences),
        "source_packet_present": packet is not None,
        "source_packet_visible": source_packet_visible,
        "source_coverage_present": coverage is not None,
        "source_coverage_visible": source_coverage_visible,
        "source_count": packet.source_count if packet else 0,
        "sources_with_evidence": coverage.sources_with_evidence if coverage else 0,
        "assigned_evidence_count": coverage.assigned_evidence_count if coverage else 0,
        "unassigned_evidence_count": len(coverage.unassigned_evidence_ids) if coverage else 0,
        "known_gap_count": packet.known_gap_count if packet else 0,
        "high_priority_gap_count": packet.high_priority_gap_count if packet else 0,
        "source_scope_visible": source_scope_visible,
        "recommendations": [] if source_scope_visible else [
            "Caveat damaging absence claims by source scope; specify where the missing trace should be found."
        ],
    }
    score += source_points

    network = _network_visual_stats(result)
    network_legend_visible = _report_has(report_html, "top driver edge", "not discarded")
    academic_review_visible = _report_has(
        report_html,
        "academic phd review",
        "recommendations by pipeline output",
        "proceed until optimal",
        "optimality gate",
        "evidence triage",
    )
    safe = "</script><script>" not in report_html and 'id="detail-' in report_html
    visual_ok = network["top_graph_connected"] and network_legend_visible and academic_review_visible
    categories["report_usability_and_safety"] = {
        "points": 5 if safe and visual_ok else 2 if safe else 0,
        "max": 5,
        "safe": safe,
        "visual_ok": visual_ok,
        "network_legend_visible": network_legend_visible,
        "academic_review_visible": academic_review_visible,
        **network,
        "recommendations": [] if safe and visual_ok else [
            "Connect the top-ranked hypothesis to visible support and disclose hidden isolated evidence as not discarded."
        ],
    }
    score += categories["report_usability_and_safety"]["points"]

    base_score = score
    (
        conditional_cap,
        conditional_caps,
        claim_scope_cap,
        claim_scope_caps,
        conditional_recommendations,
        claim_scope_recommendations,
    ) = _academic_caps(
        result,
        temporal=temporal,
        fragility=fragility,
        verdict_issues=verdict_issues,
        network=network,
    )
    conditional_score = min(base_score, conditional_cap)
    claim_scope_score = min(conditional_score, claim_scope_cap)
    academic_caps = conditional_caps + claim_scope_caps
    priority_recommendations = conditional_recommendations + claim_scope_recommendations
    optimality = _optimality_status(
        conditional_score,
        conditional_caps,
        claim_scope_caps,
    )

    return {
        "score": claim_scope_score,
        "base_score": base_score,
        "academic_cap": min(conditional_cap, claim_scope_cap),
        "grade": _grade(claim_scope_score),
        "conditional_score": conditional_score,
        "conditional_grade": _grade(conditional_score),
        "conditional_cap": conditional_cap,
        "claim_scope_score": claim_scope_score,
        "claim_scope_grade": _grade(claim_scope_score),
        "claim_scope_cap": claim_scope_cap,
        "optimality": optimality,
        "source_material_context": _source_material_context(result),
        "categories": categories,
        "academic_caps": academic_caps,
        "conditional_caps": conditional_caps,
        "claim_scope_caps": claim_scope_caps,
        "priority_recommendations": priority_recommendations,
        "conditional_priority_recommendations": conditional_recommendations,
        "claim_scope_priority_recommendations": claim_scope_recommendations,
    }


def _render_text(audit: dict[str, Any]) -> str:
    lines = [
        (
            f"Given-source grade: {audit['conditional_grade']} "
            f"({audit['conditional_score']}/100)"
        ),
        (
            f"Claim-scope grade: {audit['claim_scope_grade']} "
            f"({audit['claim_scope_score']}/100)"
        ),
    ]
    if audit.get("base_score") != audit.get("score"):
        lines.append(
            "Report-surface score: "
            f"{audit['base_score']}/100; conditional cap: "
            f"{audit['conditional_cap']}/100; claim-scope cap: "
            f"{audit['claim_scope_cap']}/100"
        )
    source_context = audit.get("source_material_context") or {}
    lines.append("- source_material_context:")
    for key in [
        "has_source_packet",
        "case_name",
        "source_count",
        "source_groups",
        "source_kinds",
        "sources_with_evidence",
        "assigned_evidence_count",
        "known_gap_count",
        "high_priority_gap_count",
        "high_priority_gaps",
    ]:
        lines.append(f"  {key}: {source_context.get(key)}")
    if source_context.get("accepted_sources"):
        lines.append("  accepted_sources:")
        for source in source_context["accepted_sources"]:
            lines.append(
                "    - "
                f"{source['source_id']}: {source['title']} "
                f"({source['source_kind']}, status={source['status']}, "
                f"evidence_count={source['evidence_count']})"
            )
    for name, details in audit["categories"].items():
        lines.append(f"- {name}: {details['points']}/{details['max']}")
        for key, value in details.items():
            if key in {"points", "max"}:
                continue
            lines.append(f"  {key}: {value}")
    if audit.get("conditional_caps"):
        lines.append("- conditional_caps:")
        for cap in audit["conditional_caps"]:
            lines.append(f"  cap {cap['cap']}: {cap['reason']}")
            lines.append(f"    recommendation: {cap['recommendation']}")
    if audit.get("claim_scope_caps"):
        lines.append("- claim_scope_caps:")
        for cap in audit["claim_scope_caps"]:
            lines.append(f"  cap {cap['cap']}: {cap['reason']}")
            lines.append(f"    recommendation: {cap['recommendation']}")
    if audit.get("priority_recommendations"):
        lines.append("- priority_recommendations:")
        for rec in audit["priority_recommendations"]:
            lines.append(f"  - {rec}")
    if audit.get("optimality"):
        lines.append("- optimality:")
        for key, value in audit["optimality"].items():
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
