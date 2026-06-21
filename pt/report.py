"""HTML report with vis.js network and Bootstrap 5."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from typing import Any, TypedDict

from pt.bayesian import INTERPRETIVE_LR_CAP, RESIDUAL_ID, lr_matrix
from pt.schemas import Evidence, Hypothesis, HypothesisPosterior, PredictionClassification, ProcessTracingResult


class _TemporalAudit(TypedDict):
    """Temporal proximity summary for the report audit card."""
    focal_year: int | None
    proximate: int
    background: int
    unknown: int
    total: int
    top_driver_background: list[str]


class _NetworkCoverage(TypedDict):
    """Connectivity summary for the network coverage audit."""
    node_count: int
    edge_count: int
    isolated_node_count: int
    isolated_evidence_count: int
    top_id: str | None
    top_degree: int


def _interpretive_caps(result: ProcessTracingResult) -> dict[str, float]:
    """Per-evidence interpretive caps, matching what the Bayesian update applied,
    so the report's displayed LRs agree with the model."""
    return {
        ev.id: INTERPRETIVE_LR_CAP
        for ev in result.extraction.evidence
        if ev.evidence_type == "interpretive"
    }


def _esc(s: str) -> str:
    return html.escape(str(s))


def _json_for_script(value: object) -> str:
    """Serialize JSON safely for embedding directly inside a script tag."""
    return json.dumps(value).replace("</", "<\\/").replace("<!--", "<\\!--")


def _dom_id(prefix: str, value: str) -> str:
    """Build a stable, selector-safe HTML id from model-provided IDs."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_") or "id"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{slug[:48]}-{digest}"


def _status_color(status: str) -> str:
    return {
        "strongly_supported": "#28a745",
        "supported": "#5cb85c",
        "weakened": "#f0ad4e",
        "eliminated": "#d9534f",
        "indeterminate": "#6c757d",
    }.get(status, "#6c757d")


_DIAGNOSTIC_TOOLTIPS = {
    "hoop": "Hoop test: necessary but not sufficient. Failing eliminates the hypothesis; passing keeps it alive but doesn't confirm it.",
    "smoking_gun": "Smoking gun: sufficient but not necessary. Passing strongly confirms the hypothesis; failing doesn't eliminate it.",
    "doubly_decisive": "Doubly decisive: both necessary and sufficient. Passing confirms, failing eliminates.",
    "straw_in_the_wind": "Straw in the wind: neither necessary nor sufficient. Provides a small update in one direction.",
}

_STATUS_TOOLTIPS = {
    "strongly_supported": "Strong Bayesian support from multiple decisive tests",
    "supported": "Moderate Bayesian support; posterior above prior",
    "weakened": "Evidence weighs against this hypothesis; posterior below prior",
    "eliminated": "Failed critical hoop or doubly decisive tests; very low posterior",
    "indeterminate": "Evidence is mixed or insufficient to move the posterior significantly",
}


def _diagnostic_badge(dtype: str) -> str:
    colors = {
        "hoop": "#17a2b8",
        "smoking_gun": "#dc3545",
        "doubly_decisive": "#6f42c1",
        "straw_in_the_wind": "#6c757d",
    }
    color = colors.get(dtype, "#6c757d")
    label = dtype.replace("_", " ").title()
    tip = _esc(_DIAGNOSTIC_TOOLTIPS.get(dtype, ""))
    return (
        f'<span class="badge" style="background:{color}" '
        f'data-bs-toggle="tooltip" data-bs-placement="top" title="{tip}">'
        f'{label}</span>'
    )


def _robustness_badge(robustness: str) -> str:
    colors = {"robust": "#28a745", "moderate": "#f0ad4e", "fragile": "#dc3545"}
    tips = {
        "robust": "Posterior driven by a few decisive evidence items with strong likelihood ratios (|log(LR)| > 1.6)",
        "moderate": "Posterior driven by a mix of strong and weak evidence",
        "fragile": "Posterior driven by accumulation of many weak evidence items (|log(LR)| < 0.7) that could individually go either way",
    }
    color = colors.get(robustness, "#6c757d")
    tip = _esc(tips.get(robustness, ""))
    return (
        f'<span class="badge" style="background:{color}" '
        f'data-bs-toggle="tooltip" data-bs-placement="top" title="{tip}">'
        f'{_esc(robustness.title())}</span>'
    )


def _status_badge(status: str) -> str:
    color = _status_color(status)
    tip = _esc(_STATUS_TOOLTIPS.get(status, ""))
    label = status.replace("_", " ").title()
    return (
        f'<span class="badge" style="background:{color}" '
        f'data-bs-toggle="tooltip" data-bs-placement="top" title="{tip}">'
        f'{_esc(label)}</span>'
    )


def _severity_badge(severity: str) -> str:
    colors = {"damaging": "#dc3545", "notable": "#f0ad4e", "minor": "#6c757d"}
    color = colors.get(severity, "#6c757d")
    return f'<span class="badge" style="background:{color}">{_esc(severity.title())}</span>'


def _th(label: str, tooltip: str = "", extra: str = "") -> str:
    """Sortable table header with optional tooltip."""
    tip_attr = ""
    if tooltip:
        tip_attr = f' data-bs-toggle="tooltip" data-bs-placement="top" title="{_esc(tooltip)}"'
    return f'<th class="sortable" role="button"{tip_attr} {extra}>{label}</th>'


def _render_narrative(text: str, ev_map: dict) -> str:
    """Render narrative with paragraph splitting, basic formatting, and evidence ID replacement."""
    paragraphs = text.split("\n\n")
    out = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        escaped = _esc(p)
        # Replace **bold** patterns
        import re
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', escaped)
        escaped = re.sub(r'\*(.+?)\*', r'<em>\1</em>', escaped)
        # Replace evidence IDs with human-readable descriptions
        for eid, ev in ev_map.items():
            esc_eid = _esc(eid)
            if esc_eid in escaped:
                desc = _esc(ev.description[:60])
                escaped = escaped.replace(
                    esc_eid,
                    f'<abbr class="text-primary" data-bs-toggle="tooltip" title="{_esc(ev.source_text[:200])}">{desc}</abbr>'
                )
        out.append(f"<p>{escaped}</p>")
    return "\n".join(out)


def _first_year(value: str | None) -> int | None:
    """Extract the first plausible four-digit year from free-text dates."""
    if not value:
        return None
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", value)
    return int(match.group(1)) if match else None


def _focal_year(result: ProcessTracingResult) -> int | None:
    """Infer the outcome year for temporal-proximity reporting."""
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


def _temporal_evidence_mix(
    result: ProcessTracingResult,
    top_id: str | None,
    ev_map: dict[str, Evidence],
    posteriors: dict[str, HypothesisPosterior],
) -> _TemporalAudit:
    """Summarize proximate/background evidence and background top drivers."""
    focal_year = _focal_year(result)
    proximate = 0
    background = 0
    unknown = 0
    if focal_year is None:
        unknown = len(result.extraction.evidence)
    else:
        for ev in result.extraction.evidence:
            year = _first_year(ev.approximate_date)
            if year is None:
                unknown += 1
            elif year >= focal_year - 2:
                proximate += 1
            elif year < focal_year - 5:
                background += 1

    top_driver_background: list[str] = []
    top = posteriors.get(top_id) if top_id else None
    if focal_year is not None and top:
        for evidence_id in top.top_drivers:
            driver_ev = ev_map.get(evidence_id)
            year = _first_year(driver_ev.approximate_date) if driver_ev else None
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


def _effective_evidence_count(result: ProcessTracingResult) -> tuple[float, int]:
    """Estimate effective observations after dependence-cluster pooling."""
    clustered: set[str] = set()
    cluster_effective = 0.0
    for cluster in result.testing.dependence_clusters:
        members = list(dict.fromkeys(cluster.evidence_ids))
        clustered.update(members)
        k = len(members)
        rho = cluster.dependence_strength
        cluster_effective += 1.0 + (k - 1) * (1.0 - rho)
    unclustered = len([ev for ev in result.extraction.evidence if ev.id not in clustered])
    return unclustered + cluster_effective, len(clustered)


def _network_coverage(
    result: ProcessTracingResult,
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
) -> _NetworkCoverage:
    """Summarize what the network hides or connects."""
    degree = {str(node["id"]): 0 for node in nodes}
    for edge in edges:
        source = str(edge["from"])
        target = str(edge["to"])
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1
    evidence_ids = {ev.id for ev in result.extraction.evidence}
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "isolated_node_count": sum(1 for node_id in degree if degree[node_id] == 0),
        "isolated_evidence_count": sum(1 for evidence_id in evidence_ids if degree.get(evidence_id, 0) == 0),
        "top_id": top_id,
        "top_degree": degree.get(top_id or "", 0),
    }


def _verdict_calibration_issues(
    result: ProcessTracingResult,
    posteriors: dict[str, HypothesisPosterior],
) -> list[str]:
    """Find synthesis labels that overstate low comparative support."""
    issues: list[str] = []
    for verdict in result.synthesis.verdicts:
        posterior = posteriors.get(verdict.hypothesis_id)
        support = posterior.final_posterior if posterior else None
        if support is None:
            continue
        if verdict.status in {"supported", "strongly_supported"} and support < 0.10:
            issues.append(
                f"{verdict.hypothesis_id} is labeled {verdict.status.replace('_', ' ')} "
                f"with comparative support {support:.3f}"
            )
    return issues


def _broad_winning_hypothesis(top_h: Hypothesis | None) -> bool:
    """Flag winners whose framing may absorb rival mechanisms."""
    if top_h is None:
        return False
    text = f"{top_h.description} {top_h.causal_mechanism}".lower()
    broad_terms = ["vacuum", "combined", "across", "all social", "confluence", "multiple"]
    return any(term in text for term in broad_terms)


def _render_output_quality_audit(
    result: ProcessTracingResult,
    *,
    top_id: str | None,
    top_h: Hypothesis | None,
    top_post: float,
    top_robust: str,
    posteriors: dict[str, HypothesisPosterior],
    ev_map: dict[str, Evidence],
    network_coverage: _NetworkCoverage,
) -> str:
    """Render caveats that keep the report from overstating its own evidence."""
    temporal = _temporal_evidence_mix(result, top_id, ev_map, posteriors)
    effective_count, clustered_count = _effective_evidence_count(result)
    verdict_issues = _verdict_calibration_issues(result, posteriors)
    damaging_absences = [ae for ae in result.absence.evaluations if ae.severity == "damaging"]
    total_evidence = len(result.extraction.evidence)
    cluster_count = len(result.testing.dependence_clusters)
    cards: list[str] = []

    if top_post >= 0.75 and top_robust == "fragile":
        cards.append(f"""
        <div class="alert alert-warning mb-2">
          <strong>High Support, Fragile.</strong>
          The top hypothesis has comparative support {top_post:.3f}, but the robustness label is fragile.
          Treat the magnitude as provisional and read it with the sensitivity range and rank-stable flag.
        </div>""")

    bg_drivers = temporal["top_driver_background"]
    bg_driver_text = ""
    if bg_drivers:
        labels = ", ".join(_esc(eid) for eid in bg_drivers)
        bg_driver_text = (
            f"<br><strong>background top-driver:</strong> {labels}. "
            "These influential items are background context rather than proximate outcome evidence."
        )
    focal = temporal["focal_year"] if temporal["focal_year"] is not None else "not inferred"
    cards.append(f"""
    <div class="alert alert-light border mb-2">
      <strong>Temporal Evidence Mix.</strong>
      Focal year: {_esc(str(focal))}. Proximate evidence: {temporal["proximate"]}/{temporal["total"]};
      background evidence: {temporal["background"]}/{temporal["total"]}; unknown dates:
      {temporal["unknown"]}/{temporal["total"]}.{bg_driver_text}
    </div>""")

    cards.append(f"""
    <div class="alert alert-light border mb-2">
      <strong>Effective Evidence vs raw counts.</strong>
      The report lists {total_evidence} raw counts of evidence items, but dependence cluster pooling estimates
      about {effective_count:.1f} effective evidence observations. Dependence: {cluster_count} cluster(s),
      {clustered_count} clustered item(s). Raw counts can overstate corroboration when items share a source,
      event, or mechanism.
    </div>""")

    cards.append(f"""
    <div class="alert alert-light border mb-2">
      <strong>Network Coverage.</strong>
      The causal network has {network_coverage["node_count"]} nodes and {network_coverage["edge_count"]} edges.
      The first view hides {network_coverage["isolated_node_count"]} isolated nodes by default, but they are
      <strong>not discarded</strong>; the network toggle restores them and the Evidence Inventory keeps the full list.
      {network_coverage["isolated_evidence_count"]} evidence item(s) currently have no visual edge because they are
      not top drivers and do not clear the displayed LR threshold. Top hypothesis {_esc(str(network_coverage["top_id"]))}
      has visual degree {network_coverage["top_degree"]}.
    </div>""")

    if verdict_issues:
        issues = "<br>".join(_esc(issue) for issue in verdict_issues)
        cards.append(f"""
        <div class="alert alert-warning mb-2">
          <strong>Verdict Calibration.</strong>
          {issues}. Read this as a secondary mechanism caveat, not as a standalone winning explanation.
        </div>""")

    if _broad_winning_hypothesis(top_h):
        cards.append("""
        <div class="alert alert-warning mb-2">
          <strong>Broad Winning Hypothesis.</strong>
          The leading hypothesis is broad enough to absorb mechanisms from rivals. Interpret the win as a
          comparative umbrella explanation unless the diagnostic matrix identifies evidence that separates it
          from narrower alternatives.
        </div>""")

    if damaging_absences:
        cards.append(f"""
        <div class="alert alert-warning mb-2">
          <strong>Source-scope Absence.</strong>
          {len(damaging_absences)} damaging absence finding(s) depend on whether the input text's scope should
          contain the missing micro-evidence. Broad overview texts may omit such details even when the mechanism
          is real, so absence should be read with source-scope caution.
        </div>""")

    if not cards:
        cards.append("""
        <div class="alert alert-success mb-2">
          <strong>No major output-quality caveats triggered.</strong>
          Contract checks, temporal proximity, robustness, dependence, verdict calibration, and absence handling
          did not raise visible report warnings.
        </div>""")

    return f"""
    <div class="card mb-4 shadow-sm border-warning">
      <div class="card-header bg-warning text-dark"><h4 class="mb-0">Output Quality Audit</h4></div>
      <div class="card-body">
        <p class="small text-muted mb-3">This audit highlights conditions that can make a ranked result misleading even when the pipeline contract is valid.</p>
        {''.join(cards)}
      </div>
    </div>"""


def _evidence_signal_badge(evidence_id: str, matrix: dict[str, dict[str, float]]) -> str:
    """Render the strongest displayed Bayesian signal for one evidence item."""
    lrs = matrix.get(evidence_id, {})
    if not lrs:
        return '<span class="badge bg-secondary">not tested</span>'
    hyp_id, lr = max(
        lrs.items(),
        key=lambda item: abs(math.log(max(item[1], 0.01))),
    )
    strength = abs(math.log(max(lr, 0.01)))
    if strength < 0.01:
        return '<span class="badge bg-secondary">neutral LR 1.00</span>'
    if lr > 1:
        return f'<span class="badge bg-success">supports {_esc(hyp_id)} LR {lr:.2f}</span>'
    return f'<span class="badge bg-danger">opposes {_esc(hyp_id)} LR {lr:.2f}</span>'


def _render_temporal_timeline(result: ProcessTracingResult) -> str:
    """Render a chronological table of extracted events and evidence."""
    hyp_ids = [h.id for h in result.hypothesis_space.hypotheses]
    matrix = dict(lr_matrix(result.testing, hyp_ids, _interpretive_caps(result)))
    top_driver_for: dict[str, list[str]] = {}
    for posterior in result.bayesian.posteriors:
        for evidence_id in posterior.top_drivers:
            top_driver_for.setdefault(evidence_id, []).append(posterior.hypothesis_id)

    edge_counts: dict[str, list[str]] = {}
    for edge in result.extraction.causal_edges:
        edge_counts.setdefault(edge.source_id, []).append(f"out: {edge.relationship}")
        edge_counts.setdefault(edge.target_id, []).append(f"in: {edge.relationship}")

    timeline_rows: list[tuple[tuple[int, int, str], str]] = []
    for event in result.extraction.events:
        year = _first_year(event.date)
        date_label = event.date or "unknown"
        role = "; ".join(edge_counts.get(event.id, [])) or "no extracted causal edge"
        row = f"""
        <tr>
          <td>{_esc(date_label)}</td>
          <td><span class="badge bg-primary">Event</span></td>
          <td><code>{_esc(event.id)}</code><br>{_esc(event.description)}</td>
          <td class="small">{_esc(role)}</td>
        </tr>"""
        timeline_rows.append(((1 if year is None else 0, year or 9999, event.id), row))

    for evidence in result.extraction.evidence:
        year = _first_year(evidence.approximate_date)
        date_label = evidence.approximate_date or "unknown"
        type_class = "bg-info" if evidence.evidence_type == "empirical" else "bg-warning text-dark"
        top_driver_badges = "".join(
            f' <span class="badge bg-dark">top driver {_esc(hyp_id)}</span>'
            for hyp_id in top_driver_for.get(evidence.id, [])
        )
        row = f"""
        <tr>
          <td>{_esc(date_label)}</td>
          <td><span class="badge {type_class}">{_esc(evidence.evidence_type.title())}</span></td>
          <td><code>{_esc(evidence.id)}</code><br>{_esc(evidence.description)}{top_driver_badges}</td>
          <td class="small">{_evidence_signal_badge(evidence.id, matrix)}</td>
        </tr>"""
        timeline_rows.append(((1 if year is None else 0, year or 9999, evidence.id), row))

    rows = "".join(row for _, row in sorted(timeline_rows, key=lambda item: item[0]))
    return f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0">Temporal Causal Timeline</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#timelineBody">Collapse</button>
      </div>
      <div class="collapse show" id="timelineBody">
      <div class="card-body">
        <p class="small text-muted">Extracted events and evidence are ordered by date before the network view. Temporal order is necessary for causal interpretation but is not sufficient by itself; use this with the diagnostic matrix and evidence links.</p>
        <div class="table-responsive" style="max-height:520px;overflow:auto">
          <table class="table table-sm table-striped sortable-table" id="timeline-table">
            <thead><tr>
              {_th("Date")}
              {_th("Type")}
              {_th("Item")}
              {_th("Causal/Bayesian signal")}
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
      </div>
    </div>"""


def _diagnostic_strength_summary(result: ProcessTracingResult) -> dict[str, int | float]:
    """Summarize diagnostic strength of evidence likelihood vectors."""
    hyp_ids = [h.id for h in result.hypothesis_space.hypotheses]
    matrix = lr_matrix(result.testing, hyp_ids, _interpretive_caps(result))
    strengths: list[float] = []
    for _, lrs in matrix:
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
        "decisive": sum(1 for strength in strengths if strength >= decisive_threshold),
        "moderate": sum(
            1 for strength in strengths if moderate_threshold <= strength < decisive_threshold
        ),
        "weak": sum(1 for strength in strengths if 0.1 < strength < moderate_threshold),
        "near_neutral": sum(1 for strength in strengths if strength <= 0.1),
        "max_log_lr": round(max(strengths), 3) if strengths else 0.0,
    }


def _evidence_triage_summary(result: ProcessTracingResult) -> dict[str, Any]:
    """Classify extracted evidence by how it is used in the current analysis."""
    focal_year = _focal_year(result)
    hyp_ids = [h.id for h in result.hypothesis_space.hypotheses]
    matrix = dict(lr_matrix(result.testing, hyp_ids, _interpretive_caps(result)))
    by_item = {item.evidence_id: item for item in result.testing.evidence_likelihoods}
    top_driver_ids = {
        evidence_id
        for posterior in result.bayesian.posteriors
        for evidence_id in posterior.top_drivers
    }
    counts = {
        "top_driver": 0,
        "displayed_discriminator": 0,
        "background_weak_signal": 0,
        "low_relevance": 0,
        "near_neutral": 0,
        "tested_unlinked": 0,
    }
    labels = {
        "top_driver": "Top driver",
        "displayed_discriminator": "Displayed discriminator",
        "background_weak_signal": "Background weak signal",
        "low_relevance": "Low relevance",
        "near_neutral": "Near-neutral",
        "tested_unlinked": "Tested, not displayed",
    }
    actions = {
        "top_driver": "Audit source quality and temporal proximity before treating as mechanism evidence.",
        "displayed_discriminator": "Check whether the signal is independent or shares a source/mechanism with other items.",
        "background_weak_signal": "Keep as enabling context unless paired with proximate mechanism traces.",
        "low_relevance": "Do not use as causal support without a better relevance justification.",
        "near_neutral": "Move to inventory/context unless a new hypothesis makes it discriminating.",
        "tested_unlinked": "Classify manually as background, discarded, or pending a sharper test.",
    }
    samples: dict[str, list[str]] = {key: [] for key in counts}

    for evidence in result.extraction.evidence:
        item = by_item.get(evidence.id)
        lrs = matrix.get(evidence.id, {})
        max_strength = max(
            (abs(math.log(max(lr, 0.01))) for lr in lrs.values()),
            default=0.0,
        )
        displayed = evidence.id in top_driver_ids or any(
            not (0.67 <= lr <= 1.5) for lr in lrs.values()
        )
        year = _first_year(evidence.approximate_date)
        if evidence.id in top_driver_ids:
            bucket = "top_driver"
        elif displayed:
            bucket = "displayed_discriminator"
        elif item is not None and item.relevance < 0.4:
            bucket = "low_relevance"
        elif max_strength <= 0.1:
            bucket = "near_neutral"
        elif focal_year is not None and year is not None and year < focal_year - 5:
            bucket = "background_weak_signal"
        else:
            bucket = "tested_unlinked"

        counts[bucket] += 1
        if len(samples[bucket]) < 3:
            samples[bucket].append(f"{evidence.id}: {evidence.description[:80]}")

    return {
        "counts": counts,
        "labels": labels,
        "actions": actions,
        "samples": samples,
    }


def _render_academic_review(
    result: ProcessTracingResult,
    *,
    top_id: str | None,
    top_h: Hypothesis | None,
    top_post: float,
    top_robust: str,
    posteriors: dict[str, HypothesisPosterior],
    ev_map: dict[str, Evidence],
    network_coverage: _NetworkCoverage,
) -> str:
    """Render a PhD-level methods critique with concrete next steps."""
    temporal = _temporal_evidence_mix(result, top_id, ev_map, posteriors)
    diagnostic = _diagnostic_strength_summary(result)
    verdict_issues = _verdict_calibration_issues(result, posteriors)
    limitations_text = " ".join(result.synthesis.limitations).lower()
    single_source_limited = any(
        term in limitations_text
        for term in ("single historical text", "single source", "single text")
    )
    broad_winner = _broad_winning_hypothesis(top_h)
    triage = _evidence_triage_summary(result)
    triage_counts = triage["counts"]
    triage_rows = "".join(
        f"""
        <tr>
          <td><strong>{_esc(triage['labels'][key])}</strong></td>
          <td>{count}</td>
          <td>{_esc('; '.join(triage['samples'][key]) or 'None')}</td>
          <td>{_esc(triage['actions'][key])}</td>
        </tr>"""
        for key, count in triage_counts.items()
    )
    total_evidence = temporal["total"] or 1
    proximate_share = temporal["proximate"] / total_evidence
    high_fragile = top_post >= 0.75 and top_robust == "fragile"
    too_many_unlinked = network_coverage["isolated_evidence_count"] > len(result.extraction.evidence) * 0.5
    external_blockers: list[str] = []
    if single_source_limited:
        external_blockers.append("single-source corpus")
    if diagnostic["decisive"] == 0 and diagnostic["moderate"] == 0:
        external_blockers.append("weak diagnostic tests")
    elif diagnostic["decisive"] == 0:
        external_blockers.append("no decisive diagnostic test")
    if proximate_share < 0.20:
        external_blockers.append("thin proximate evidence")
    if temporal["top_driver_background"]:
        external_blockers.append("background top drivers")
    if broad_winner:
        external_blockers.append("broad hypothesis design")
    if high_fragile:
        external_blockers.append("high-support fragile winner")
    if verdict_issues:
        external_blockers.append("verdict calibration")
    if too_many_unlinked:
        external_blockers.append("untriaged isolated evidence")
    optimal_for_corpus = not external_blockers
    rows = [
        (
            "Input corpus and source base",
            "Single-text or broad-overview input is not enough for PhD-level causal identification." if single_source_limited else "Source scope does not trigger an active cap; limitations are documented in synthesis.",
            "Build or preserve a source packet with primary documents, rival secondary accounts, source genre metadata, and a note on what each source can and cannot reveal.",
        ),
        (
            "Extraction and provenance",
            f"{len(result.extraction.evidence)} evidence items extracted; {network_coverage['isolated_evidence_count']} currently have no displayed graph edge.",
            "Classify every evidence item as mechanism trace, background condition, context, discarded, or pending-test evidence; preserve source quote and date confidence.",
        ),
        (
            "Hypothesis space",
            "The leading hypothesis is broad or absorptive." if broad_winner else "Leading hypothesis is not flagged as broad or absorptive; rival mechanisms are visibly separated.",
            "Split broad hypotheses when flagged; otherwise preserve pairwise discriminators and keep overlap visible in synthesis.",
        ),
        (
            "Diagnostic tests",
            f"Diagnostic strength: {diagnostic['decisive']} decisive, {diagnostic['moderate']} moderate, {diagnostic['weak']} weak, {diagnostic['near_neutral']} near-neutral items.",
            "Pre-register hoop and smoking-gun tests before likelihood scoring; seek direct traces unlikely under rival hypotheses when decisive counts are low.",
        ),
        (
            "Temporal process sequence",
            f"Only {temporal['proximate']}/{temporal['total']} evidence items are proximate to the focal outcome; background top drivers: {len(temporal['top_driver_background'])}.",
            "Construct and preserve a dated mechanism sequence for the final decision window; distinguish enabling background from proximate causal action.",
        ),
        (
            "Bayesian update and dependence",
            f"Top hypothesis {top_id or 'N/A'} has support {top_post:.3f} with robustness {top_robust}.",
            "Treat high-support fragile results as rankings, not calibrated truth probabilities; audit dependence clusters and rerun sensitivity after adding stronger traces.",
        ),
        (
            "Absence, counterfactuals, and scope",
            f"{sum(1 for ae in result.absence.evaluations if ae.severity == 'damaging')} damaging absence finding(s); source scope determines whether absence is meaningful.",
            "Specify the archive or source genre where the missing trace should appear; add counterfactual evidence on failed alternatives and non-events.",
        ),
        (
            "Synthesis and claims",
            f"{len(verdict_issues)} verdict calibration issue(s) detected.",
            "Phrase conclusions as exploratory under the present corpus; separate mechanism plausibility from publication-strength causal identification.",
        ),
    ]
    row_html = "".join(
        f"""
        <tr>
          <td><strong>{_esc(output)}</strong></td>
          <td>{_esc(critique)}</td>
          <td>{_esc(recommendation)}</td>
        </tr>"""
        for output, critique, recommendation in rows
    )
    optimal_steps = [
        "Freeze a sharper research question and focal decision window.",
        "Assemble a source packet with independent primary and secondary evidence.",
        "Split broad hypotheses and define pairwise discriminators before testing.",
        "Collect proximate dated traces for the decisive sequence.",
        "Rerun likelihood scoring, dependence clustering, sensitivity, and this audit.",
        "Only then upgrade from exploratory ranking to a PhD-level causal claim.",
    ]
    steps_html = "".join(f"<li>{_esc(step)}</li>" for step in optimal_steps)
    blockers_html = "".join(f"<li>{_esc(blocker)}</li>" for blocker in external_blockers)
    if optimal_for_corpus:
        status_text = (
            "PhD-review-ready under the current corpus. This is not a claim of historical truth; "
            "it means the report, source scope, temporal sequence, discriminators, and diagnostic "
            "tests clear the active audit gates."
        )
        gate_html = """
        <div class="alert alert-success">
          <strong>Optimality Gate:</strong> optimal_for_current_corpus. Next iteration mode:
          <strong>none</strong>. No active academic evidence caps remain.
        </div>"""
        proceed_html = "<p class=\"small mb-0\">No required iteration remains under the current corpus. Future work should add archival sources only if the research question needs stronger external validation.</p>"
    else:
        status_text = (
            "Exploratory process-tracing output under a limited input corpus. It is useful for "
            "hypothesis generation and audit planning, not yet a PhD-level causal demonstration."
        )
        gate_html = f"""
        <div class="alert alert-danger">
          <strong>Optimality Gate:</strong> not optimal for a PhD-level causal claim. Next iteration mode:
          <strong>collect or design evidence</strong>, not report polishing. Blocking conditions:
          <ul class="mb-0">{blockers_html}</ul>
        </div>"""
        proceed_html = f"<ol>{steps_html}</ol>"
    card_border = "border-success" if optimal_for_corpus else "border-danger"
    header_class = "bg-success" if optimal_for_corpus else "bg-danger"
    return f"""
    <div class="card mb-4 shadow-sm {card_border}">
      <div class="card-header {header_class} text-white"><h4 class="mb-0">Academic PhD Review</h4></div>
      <div class="card-body">
        <p><strong>Current scholarly status:</strong> {_esc(status_text)}</p>
        {gate_html}
        <h5>Recommendations by Pipeline Output</h5>
        <div class="table-responsive">
          <table class="table table-sm table-bordered">
            <thead><tr>
              <th>Output</th>
              <th>PhD-level critique</th>
              <th>How to improve</th>
            </tr></thead>
            <tbody>{row_html}</tbody>
          </table>
        </div>
        <h5>Evidence Triage</h5>
        <div class="table-responsive">
          <table class="table table-sm table-bordered">
            <thead><tr>
              <th>Class</th>
              <th>Count</th>
              <th>Examples</th>
              <th>Next action</th>
            </tr></thead>
            <tbody>{triage_rows}</tbody>
          </table>
        </div>
        <h5>Proceed Until Optimal</h5>
        {proceed_html}
      </div>
    </div>"""


def _build_vis_data(result: ProcessTracingResult) -> tuple[list[dict], list[dict]]:
    """Build vis.js nodes and edges including evidence-hypothesis links."""
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    ext = result.extraction
    posteriors = {p.hypothesis_id: p.final_posterior for p in result.bayesian.posteriors}
    posterior_objs = {p.hypothesis_id: p for p in result.bayesian.posteriors}
    node_ids = set()

    for e in ext.events:
        nodes.append({
            "id": e.id, "label": e.description[:40], "title": _esc(e.description),
            "color": "#66b3ff", "shape": "dot", "size": 15, "group": "event",
        })
        node_ids.add(e.id)

    for a in ext.actors:
        nodes.append({
            "id": a.id, "label": a.name[:30], "title": _esc(a.description),
            "color": "#ff99cc", "shape": "dot", "size": 12, "group": "actor",
            "hidden": True,
        })
        node_ids.add(a.id)

    for ev in ext.evidence:
        nodes.append({
            "id": ev.id, "label": ev.description[:40], "title": _esc(ev.source_text[:200]),
            "color": "#ff6666", "shape": "diamond", "size": 10, "group": "evidence",
        })
        node_ids.add(ev.id)

    for m in ext.mechanisms:
        nodes.append({
            "id": m.id, "label": m.description[:40], "title": _esc(m.description),
            "color": "#99ff99", "shape": "dot", "size": 12, "group": "mechanism",
            "hidden": True,
        })
        node_ids.add(m.id)

    for h in result.hypothesis_space.hypotheses:
        post = posteriors.get(h.id, 0)
        size = 15 + post * 30
        nodes.append({
            "id": h.id, "label": f"{h.id}: {h.description[:30]}",
            "title": _esc(f"{h.description}\nSupport: {post:.3f}"),
            "color": "#ffcc00", "shape": "star", "size": int(size),
            "group": "hypothesis",
        })
        node_ids.add(h.id)

    # Causal edges from extraction
    for ce in ext.causal_edges:
        if ce.source_id in node_ids and ce.target_id in node_ids:
            edges.append({
                "from": ce.source_id, "to": ce.target_id,
                "label": ce.relationship[:20], "arrows": "to",
                "color": {"color": "#999"}, "group": "causal",
            })

    # Evidence-hypothesis edges from the likelihood matrix (same caps as the update)
    hyp_ids = [h.id for h in result.hypothesis_space.hypotheses]
    top_driver_pairs = {
        (hyp_id, evidence_id)
        for hyp_id, posterior in posterior_objs.items()
        if hyp_id in hyp_ids
        for evidence_id in posterior.top_drivers
    }
    for evidence_id, lrs in lr_matrix(result.testing, hyp_ids, _interpretive_caps(result)):
        if evidence_id not in node_ids:
            continue
        for hyp_id, lr in lrs.items():
            if hyp_id not in node_ids:
                continue
            is_top_driver = (hyp_id, evidence_id) in top_driver_pairs
            if 0.67 <= lr <= 1.5 and not is_top_driver:
                continue  # Skip uninformative
            log_lr = abs(math.log(max(lr, 0.01)))
            width = max(0.9 if is_top_driver else 0.5, min(4, log_lr))
            color = "#28a745" if lr > 1 else "#dc3545"
            edge: dict[str, object] = {
                "from": evidence_id, "to": hyp_id,
                "arrows": "to", "width": round(width, 1),
                "color": {"color": color, "opacity": 0.85 if is_top_driver else 0.6},
                "group": "evidence_link",
                "title": f"LR={lr:.2f}" + ("; top driver edge" if is_top_driver else ""),
            }
            if is_top_driver and 0.67 <= lr <= 1.5:
                edge["dashes"] = [5, 5]
            edges.append(edge)

    return nodes, edges


def generate_report(result: ProcessTracingResult) -> str:
    """Generate self-contained HTML report."""

    posteriors = {p.hypothesis_id: p for p in result.bayesian.posteriors}
    sensitivity = {s.hypothesis_id: s for s in result.bayesian.sensitivity}
    h_map = {h.id: h for h in result.hypothesis_space.hypotheses}
    # The residual hypothesis is added only at the Bayesian stage; give the report a
    # description so it renders in the posterior table / network.
    if any(p.hypothesis_id == RESIDUAL_ID for p in result.bayesian.posteriors) and RESIDUAL_ID not in h_map:
        h_map[RESIDUAL_ID] = Hypothesis(
            id=RESIDUAL_ID,
            description="None of the listed explanations (other cause, or a genuinely conjunctural combination)",
            source="residual",
            theoretical_basis="Exhaustiveness: reserve mass so the listed set need not contain the truth.",
            causal_mechanism="(residual — no specific mechanism)",
            observable_predictions=[],
        )
    ev_map = {e.id: e for e in result.extraction.evidence}

    vis_nodes, vis_edges = _build_vis_data(result)
    network_coverage = _network_coverage(result, vis_nodes, vis_edges)
    nodes_json = _json_for_script(vis_nodes)
    edges_json = _json_for_script(vis_edges)

    # -- Build prediction lookup: (hypothesis_id, prediction_id) -> PredictionClassification
    pred_class_map: dict[tuple[str, str], PredictionClassification] = {}
    for pc in result.testing.prediction_classifications:
        pred_class_map[(pc.hypothesis_id, pc.prediction_id)] = pc

    # -- Build prediction descriptions: pred_id -> description
    pred_desc_map: dict[str, str] = {}
    for h in result.hypothesis_space.hypotheses:
        for pred in h.observable_predictions:
            pred_desc_map[pred.id] = pred.description

    # ===== Section 1: Executive Summary =====
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    top_h = h_map.get(top_id) if top_id else None
    top_post = posteriors[top_id].final_posterior if top_id and top_id in posteriors else 0
    top_robust = posteriors[top_id].robustness if top_id and top_id in posteriors else "unknown"

    # Slice 4: surface the support interval + rank/prior stability as the headline.
    top_sens = sensitivity.get(top_id) if top_id else None
    interval_badge = ""
    if top_sens:
        rank_txt = "rank-stable" if top_sens.rank_stable else "rank NOT stable"
        interval_badge = (
            f'<span class="badge bg-light text-dark border" data-bs-toggle="tooltip" '
            f'title="Support range when the most influential likelihoods are perturbed ±50%, and whether the ranking holds.">'
            f'range {top_sens.posterior_low:.3f}–{top_sens.posterior_high:.3f} · {rank_txt}</span>'
        )
    prior_badge = ""
    ps = result.bayesian.prior_sensitivity
    if ps:
        ptxt = "robust to prior" if ps.stable_under_prior_perturbation else "prior-sensitive"
        pcolor = "bg-success" if ps.stable_under_prior_perturbation else "bg-warning text-dark"
        prior_badge = (
            f'<span class="badge {pcolor}" data-bs-toggle="tooltip" '
            f'title="Whether the top hypothesis stays on top when each prior is up/down-weighted {ps.perturbation_factor:g}×.">'
            f'{ptxt}</span>'
        )

    # Honest overconfidence flag: a near-degenerate top support that is still "fragile"
    # after dependence pooling means the items weren't grouped strongly enough to offset
    # accumulation. Treat such a result as a ranking, not a calibrated number.
    overconfidence_banner = ""
    if top_post > 0.99 and top_robust == "fragile":
        overconfidence_banner = (
            '<div class="alert alert-warning mb-3"><strong>Likely overconfident.</strong> '
            'The top support is near-degenerate and the posterior is <em>fragile</em> - driven by '
            'accumulation of many weakly-discriminating (and possibly correlated) evidence items. '
            'Dependence pooling is applied, but if it did not group these items strongly enough the '
            'magnitude remains unreliable. Read this as a <strong>ranking</strong>, not a calibrated probability.</div>'
        )

    exec_summary = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header bg-primary text-white"><h4 class="mb-0">Executive Summary</h4></div>
      <div class="card-body">
        {overconfidence_banner}
        <p><strong>Research Question:</strong> {_esc(result.hypothesis_space.research_question)}</p>
        <p><strong>Top Hypothesis:</strong> {_esc(top_h.description) if top_h else 'N/A'}
           <span class="badge bg-warning text-dark"
             data-bs-toggle="tooltip" title="Comparative support: normalized odds across the listed hypotheses after Bayesian updating. NOT an absolute probability that the hypothesis is true, nor a causal-effect estimate.">
             Support: {top_post:.3f}</span>
           {interval_badge}
           {_robustness_badge(top_robust)}
           {prior_badge}
           {'<span class="badge bg-info" data-bs-toggle="tooltip" title="This analysis includes an analytical refinement pass (second reading)">Refined</span>' if result.is_refined else ''}</p>
        <p><strong>Hypotheses evaluated:</strong> {len(result.hypothesis_space.hypotheses)} &nbsp;|&nbsp;
           <strong>Evidence items:</strong> {len(result.extraction.evidence)} &nbsp;|&nbsp;
           <strong>Causal edges:</strong> {len(result.extraction.causal_edges)}</p>
        <p>{_esc(result.extraction.summary)}</p>
        <p class="small text-muted mb-0"><em>How to read the numbers:</em> values are <strong>comparative support</strong> &mdash;
           each hypothesis's share of the evidence-weighted odds <em>among the hypotheses listed here</em>. They are not
           absolute probabilities of truth, not causal-effect sizes, and not counterfactual quantities; a hypothesis not
           on the list cannot be supported. Read the ranking and the support range / robustness / stability flags, not the third decimal.</p>
      </div>
    </div>"""

    output_quality_audit = _render_output_quality_audit(
        result,
        top_id=top_id,
        top_h=top_h,
        top_post=top_post,
        top_robust=top_robust,
        posteriors=posteriors,
        ev_map=ev_map,
        network_coverage=network_coverage,
    )

    academic_review = _render_academic_review(
        result,
        top_id=top_id,
        top_h=top_h,
        top_post=top_post,
        top_robust=top_robust,
        posteriors=posteriors,
        ev_map=ev_map,
        network_coverage=network_coverage,
    )

    temporal_timeline = _render_temporal_timeline(result)

    # ===== Section 2: Interactive Network =====
    network_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0">Interactive Causal Network</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#networkBody">Expand</button>
      </div>
      <div class="collapse" id="networkBody">
      <div class="card-body">
        <div class="d-flex flex-wrap gap-3 mb-3 align-items-center">
          <strong>Show:</strong>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-event" checked>
            <label class="form-check-label" for="toggle-event"><span style="display:inline-block;width:12px;height:12px;background:#66b3ff;border-radius:50%"></span> Events</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-hypothesis" checked>
            <label class="form-check-label" for="toggle-hypothesis"><span style="display:inline-block;width:12px;height:12px;background:#ffcc00;border-radius:50%"></span> Hypotheses</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-evidence" checked>
            <label class="form-check-label" for="toggle-evidence"><span style="display:inline-block;width:12px;height:12px;background:#ff6666;border-radius:50%"></span> Evidence</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-actor">
            <label class="form-check-label" for="toggle-actor"><span style="display:inline-block;width:12px;height:12px;background:#ff99cc;border-radius:50%"></span> Actors</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-mechanism">
            <label class="form-check-label" for="toggle-mechanism"><span style="display:inline-block;width:12px;height:12px;background:#99ff99;border-radius:50%"></span> Mechanisms</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-evidence_link" checked>
            <label class="form-check-label" for="toggle-evidence_link">
              <span style="display:inline-block;width:12px;height:3px;background:#28a745;vertical-align:middle"></span>/<span style="display:inline-block;width:12px;height:3px;background:#dc3545;vertical-align:middle"></span> Evidence Links</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-isolated" checked>
            <label class="form-check-label" for="toggle-isolated">Hide isolated nodes</label>
          </div>
          <div class="btn-group btn-group-sm ms-auto">
            <button class="btn btn-outline-secondary" id="net-zoom-in" title="Zoom in">+</button>
            <button class="btn btn-outline-secondary" id="net-zoom-out" title="Zoom out">&minus;</button>
            <button class="btn btn-outline-secondary" id="net-fit" title="Fit all">Fit</button>
          </div>
        </div>
        <div id="network" style="width:100%;height:600px;border:1px solid #ddd;border-radius:4px"></div>
        <div id="network-info" class="alert alert-light mt-2 small">Click a node for details. Green edges = supporting evidence (LR &gt; 1). Red edges = opposing evidence (LR &lt; 1). A dashed top driver edge is shown even when the LR is weak because it is one of that hypothesis's largest updates.</div>
      </div>
      </div>
    </div>"""

    # ===== Section 3: Hypothesis Comparison Table =====
    h_rows = []
    for rank, hid in enumerate(result.bayesian.ranking, 1):
        hyp = h_map.get(hid)
        p = posteriors.get(hid)
        s = sensitivity.get(hid)
        if not hyp or not p:
            continue
        verdict = next((v for v in result.synthesis.verdicts if v.hypothesis_id == hid), None)
        status = verdict.status if verdict else "indeterminate"
        status_cell = _status_badge(status)
        if status in {"supported", "strongly_supported"} and p.final_posterior < 0.10:
            status_cell += (
                '<div class="small text-warning mt-1">secondary mechanism; low comparative support</div>'
            )
        sens_range = f"[{s.posterior_low:.3f}, {s.posterior_high:.3f}]" if s else "N/A"
        rank_badge = f'<span class="badge bg-{"success" if s and s.rank_stable else "warning"} bg-opacity-75" data-bs-toggle="tooltip" title="{"Rank is stable under sensitivity perturbation" if s and s.rank_stable else "Rank may change under sensitivity perturbation"}">{"Stable" if s and s.rank_stable else "Unstable"}</span>' if s else ""
        steelman_html = _esc(verdict.steelman) if verdict else ""
        mechanism_html = _esc(hyp.causal_mechanism)
        basis_html = _esc(hyp.theoretical_basis)

        source_label = hyp.source
        source_badge_class = "bg-secondary"
        if hyp.source == "text":
            source_badge_class = "bg-info"
        elif hyp.source == "generated":
            source_badge_class = "bg-warning text-dark"
        elif "theory" in hyp.source.lower():
            source_badge_class = "bg-purple"
        detail_id = _dom_id("detail", hid)

        h_rows.append(f"""
        <tr>
          <td>{rank}</td>
          <td><strong>{_esc(hid)}</strong></td>
          <td>{_esc(hyp.description)}</td>
          <td><span class="badge {source_badge_class}" data-bs-toggle="tooltip" title="{_esc(hyp.theoretical_basis)}">{_esc(source_label)}</span></td>
          <td>{p.prior:.3f}</td>
          <td><strong>{p.final_posterior:.3f}</strong></td>
          <td>{status_cell}</td>
          <td>{_robustness_badge(p.robustness)}</td>
          <td><span data-bs-toggle="tooltip" title="Posterior range under ±50% perturbation of top drivers">{sens_range}</span> {rank_badge}</td>
          <td>
            <a class="btn btn-sm btn-outline-secondary" data-bs-toggle="collapse" href="#{detail_id}" role="button">Details</a>
          </td>
        </tr>
        <tr class="collapse" id="{detail_id}">
          <td colspan="10" class="bg-light">
            <div class="row p-2">
              <div class="col-md-4"><strong>Causal Mechanism:</strong><br>{mechanism_html}</div>
              <div class="col-md-4"><strong>Theoretical Basis:</strong><br>{basis_html}</div>
              <div class="col-md-4"><strong>Steelman Case:</strong><br>{steelman_html}</div>
            </div>
          </td>
        </tr>""")

    comparison_table = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0">Hypothesis Comparison</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#comparisonBody">Expand</button>
      </div>
      <div class="collapse" id="comparisonBody">
      <div class="card-body table-responsive">
        <table class="table table-striped table-hover sortable-table" id="hypothesis-table">
          <thead><tr>
            {_th("Rank")}
            {_th("ID")}
            {_th("Hypothesis")}
            {_th("Source")}
            {_th("Prior", "Starting support before any evidence is considered (uniform across hypotheses by default)")}
            {_th("Support", "Comparative support: normalized odds across the listed hypotheses after Bayesian updating — not an absolute probability of truth")}
            {_th("Status")}
            {_th("Robustness", "Whether the posterior is driven by few decisive tests (robust) or many weak ones (fragile)")}
            {_th("Sensitivity", "Range of posterior values under ±50% perturbation of the most influential evidence")}
            <th></th>
          </tr></thead>
          <tbody>{''.join(h_rows)}</tbody>
        </table>
      </div>
      </div>
    </div>"""

    # ===== Section 4: Robustness & Sensitivity =====
    sens_rows = []
    for hid in result.bayesian.ranking:
        p = posteriors.get(hid)
        s = sensitivity.get(hid)
        hyp = h_map.get(hid)
        if not p or not hyp:
            continue
        # Top drivers with descriptions
        drivers_html = ""
        for did in p.top_drivers[:3]:
            ev = ev_map.get(did)
            desc = ev.description[:60] if ev else did
            # Find the LR for this driver
            lr_val = None
            for u in p.updates:
                if u.evidence_id == did:
                    lr_val = u.likelihood_ratio
                    break
            lr_str = f" (LR={lr_val:.2f})" if lr_val is not None else ""
            drivers_html += f'<div class="small mb-1"><code>{_esc(did)}</code>: {_esc(desc)}{lr_str}</div>'

        baseline = s.baseline_posterior if s else p.final_posterior
        low = s.posterior_low if s else baseline
        high = s.posterior_high if s else baseline
        # Scale to percentage for bar widths (max 100%)
        bar_low_pct = low * 100
        bar_high_pct = high * 100
        bar_base_pct = baseline * 100

        sens_rows.append(f"""
        <div class="mb-3 p-3 border rounded">
          <div class="d-flex justify-content-between align-items-center mb-2">
            <div><strong>{_esc(hid)}</strong>: {_esc(hyp.description[:80])}
              {_robustness_badge(p.robustness)}</div>
            <div class="text-end">
              <span class="badge bg-primary">Support: {baseline:.3f}</span>
              {'<span class="badge bg-' + ('success' if s.rank_stable else 'warning') + '">Rank ' + ('stable' if s.rank_stable else 'unstable') + '</span>' if s else ''}
            </div>
          </div>
          <div class="position-relative" style="height:24px;background:#e9ecef;border-radius:4px;overflow:hidden">
            <div style="position:absolute;left:{bar_low_pct:.1f}%;width:{(bar_high_pct - bar_low_pct):.1f}%;height:100%;background:rgba(0,123,255,0.2);border-radius:4px"
              data-bs-toggle="tooltip" title="Sensitivity range: [{low:.3f}, {high:.3f}]"></div>
            <div style="position:absolute;left:0;width:{bar_base_pct:.1f}%;height:100%;background:#007bff;border-radius:4px;opacity:0.8"></div>
            <span style="position:absolute;right:8px;top:2px;font-size:0.8em;font-weight:bold;color:#333">{baseline:.3f} [{low:.3f} – {high:.3f}]</span>
          </div>
          <div class="mt-2"><strong class="small">Top drivers:</strong>{drivers_html}</div>
        </div>""")

    robustness_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="How stable are the posteriors? Robustness measures whether results depend on a few strong pieces of evidence or many weak ones. Sensitivity shows how much posteriors change under perturbation.">Robustness &amp; Sensitivity Analysis</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#robustnessBody">Expand</button>
      </div>
      <div class="collapse" id="robustnessBody">
      <div class="card-body">
        {''.join(sens_rows)}
      </div>
      </div>
    </div>"""

    # ===== Section 5: Diagnostic Test Matrix (Accordion by hypothesis) =====
    # Built from the coherent likelihood vectors: per evidence item the relative
    # likelihood under each hypothesis, and the derived per-hypothesis LR
    # (relative_likelihood / geometric mean of the item's vector).
    hyp_ids = [h.id for h in result.hypothesis_space.hypotheses]
    matrix = dict(lr_matrix(result.testing, hyp_ids, _interpretive_caps(result)))  # evidence_id -> {hyp_id: lr}
    # evidence_id -> {hyp_id: (relative_likelihood, diagnostic_type)}, plus item meta
    vec_by_ev: dict[str, dict] = {}
    for item in result.testing.evidence_likelihoods:
        vec_by_ev[item.evidence_id] = {
            "relevance": item.relevance,
            "justification": item.justification,
            "cells": {
                hl.hypothesis_id: (hl.relative_likelihood, hl.diagnostic_type)
                for hl in item.hypothesis_likelihoods
            },
        }

    test_accordion_items = []
    total_evals = 0
    informative_evals = 0

    for h in result.hypothesis_space.hypotheses:
        h_label = f"{h.id}: {h.description[:60]}"
        matrix_rows: list[tuple[float, str]] = []
        for evidence_id, lrs in matrix.items():
            lr = lrs.get(h.id, 1.0)
            total_evals += 1
            if 0.9 < lr < 1.1:
                continue
            informative_evals += 1
            matrix_rows.append((lr, evidence_id))
        matrix_rows.sort(key=lambda x: abs(math.log(max(x[0], 0.01))), reverse=True)

        body = f"""
            <table class="table table-sm table-bordered mb-0 sortable-table">
              <thead><tr>
                {_th("Evidence")}
                {_th("Direction", "Whether this evidence favors (LR>1) or weighs against (LR<1) this hypothesis relative to the others")}
                {_th("Rel. likelihood", "Relative likelihood of this evidence under this hypothesis on the item's shared scale (only ratios across hypotheses matter)")}
                {_th("LR", "Derived likelihood ratio: relative likelihood / geometric mean across hypotheses, after relevance gating and capping. LR>1 favors this hypothesis")}
                {_th("Diagnostic", "Van Evera character of this evidence for this hypothesis")}
                {_th("Relevance", "How discriminating this evidence is (0-1). Below 0.4 = forced uninformative")}
                {_th("Justification")}
              </tr></thead>
              <tbody>"""
        for lr, evidence_id in matrix_rows:
            ev = ev_map.get(evidence_id)
            ev_label = ev.description[:80] if ev else evidence_id
            meta = vec_by_ev.get(evidence_id, {})
            rel_like, dtype = meta.get("cells", {}).get(h.id, (None, "straw_in_the_wind"))
            relevance = meta.get("relevance", 1.0)
            justification = meta.get("justification", "")
            lr_color = "#28a745" if lr > 1.1 else "#dc3545"
            direction = "favors" if lr > 1.1 else "against"
            dir_color = "#28a745" if lr > 1.1 else "#dc3545"
            rel_like_str = f"{rel_like:.2f}" if rel_like is not None else "—"
            body += f"""
                <tr>
                  <td class="small" data-bs-toggle="tooltip" title="{_esc(ev.source_text[:200]) if ev else ''}">{_esc(ev_label)}</td>
                  <td><span class="badge" style="background:{dir_color}">{direction}</span></td>
                  <td>{rel_like_str}</td>
                  <td><strong style="color:{lr_color}">{lr:.2f}</strong></td>
                  <td>{_diagnostic_badge(dtype)}</td>
                  <td>{relevance:.2f}</td>
                  <td class="small">{_esc(justification)}</td>
                </tr>"""
        body += "</tbody></table>"

        collapsed = "" if h == result.hypothesis_space.hypotheses[0] else "collapsed"
        show = "show" if h == result.hypothesis_space.hypotheses[0] else ""
        test_id = _dom_id("test", h.id)
        test_accordion_items.append(f"""
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button {collapsed}" type="button" data-bs-toggle="collapse" data-bs-target="#{test_id}">
              {_esc(h_label)} <span class="badge bg-info ms-2">{len(matrix_rows)} informative</span>
            </button>
          </h2>
          <div id="{test_id}" class="accordion-collapse collapse {show}" data-bs-parent="#testAccordion">
            <div class="accordion-body">{body}</div>
          </div>
        </div>""")

    skipped_count = total_evals - informative_evals
    skipped_note = f'<p class="text-muted mb-3">{informative_evals} informative / {total_evals} total evaluations shown. {skipped_count} uninformative (LR ≈ 1.0) hidden.</p>'

    test_matrix = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Each evidence item gets a likelihood vector across all hypotheses; the per-hypothesis LR is derived from that vector (relative likelihood over the geometric mean), so the comparisons are coherent.">Diagnostic Test Matrix</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#testMatrixBody">Expand</button>
      </div>
      <div class="collapse" id="testMatrixBody">
      <div class="card-body">
        {skipped_note}
        <div class="accordion" id="testAccordion">
          {''.join(test_accordion_items)}
        </div>
      </div>
      </div>
    </div>"""

    # ===== Section 6: Bayesian Update Summary =====
    trail_items = []
    for p_obj in result.bayesian.posteriors:
        hyp = h_map.get(p_obj.hypothesis_id)
        h_label = f"{p_obj.hypothesis_id}: {hyp.description[:50]}" if hyp else p_obj.hypothesis_id
        final_width = max(2, int(p_obj.final_posterior * 100))

        # Build SVG sparkline of posterior progression
        svg_points: list[tuple[float, float]] = []
        svg_w, svg_h = 500, 80
        updates_with_movement = [u for u in p_obj.updates if not (0.95 < u.likelihood_ratio < 1.05)]
        n_points = len(updates_with_movement) + 1  # +1 for the prior
        if n_points > 1:
            x_step = svg_w / max(n_points - 1, 1)
            # Start with prior
            svg_points.append((0, svg_h - p_obj.prior * svg_h))
            for i, u in enumerate(updates_with_movement):
                x = (i + 1) * x_step
                y = svg_h - u.posterior * svg_h
                svg_points.append((x, y))
            polyline_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in svg_points)
            # Color dots by LR direction
            svg_dots = ""
            for i, u in enumerate(updates_with_movement):
                x = (i + 1) * x_step
                y = svg_h - u.posterior * svg_h
                dot_color = "#28a745" if u.likelihood_ratio > 1 else "#dc3545"
                ev = ev_map.get(u.evidence_id)
                tip = f"{ev.description[:40] if ev else u.evidence_id} (LR={u.likelihood_ratio:.2f})"
                svg_dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{dot_color}"><title>{_esc(tip)}</title></circle>'
            sparkline_svg = f"""
            <svg width="100%" viewBox="0 0 {svg_w} {svg_h + 10}" style="max-height:100px" class="mt-1">
              <rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="#f8f9fa" rx="3"/>
              <line x1="0" y1="{svg_h - p_obj.prior * svg_h:.1f}" x2="{svg_w}" y2="{svg_h - p_obj.prior * svg_h:.1f}" stroke="#ccc" stroke-dasharray="4"/>
              <text x="{svg_w - 2}" y="{svg_h - p_obj.prior * svg_h - 3:.1f}" font-size="9" fill="#999" text-anchor="end">prior {p_obj.prior:.3f}</text>
              <polyline points="{polyline_str}" fill="none" stroke="#007bff" stroke-width="1.5"/>
              <circle cx="0" cy="{svg_h - p_obj.prior * svg_h:.1f}" r="3" fill="#007bff"><title>Prior: {p_obj.prior:.3f}</title></circle>
              {svg_dots}
              <text x="{svg_w - 2}" y="{svg_h + 9}" font-size="9" fill="#333" text-anchor="end">{n_points - 1} updates</text>
            </svg>"""
        else:
            sparkline_svg = ""
        trail_graph_id = _dom_id("trailGraph", p_obj.hypothesis_id)
        trail_table_id = _dom_id("trail", p_obj.hypothesis_id)

        # Build collapsible update trail table
        trail_rows = ""
        for u in p_obj.updates:
            ev = ev_map.get(u.evidence_id)
            ev_label = ev.description[:60] if ev else u.evidence_id
            lr_color = "#28a745" if u.likelihood_ratio > 1 else "#dc3545" if u.likelihood_ratio < 1 else "#6c757d"
            # Skip LR ≈ 1 in the detail table too
            if 0.95 < u.likelihood_ratio < 1.05:
                continue
            trail_rows += f"""
            <tr>
              <td class="small">{_esc(ev_label)}</td>
              <td style="color:{lr_color};font-weight:bold">{u.likelihood_ratio:.3f}</td>
              <td>{u.posterior:.4f}</td>
            </tr>"""

        trail_items.append(f"""
        <div class="mb-3 p-2 border-bottom">
          <div class="d-flex justify-content-between align-items-center mb-1">
            <strong>{_esc(h_label)}</strong>
            <div>
              {_robustness_badge(p_obj.robustness)}
              <span class="badge bg-primary ms-1">{p_obj.final_posterior:.3f}</span>
            </div>
          </div>
          <div style="background:#e9ecef;border-radius:4px;overflow:hidden;height:22px;position:relative">
            <div style="width:{final_width}%;height:100%;background:#007bff;border-radius:4px;transition:width 0.3s"></div>
            <span style="position:absolute;right:8px;top:1px;font-size:0.8em;font-weight:bold">{p_obj.prior:.3f} → {p_obj.final_posterior:.3f}</span>
          </div>
          <div class="mt-1">
            <a class="btn btn-sm btn-link p-0" data-bs-toggle="collapse" href="#{trail_graph_id}">Show update graph</a>
            <span class="text-muted mx-1">|</span>
            <a class="btn btn-sm btn-link p-0" data-bs-toggle="collapse" href="#{trail_table_id}">Show update table</a>
          </div>
          <div class="collapse" id="{trail_graph_id}">
            {sparkline_svg}
            <div class="small text-muted mt-1">Each dot is an evidence update. <span style="color:#28a745">Green</span> = LR &gt; 1 (supports). <span style="color:#dc3545">Red</span> = LR &lt; 1 (opposes). Dashed line = prior. Hover dots for details.</div>
          </div>
          <div class="collapse" id="{trail_table_id}">
            <table class="table table-sm mt-1 sortable-table" style="font-size:0.85em">
              <thead><tr>{_th("Evidence")}{_th("LR", "Likelihood Ratio after relevance gating and capping")}{_th("Cumulative Posterior", "Running posterior after this update")}</tr></thead>
              <tbody>{trail_rows}</tbody>
            </table>
          </div>
        </div>""")

    bayesian_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Starting from the prior, each evidence item updates the comparative support via its likelihood ratio. LR > 1 increases support, LR < 1 decreases it. The result is support relative to the listed hypotheses, not an absolute probability.">Bayesian Update Summary</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#bayesianBody">Expand</button>
      </div>
      <div class="collapse" id="bayesianBody">
      <div class="card-body">{''.join(trail_items)}</div>
      </div>
    </div>"""

    # ===== Section 7: Evidence List (collapsed by default) =====
    ev_rows = []
    for ev in result.extraction.evidence:
        type_badge = '<span class="badge bg-info">Empirical</span>' if ev.evidence_type == "empirical" else '<span class="badge bg-warning text-dark">Interpretive</span>'
        ev_rows.append(f"""
        <tr>
          <td><code class="small">{_esc(ev.id)}</code></td>
          <td>{_esc(ev.description)}</td>
          <td>{type_badge}</td>
          <td>{_esc(ev.approximate_date or 'N/A')}</td>
          <td>
            <span class="d-inline-block text-truncate" style="max-width:250px" data-bs-toggle="tooltip" title="{_esc(ev.source_text)}">{_esc(ev.source_text[:80])}...</span>
          </td>
        </tr>""")

    evidence_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0">Evidence Inventory</h4>
        <button class="btn btn-sm btn-outline-primary" type="button" data-bs-toggle="collapse" data-bs-target="#evidenceBody">
          Show/Hide ({len(result.extraction.evidence)} items)
        </button>
      </div>
      <div class="collapse" id="evidenceBody">
        <div class="card-body table-responsive">
          <table class="table table-sm table-striped sortable-table" id="evidence-table">
            <thead><tr>
              {_th("ID")}
              {_th("Description")}
              {_th("Type", "Empirical = facts/events/actions. Interpretive = historian arguments/scholarly claims.")}
              {_th("Date")}
              {_th("Source Text")}
            </tr></thead>
            <tbody>{''.join(ev_rows)}</tbody>
          </table>
        </div>
      </div>
    </div>"""

    # ===== Section 8: Absence-of-Evidence Findings =====
    absence_section = ""
    if result.absence and result.absence.evaluations:
        # Group by hypothesis
        absence_by_hyp: dict[str, list] = {}
        for ae in result.absence.evaluations:
            absence_by_hyp.setdefault(ae.hypothesis_id, []).append(ae)

        absence_items = []
        for hid, aes in absence_by_hyp.items():
            hyp = h_map.get(hid)
            h_label = f"{hid}: {hyp.description[:60]}" if hyp else hid
            rows = ""
            for ae in aes:
                pred_desc = pred_desc_map.get(ae.prediction_id, ae.prediction_id)
                extractable_tip = "Would this evidence appear in a text of this scope if it existed?"
                extractable_badge = (
                    f'<span class="badge bg-warning text-dark" data-bs-toggle="tooltip" title="{extractable_tip}">Yes</span>'
                    if ae.would_be_extractable
                    else f'<span class="badge bg-secondary" data-bs-toggle="tooltip" title="{extractable_tip}">No</span>'
                )
                rows += f"""
                <tr>
                  <td class="small">{_esc(pred_desc[:100])}</td>
                  <td>{_esc(ae.missing_evidence)}</td>
                  <td>{_severity_badge(ae.severity)}</td>
                  <td>{extractable_badge}</td>
                  <td class="small">{_esc(ae.reasoning)}</td>
                </tr>"""

            absence_items.append(f"""
            <div class="mb-3">
              <h6>{_esc(h_label)}</h6>
              <table class="table table-sm table-bordered">
                <thead><tr>
                  {_th("Prediction")}
                  {_th("Missing Evidence")}
                  {_th("Severity")}
                  {_th("Extractable?", "Would this evidence appear in a text of this scope and genre if it actually existed?")}
                  {_th("Reasoning")}
                </tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>""")

        absence_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Evidence that hypotheses predict should exist but was not found in the text. These are 'failed hoop tests' — the absence of expected evidence is informative about hypothesis validity.">Absence-of-Evidence Findings</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#absenceBody">Expand</button>
      </div>
      <div class="collapse" id="absenceBody">
      <div class="card-body">
        <p class="text-muted">Predicted evidence not found in the text. These findings inform the synthesis qualitatively but do <strong>not</strong> affect Bayesian posteriors.</p>
        {''.join(absence_items)}
      </div>
      </div>
    </div>"""

    # ===== Section 8b: Analytical Refinement =====
    refinement_section = ""
    if result.refinement is not None:
        ref = result.refinement

        # Summary stats
        stats_parts = []
        if ref.new_evidence:
            stats_parts.append(f"{len(ref.new_evidence)} new evidence")
        if ref.reinterpreted_evidence:
            stats_parts.append(f"{len(ref.reinterpreted_evidence)} reinterpretations")
        if ref.new_causal_edges:
            stats_parts.append(f"{len(ref.new_causal_edges)} new edges")
        if ref.spurious_extractions:
            stats_parts.append(f"{len(ref.spurious_extractions)} removed")
        if ref.hypothesis_refinements:
            stats_parts.append(f"{len(ref.hypothesis_refinements)} refinements")
        if ref.missing_mechanisms:
            stats_parts.append(f"{len(ref.missing_mechanisms)} missing mechanisms")
        stats_bar = " | ".join(stats_parts) if stats_parts else "No changes"

        # New evidence table
        new_ev_html = ""
        if ref.new_evidence:
            new_ev_rows = ""
            for ne in ref.new_evidence:
                type_badge = '<span class="badge bg-info">Empirical</span>' if ne.evidence_type == "empirical" else '<span class="badge bg-warning text-dark">Interpretive</span>'
                new_ev_rows += f"""
                <tr>
                  <td><code class="small">{_esc(ne.id)}</code></td>
                  <td>{_esc(ne.description)}</td>
                  <td>{type_badge}</td>
                  <td class="small"><em>{_esc(ne.source_text[:200])}</em></td>
                  <td class="small">{_esc(ne.rationale)}</td>
                </tr>"""
            new_ev_html = f"""
            <h6>New Evidence</h6>
            <table class="table table-sm table-striped">
              <thead><tr>
                {_th("ID")}{_th("Description")}{_th("Type")}{_th("Source Text")}{_th("Rationale")}
              </tr></thead>
              <tbody>{new_ev_rows}</tbody>
            </table>"""

        # Reinterpretations table
        reint_html = ""
        if ref.reinterpreted_evidence:
            reint_rows = ""
            for ri in ref.reinterpreted_evidence:
                reint_rows += f"""
                <tr>
                  <td><code class="small">{_esc(ri.evidence_id)}</code></td>
                  <td><span class="badge bg-secondary">{_esc(ri.original_type)}</span> &rarr; <span class="badge bg-info">{_esc(ri.new_type)}</span></td>
                  <td class="small">{_esc(ri.reinterpretation)}</td>
                </tr>"""
            reint_html = f"""
            <h6>Reinterpretations</h6>
            <table class="table table-sm table-striped">
              <thead><tr>
                {_th("Evidence ID")}{_th("Type Change")}{_th("Reinterpretation")}
              </tr></thead>
              <tbody>{reint_rows}</tbody>
            </table>"""

        # Spurious removals table
        spur_html = ""
        if ref.spurious_extractions:
            spur_rows = ""
            for se in ref.spurious_extractions:
                type_badge = '<span class="badge bg-danger">Evidence</span>' if se.item_type == "evidence" else '<span class="badge bg-warning text-dark">Causal Edge</span>'
                spur_rows += f"""
                <tr>
                  <td><code class="small" style="text-decoration:line-through">{_esc(se.item_id)}</code></td>
                  <td>{type_badge}</td>
                  <td class="small">{_esc(se.reason)}</td>
                </tr>"""
            spur_html = f"""
            <h6>Spurious Removals</h6>
            <table class="table table-sm table-striped">
              <thead><tr>
                {_th("Item ID")}{_th("Type")}{_th("Reason")}
              </tr></thead>
              <tbody>{spur_rows}</tbody>
            </table>"""

        # Hypothesis refinements (grouped by hypothesis)
        hyp_ref_html = ""
        if ref.hypothesis_refinements:
            by_hyp: dict[str, list] = {}
            for hr in ref.hypothesis_refinements:
                by_hyp.setdefault(hr.hypothesis_id, []).append(hr)

            hyp_ref_items = ""
            for hid, refinements in by_hyp.items():
                hyp = h_map.get(hid)
                h_label = f"{hid}: {hyp.description[:60]}" if hyp else hid
                ref_rows = ""
                for hr in refinements:
                    type_colors = {
                        "sharpen_mechanism": "bg-info",
                        "add_prediction": "bg-success",
                        "reframe": "bg-primary",
                        "merge_suggestion": "bg-warning text-dark",
                    }
                    badge_class = type_colors.get(hr.refinement_type, "bg-secondary")
                    label = hr.refinement_type.replace("_", " ").title()

                    if hr.refinement_type == "merge_suggestion":
                        ref_rows += f"""
                        <div class="alert alert-warning small mb-2">
                          <span class="badge {badge_class}">{label}</span>
                          {_esc(hr.description)}
                        </div>"""
                    else:
                        ref_rows += f"""
                        <div class="mb-2 small">
                          <span class="badge {badge_class}">{label}</span>
                          {_esc(hr.description)}
                        </div>"""

                hyp_ref_items += f"""
                <div class="mb-3">
                  <strong>{_esc(h_label)}</strong>
                  {ref_rows}
                </div>"""

            hyp_ref_html = f"""
            <h6>Hypothesis Refinements</h6>
            {hyp_ref_items}"""

        # Missing mechanisms table
        mech_html = ""
        if ref.missing_mechanisms:
            mech_rows = ""
            for mm in ref.missing_mechanisms:
                hyps = ", ".join(mm.relevant_hypotheses)
                mech_rows += f"""
                <tr>
                  <td class="small">{_esc(mm.description)}</td>
                  <td class="small"><em>{_esc(mm.source_text_support[:200])}</em></td>
                  <td class="small">{_esc(hyps)}</td>
                </tr>"""
            mech_html = f"""
            <h6>Missing Mechanisms</h6>
            <table class="table table-sm table-striped">
              <thead><tr>
                {_th("Description")}{_th("Source Text Support")}{_th("Relevant Hypotheses")}
              </tr></thead>
              <tbody>{mech_rows}</tbody>
            </table>"""

        # Analyst notes
        notes_html = ""
        if ref.analyst_notes:
            notes_html = f"""
            <h6>Analyst Notes</h6>
            <div class="narrative-body">{_render_narrative(ref.analyst_notes, ev_map)}</div>"""

        refinement_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center" style="background-color: #17a2b8; color: white;">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Second reading of the source text, informed by the full first-pass analysis. Surfaces missed evidence, reinterpretations, and hypothesis refinements.">Analytical Refinement (Second Reading)</h4>
        <button class="btn btn-sm btn-outline-light section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#refinementBody">Expand</button>
      </div>
      <div class="collapse" id="refinementBody">
      <div class="card-body">
        <p class="fw-bold">{_esc(stats_bar)}</p>
        {new_ev_html}
        {reint_html}
        {spur_html}
        {hyp_ref_html}
        {mech_html}
        {notes_html}
      </div>
      </div>
    </div>"""

    # ===== Section 9: Analytical Narrative =====
    narrative_html = _render_narrative(result.synthesis.analytical_narrative, ev_map)
    comparative_html = _render_narrative(result.synthesis.comparative_analysis, ev_map)

    narrative_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header bg-dark text-white"><h4 class="mb-0">Analytical Narrative</h4></div>
      <div class="card-body narrative-body">
        {narrative_html}
        <hr>
        <h5>Comparative Analysis</h5>
        {comparative_html}
      </div>
    </div>"""

    # ===== Section 10: Limitations & Further Research =====
    lim_items = "".join(f"<li>{_esc(l)}</li>" for l in result.synthesis.limitations)
    test_items = "".join(f"<li>{_esc(t)}</li>" for t in result.synthesis.suggested_further_tests)

    limitations_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header"><h4 class="mb-0">Limitations &amp; Further Research</h4></div>
      <div class="card-body">
        <h5>Limitations</h5>
        <ul>{lim_items}</ul>
        <h5>Suggested Further Tests</h5>
        <ul>{test_items}</ul>
      </div>
    </div>"""

    # ===== Assemble =====
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Process Tracing Analysis Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8f9fa; }}
    .card-header h4 {{ font-size: 1.1rem; }}
    .sortable th.sortable {{ cursor: pointer; user-select: none; white-space: nowrap; }}
    .sortable th.sortable:hover {{ background: rgba(0,0,0,0.05); }}
    .sortable th.sortable::after {{ content: ' ⇅'; opacity: 0.3; font-size: 0.8em; }}
    .sortable th.sortable.asc::after {{ content: ' ▲'; opacity: 0.7; }}
    .sortable th.sortable.desc::after {{ content: ' ▼'; opacity: 0.7; }}
    .narrative-body p {{ line-height: 1.7; margin-bottom: 1em; }}
    .narrative-body abbr {{ text-decoration: underline dotted; cursor: help; }}
    .badge {{ font-size: 0.8em; }}
    code {{ font-size: 0.85em; }}
    .accordion-button:not(.collapsed) {{ background-color: rgba(13,110,253,0.1); }}
  </style>
</head>
<body>
<div class="container-fluid py-4" style="max-width:1400px">
  <div class="d-flex justify-content-between align-items-center mb-4">
    <h1 class="mb-0">Process Tracing Analysis Report</h1>
    <button class="btn btn-sm btn-outline-secondary" id="expand-all-btn">Expand All Sections</button>
  </div>
  {exec_summary}
  {output_quality_audit}
  {academic_review}
  {temporal_timeline}
  {network_section}
  {comparison_table}
  {robustness_section}
  {test_matrix}
  {bayesian_section}
  {evidence_section}
  {absence_section}
  {refinement_section}
  {narrative_section}
  {limitations_section}
</div>

<script>
// === Tooltip initialization ===
document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function(el) {{
  new bootstrap.Tooltip(el);
}});

// === Section collapse toggle button text ===
document.querySelectorAll('.section-toggle').forEach(function(btn) {{
  var targetId = btn.getAttribute('data-bs-target');
  var target = document.querySelector(targetId);
  if (!target) return;
  target.addEventListener('shown.bs.collapse', function() {{ btn.textContent = 'Collapse'; }});
  target.addEventListener('hidden.bs.collapse', function() {{ btn.textContent = 'Expand'; }});
}});

// === Expand All / Collapse All ===
(function() {{
  var expandAllBtn = document.getElementById('expand-all-btn');
  var allExpanded = false;
  var sectionIds = ['#networkBody','#comparisonBody','#robustnessBody','#testMatrixBody','#bayesianBody','#evidenceBody','#absenceBody','#refinementBody'];
  expandAllBtn.addEventListener('click', function() {{
    allExpanded = !allExpanded;
    sectionIds.forEach(function(id) {{
      var el = document.querySelector(id);
      if (!el) return;
      var inst = bootstrap.Collapse.getOrCreateInstance(el, {{toggle: false}});
      allExpanded ? inst.show() : inst.hide();
    }});
    expandAllBtn.textContent = allExpanded ? 'Collapse All Sections' : 'Expand All Sections';
  }});
}})();
// Re-init tooltips when accordions/collapses open (new content becomes visible)
document.addEventListener('shown.bs.collapse', function() {{
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function(el) {{
    if (!bootstrap.Tooltip.getInstance(el)) new bootstrap.Tooltip(el);
  }});
}});

// === Sortable tables ===
document.querySelectorAll('.sortable-table').forEach(function(table) {{
  var headers = table.querySelectorAll('th.sortable');
  headers.forEach(function(th, colIdx) {{
    th.addEventListener('click', function() {{
      var tbody = table.querySelector('tbody');
      if (!tbody) return;
      var rows = Array.from(tbody.querySelectorAll('tr:not(.collapse)'));
      var isAsc = th.classList.contains('asc');
      // Clear all sort indicators in this table
      headers.forEach(function(h) {{ h.classList.remove('asc', 'desc'); }});
      th.classList.add(isAsc ? 'desc' : 'asc');
      var dir = isAsc ? -1 : 1;
      rows.sort(function(a, b) {{
        var aCell = a.cells[colIdx];
        var bCell = b.cells[colIdx];
        if (!aCell || !bCell) return 0;
        var aText = aCell.textContent.trim();
        var bText = bCell.textContent.trim();
        var aNum = parseFloat(aText);
        var bNum = parseFloat(bText);
        if (!isNaN(aNum) && !isNaN(bNum)) return (aNum - bNum) * dir;
        return aText.localeCompare(bText) * dir;
      }});
      rows.forEach(function(row) {{
        // Also move the detail row if it exists right after
        var next = row.nextElementSibling;
        tbody.appendChild(row);
        if (next && next.classList.contains('collapse')) tbody.appendChild(next);
      }});
    }});
  }});
}});

// === Network graph ===
document.addEventListener('DOMContentLoaded', function() {{
  var nodesData = {nodes_json};
  var edgesData = {edges_json};
  function escapeHtml(value) {{
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }}
  var nodes = new vis.DataSet(nodesData);
  var edges = new vis.DataSet(edgesData);
  var container = document.getElementById('network');
  var network = new vis.Network(container, {{nodes: nodes, edges: edges}}, {{
    nodes: {{font: {{size: 11}}, borderWidth: 2}},
    edges: {{
      arrows: {{to: {{enabled: true, scaleFactor: 0.7}}}},
      font: {{size: 8, align: 'middle'}},
      width: 1.2,
      smooth: {{type: 'continuous'}}
    }},
    physics: {{
      enabled: true,
      stabilization: {{iterations: 300}},
      barnesHut: {{
        gravitationalConstant: -5000,
        springLength: 250,
        springConstant: 0.04,
        damping: 0.09
      }}
    }},
    interaction: {{hover: true, tooltipDelay: 200, zoomView: true, dragView: true}}
  }});

  network.on('click', function(params) {{
    var info = document.getElementById('network-info');
    if (params.nodes.length > 0) {{
      var node = nodes.get(params.nodes[0]);
      info.innerHTML = '<strong>' + escapeHtml(node.label) + '</strong><br>' + escapeHtml(node.title);
      info.className = 'alert alert-info mt-2 small';
    }} else if (params.edges.length > 0) {{
      var edge = edges.get(params.edges[0]);
      info.innerHTML = '<strong>Edge:</strong> ' + escapeHtml(edge.label || edge.title || 'relationship');
      info.className = 'alert alert-secondary mt-2 small';
    }}
  }});

  // Zoom controls
  document.getElementById('net-zoom-in').addEventListener('click', function() {{
    var scale = network.getScale();
    network.moveTo({{scale: scale * 1.3}});
  }});
  document.getElementById('net-zoom-out').addEventListener('click', function() {{
    var scale = network.getScale();
    network.moveTo({{scale: scale / 1.3}});
  }});
  document.getElementById('net-fit').addEventListener('click', function() {{
    network.fit();
  }});

  var evLinkCb = document.getElementById('toggle-evidence_link');
  var isolatedCb = document.getElementById('toggle-isolated');

  function edgeVisible(edge) {{
    if (edge.group === 'evidence_link' && evLinkCb && !evLinkCb.checked) return false;
    return !edge.hidden;
  }}

  function updateNodeVisibility() {{
    var connected = new Set();
    edges.forEach(function(e) {{
      if (edgeVisible(e)) {{
        connected.add(e.from);
        connected.add(e.to);
      }}
    }});
    var hideIsolated = isolatedCb && isolatedCb.checked;
    var updates = [];
    nodes.forEach(function(n) {{
      var groupCb = document.getElementById('toggle-' + n.group);
      var groupVisible = !groupCb || groupCb.checked;
      var isIsolated = !connected.has(n.id);
      updates.push({{id: n.id, hidden: !groupVisible || (hideIsolated && isIsolated)}});
    }});
    nodes.update(updates);
  }}

  // Toggle node groups
  ['event','hypothesis','evidence','actor','mechanism'].forEach(function(group) {{
    var cb = document.getElementById('toggle-' + group);
    if (!cb) return;
    cb.addEventListener('change', updateNodeVisibility);
  }});

  // Toggle evidence link edges
  if (evLinkCb) {{
    evLinkCb.addEventListener('change', function() {{
      var show = evLinkCb.checked;
      var updates = [];
      edges.forEach(function(e) {{
        if (e.group === 'evidence_link') updates.push({{id: e.id, hidden: !show}});
      }});
      edges.update(updates);
      updateNodeVisibility();
    }});
  }}

  // Toggle isolated nodes (nodes with no visible edges)
  if (isolatedCb) {{
    isolatedCb.addEventListener('change', updateNodeVisibility);
  }}

  updateNodeVisibility();
}});
</script>
</body>
</html>"""
