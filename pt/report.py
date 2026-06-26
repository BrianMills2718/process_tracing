"""HTML report with vis.js network and Bootstrap 5."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import defaultdict
from typing import Any, TypedDict

from pt.bayesian import INTERPRETIVE_LR_CAP, RESIDUAL_ID, lr_matrix
from pt.schemas import Evidence, Hypothesis, HypothesisPosterior, PredictionClassification, ProcessTracingResult

_BACKGROUND_DRIVER_LEVEL_GAP = 18
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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


def _clean_text(value: object) -> str:
    """Remove non-printing control characters before rendering model text."""
    return _CONTROL_CHARS_RE.sub("", str(value))


def _clean_for_json(value: object) -> object:
    """Recursively remove control characters from values embedded in script JSON."""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, dict):
        return {
            _clean_text(key) if isinstance(key, str) else key: _clean_for_json(item)
            for key, item in value.items()
        }
    return value


def _esc(s: object) -> str:
    return html.escape(_clean_text(s))


def _json_for_script(value: object) -> str:
    """Serialize JSON safely for embedding directly inside a script tag."""
    return json.dumps(_clean_for_json(value)).replace("</", "<\\/").replace("<!--", "<\\!--")


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


_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _temporal_position(value: str | None) -> int | None:
    """Return a sortable coarse date key for DAG levels."""
    year = _first_year(value)
    if year is None:
        return None
    month = 6
    day = 15
    lowered = value.lower() if value else ""

    iso_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})-(\d{1,2})(?:-(\d{1,2}))?\b", lowered)
    if iso_match:
        month = int(iso_match.group(2))
        if iso_match.group(3):
            day = int(iso_match.group(3))
    else:
        for name, number in _MONTHS.items():
            if name in lowered:
                month = number
                day_match = re.search(rf"\b(\d{{1,2}})\s+{name}\b", lowered)
                if day_match:
                    day = int(day_match.group(1))
                break

    brumaire_match = re.search(r"\b(17|18|19)\s+brumaire\b", lowered)
    if brumaire_match and year == 1799:
        month = 11
        day = {"17": 8, "18": 9, "19": 10}[brumaire_match.group(1)]

    if "before brumaire" in lowered and year == 1799:
        month = min(month, 10)
        day = min(day, 1)
    if "weeks before" in lowered and year == 1799:
        month = min(month, 10)
        day = min(day, 20)

    return year * 10000 + month * 100 + day


def _layout_temporal_position(primary: str | None, *context: str | None) -> int | None:
    """Return a sortable date key using contextual before/after cues for layout."""
    primary_position = _temporal_position(primary)
    context_text = " ".join(part for part in context if part)
    if not context_text:
        return primary_position
    contextual_position = _temporal_position(f"{primary or ''} {context_text}")
    if primary_position is None:
        return contextual_position
    lowered = context_text.lower()
    has_prior_cue = any(
        cue in lowered
        for cue in ("before", "prior to", "earlier than", "weeks before", "months before")
    )
    if has_prior_cue and contextual_position is not None:
        return min(primary_position, contextual_position)
    return primary_position


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


def _assign_temporal_dag_positions(nodes: list[dict], edges: list[dict]) -> None:
    """Assign deterministic left-to-right DAG coordinates to network nodes."""
    if not nodes:
        return

    node_by_id = {str(node["id"]): node for node in nodes}
    level_spacing = 155
    row_spacing = 54
    occupied_by_level: dict[int, list[float]] = defaultdict(list)

    def node_level(node: dict) -> int:
        return int(node.get("level") or 0)

    def reserve_y(level: int, desired: float, spacing: int = row_spacing) -> float:
        used = occupied_by_level[level]
        candidates = [desired]
        for step in range(1, 24):
            candidates.extend((desired + step * spacing, desired - step * spacing))
        for candidate in candidates:
            if all(abs(candidate - existing) >= spacing * 0.72 for existing in used):
                used.append(candidate)
                return candidate
        fallback = desired + len(used) * spacing
        used.append(fallback)
        return fallback

    def pin(node: dict, y: float) -> None:
        node["x"] = node_level(node) * level_spacing
        node["y"] = int(round(y))
        node["fixed"] = {"x": True, "y": True}

    hypothesis_nodes = sorted(
        (node for node in nodes if node.get("group") == "hypothesis"),
        key=lambda node: str(node["id"]),
    )
    hypothesis_y: dict[str, float] = {}
    if hypothesis_nodes:
        start = -((len(hypothesis_nodes) - 1) * row_spacing) / 2
        for idx, node in enumerate(hypothesis_nodes):
            y = start + idx * row_spacing
            pin(node, y)
            hypothesis_y[str(node["id"])] = y

    top_driver_targets: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.get("group") == "top_driver_link":
            top_driver_targets[str(edge["from"])].append(str(edge["to"]))

    for node_id, target_ids in sorted(top_driver_targets.items()):
        top_driver_node = node_by_id.get(node_id)
        if top_driver_node is None:
            continue
        target_ys = [hypothesis_y[target_id] for target_id in target_ids if target_id in hypothesis_y]
        desired_y = sum(target_ys) / len(target_ys) if target_ys else 0
        pin(top_driver_node, reserve_y(node_level(top_driver_node), desired_y))

    group_base_y = {
        "event": -260,
        "evidence": 250,
        "mechanism": -140,
        "actor": 380,
    }
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for node in nodes:
        if "fixed" in node:
            continue
        buckets[(node_level(node), str(node.get("group") or ""))].append(node)

    for (level, group), bucket in sorted(buckets.items()):
        base_y = group_base_y.get(group, 0)
        start = base_y - ((len(bucket) - 1) * row_spacing) / 2
        for idx, node in enumerate(sorted(bucket, key=lambda item: str(item["id"]))):
            desired_y = start + idx * row_spacing
            pin(node, reserve_y(level, desired_y))


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
    source_packet = result.source_packet
    source_coverage = result.source_coverage
    if source_packet is None:
        source_scope_status = (
            "No accepted source-packet contract is stored with this result."
        )
        source_scope_recommendation = (
            "Claims are conditional on the supplied text. Build or load a source "
            "packet only when extending beyond this accepted-source critique."
        )
    else:
        source_kinds = ", ".join(source_packet.source_kinds) or "unspecified"
        if source_coverage is None:
            coverage_phrase = "source coverage was not computed"
        else:
            coverage_phrase = (
                f"{source_coverage.sources_with_evidence}/"
                f"{source_coverage.source_count} packet source(s) represented "
                "in extracted evidence"
            )
        source_scope_status = (
            f"Source packet accepted for {source_packet.case_name}: "
            f"{source_packet.source_count} source(s), kinds: {source_kinds}; "
            f"high-priority gaps: {source_packet.unresolved_high_priority_gap_count} unresolved "
            f"of {source_packet.high_priority_gap_count}; "
            f"{coverage_phrase}."
        )
        if source_packet.unresolved_high_priority_gap_count:
            gap_text = ", ".join(source_packet.high_priority_gaps)
            source_scope_recommendation = (
                f"Do not treat this as a flaw in the given-source critique. Treat "
                f"it as a claim-scope limit: resolve or explicitly accept these "
                f"gaps before broader publication-strength claims: {gap_text}."
            )
        else:
            source_scope_recommendation = (
                "Use the packet to constrain absence claims and verify that extracted "
                "evidence quotes come from the represented source groups."
            )
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
    source_scope_blockers: list[str] = []
    given_source_blockers: list[str] = []
    if source_packet is None and single_source_limited:
        source_scope_blockers.append("single-source corpus without source packet")
    elif source_packet is not None and (
        source_packet.source_count < 3
        or source_packet.unresolved_high_priority_gap_count > 0
        or source_packet.limitations
    ):
        source_scope_blockers.append("source-packet gaps or limitations")
    if source_packet is not None and source_coverage is None:
        given_source_blockers.append("accepted-source coverage not computed")
    elif source_coverage is not None and (
        source_coverage.sources_with_evidence < source_coverage.source_count
        or source_coverage.unconfigured_source_ids
    ):
        given_source_blockers.append("accepted-source coverage gaps")
    if diagnostic["decisive"] == 0 and diagnostic["moderate"] == 0:
        given_source_blockers.append("weak diagnostic tests")
    elif diagnostic["decisive"] == 0:
        given_source_blockers.append("no decisive diagnostic test")
    if proximate_share < 0.20:
        given_source_blockers.append("thin proximate evidence")
    if temporal["top_driver_background"]:
        given_source_blockers.append("background top drivers")
    if broad_winner:
        given_source_blockers.append("broad hypothesis design")
    if high_fragile:
        given_source_blockers.append("high-support fragile winner")
    if verdict_issues:
        given_source_blockers.append("verdict calibration")
    if too_many_unlinked:
        given_source_blockers.append("untriaged isolated evidence")
    optimal_given_sources = not given_source_blockers
    rows = [
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
    if source_packet is None:
        packet_html = """
        <h5>Source Material Known to the Grader</h5>
        <p class="small text-muted">No source packet is stored in this result. Source-scope
        caps therefore depend on extracted evidence and synthesis limitations. The
        given-source critique is conditional on the supplied input text.</p>"""
    else:
        packet_limitations = "; ".join(source_packet.limitations) or "None stated"
        packet_gaps = "; ".join(source_packet.high_priority_gaps) or "None"
        packet_groups = ", ".join(source_packet.source_groups) or "No groups specified"
        packet_path = source_packet.source_packet_path or "not stored"
        packet_html = f"""
        <h5>Source Material Known to the Grader</h5>
        <p class="small text-muted">The given-source critique below is graded
        conditional on this accepted source material and on the source coverage
        table. Known gaps are claim-scope caveats, not automatic failures of the
        analysis given these sources.</p>
        <div class="table-responsive">
          <table class="table table-sm table-bordered">
            <tbody>
              <tr><th>Case</th><td>{_esc(source_packet.case_name)}</td></tr>
              <tr><th>Focal Window</th><td>{_esc(source_packet.focal_window)}</td></tr>
              <tr><th>Outcome</th><td>{_esc(source_packet.outcome)}</td></tr>
              <tr><th>Sources</th><td>{source_packet.source_count} source(s); groups: {_esc(packet_groups)}; kinds: {_esc(', '.join(source_packet.source_kinds) or 'unspecified')}</td></tr>
              <tr><th>Known Gaps</th><td>{source_packet.known_gap_count} total; high priority: {_esc(packet_gaps)}; unresolved high priority: {source_packet.unresolved_high_priority_gap_count}</td></tr>
              <tr><th>Pre-specified Tests</th><td>{source_packet.pre_specified_test_count}</td></tr>
              <tr><th>Packet Limitations</th><td>{_esc(packet_limitations)}</td></tr>
              <tr><th>Packet Path</th><td>{_esc(packet_path)}</td></tr>
            </tbody>
          </table>
        </div>
        <p class="small text-muted">Interpretation rule: the packet governs source scope,
        observability, and missing-source claims. Packet metadata is not itself evidence;
        evidence still must appear in the input text and likelihood matrix.</p>"""
        if source_packet.source_gap_dispositions:
            disposition_rows = "".join(
                f"""
                <tr>
                  <td>{_esc(disposition.missing_source_class)}</td>
                  <td>{_esc(disposition.status)}</td>
                  <td class="small">{_esc(', '.join(disposition.relevant_source_ids) or 'None')}</td>
                  <td class="small">{_esc(disposition.claim_implications)}</td>
                  <td class="small">{_esc(disposition.disposition_reason)}</td>
                </tr>"""
                for disposition in source_packet.source_gap_dispositions
            )
            packet_html += f"""
            <h5>Source Gap Dispositions</h5>
            <p class="small text-muted">Dispositions distinguish acquired evidence,
            partial mitigation, unresolved source classes, unavailable sources, and
            explicitly accepted limits. Partial mitigation does not clear
            publication-strength claim-scope caps.</p>
            <div class="table-responsive">
              <table class="table table-sm table-bordered">
                <thead><tr>
                  <th>Source Gap</th>
                  <th>Status</th>
                  <th>Relevant Sources</th>
                  <th>Claim Implications</th>
                  <th>Reason</th>
                </tr></thead>
                <tbody>{disposition_rows}</tbody>
              </table>
            </div>"""
    if source_coverage is None:
        coverage_html = """
        <h5>Packet Source Coverage</h5>
        <p class="small text-muted">No packet source coverage report is stored in this result.</p>"""
    else:
        coverage_rows = "".join(
            f"""
            <tr>
              <td><code>{_esc(item.source_id)}</code></td>
              <td>{_esc(item.title)}</td>
              <td>{_esc(item.status)}</td>
              <td>{item.input_marker_hits}</td>
              <td>{item.evidence_count}</td>
              <td class="small">{_esc(', '.join(item.evidence_ids[:8]) or 'None')}</td>
            </tr>"""
            for item in source_coverage.items
        )
        unassigned = ", ".join(source_coverage.unassigned_evidence_ids[:12]) or "None"
        coverage_html = f"""
        <h5>Packet Source Coverage</h5>
        <p class="small text-muted">Coverage is matched from exact packet text markers in
        the input text and extracted evidence quotes. It is provenance plumbing, not a
        semantic judgment about evidentiary quality.</p>
        <div class="table-responsive">
          <table class="table table-sm table-bordered">
            <thead><tr>
              <th>Source ID</th>
              <th>Title</th>
              <th>Status</th>
              <th>Input Marker Hits</th>
              <th>Evidence Count</th>
              <th>Evidence IDs</th>
            </tr></thead>
            <tbody>{coverage_rows}</tbody>
          </table>
        </div>
        <p class="small text-muted">Assigned evidence:
        {source_coverage.assigned_evidence_count}/{source_coverage.evidence_count};
        unassigned evidence sample: {_esc(unassigned)}</p>"""
    given_source_steps = [
        "Freeze a sharper research question and focal decision window.",
        "Split broad hypotheses and define pairwise discriminators before testing.",
        "Collect proximate dated traces for the decisive sequence.",
        "Rerun likelihood scoring, dependence clustering, sensitivity, and this audit.",
        "Only then upgrade the given-source analysis from exploratory ranking to a stronger conditional claim.",
    ]
    source_scope_steps = [
        "Keep the current conclusions explicitly conditional on the accepted source set.",
        "Acquire, add, or explicitly disposition the high-priority source gaps before claims that exceed the current corpus.",
        "After source changes, rerun extraction through audit rather than treating packet metadata as evidence.",
    ]
    steps_html = "".join(f"<li>{_esc(step)}</li>" for step in given_source_steps)
    source_steps_html = "".join(f"<li>{_esc(step)}</li>" for step in source_scope_steps)
    given_blockers_html = "".join(f"<li>{_esc(blocker)}</li>" for blocker in given_source_blockers)
    source_blockers_html = "".join(f"<li>{_esc(blocker)}</li>" for blocker in source_scope_blockers)
    if optimal_given_sources:
        status_text = (
            "The analysis clears the given-source critique gate. This is not a claim of "
            "historical truth or source completeness; it means the report, temporal sequence, "
            "discriminators, and diagnostic tests clear the active audit gates conditional on "
            "the accepted source set."
        )
        gate_html = """
        <div class="alert alert-success">
          <strong>Optimality Gate:</strong> optimal_given_accepted_sources. Next iteration mode:
          <strong>none for the given-source critique</strong>. No active conditional-analysis blockers remain.
        </div>"""
        proceed_html = "<p class=\"small mb-0\">No required report/model iteration remains for the accepted-source critique. Additional sources change claim scope; they are not a prerequisite for interpreting this run conditional on the current corpus.</p>"
    else:
        status_text = (
            "Exploratory process-tracing output under the accepted sources. The critique below "
            "evaluates the analysis given the current corpus; source-scope limits are recorded "
            "separately as claim-strength caveats."
        )
        gate_html = f"""
        <div class="alert alert-danger">
          <strong>Optimality Gate:</strong> not optimal for the given-source critique. Next iteration mode:
          <strong>repair analysis or design stronger traces</strong>. Conditional-analysis blockers:
          <ul class="mb-0">{given_blockers_html}</ul>
        </div>"""
        proceed_html = f"<ol>{steps_html}</ol>"
    if source_scope_blockers:
        source_scope_html = f"""
        <div class="alert alert-warning">
          <strong>Claim-Scope Caveat:</strong> the critique above is conditional on the accepted
          source set. These source-scope limits cap broader publication-strength claims, but
          they are not themselves criticisms of whether the analysis is coherent given the
          supplied sources:
          <ul class="mb-0">{source_blockers_html}</ul>
          <p class="small mb-0 mt-2">Accepted-source status: {_esc(source_scope_status)}</p>
          <p class="small mb-0 mt-2">{_esc(source_scope_recommendation)}</p>
        </div>"""
    else:
        source_scope_html = f"""
        <div class="alert alert-success">
          <strong>Claim-Scope Caveat:</strong> no separate source-scope blocker is active beyond
          the accepted-source critique. Accepted-source status: {_esc(source_scope_status)}
          {_esc(source_scope_recommendation)}
        </div>"""
    card_border = "border-success" if optimal_given_sources else "border-danger"
    header_class = "bg-success" if optimal_given_sources else "bg-danger"
    return f"""
    <div class="card mb-4 shadow-sm {card_border}">
      <div class="card-header {header_class} text-white"><h4 class="mb-0">Academic PhD Review</h4></div>
      <div class="card-body">
        <p><strong>Current scholarly status:</strong> {_esc(status_text)}</p>
        {gate_html}
        {source_scope_html}
        {packet_html}
        {coverage_html}
        <h5>Given-Source Recommendations by Pipeline Output</h5>
        <div class="table-responsive">
          <table class="table table-sm table-bordered">
            <thead><tr>
              <th>Output</th>
              <th>PhD-level critique given accepted sources</th>
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
        <h5>When Extending Beyond These Sources</h5>
        <ol>{source_steps_html}</ol>
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
    node_levels: dict[str, int] = {}
    dated_positions = sorted({
        pos for item in ext.events for pos in [_temporal_position(item.date)] if pos is not None
    } | {
        pos
        for item in ext.evidence
        for pos in [_layout_temporal_position(item.approximate_date, item.description, item.source_text)]
        if pos is not None
    })
    level_by_position = {position: idx for idx, position in enumerate(dated_positions)}
    unknown_temporal_level = len(level_by_position)
    mechanism_level = unknown_temporal_level + 1
    hypothesis_level = unknown_temporal_level + 2

    def position_level(pos: int | None) -> int:
        if pos is None:
            return unknown_temporal_level
        return level_by_position.get(pos, unknown_temporal_level)

    def temporal_level(value: str | None) -> int:
        return position_level(_temporal_position(value))

    for e in ext.events:
        level = temporal_level(e.date)
        nodes.append({
            "id": e.id, "label": e.description[:40], "title": _esc(e.description),
            "color": "#66b3ff", "shape": "dot", "size": 15, "group": "event",
            "level": level,
        })
        node_ids.add(e.id)
        node_levels[e.id] = level

    for a in ext.actors:
        nodes.append({
            "id": a.id, "label": a.name[:30], "title": _esc(a.description),
            "color": "#ff99cc", "shape": "dot", "size": 12, "group": "actor",
            "hidden": True, "level": 0,
        })
        node_ids.add(a.id)
        node_levels[a.id] = 0

    for ev in ext.evidence:
        level = position_level(_layout_temporal_position(ev.approximate_date, ev.description, ev.source_text))
        nodes.append({
            "id": ev.id, "label": ev.description[:40], "title": _esc(ev.source_text[:200]),
            "color": "#ff6666", "shape": "diamond", "size": 10, "group": "evidence",
            "level": level,
        })
        node_ids.add(ev.id)
        node_levels[ev.id] = level

    for m in ext.mechanisms:
        nodes.append({
            "id": m.id, "label": m.description[:40], "title": _esc(m.description),
            "color": "#99ff99", "shape": "dot", "size": 12, "group": "mechanism",
            "hidden": True, "level": mechanism_level,
        })
        node_ids.add(m.id)
        node_levels[m.id] = mechanism_level

    for h in result.hypothesis_space.hypotheses:
        post = posteriors.get(h.id, 0)
        size = 15 + post * 30
        nodes.append({
            "id": h.id, "label": f"{h.id}: {h.description[:30]}",
            "title": _esc(f"{h.description}\nSupport: {post:.3f}"),
            "color": "#ffcc00", "shape": "star", "size": int(size),
            "group": "hypothesis", "level": hypothesis_level,
        })
        node_ids.add(h.id)
        node_levels[h.id] = hypothesis_level

    # Causal edges from extraction
    for ce in ext.causal_edges:
        if ce.source_id in node_ids and ce.target_id in node_ids:
            source_level = node_levels.get(ce.source_id)
            target_level = node_levels.get(ce.target_id)
            temporal_conflict = (
                source_level is not None
                and target_level is not None
                and source_level > target_level
            )
            edges.append({
                "from": ce.source_id, "to": ce.target_id,
                "arrows": "to",
                "color": {"color": "#b23a48" if temporal_conflict else "#5f6670", "opacity": 0.82 if temporal_conflict else 0.78},
                "width": 1.5 if temporal_conflict else 1.6,
                "dashes": temporal_conflict,
                "hidden": temporal_conflict,
                "title": _esc("Temporal-order warning: extracted relationship points backward in the dated layout." if temporal_conflict else ce.relationship),
                "group": "temporal_conflict" if temporal_conflict else "causal",
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
            source_level = node_levels.get(evidence_id)
            target_level = node_levels.get(hyp_id)
            is_background_driver = (
                is_top_driver
                and source_level is not None
                and target_level is not None
                and target_level - source_level >= _BACKGROUND_DRIVER_LEVEL_GAP
            )
            edge_group = (
                "background_driver_link"
                if is_background_driver
                else "top_driver_link"
                if is_top_driver
                else "evidence_link"
            )
            edge: dict[str, object] = {
                "from": evidence_id, "to": hyp_id,
                "arrows": "to", "width": round(width, 1),
                "color": {"color": color, "opacity": 0.85 if is_top_driver else 0.6},
                "group": edge_group,
                "hidden": edge_group != "top_driver_link",
                "title": f"LR={lr:.2f}" + (
                    "; background top driver" if is_background_driver
                    else "; top driver edge" if is_top_driver
                    else ""
                ),
            }
            if is_top_driver and 0.67 <= lr <= 1.5:
                edge["dashes"] = [5, 5]
            edges.append(edge)

    _assign_temporal_dag_positions(nodes, edges)
    return nodes, edges


def _network_edge_example(edges: list[dict], nodes: list[dict], group: str) -> str:
    """Return a compact concrete example for a network edge group."""
    node_by_id = {str(node["id"]): node for node in nodes}
    edge = next((item for item in edges if item.get("group") == group), None)
    if edge is None:
        return "No edge of this type was generated for this run."

    source = node_by_id.get(str(edge["from"]), {})
    target = node_by_id.get(str(edge["to"]), {})
    source_label = str(source.get("label") or edge["from"])
    target_label = str(target.get("label") or edge["to"])
    title = html.unescape(str(edge.get("title") or ""))
    title_html = f' <span class="text-muted">({_esc(title)})</span>' if title else ""
    return f"{_esc(source_label)} &rarr; {_esc(target_label)}{title_html}"


def _render_network_interpretation_guide(
    result: ProcessTracingResult, nodes: list[dict], edges: list[dict]
) -> str:
    """Explain the network ontology, edge layers, and interpretation rules."""
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for node in nodes:
        group = str(node.get("group") or "unknown")
        node_counts[group] = node_counts.get(group, 0) + 1
    for edge in edges:
        group = str(edge.get("group") or "unknown")
        edge_counts[group] = edge_counts.get(group, 0) + 1

    return f"""
        <div class="border rounded p-3 mb-3 bg-light">
          <h5 class="mb-2">How to Read This Network</h5>
          <p class="small mb-2">
            This network is an analyst-facing projection of the mixed-methods research ontology.
            <strong>Events</strong>, <strong>evidence</strong>, <strong>hypotheses</strong>,
            <strong>actors</strong>, and <strong>mechanisms</strong> are node types. The driver/link
            toggles are edge layers and review filters; they are not additional kinds of causal things.
            The goal is to make within-case causal inference inspectable enough to automate, audit, and
            connect to quantitative designs.
          </p>
          <div class="row g-3 small">
            <div class="col-lg-6">
              <table class="table table-sm mb-0">
                <thead><tr><th>Node type</th><th>Meaning</th><th>Count</th></tr></thead>
                <tbody>
                  <tr><td><strong>Event</strong></td><td>Dated occurrence that can participate in a causal sequence.</td><td>{node_counts.get("event", 0)}</td></tr>
                  <tr><td><strong>Evidence</strong></td><td>Observation, quotation, claim, or fact used to update hypotheses. Evidence is not itself the same as a cause.</td><td>{node_counts.get("evidence", 0)}</td></tr>
                  <tr><td><strong>Hypothesis</strong></td><td>Candidate explanation being compared by the Bayesian update.</td><td>{node_counts.get("hypothesis", 0)}</td></tr>
                  <tr><td><strong>Actor</strong></td><td>Person or institution extracted from the text; hidden by default until agency needs inspection.</td><td>{node_counts.get("actor", 0)}</td></tr>
                  <tr><td><strong>Mechanism</strong></td><td>Abstract process claim; hidden by default because it needs evidence/event anchoring.</td><td>{node_counts.get("mechanism", 0)}</td></tr>
                </tbody>
              </table>
            </div>
            <div class="col-lg-6">
              <table class="table table-sm mb-0">
                <thead><tr><th>Edge layer</th><th>How it is determined</th><th>Count</th></tr></thead>
                <tbody>
                  <tr><td><strong>Extracted process edge</strong></td><td>Extractor-produced source &rarr; target relation whose dated source is not after its target. It may connect events, evidence, mechanisms, or hypotheses depending on what the extractor grounded.</td><td>{edge_counts.get("causal", 0)}</td></tr>
                  <tr><td><strong>Top drivers</strong></td><td>Evidence &rarr; hypothesis links named in that hypothesis posterior's <code>top_drivers</code>. They are the largest update contributors, not necessarily favorable evidence.</td><td>{edge_counts.get("top_driver_link", 0)}</td></tr>
                  <tr><td><strong>Background drivers</strong></td><td>Top-driver links whose evidence sits far upstream in the temporal layout. They are preserved but hidden by default so background conditions do not dominate the proximate sequence.</td><td>{edge_counts.get("background_driver_link", 0)}</td></tr>
                  <tr><td><strong>Additional evidence links</strong></td><td>Informative evidence &rarr; hypothesis links outside the neutral LR band [0.67, 1.50] that were not selected as top drivers.</td><td>{edge_counts.get("evidence_link", 0)}</td></tr>
                  <tr><td><strong>Temporal conflicts</strong></td><td>Extractor-produced causal relations where the source is dated after the target. These are audit warnings, not normal causal arrows.</td><td>{edge_counts.get("temporal_conflict", 0)}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
          <div class="small mt-3">
            <strong>Concrete examples from this run:</strong>
            <ul class="mb-2">
              <li><strong>Extracted process edge:</strong> {_network_edge_example(edges, nodes, "causal")}</li>
              <li><strong>Top driver:</strong> {_network_edge_example(edges, nodes, "top_driver_link")} &mdash; this is one of the largest updates for its target hypothesis; LR &gt; 1 favors that target, LR &lt; 1 weighs against it.</li>
              <li><strong>Background driver:</strong> {_network_edge_example(edges, nodes, "background_driver_link")}</li>
              <li><strong>Additional evidence link:</strong> {_network_edge_example(edges, nodes, "evidence_link")}</li>
              <li><strong>Temporal conflict:</strong> {_network_edge_example(edges, nodes, "temporal_conflict")} &mdash; toggle this on to inspect likely extraction or date-direction problems.</li>
            </ul>
            <strong>Interpretation rule:</strong> read gray process arrows as extracted chronological process claims that still need review.
            Read green/red evidence-to-hypothesis arrows as diagnostic support or opposition, not as a claim that
            the evidence node caused the hypothesis. A dashed driver can still be important because it is one of the
            largest updates available in a limited corpus, even if its absolute LR is weak.
          </div>
        </div>"""


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
    network_guide = _render_network_interpretation_guide(result, vis_nodes, vis_edges)
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
            <input class="form-check-input" type="checkbox" id="toggle-top_driver_link" checked>
            <label class="form-check-label" for="toggle-top_driver_link">
              <span style="display:inline-block;width:12px;height:3px;background:#28a745;vertical-align:middle"></span> Top drivers</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-background_driver_link">
            <label class="form-check-label" for="toggle-background_driver_link">
              <span style="display:inline-block;width:12px;height:3px;background:#6f8f45;vertical-align:middle"></span> Background drivers</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-evidence_link">
            <label class="form-check-label" for="toggle-evidence_link">
              <span style="display:inline-block;width:12px;height:3px;background:#28a745;vertical-align:middle"></span>/<span style="display:inline-block;width:12px;height:3px;background:#dc3545;vertical-align:middle"></span> Additional evidence links</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-temporal_conflict">
            <label class="form-check-label" for="toggle-temporal_conflict">
              <span style="display:inline-block;width:12px;height:3px;border-top:2px dashed #b23a48;vertical-align:middle"></span> Temporal conflicts</label>
          </div>
          <div class="form-check form-check-inline">
            <input class="form-check-input" type="checkbox" id="toggle-isolated" checked>
            <label class="form-check-label" for="toggle-isolated">Hide isolated nodes</label>
          </div>
          <div class="btn-group btn-group-sm ms-auto">
            <button class="btn btn-outline-secondary" id="net-zoom-in" title="Zoom in">+</button>
            <button class="btn btn-outline-secondary" id="net-zoom-out" title="Zoom out">&minus;</button>
            <button class="btn btn-outline-secondary" id="net-focus" title="Focus on the downstream causal window">Focus</button>
            <button class="btn btn-outline-secondary" id="net-fit" title="Fit all">Fit</button>
          </div>
        </div>
        {network_guide}
        <div id="network" style="width:100%;height:720px;border:1px solid #ddd;border-radius:4px"></div>
        <div id="network-info" class="alert alert-light mt-2 small">Temporal DAG layout: left-to-right columns follow extracted dates, with mechanisms and hypotheses downstream. Proximate top-driver evidence links are shown by default; green links support a hypothesis (LR &gt; 1), red links oppose it (LR &lt; 1), and dashed links mark weak top drivers. Background drivers, additional evidence links, and temporal conflicts remain available through toggles.</div>
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
        <tr id="hyp-{_esc(hid)}">
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

    # ===== Section 6b: Diagnostic Test Matrix =====
    diagnostic_section = ""
    if result.diagnostic_matrix:
        dm = result.diagnostic_matrix
        ev_lookup = {ev.id: ev for ev in result.extraction.evidence}
        h_lookup = {h.id: h.description[:50] for h in result.hypothesis_space.hypotheses}
        dm_rows = []
        for pair in dm.rival_pair_diagnostics:
            cap_badge = '<span class="badge bg-danger ms-1">NO DISCRIMINATORS — CAPPED</span>' if pair.grade_capped else ""
            disc_items = []
            for d in sorted(pair.discriminators, key=lambda x: -abs(x.log_lr_h1_over_h2))[:5]:
                ev = ev_lookup.get(d.evidence_id)
                ev_label = (ev.description[:55] + "…") if ev else d.evidence_id
                strength_cls = "bg-danger" if d.strength == "decisive" else "bg-warning text-dark"
                favors_label = pair.h1_id if d.favors == "h1" else pair.h2_id
                lr_ratio = abs(d.log_lr_h1_over_h2)
                disc_items.append(
                    f'<li class="small">'
                    f'<span class="badge {strength_cls}">{d.strength}</span> '
                    f'<span data-bs-toggle="tooltip" title="{_esc(ev.source_text[:200]) if ev else ""}">{_esc(ev_label)}</span> '
                    f'→ favors <strong>{_esc(favors_label)}</strong> '
                    f'(log-ratio={lr_ratio:.2f})'
                    f'</li>'
                )
            overflow = len(pair.discriminators) - 5
            if overflow > 0:
                disc_items.append(f'<li class="small text-muted">…and {overflow} more</li>')
            disc_html = (
                f'<ul class="mb-0 ps-3">{"".join(disc_items)}</ul>'
                if disc_items
                else '<span class="text-muted small">none</span>'
            )
            dm_rows.append(f"""
            <tr>
              <td class="small"><code>{_esc(pair.h1_id)}</code> <span class="text-muted">vs</span> <code>{_esc(pair.h2_id)}</code>{cap_badge}</td>
              <td class="small">{_esc(h_lookup.get(pair.h1_id, pair.h1_id))}</td>
              <td class="small">{_esc(h_lookup.get(pair.h2_id, pair.h2_id))}</td>
              <td>{_esc(str(pair.discriminator_count))}</td>
              <td>{disc_html}</td>
            </tr>""")
        cap_note = (
            f'<div class="alert alert-danger mb-3 small">A-level claim blocked: '
            f'{len(dm.pairs_without_discriminators)} rival pair(s) have no source-grounded discriminators. '
            f'These pairs: {", ".join(f"{p[0]}↔{p[1]}" for p in dm.pairs_without_discriminators)}.'
            f'</div>'
            if dm.grade_cap_applied else ""
        )
        diagnostic_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="For each rival pair, which evidence items discriminate between them (LR ratio ≥ 2×). An A-level claim requires at least one source-grounded discriminator per pair. Derived deterministically from the likelihood matrix — no additional LLM call.">Diagnostic Test Matrix</h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#diagnosticBody">Expand</button>
      </div>
      <div class="collapse" id="diagnosticBody">
        <div class="card-body">
          {cap_note}
          <p class="small text-muted">Discriminators are evidence items where |log(LR_h1/LR_h2)| ≥ log(2). Strong = 2–5× LR ratio; decisive = 5×+. Up to 5 strongest shown per pair.</p>
          <div class="table-responsive">
            <table class="table table-sm table-striped sortable-table" id="diagnostic-matrix-table">
              <thead><tr>
                {_th("Rival pair")}
                {_th("H1 description")}
                {_th("H2 description")}
                {_th("# discriminators")}
                {_th("Top discriminators")}
              </tr></thead>
              <tbody>{''.join(dm_rows)}</tbody>
            </table>
          </div>
        </div>
      </div>
    </div>"""

    # ===== Section 6c: Structural Critic (collapsed by default) =====
    _CRITIC_SEVERITY_COLOR = {"high": "danger", "medium": "warning", "low": "secondary"}
    _CRITIC_TYPE_COLOR = {
        "confound": "danger",
        "missing_pathway": "warning",
        "void_link": "secondary",
        "too_strong_claim": "warning",
        "confirmed_link": "success",
    }

    critic_section = ""
    if result.critic:
        cr = result.critic
        reelicit_note = ""
        if cr.re_elicitation_needed:
            reelicit_note = (
                '<div class="alert alert-info mt-2 mb-2 py-2 px-3">'
                '<strong>Pass 3 was re-elicited</strong> after high-severity findings. '
                'result.json contains updated posteriors; result_base.json has the pre-critic snapshot.'
                '</div>'
            )

        # Build findings table
        ev_ids = {ev.id for ev in result.extraction.evidence}
        hyp_ids = {h.id for h in result.hypothesis_space.hypotheses}
        defect_rows = []
        confirmed_rows = []
        for f in cr.findings:
            sev_color = _CRITIC_SEVERITY_COLOR.get(f.severity, "secondary")
            type_color = _CRITIC_TYPE_COLOR.get(f.finding_type, "secondary")
            target_html = _esc(f.target)
            if f.target_type == "evidence" and f.target in ev_ids:
                target_html = f'<a href="#ev-{_esc(f.target)}">{_esc(f.target)}</a>'
            elif f.target_type == "hypothesis" and f.target in hyp_ids:
                target_html = f'<a href="#hyp-{_esc(f.target)}">{_esc(f.target)}</a>'

            row = f"""
            <tr>
              <td><span class="badge bg-{type_color}">{_esc(f.finding_type)}</span></td>
              <td class="small font-monospace">{target_html}</td>
              <td><span class="badge bg-{sev_color}">{_esc(f.severity)}</span></td>
              <td class="small">{_esc(f.reasoning)}</td>
              <td class="small">{_esc(f.recommendation)}</td>
            </tr>"""

            if f.finding_type == "confirmed_link":
                confirmed_rows.append(row)
            else:
                defect_rows.append(row)

        findings_html = ""
        if defect_rows:
            findings_html = f"""
          <div class="table-responsive mt-3">
            <table class="table table-sm table-striped">
              <thead><tr>
                <th>Type</th><th>Target</th><th>Severity</th><th>Reasoning</th><th>Recommendation</th>
              </tr></thead>
              <tbody>{''.join(defect_rows)}</tbody>
            </table>
          </div>"""
        else:
            findings_html = '<p class="text-muted small mt-2">No structural defects found.</p>'

        confirmed_html = ""
        if confirmed_rows:
            confirmed_html = f"""
          <div class="mt-3">
            <h6 class="text-success">&#10003; Structural Anchors (confirmed_link)</h6>
            <div class="table-responsive">
              <table class="table table-sm table-bordered border-success">
                <thead class="table-success"><tr>
                  <th>Type</th><th>Target</th><th>Severity</th><th>Reasoning</th><th>Recommendation</th>
                </tr></thead>
                <tbody>{''.join(confirmed_rows)}</tbody>
              </table>
            </div>
          </div>"""

        # Badge counts only defects (confirmed_link is positive, not a problem)
        n_high = sum(1 for f in cr.findings if f.severity == "high" and f.finding_type != "confirmed_link")
        n_defects = sum(1 for f in cr.findings if f.finding_type != "confirmed_link")
        n_finding_badge = (
            f'<span class="badge bg-danger ms-2">{n_high} high</span>'
            if n_high else
            f'<span class="badge bg-secondary ms-2">{n_defects} findings</span>'
        )

        critic_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Structural review of causal graph and likelihood claims. Flags confounds, missing pathways, void links, too-strong claims, and confirmed links. Advisory only — numeric changes route through re-elicitation of Pass 3.">
          Structural Critic{n_finding_badge}
        </h4>
        <button class="btn btn-sm btn-outline-primary section-toggle" type="button" data-bs-toggle="collapse" data-bs-target="#criticBody">Expand</button>
      </div>
      <div class="collapse" id="criticBody">
        <div class="card-body">
          {reelicit_note}
          <p class="small text-muted mb-2">{_esc(cr.summary)}</p>
          {findings_html}
          {confirmed_html}
        </div>
      </div>
    </div>"""

    # ===== Section 7: Evidence List (collapsed by default) =====
    _GENRE_BADGE: dict[str, str] = {
        "overview": "bg-secondary",
        "primary_document": "bg-success",
        "speech": "bg-info text-dark",
        "legal_constitutional": "bg-primary",
        "memoir": "bg-warning text-dark",
        "parliamentary_record": "bg-dark",
        "secondary_analysis": "bg-secondary",
        "news_dispatch": "bg-danger",
        "other": "bg-light text-dark border",
    }
    _TRACE_BADGE: dict[str, str] = {
        "direct": "bg-success",
        "indirect": "bg-warning text-dark",
        "background": "bg-light text-dark border",
    }

    ev_rows = []
    for ev in result.extraction.evidence:
        type_badge = '<span class="badge bg-info">Empirical</span>' if ev.evidence_type == "empirical" else '<span class="badge bg-warning text-dark">Interpretive</span>'
        genre_html = (
            f'<span class="badge {_GENRE_BADGE.get(ev.source_genre, "bg-secondary")}">{_esc(ev.source_genre)}</span>'
            if ev.source_genre
            else '<span class="text-muted small">—</span>'
        )
        trace_html = (
            f'<span class="badge {_TRACE_BADGE.get(ev.trace_production_relevance, "bg-secondary")}">{_esc(ev.trace_production_relevance)}</span>'
            if ev.trace_production_relevance
            else '<span class="text-muted small">—</span>'
        )
        group_html = f'<span class="small text-muted">{_esc(ev.source_group)}</span>' if ev.source_group else ""
        ev_rows.append(f"""
        <tr id="ev-{_esc(ev.id)}">
          <td><code class="small">{_esc(ev.id)}</code></td>
          <td>{_esc(ev.description)}</td>
          <td>{type_badge}</td>
          <td>{_esc(ev.approximate_date or 'N/A')}</td>
          <td>{genre_html}{(' ' + group_html) if group_html else ''}</td>
          <td>{trace_html}</td>
          <td>
            <span class="d-inline-block text-truncate" style="max-width:200px" data-bs-toggle="tooltip" title="{_esc(ev.source_text)}">{_esc(ev.source_text[:80])}...</span>
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
              {_th("Genre / Group", "Source genre and section group for provenance auditing.")}
              {_th("Trace role", "direct = mechanism acting; indirect = circumstantial; background = context.")}
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
                pred_desc = pred_desc_map.get(ae.prediction_id) or ae.prediction_id
                extractable_tip = "Would this evidence appear in a text of this scope if it existed?"
                extractable_badge = (
                    f'<span class="badge bg-warning text-dark" data-bs-toggle="tooltip" title="{extractable_tip}">Yes</span>'
                    if ae.would_be_extractable
                    else f'<span class="badge bg-secondary" data-bs-toggle="tooltip" title="{extractable_tip}">No</span>'
                )
                genre_cell = ""
                if ae.expected_source_genre:
                    genre_cell = f'<span class="badge bg-info text-dark">{_esc(ae.expected_source_genre)}</span>'
                    if ae.expected_source_location:
                        genre_cell += f'<br><small class="text-muted">{_esc(ae.expected_source_location)}</small>'
                else:
                    genre_cell = '<span class="text-muted small">—</span>'
                rows += f"""
                <tr>
                  <td class="small">{_esc(pred_desc[:100])}</td>
                  <td>{_esc(ae.missing_evidence)}</td>
                  <td>{_severity_badge(ae.severity)}</td>
                  <td>{extractable_badge}</td>
                  <td>{genre_cell}</td>
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
                  {_th("Acquire from", "Source genre and collection where this missing trace would appear — use to plan source acquisition.")}
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
  {diagnostic_section}
  {critic_section}
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
    nodes: {{font: {{size: 13, strokeWidth: 3, strokeColor: '#ffffff'}}, borderWidth: 2}},
    edges: {{
      arrows: {{to: {{enabled: true, scaleFactor: 0.7}}}},
      font: {{size: 8, align: 'middle'}},
      width: 1.2,
      smooth: {{type: 'cubicBezier', forceDirection: 'horizontal', roundness: 0.35}}
    }},
    layout: {{
      improvedLayout: false,
      randomSeed: 42
    }},
    physics: {{
      enabled: false
    }},
    interaction: {{hover: true, tooltipDelay: 200, zoomView: true, dragView: true}}
  }});
  window.ptNetwork = network;
  window.ptNodes = nodes;
  window.ptEdges = edges;

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
  function focusNetwork() {{
    var visibleNodes = nodes.get().filter(function(n) {{ return n.hidden !== true; }});
    if (!visibleNodes.length) {{
      network.fit({{animation: false}});
      return;
    }}
    var topDriverNodeIds = new Set();
    edges.forEach(function(e) {{
      if (e.hidden !== true && e.group === 'top_driver_link') {{
        topDriverNodeIds.add(e.from);
        topDriverNodeIds.add(e.to);
      }}
    }});
    var maxLevel = Math.max.apply(null, visibleNodes.map(function(n) {{ return Number(n.level || 0); }}));
    var focusNodes = visibleNodes.filter(function(n) {{
      var level = Number(n.level || 0);
      return n.group === 'hypothesis' || topDriverNodeIds.has(n.id) || level >= maxLevel - 8;
    }});
    if (!focusNodes.length) focusNodes = visibleNodes;
    var focusIds = focusNodes.map(function(n) {{ return n.id; }});
    var positions = network.getPositions(focusIds);
    var xs = [];
    var ys = [];
    Object.keys(positions).forEach(function(id) {{
      xs.push(positions[id].x);
      ys.push(positions[id].y);
    }});
    if (!xs.length) {{
      network.fit({{animation: false}});
      return;
    }}
    var minX = Math.min.apply(null, xs);
    var maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys);
    var maxY = Math.max.apply(null, ys);
    var width = Math.max(1, maxX - minX);
    var height = Math.max(1, maxY - minY);
    var scaleX = container.clientWidth / (width + 460);
    var scaleY = container.clientHeight / (height + 260);
    var scale = Math.max(0.42, Math.min(0.9, Math.min(scaleX, scaleY)));
    network.moveTo({{
      position: {{x: (minX + maxX) / 2, y: (minY + maxY) / 2}},
      scale: scale,
      animation: false
    }});
  }}
  document.getElementById('net-focus').addEventListener('click', function() {{
    focusNetwork();
  }});
  document.getElementById('net-fit').addEventListener('click', function() {{
    network.fit();
  }});
  var networkBody = document.getElementById('networkBody');
  if (networkBody) {{
    networkBody.addEventListener('shown.bs.collapse', function() {{
      setTimeout(focusNetwork, 100);
    }});
  }}

  var topDriverCb = document.getElementById('toggle-top_driver_link');
  var backgroundDriverCb = document.getElementById('toggle-background_driver_link');
  var evLinkCb = document.getElementById('toggle-evidence_link');
  var temporalConflictCb = document.getElementById('toggle-temporal_conflict');
  var isolatedCb = document.getElementById('toggle-isolated');

  function edgeVisible(edge) {{
    if (edge.group === 'top_driver_link' && topDriverCb && !topDriverCb.checked) return false;
    if (edge.group === 'background_driver_link' && backgroundDriverCb && !backgroundDriverCb.checked) return false;
    if (edge.group === 'evidence_link' && evLinkCb && !evLinkCb.checked) return false;
    if (edge.group === 'temporal_conflict' && temporalConflictCb && !temporalConflictCb.checked) return false;
    return !edge.hidden;
  }}

  function syncOptionalEdgeGroup(group, cb) {{
    if (!cb) return;
    var show = cb.checked;
    var updates = [];
    edges.forEach(function(e) {{
      if (e.group === group) updates.push({{id: e.id, hidden: !show}});
    }});
    edges.update(updates);
  }}

  function syncOptionalEdges() {{
    syncOptionalEdgeGroup('top_driver_link', topDriverCb);
    syncOptionalEdgeGroup('background_driver_link', backgroundDriverCb);
    syncOptionalEdgeGroup('evidence_link', evLinkCb);
    syncOptionalEdgeGroup('temporal_conflict', temporalConflictCb);
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
      var preserveIsolated = n.group === 'hypothesis';
      updates.push({{id: n.id, hidden: !groupVisible || (hideIsolated && isIsolated && !preserveIsolated)}});
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
  if (topDriverCb) {{
    topDriverCb.addEventListener('change', function() {{
      syncOptionalEdges();
      updateNodeVisibility();
      focusNetwork();
    }});
  }}
  if (backgroundDriverCb) {{
    backgroundDriverCb.addEventListener('change', function() {{
      syncOptionalEdges();
      updateNodeVisibility();
      focusNetwork();
    }});
  }}
  if (evLinkCb) {{
    evLinkCb.addEventListener('change', function() {{
      syncOptionalEdges();
      updateNodeVisibility();
      focusNetwork();
    }});
  }}
  if (temporalConflictCb) {{
    temporalConflictCb.addEventListener('change', function() {{
      syncOptionalEdges();
      updateNodeVisibility();
      focusNetwork();
    }});
  }}

  // Toggle isolated nodes (nodes with no visible edges)
  if (isolatedCb) {{
    isolatedCb.addEventListener('change', updateNodeVisibility);
  }}

  syncOptionalEdges();
  updateNodeVisibility();
  setTimeout(focusNetwork, 100);
}});
</script>
</body>
</html>"""
