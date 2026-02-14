"""HTML report with vis.js network and Bootstrap 5."""

from __future__ import annotations

import html
import json
import math

from pt.schemas import ProcessTracingResult


def _esc(s: str) -> str:
    return html.escape(str(s))


def _status_color(status: str) -> str:
    return {
        "strongly_supported": "#28a745",
        "supported": "#5cb85c",
        "weakened": "#f0ad4e",
        "eliminated": "#d9534f",
        "indeterminate": "#6c757d",
    }.get(status, "#6c757d")


def _status_bg(status: str) -> str:
    return {
        "strongly_supported": "bg-success",
        "supported": "bg-success bg-opacity-75",
        "weakened": "bg-warning",
        "eliminated": "bg-danger",
        "indeterminate": "bg-secondary",
    }.get(status, "bg-secondary")


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


def _finding_badge(finding: str) -> str:
    colors = {"pass": "#28a745", "fail": "#dc3545", "ambiguous": "#f0ad4e"}
    color = colors.get(finding, "#6c757d")
    return f'<span class="badge" style="background:{color}">{_esc(finding)}</span>'


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


def _build_vis_data(result: ProcessTracingResult) -> tuple[list[dict], list[dict]]:
    """Build vis.js nodes and edges including evidence-hypothesis links."""
    nodes = []
    edges = []
    ext = result.extraction
    posteriors = {p.hypothesis_id: p.final_posterior for p in result.bayesian.posteriors}
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
            "title": _esc(f"{h.description}\nPosterior: {post:.3f}"),
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

    # Evidence-hypothesis edges from testing
    for ht in result.testing.hypothesis_tests:
        for ev_eval in ht.evidence_evaluations:
            if ev_eval.evidence_id not in node_ids or ht.hypothesis_id not in node_ids:
                continue
            lr = ev_eval.p_e_given_h / max(ev_eval.p_e_given_not_h, 0.001)
            if 0.67 <= lr <= 1.5:
                continue  # Skip uninformative
            log_lr = abs(math.log(max(lr, 0.01)))
            width = max(0.5, min(4, log_lr))
            color = "#28a745" if lr > 1 else "#dc3545"
            edges.append({
                "from": ev_eval.evidence_id, "to": ht.hypothesis_id,
                "arrows": "to", "width": round(width, 1),
                "color": {"color": color, "opacity": 0.6},
                "group": "evidence_link",
                "title": f"LR={lr:.2f}",
            })

    return nodes, edges


def generate_report(result: ProcessTracingResult) -> str:
    """Generate self-contained HTML report."""

    posteriors = {p.hypothesis_id: p for p in result.bayesian.posteriors}
    sensitivity = {s.hypothesis_id: s for s in result.bayesian.sensitivity}
    h_map = {h.id: h for h in result.hypothesis_space.hypotheses}
    ev_map = {e.id: e for e in result.extraction.evidence}

    vis_nodes, vis_edges = _build_vis_data(result)
    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    # -- Build prediction lookup: hypothesis_id -> prediction_id -> PredictionClassification
    pred_class_map: dict[str, dict] = {}  # (hyp_id, pred_id) -> PredictionClassification
    for ht in result.testing.hypothesis_tests:
        for pc in ht.prediction_classifications:
            pred_class_map[(ht.hypothesis_id, pc.prediction_id)] = pc

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

    exec_summary = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header bg-primary text-white"><h4 class="mb-0">Executive Summary</h4></div>
      <div class="card-body">
        <p><strong>Research Question:</strong> {_esc(result.hypothesis_space.research_question)}</p>
        <p><strong>Top Hypothesis:</strong> {_esc(top_h.description) if top_h else 'N/A'}
           <span class="badge bg-warning text-dark"
             data-bs-toggle="tooltip" title="Posterior probability after Bayesian updating with all evidence">
             Posterior: {top_post:.3f}</span>
           {_robustness_badge(top_robust)}
           {'<span class="badge bg-info" data-bs-toggle="tooltip" title="This analysis includes an analytical refinement pass (second reading)">Refined</span>' if result.is_refined else ''}</p>
        <p><strong>Hypotheses evaluated:</strong> {len(result.hypothesis_space.hypotheses)} &nbsp;|&nbsp;
           <strong>Evidence items:</strong> {len(result.extraction.evidence)} &nbsp;|&nbsp;
           <strong>Causal edges:</strong> {len(result.extraction.causal_edges)}</p>
        <p>{_esc(result.extraction.summary)}</p>
      </div>
    </div>"""

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
            <input class="form-check-input" type="checkbox" id="toggle-isolated">
            <label class="form-check-label" for="toggle-isolated">Hide isolated nodes</label>
          </div>
          <div class="btn-group btn-group-sm ms-auto">
            <button class="btn btn-outline-secondary" id="net-zoom-in" title="Zoom in">+</button>
            <button class="btn btn-outline-secondary" id="net-zoom-out" title="Zoom out">&minus;</button>
            <button class="btn btn-outline-secondary" id="net-fit" title="Fit all">Fit</button>
          </div>
        </div>
        <div id="network" style="width:100%;height:600px;border:1px solid #ddd;border-radius:4px"></div>
        <div id="network-info" class="alert alert-light mt-2 small">Click a node for details. Green edges = supporting evidence (LR &gt; 1). Red edges = opposing evidence (LR &lt; 1).</div>
      </div>
      </div>
    </div>"""

    # ===== Section 3: Hypothesis Comparison Table =====
    h_rows = []
    for rank, hid in enumerate(result.bayesian.ranking, 1):
        h = h_map.get(hid)
        p = posteriors.get(hid)
        s = sensitivity.get(hid)
        if not h or not p:
            continue
        verdict = next((v for v in result.synthesis.verdicts if v.hypothesis_id == hid), None)
        status = verdict.status if verdict else "indeterminate"
        sens_range = f"[{s.posterior_low:.3f}, {s.posterior_high:.3f}]" if s else "N/A"
        rank_badge = f'<span class="badge bg-{"success" if s and s.rank_stable else "warning"} bg-opacity-75" data-bs-toggle="tooltip" title="{"Rank is stable under sensitivity perturbation" if s and s.rank_stable else "Rank may change under sensitivity perturbation"}">{"Stable" if s and s.rank_stable else "Unstable"}</span>' if s else ""
        steelman_html = _esc(verdict.steelman) if verdict else ""
        mechanism_html = _esc(h.causal_mechanism)
        basis_html = _esc(h.theoretical_basis)

        source_label = h.source
        source_badge_class = "bg-secondary"
        if h.source == "text":
            source_badge_class = "bg-info"
        elif h.source == "generated":
            source_badge_class = "bg-warning text-dark"
        elif "theory" in h.source.lower():
            source_badge_class = "bg-purple"

        h_rows.append(f"""
        <tr>
          <td>{rank}</td>
          <td><strong>{_esc(hid)}</strong></td>
          <td>{_esc(h.description)}</td>
          <td><span class="badge {source_badge_class}" data-bs-toggle="tooltip" title="{_esc(h.theoretical_basis)}">{_esc(source_label)}</span></td>
          <td>{p.prior:.3f}</td>
          <td><strong>{p.final_posterior:.3f}</strong></td>
          <td>{_status_badge(status)}</td>
          <td>{_robustness_badge(p.robustness)}</td>
          <td><span data-bs-toggle="tooltip" title="Posterior range under ±50% perturbation of top drivers">{sens_range}</span> {rank_badge}</td>
          <td>
            <a class="btn btn-sm btn-outline-secondary" data-bs-toggle="collapse" href="#detail-{_esc(hid)}" role="button">Details</a>
          </td>
        </tr>
        <tr class="collapse" id="detail-{_esc(hid)}">
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
            {_th("Prior", "Starting probability before any evidence is considered")}
            {_th("Posterior", "Final probability after Bayesian updating with all evidence")}
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
        h = h_map.get(hid)
        if not p or not h:
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
            <div><strong>{_esc(hid)}</strong>: {_esc(h.description[:80])}
              {_robustness_badge(p.robustness)}</div>
            <div class="text-end">
              <span class="badge bg-primary">Posterior: {baseline:.3f}</span>
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
    test_accordion_items = []
    total_evals = 0
    informative_evals = 0

    for ht in result.testing.hypothesis_tests:
        h = h_map.get(ht.hypothesis_id)
        h_label = f"{ht.hypothesis_id}: {h.description[:60]}" if h else ht.hypothesis_id

        # Group evaluations by prediction_id
        by_pred: dict[str | None, list] = {}
        for ev_eval in ht.evidence_evaluations:
            total_evals += 1
            lr = ev_eval.p_e_given_h / max(ev_eval.p_e_given_not_h, 0.001)
            if 0.9 < lr < 1.1:
                continue
            informative_evals += 1
            pid = ev_eval.prediction_id
            by_pred.setdefault(pid, []).append((ev_eval, lr))

        # Sort each group by |log(LR)| descending
        for pid in by_pred:
            by_pred[pid].sort(key=lambda x: abs(math.log(max(x[1], 0.01))), reverse=True)

        # Build inner HTML
        inner_html = ""
        # First: predictions with evaluations
        for pid, evals in by_pred.items():
            if pid is None:
                continue
            pc = pred_class_map.get((ht.hypothesis_id, pid))
            dtype = pc.diagnostic_type if pc else "unknown"
            pred_desc = pred_desc_map.get(pid, pid)
            inner_html += f"""
            <div class="mb-3">
              <div class="fw-bold small text-muted mb-1">
                {_diagnostic_badge(dtype)}
                <span class="ms-1">{_esc(pred_desc[:120])}</span>
              </div>
              <table class="table table-sm table-bordered mb-0 sortable-table">
                <thead><tr>
                  {_th("Evidence")}
                  {_th("Finding")}
                  {_th("P(E|H)", "Probability of observing this evidence if the hypothesis is true")}
                  {_th("P(E|~H)", "Probability of observing this evidence if the hypothesis is false")}
                  {_th("LR", "Likelihood Ratio = P(E|H)/P(E|~H). LR>1 supports, LR<1 opposes the hypothesis")}
                  {_th("Relevance", "How relevant this evidence is to the hypothesis (0-1). Below 0.4 = forced uninformative")}
                  {_th("Justification")}
                </tr></thead>
                <tbody>"""
            for ev_eval, lr in evals:
                ev = ev_map.get(ev_eval.evidence_id)
                ev_label = ev.description[:80] if ev else ev_eval.evidence_id
                lr_color = "#28a745" if lr > 1.1 else "#dc3545"
                inner_html += f"""
                  <tr>
                    <td class="small" data-bs-toggle="tooltip" title="{_esc(ev.source_text[:200]) if ev else ''}">{_esc(ev_label)}</td>
                    <td>{_finding_badge(ev_eval.finding)}</td>
                    <td>{ev_eval.p_e_given_h:.2f}</td>
                    <td>{ev_eval.p_e_given_not_h:.2f}</td>
                    <td><strong style="color:{lr_color}">{lr:.2f}</strong></td>
                    <td>{ev_eval.relevance:.2f}</td>
                    <td class="small">{_esc(ev_eval.justification)}</td>
                  </tr>"""
            inner_html += "</tbody></table></div>"

        # Unlinked evidence (prediction_id is None)
        if None in by_pred:
            evals = by_pred[None]
            inner_html += f"""
            <div class="mb-3">
              <div class="fw-bold small text-muted mb-1">Unlinked Evidence
                <span class="badge bg-secondary" data-bs-toggle="tooltip" title="Evidence not mapped to a specific prediction. Diagnostic type determined by the overall relationship to the hypothesis.">No specific prediction</span>
              </div>
              <table class="table table-sm table-bordered mb-0 sortable-table">
                <thead><tr>
                  {_th("Evidence")}
                  {_th("Finding")}
                  {_th("P(E|H)", "Probability of observing this evidence if the hypothesis is true")}
                  {_th("P(E|~H)", "Probability of observing this evidence if the hypothesis is false")}
                  {_th("LR", "Likelihood Ratio = P(E|H)/P(E|~H). LR>1 supports, LR<1 opposes the hypothesis")}
                  {_th("Relevance", "How relevant this evidence is to the hypothesis (0-1). Below 0.4 = forced uninformative")}
                  {_th("Justification")}
                </tr></thead>
                <tbody>"""
            for ev_eval, lr in evals:
                ev = ev_map.get(ev_eval.evidence_id)
                ev_label = ev.description[:80] if ev else ev_eval.evidence_id
                lr_color = "#28a745" if lr > 1.1 else "#dc3545"
                inner_html += f"""
                  <tr>
                    <td class="small" data-bs-toggle="tooltip" title="{_esc(ev.source_text[:200]) if ev else ''}">{_esc(ev_label)}</td>
                    <td>{_finding_badge(ev_eval.finding)}</td>
                    <td>{ev_eval.p_e_given_h:.2f}</td>
                    <td>{ev_eval.p_e_given_not_h:.2f}</td>
                    <td><strong style="color:{lr_color}">{lr:.2f}</strong></td>
                    <td>{ev_eval.relevance:.2f}</td>
                    <td class="small">{_esc(ev_eval.justification)}</td>
                  </tr>"""
            inner_html += "</tbody></table></div>"

        hyp_eval_count = sum(len(v) for v in by_pred.values())
        collapsed = "" if ht == result.testing.hypothesis_tests[0] else "collapsed"
        show = "show" if ht == result.testing.hypothesis_tests[0] else ""
        test_accordion_items.append(f"""
        <div class="accordion-item">
          <h2 class="accordion-header">
            <button class="accordion-button {collapsed}" type="button" data-bs-toggle="collapse" data-bs-target="#test-{_esc(ht.hypothesis_id)}">
              {_esc(h_label)} <span class="badge bg-info ms-2">{hyp_eval_count} informative</span>
            </button>
          </h2>
          <div id="test-{_esc(ht.hypothesis_id)}" class="accordion-collapse collapse {show}" data-bs-parent="#testAccordion">
            <div class="accordion-body">{inner_html}</div>
          </div>
        </div>""")

    skipped_count = total_evals - informative_evals
    skipped_note = f'<p class="text-muted mb-3">{informative_evals} informative / {total_evals} total evaluations shown. {skipped_count} uninformative (LR ≈ 1.0) hidden.</p>'

    test_matrix = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Each piece of evidence is evaluated against each hypothesis using Van Evera's diagnostic tests. The likelihood ratio (LR) measures how much the evidence shifts the probability.">Diagnostic Test Matrix</h4>
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
        h = h_map.get(p_obj.hypothesis_id)
        h_label = f"{p_obj.hypothesis_id}: {h.description[:50]}" if h else p_obj.hypothesis_id
        final_width = max(2, int(p_obj.final_posterior * 100))

        # Build SVG sparkline of posterior progression
        svg_points = []
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
            <a class="btn btn-sm btn-link p-0" data-bs-toggle="collapse" href="#trailGraph-{_esc(p_obj.hypothesis_id)}">Show update graph</a>
            <span class="text-muted mx-1">|</span>
            <a class="btn btn-sm btn-link p-0" data-bs-toggle="collapse" href="#trail-{_esc(p_obj.hypothesis_id)}">Show update table</a>
          </div>
          <div class="collapse" id="trailGraph-{_esc(p_obj.hypothesis_id)}">
            {sparkline_svg}
            <div class="small text-muted mt-1">Each dot is an evidence update. <span style="color:#28a745">Green</span> = LR &gt; 1 (supports). <span style="color:#dc3545">Red</span> = LR &lt; 1 (opposes). Dashed line = prior. Hover dots for details.</div>
          </div>
          <div class="collapse" id="trail-{_esc(p_obj.hypothesis_id)}">
            <table class="table table-sm mt-1 sortable-table" style="font-size:0.85em">
              <thead><tr>{_th("Evidence")}{_th("LR", "Likelihood Ratio after relevance gating and capping")}{_th("Cumulative Posterior", "Running posterior after this update")}</tr></thead>
              <tbody>{trail_rows}</tbody>
            </table>
          </div>
        </div>""")

    bayesian_section = f"""
    <div class="card mb-4 shadow-sm">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h4 class="mb-0" data-bs-toggle="tooltip" title="Starting from equal priors, each evidence item updates the posterior probability via its likelihood ratio. LR > 1 increases support, LR < 1 decreases it.">Bayesian Update Summary</h4>
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
            h = h_map.get(hid)
            h_label = f"{hid}: {h.description[:60]}" if h else hid
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
                h = h_map.get(hid)
                h_label = f"{hid}: {h.description[:60]}" if h else hid
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
      info.innerHTML = '<strong>' + node.label + '</strong><br>' + (node.title || '');
      info.className = 'alert alert-info mt-2 small';
    }} else if (params.edges.length > 0) {{
      var edge = edges.get(params.edges[0]);
      info.innerHTML = '<strong>Edge:</strong> ' + (edge.label || edge.title || 'relationship');
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

  // Toggle node groups
  ['event','hypothesis','evidence','actor','mechanism'].forEach(function(group) {{
    var cb = document.getElementById('toggle-' + group);
    if (!cb) return;
    cb.addEventListener('change', function() {{
      var show = cb.checked;
      var updates = [];
      nodes.forEach(function(n) {{
        if (n.group === group) updates.push({{id: n.id, hidden: !show}});
      }});
      nodes.update(updates);
    }});
  }});

  // Toggle evidence link edges
  var evLinkCb = document.getElementById('toggle-evidence_link');
  if (evLinkCb) {{
    evLinkCb.addEventListener('change', function() {{
      var show = evLinkCb.checked;
      var updates = [];
      edges.forEach(function(e) {{
        if (e.group === 'evidence_link') updates.push({{id: e.id, hidden: !show}});
      }});
      edges.update(updates);
    }});
  }}

  // Toggle isolated nodes (nodes with no visible edges)
  var isolatedCb = document.getElementById('toggle-isolated');
  if (isolatedCb) {{
    isolatedCb.addEventListener('change', function() {{
      var hideIsolated = isolatedCb.checked;
      // Build set of node IDs that have at least one visible edge
      var connected = new Set();
      edges.forEach(function(e) {{
        if (!e.hidden) {{
          connected.add(e.from);
          connected.add(e.to);
        }}
      }});
      var updates = [];
      nodes.forEach(function(n) {{
        // Don't touch nodes already hidden by group toggles
        var groupCb = document.getElementById('toggle-' + n.group);
        if (groupCb && !groupCb.checked) return;
        var isIsolated = !connected.has(n.id);
        if (isIsolated) updates.push({{id: n.id, hidden: hideIsolated}});
      }});
      nodes.update(updates);
    }});
  }}
}});
</script>
</body>
</html>"""
