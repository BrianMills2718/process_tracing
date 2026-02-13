"""HTML report with vis.js network and Bootstrap 5."""

from __future__ import annotations

import html
import json

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


def _diagnostic_badge(dtype: str) -> str:
    colors = {
        "hoop": "#17a2b8",
        "smoking_gun": "#dc3545",
        "doubly_decisive": "#6f42c1",
        "straw_in_the_wind": "#6c757d",
    }
    color = colors.get(dtype, "#6c757d")
    label = dtype.replace("_", " ").title()
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em">{label}</span>'


def _finding_badge(finding: str) -> str:
    colors = {"pass": "#28a745", "fail": "#dc3545", "ambiguous": "#f0ad4e"}
    color = colors.get(finding, "#6c757d")
    return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em">{_esc(finding)}</span>'


def _build_vis_nodes(result: ProcessTracingResult) -> list[dict]:
    """Build vis.js node list."""
    nodes = []
    ext = result.extraction
    posteriors = {p.hypothesis_id: p.final_posterior for p in result.bayesian.posteriors}

    for e in ext.events:
        nodes.append({
            "id": e.id, "label": e.description[:40], "title": _esc(e.description),
            "color": "#66b3ff", "shape": "dot", "size": 15,
            "group": "event",
        })
    for a in ext.actors:
        nodes.append({
            "id": a.id, "label": a.name[:30], "title": _esc(a.description),
            "color": "#ff99cc", "shape": "dot", "size": 12,
            "group": "actor",
        })
    for ev in ext.evidence:
        nodes.append({
            "id": ev.id, "label": ev.description[:40], "title": _esc(ev.source_text),
            "color": "#ff6666", "shape": "diamond", "size": 12,
            "group": "evidence",
        })
    for m in ext.mechanisms:
        nodes.append({
            "id": m.id, "label": m.description[:40], "title": _esc(m.description),
            "color": "#99ff99", "shape": "dot", "size": 12,
            "group": "mechanism",
        })
    for h in result.hypothesis_space.hypotheses:
        post = posteriors.get(h.id, 0)
        size = 15 + post * 30  # Scale node by posterior
        nodes.append({
            "id": h.id, "label": f"{h.id}: {h.description[:30]}",
            "title": _esc(f"{h.description}\nPosterior: {post:.3f}"),
            "color": "#ffcc00", "shape": "star", "size": int(size),
            "group": "hypothesis",
        })
    return nodes


def _build_vis_edges(result: ProcessTracingResult) -> list[dict]:
    """Build vis.js edge list."""
    edges = []
    node_ids = set()
    for n in _build_vis_nodes(result):
        node_ids.add(n["id"])

    for ce in result.extraction.causal_edges:
        if ce.source_id in node_ids and ce.target_id in node_ids:
            edges.append({
                "from": ce.source_id, "to": ce.target_id,
                "label": ce.relationship[:20], "arrows": "to",
                "color": {"color": "#666"},
            })
    return edges


def generate_report(result: ProcessTracingResult) -> str:
    """Generate self-contained HTML report."""
    vis_nodes = _build_vis_nodes(result)
    vis_edges = _build_vis_edges(result)
    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    posteriors = {p.hypothesis_id: p for p in result.bayesian.posteriors}
    h_map = {h.id: h for h in result.hypothesis_space.hypotheses}
    ev_map = {e.id: e for e in result.extraction.evidence}

    # --- Section 1: Executive Summary ---
    top_id = result.bayesian.ranking[0] if result.bayesian.ranking else None
    top_h = h_map.get(top_id) if top_id else None
    top_post = posteriors[top_id].final_posterior if top_id and top_id in posteriors else 0

    exec_summary = f"""
    <div class="card mb-4">
      <div class="card-header bg-primary text-white"><h4 class="mb-0">Executive Summary</h4></div>
      <div class="card-body">
        <p><strong>Research Question:</strong> {_esc(result.hypothesis_space.research_question)}</p>
        <p><strong>Top Hypothesis:</strong> {_esc(top_h.description) if top_h else 'N/A'}
           <span class="badge bg-warning text-dark">Posterior: {top_post:.3f}</span></p>
        <p><strong>Summary:</strong> {_esc(result.extraction.summary)}</p>
      </div>
    </div>"""

    # --- Section 2: Interactive Network ---
    network_section = f"""
    <div class="card mb-4">
      <div class="card-header"><h4 class="mb-0">Interactive Causal Network</h4></div>
      <div class="card-body">
        <div class="d-flex flex-wrap gap-3 mb-3">
          <span><span style="display:inline-block;width:14px;height:14px;background:#66b3ff;border-radius:50%"></span> Events</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#ffcc00;border-radius:50%"></span> Hypotheses</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#ff6666;border-radius:50%"></span> Evidence</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#ff99cc;border-radius:50%"></span> Actors</span>
          <span><span style="display:inline-block;width:14px;height:14px;background:#99ff99;border-radius:50%"></span> Mechanisms</span>
        </div>
        <div id="network" style="width:100%;height:600px;border:1px solid #ddd"></div>
        <div id="network-info" class="alert alert-light mt-2">Click a node for details</div>
      </div>
    </div>"""

    # --- Section 3: Hypothesis Comparison Table ---
    h_rows = []
    for hid in result.bayesian.ranking:
        h = h_map.get(hid)
        p = posteriors.get(hid)
        if not h or not p:
            continue
        verdict = next((v for v in result.synthesis.verdicts if v.hypothesis_id == hid), None)
        status = verdict.status if verdict else "indeterminate"
        color = _status_color(status)
        h_rows.append(f"""
        <tr>
          <td><strong>{_esc(hid)}</strong></td>
          <td>{_esc(h.description[:80])}</td>
          <td>{_esc(h.source)}</td>
          <td>{p.prior:.3f}</td>
          <td><strong>{p.final_posterior:.3f}</strong></td>
          <td><span style="color:{color};font-weight:bold">{_esc(status.replace('_',' ').title())}</span></td>
        </tr>""")

    comparison_table = f"""
    <div class="card mb-4">
      <div class="card-header"><h4 class="mb-0">Hypothesis Comparison</h4></div>
      <div class="card-body table-responsive">
        <table class="table table-striped table-hover">
          <thead><tr><th>ID</th><th>Hypothesis</th><th>Source</th><th>Prior</th><th>Posterior</th><th>Status</th></tr></thead>
          <tbody>{''.join(h_rows)}</tbody>
        </table>
      </div>
    </div>"""

    # --- Section 4: Diagnostic Test Matrix ---
    matrix_rows = []
    for ht in result.testing.hypothesis_tests:
        for ev_eval in ht.evidence_evaluations:
            pc = next(
                (pc for pc in ht.prediction_classifications if pc.prediction_id == ev_eval.prediction_id),
                None,
            )
            dtype = pc.diagnostic_type if pc else "unknown"
            lr = ev_eval.p_e_given_h / max(ev_eval.p_e_given_not_h, 0.001)
            ev_desc = ev_map.get(ev_eval.evidence_id)
            ev_label = ev_desc.description[:50] if ev_desc else ev_eval.evidence_id
            matrix_rows.append(f"""
            <tr>
              <td>{_esc(ht.hypothesis_id)}</td>
              <td>{_esc(ev_label)}</td>
              <td>{_diagnostic_badge(dtype)}</td>
              <td>{_finding_badge(ev_eval.finding)}</td>
              <td>{ev_eval.p_e_given_h:.2f}</td>
              <td>{ev_eval.p_e_given_not_h:.2f}</td>
              <td><strong>{lr:.2f}</strong></td>
              <td><small>{_esc(ev_eval.justification[:100])}</small></td>
            </tr>""")

    test_matrix = f"""
    <div class="card mb-4">
      <div class="card-header"><h4 class="mb-0">Diagnostic Test Matrix</h4></div>
      <div class="card-body table-responsive">
        <table class="table table-sm table-bordered">
          <thead><tr><th>Hypothesis</th><th>Evidence</th><th>Test Type</th><th>Finding</th>
                     <th>P(E|H)</th><th>P(E|~H)</th><th>LR</th><th>Justification</th></tr></thead>
          <tbody>{''.join(matrix_rows)}</tbody>
        </table>
      </div>
    </div>"""

    # --- Section 5: Bayesian Update Trail ---
    trail_items = []
    for p in result.bayesian.posteriors:
        h = h_map.get(p.hypothesis_id)
        h_label = f"{p.hypothesis_id}: {h.description[:50]}" if h else p.hypothesis_id
        bars = []
        for u in p.updates:
            width = max(2, int(u.posterior * 100))
            color = "#28a745" if u.likelihood_ratio > 1 else "#dc3545"
            bars.append(
                f'<div style="display:inline-block;width:{width}%;height:20px;background:{color};'
                f'margin-right:1px" title="After {_esc(u.evidence_id)}: {u.posterior:.3f} (LR={u.likelihood_ratio:.2f})"></div>'
            )
        final_width = max(2, int(p.final_posterior * 100))
        trail_items.append(f"""
        <div class="mb-3">
          <strong>{_esc(h_label)}</strong>
          <div style="background:#eee;border-radius:4px;overflow:hidden;height:24px;position:relative">
            <div style="width:{final_width}%;height:100%;background:#007bff;border-radius:4px"></div>
            <span style="position:absolute;right:8px;top:2px;font-size:0.8em">{p.final_posterior:.3f}</span>
          </div>
          <div class="mt-1" style="font-size:0.75em">{''.join(bars)}</div>
        </div>""")

    bayesian_trail = f"""
    <div class="card mb-4">
      <div class="card-header"><h4 class="mb-0">Bayesian Update Trail</h4></div>
      <div class="card-body">{''.join(trail_items)}</div>
    </div>"""

    # --- Section 5b: Absence-of-Evidence Findings ---
    absence_rows = []
    if result.absence and result.absence.evaluations:
        for ae in result.absence.evaluations:
            sev_colors = {"damaging": "#dc3545", "notable": "#f0ad4e", "minor": "#6c757d"}
            sev_color = sev_colors.get(ae.severity, "#6c757d")
            extractable = "Yes" if ae.would_be_extractable else "No"
            absence_rows.append(f"""
            <tr>
              <td>{_esc(ae.hypothesis_id)}</td>
              <td>{_esc(ae.prediction_id)}</td>
              <td>{_esc(ae.missing_evidence)}</td>
              <td><span style="color:{sev_color};font-weight:bold">{_esc(ae.severity.title())}</span></td>
              <td>{extractable}</td>
              <td><small>{_esc(ae.reasoning[:150])}</small></td>
            </tr>""")

    absence_section = ""
    if absence_rows:
        absence_section = f"""
    <div class="card mb-4">
      <div class="card-header"><h4 class="mb-0">Absence-of-Evidence Findings</h4></div>
      <div class="card-body table-responsive">
        <p class="text-muted">Evidence that hypotheses predict should exist but was not found in the text. These findings inform the synthesis qualitatively but do not affect Bayesian posteriors.</p>
        <table class="table table-sm table-bordered">
          <thead><tr><th>Hypothesis</th><th>Prediction</th><th>Missing Evidence</th><th>Severity</th><th>Extractable?</th><th>Reasoning</th></tr></thead>
          <tbody>{''.join(absence_rows)}</tbody>
        </table>
      </div>
    </div>"""

    # --- Section 6: Analytical Narrative ---
    narrative_paras = result.synthesis.analytical_narrative.split("\n\n")
    narrative_html = "".join(f"<p>{_esc(p.strip())}</p>" for p in narrative_paras if p.strip())

    narrative_section = f"""
    <div class="card mb-4">
      <div class="card-header bg-dark text-white"><h4 class="mb-0">Analytical Narrative</h4></div>
      <div class="card-body">
        {narrative_html}
        <hr>
        <h5>Comparative Analysis</h5>
        <p>{_esc(result.synthesis.comparative_analysis)}</p>
      </div>
    </div>"""

    # --- Section 7: Limitations & Further Research ---
    lim_items = "".join(f"<li>{_esc(l)}</li>" for l in result.synthesis.limitations)
    test_items = "".join(f"<li>{_esc(t)}</li>" for t in result.synthesis.suggested_further_tests)

    limitations_section = f"""
    <div class="card mb-4">
      <div class="card-header"><h4 class="mb-0">Limitations & Further Research</h4></div>
      <div class="card-body">
        <h5>Limitations</h5>
        <ul>{lim_items}</ul>
        <h5>Suggested Further Tests</h5>
        <ul>{test_items}</ul>
      </div>
    </div>"""

    # --- Assemble full HTML ---
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Process Tracing Analysis</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    .card-header h4 {{ font-size: 1.1rem; }}
  </style>
</head>
<body>
<div class="container-fluid py-4" style="max-width:1400px">
  <h1 class="mb-4">Process Tracing Analysis Report</h1>
  {exec_summary}
  {network_section}
  {comparison_table}
  {test_matrix}
  {bayesian_trail}
  {absence_section}
  {narrative_section}
  {limitations_section}
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {{
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});
  var container = document.getElementById('network');
  var network = new vis.Network(container, {{nodes: nodes, edges: edges}}, {{
    nodes: {{font: {{size: 12}}, borderWidth: 2}},
    edges: {{arrows: {{to: {{enabled: true, scaleFactor: 1}}}}, font: {{size: 9, align: 'middle'}}, width: 1.5}},
    physics: {{
      enabled: true,
      stabilization: {{iterations: 200}},
      barnesHut: {{gravitationalConstant: -3000, springLength: 200, springConstant: 0.05}}
    }},
    interaction: {{hover: true, tooltipDelay: 200}}
  }});
  network.on('click', function(params) {{
    var info = document.getElementById('network-info');
    if (params.nodes.length > 0) {{
      var node = nodes.get(params.nodes[0]);
      info.innerHTML = '<strong>' + node.label + '</strong><br>' + (node.title || '');
      info.className = 'alert alert-info mt-2';
    }} else if (params.edges.length > 0) {{
      var edge = edges.get(params.edges[0]);
      info.innerHTML = '<strong>Edge:</strong> ' + (edge.label || 'relationship');
      info.className = 'alert alert-secondary mt-2';
    }}
  }});
}});
</script>
</body>
</html>"""
