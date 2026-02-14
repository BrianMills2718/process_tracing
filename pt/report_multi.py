"""Cross-case HTML report with vis.js DAG, binarization matrix, and CQ results."""

from __future__ import annotations

import html
import json
import os

from pt.schemas_multi import MultiDocResult


def _esc(s: str) -> str:
    return html.escape(str(s))


def _cell_color(value: int | None, confidence: float = 1.0) -> str:
    """Color for a binarization cell."""
    if value is None:
        return "#6c757d"  # grey for NA
    if confidence < 0.7:
        return "#f0ad4e"  # amber for low confidence
    return "#28a745" if value == 1 else "#dc3545"


def _cell_text(value: int | None) -> str:
    if value is None:
        return "NA"
    return str(value)


def generate_multi_report(result: MultiDocResult, output_dir: str = "") -> str:
    """Generate a self-contained HTML report for multi-document analysis."""
    parts: list[str] = []

    # ── HTML Head ──────────────────────────────────────────────────
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-Case Process Tracing Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.6/dist/vis-network.min.js"></script>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; }
  .cell-badge { display: inline-block; width: 40px; text-align: center;
                padding: 4px 8px; border-radius: 4px; color: white; font-weight: bold; }
  .justification { font-size: 0.85rem; color: #555; margin-top: 4px; }
  #dag-container { height: 400px; border: 1px solid #ddd; border-radius: 8px; }
  .sensitivity-table td { text-align: center; }
  .section-header { border-bottom: 2px solid #0d6efd; padding-bottom: 8px; margin-top: 2rem; }
  .case-link { font-size: 0.85rem; }
</style>
</head>
<body>
<div class="container-fluid py-4" style="max-width: 1400px;">
""")

    # ── Title ──────────────────────────────────────────────────────
    model = result.causal_model
    parts.append(f"""
<h1 class="mb-1">Cross-Case Analysis: {_esc(model.name)}</h1>
<p class="text-muted">{_esc(model.description[:200])}</p>
<div class="row mb-3">
  <div class="col-auto"><span class="badge bg-primary">{result.workflow.replace('_', ' ').title()}</span></div>
  <div class="col-auto"><span class="badge bg-secondary">{len(result.binarizations)} cases</span></div>
  <div class="col-auto"><span class="badge bg-secondary">{len(model.variables)} variables</span></div>
  {"<div class='col-auto'><span class='badge bg-success'>CQ estimated</span></div>" if result.cq_result else "<div class='col-auto'><span class='badge bg-warning text-dark'>No CQ</span></div>"}
</div>
""")

    # ── Section 1: Causal Model DAG ───────────────────────────────
    parts.append("""
<h2 class="section-header">1. Causal Model DAG</h2>
<div id="dag-container" class="mb-3"></div>
""")

    # Build vis.js data
    nodes = []
    for v in model.variables:
        color = "#0d6efd" if v.name == model.outcome_variable else "#6c757d"
        shape = "box" if v.name == model.outcome_variable else "ellipse"
        nodes.append({
            "id": v.name,
            "label": v.name.replace("_", " ").title(),
            "color": color,
            "shape": shape,
            "title": f"1: {v.description}\\n0: {v.description_zero}",
        })
    edges_vis = []
    for e in model.edges:
        edges_vis.append({"from": e.parent, "to": e.child, "arrows": "to"})

    parts.append(f"""
<script>
var dagNodes = new vis.DataSet({json.dumps(nodes)});
var dagEdges = new vis.DataSet({json.dumps(edges_vis)});
var dagContainer = document.getElementById('dag-container');
var dagData = {{ nodes: dagNodes, edges: dagEdges }};
var dagOptions = {{
    layout: {{ hierarchical: {{ direction: 'LR', sortMethod: 'directed', levelSeparation: 200 }} }},
    physics: false,
    nodes: {{ font: {{ size: 14 }}, borderWidth: 2 }},
    edges: {{ color: '#333', width: 2 }}
}};
new vis.Network(dagContainer, dagData, dagOptions);
</script>
""")

    # Variable descriptions table
    parts.append("""
<h5>Variable Definitions</h5>
<table class="table table-sm table-bordered">
<thead><tr><th>Variable</th><th>1 (Present)</th><th>0 (Absent)</th><th>Indicators</th></tr></thead>
<tbody>
""")
    for v in model.variables:
        indicators = "<br>".join(f"&bull; {_esc(ind)}" for ind in v.observable_indicators[:4])
        outcome_badge = ' <span class="badge bg-primary">outcome</span>' if v.name == model.outcome_variable else ""
        parts.append(f"""<tr>
  <td><strong>{_esc(v.name)}</strong>{outcome_badge}</td>
  <td>{_esc(v.description[:120])}</td>
  <td>{_esc(v.description_zero[:120])}</td>
  <td style="font-size:0.85rem">{indicators}</td>
</tr>""")
    parts.append("</tbody></table>")

    # ── Section 2: Binary Data Matrix ─────────────────────────────
    parts.append("""
<h2 class="section-header">2. Binary Data Matrix</h2>
<table class="table table-sm table-bordered text-center">
<thead><tr><th>Case</th>
""")
    for v in model.variables:
        parts.append(f"<th>{_esc(v.name)}</th>")
    parts.append("</tr></thead><tbody>")

    # Build a lookup for quick access to codings
    for binarization in result.binarizations:
        coding_map = {c.variable_name: c for c in binarization.codings}
        parts.append(f"<tr><td class='text-start'><strong>{_esc(binarization.case_id)}</strong></td>")
        for v in model.variables:
            c = coding_map.get(v.name)
            if c:
                color = _cell_color(c.value, c.confidence)
                text = _cell_text(c.value)
                conf_pct = f"{c.confidence:.0%}"
                parts.append(
                    f'<td><span class="cell-badge" style="background:{color}" '
                    f'title="Confidence: {conf_pct}">{text}</span></td>'
                )
            else:
                parts.append('<td><span class="cell-badge" style="background:#6c757d">?</span></td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")

    # ── Section 3: Coding Justifications ──────────────────────────
    parts.append("""
<h2 class="section-header">3. Coding Justifications</h2>
<div class="accordion" id="justifications-accordion">
""")
    for i, binarization in enumerate(result.binarizations):
        case_id = binarization.case_id
        parts.append(f"""
<div class="accordion-item">
  <h2 class="accordion-header">
    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
            data-bs-target="#just-{i}" aria-expanded="false">{_esc(case_id)}</button>
  </h2>
  <div id="just-{i}" class="accordion-collapse collapse" data-bs-parent="#justifications-accordion">
    <div class="accordion-body">
      <table class="table table-sm">
      <thead><tr><th>Variable</th><th>Value</th><th>Conf.</th><th>Justification</th><th>Evidence</th></tr></thead>
      <tbody>
""")
        for c in binarization.codings:
            color = _cell_color(c.value, c.confidence)
            text = _cell_text(c.value)
            evi_str = ", ".join(c.evidence_ids[:5]) if c.evidence_ids else "-"
            parts.append(f"""<tr>
  <td>{_esc(c.variable_name)}</td>
  <td><span class="cell-badge" style="background:{color}">{text}</span></td>
  <td>{c.confidence:.2f}</td>
  <td style="font-size:0.85rem">{_esc(c.justification[:200])}</td>
  <td style="font-size:0.8rem">{_esc(evi_str)}</td>
</tr>""")
        parts.append("</tbody></table>")
        if binarization.analyst_notes:
            parts.append(f'<p class="justification"><em>{_esc(binarization.analyst_notes[:300])}</em></p>')
        parts.append("</div></div></div>")
    parts.append("</div>")

    # ── Section 4: CausalQueries Results ──────────────────────────
    if result.cq_result:
        cq = result.cq_result
        parts.append(f"""
<h2 class="section-header">4. CausalQueries Estimates</h2>
<p class="text-muted">Model: <code>{_esc(cq.model_statement)}</code> | {cq.n_cases} cases</p>
""")

        # Population ATEs
        pop_posteriors = [e for e in cq.population_estimands if e.using == "posteriors"]
        if pop_posteriors:
            parts.append("""
<h5>Population Average Treatment Effects</h5>
<table class="table table-sm table-bordered">
<thead><tr><th>Query</th><th>Mean</th><th>SD</th><th>95% CI</th></tr></thead>
<tbody>
""")
            for est in pop_posteriors:
                ci = f"[{est.cred_low:.3f}, {est.cred_high:.3f}]" if est.cred_low is not None and est.cred_high is not None else "-"
                sd_str = f"{est.sd:.3f}" if est.sd is not None else "-"
                parts.append(f"""<tr>
  <td><code>{_esc(est.query)}</code></td>
  <td><strong>{est.mean:.3f}</strong></td>
  <td>{sd_str}</td>
  <td>{ci}</td>
</tr>""")
            parts.append("</tbody></table>")

        # Case-level
        if cq.case_level_estimands:
            parts.append("""
<h5>Case-Level Causal Attribution</h5>
<table class="table table-sm table-bordered">
<thead><tr><th>Case</th><th>Query</th><th>Mean</th><th>SD</th><th>95% CI</th></tr></thead>
<tbody>
""")
            for est in cq.case_level_estimands:
                ci = f"[{est.cred_low:.3f}, {est.cred_high:.3f}]" if est.cred_low is not None and est.cred_high is not None else "-"
                sd_str = f"{est.sd:.3f}" if est.sd is not None else "-"
                parts.append(f"""<tr>
  <td>{_esc(est.case_id)}</td>
  <td><code>{_esc(est.query)}</code></td>
  <td><strong>{est.mean:.3f}</strong></td>
  <td>{sd_str}</td>
  <td>{ci}</td>
</tr>""")
            parts.append("</tbody></table>")
    else:
        parts.append("""
<h2 class="section-header">4. CausalQueries Estimates</h2>
<div class="alert alert-warning">CausalQueries was not run. Use without <code>--skip-cq</code> and with R installed to get causal estimates.</div>
""")

    # ── Section 5: Binarization Sensitivity ───────────────────────
    if result.sensitivity:
        sens = result.sensitivity
        parts.append("""
<h2 class="section-header">5. Binarization Sensitivity Analysis</h2>
<p class="text-muted">Shows how CQ posteriors change when the confidence threshold for binarization varies.</p>
""")
        # Summary
        if sens.stable_estimands or sens.fragile_estimands:
            parts.append('<div class="row mb-3">')
            if sens.stable_estimands:
                parts.append(f'<div class="col-auto"><span class="badge bg-success">{len(sens.stable_estimands)} stable</span></div>')
            if sens.fragile_estimands:
                parts.append(f'<div class="col-auto"><span class="badge bg-danger">{len(sens.fragile_estimands)} fragile</span></div>')
            parts.append('</div>')

        # Threshold runs table
        parts.append("""
<table class="table table-sm table-bordered sensitivity-table">
<thead><tr><th>Threshold</th><th>NA Codings</th>
""")
        # Collect all queries from first run that has CQ
        all_queries: list[str] = []
        for run in sens.runs:
            if run.cq_result:
                all_queries = [e.query for e in run.cq_result.population_estimands if e.using == "posteriors"]
                break
        for q in all_queries:
            short_q = q.split("[")[0] if "[" in q else q
            parts.append(f"<th>{_esc(short_q)}</th>")
        parts.append("</tr></thead><tbody>")

        for run in sens.runs:
            parts.append(f"<tr><td>{run.confidence_threshold}</td><td>{run.n_na_codings}</td>")
            if run.cq_result:
                est_map = {e.query: e for e in run.cq_result.population_estimands if e.using == "posteriors"}
                for q in all_queries:
                    e = est_map.get(q)
                    if e:
                        parts.append(f"<td>{e.mean:.3f}</td>")
                    else:
                        parts.append("<td>-</td>")
            else:
                for _ in all_queries:
                    parts.append("<td>-</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")

        # Fragile estimands detail
        if sens.fragile_estimands:
            parts.append('<div class="alert alert-warning"><strong>Fragile estimands</strong> (posteriors vary &ge; 0.1 across thresholds):<ul>')
            for q in sens.fragile_estimands:
                parts.append(f"<li><code>{_esc(q)}</code></li>")
            parts.append("</ul></div>")

    # ── Section 6: Cross-Case Comparison ──────────────────────────
    parts.append("""
<h2 class="section-header">6. Cross-Case Comparison</h2>
""")
    for v in result.causal_model.variables:
        parts.append(f"<h5>{_esc(v.name.replace('_', ' ').title())}</h5>")
        parts.append("<ul>")
        for binarization in result.binarizations:
            coding_map = {c.variable_name: c for c in binarization.codings}
            c = coding_map.get(v.name)
            if c:
                val_str = f"<strong>{'Present' if c.value == 1 else 'Absent' if c.value == 0 else 'NA'}</strong>"
                parts.append(f"<li>{_esc(binarization.case_id)}: {val_str} (conf: {c.confidence:.2f}) — {_esc(c.justification[:100])}</li>")
        parts.append("</ul>")

    # ── Section 7: Links to Individual Reports ────────────────────
    parts.append("""
<h2 class="section-header">7. Individual Case Reports</h2>
<ul>
""")
    for case_id, result_path in result.case_results.items():
        case_dir = os.path.dirname(result_path)
        report_path = os.path.join(case_dir, "report.html")
        rel_path = os.path.relpath(report_path, output_dir) if output_dir else report_path
        parts.append(f'<li><a href="{_esc(rel_path)}" class="case-link">{_esc(case_id)}</a> — <code>{_esc(result_path)}</code></li>')
    parts.append("</ul>")

    # ── Footer ────────────────────────────────────────────────────
    parts.append("""
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
""")

    return "".join(parts)
