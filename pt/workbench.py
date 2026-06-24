"""Local process-tracing workbench server.

The workbench is a deliberately small HTTP surface over the existing pipeline
artifacts. It gives humans a button-driven loop while keeping every action
available through JSON endpoints for agents and tests.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pt.source_acquisition import (
    available_retrieval_providers,
    build_acquisition_plan,
    load_packet_for_result,
    load_process_result,
    retrieve_for_plan,
)
from pt.source_design import build_source_design_state
from pt.trace_host import StageGuide, TraceHostError, TraceHostStore, TraceRunRequest, build_stage_guides


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = "output/live_plan003_source_expansion_20260623_001/result.json"
DEFAULT_REPORT = "output/live_plan003_source_expansion_20260623_001/report.html"
DEFAULT_SOURCE_PACKET = "docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json"
DEFAULT_OUTPUT = "output/source_acquisition/workbench_latest.json"


def build_app_payload(
    *,
    result_path: str,
    source_packet_path: str | None,
    max_targets: int,
    retrieve: bool,
    top_k: int = 3,
    queries_per_target: int = 1,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Build the acquisition payload, optionally with live retrieval hits."""

    result = load_process_result(_resolve_repo_path(result_path))
    packet = load_packet_for_result(
        _resolve_repo_path(source_packet_path) if source_packet_path else None,
        result,
        repo_root=REPO_ROOT,
    )
    plan = build_acquisition_plan(result, source_packet=packet, max_targets=max_targets)
    payload: dict[str, Any] = {"plan": plan.model_dump()}
    if packet is not None:
        payload["design_state"] = build_source_design_state(
            result,
            source_packet=packet,
            max_targets=max_targets,
        ).model_dump()
    if retrieve:
        payload["retrieval"] = retrieve_for_plan(
            plan,
            providers=available_retrieval_providers(),
            top_k=top_k,
            queries_per_target=queries_per_target,
            cache_dir=REPO_ROOT / "output" / "open_web_cache",
        )
    if output_path:
        resolved_output = _resolve_repo_path(output_path)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["output_path"] = str(resolved_output.relative_to(REPO_ROOT))
    return payload


def make_handler() -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to the current repository root."""

    store = TraceHostStore(REPO_ROOT)
    stage_guides = build_stage_guides()

    class WorkbenchHandler(BaseHTTPRequestHandler):
        server_version = "ProcessTracingWorkbench/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(_html(stage_guides))
                return
            if parsed.path == "/artifact":
                query = parse_qs(parsed.query)
                try:
                    artifact_path = _resolve_repo_path(query.get("path", [""])[0])
                    artifact_path.relative_to(REPO_ROOT)
                except Exception:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                if not artifact_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content_type = "text/html; charset=utf-8" if artifact_path.suffix == ".html" else "text/plain; charset=utf-8"
                self._send_bytes(artifact_path.read_bytes(), content_type=content_type)
                return
            if parsed.path == "/api/health":
                self._send_json({"ok": True})
                return
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                try:
                    run = store.get_run(parts[2])
                except Exception as exc:
                    self._send_json(
                        {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                self._send_json({"ok": True, "run": run.model_dump()})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                request = self._read_json()
                if self.path == "/api/runs":
                    payload = {
                        "ok": True,
                        "run": store.create_run(TraceRunRequest.model_validate(request)).model_dump(),
                    }
                elif self.path in {"/api/acquisition-plan", "/api/enrich"}:
                    payload = {
                        "ok": True,
                        **build_app_payload(
                            result_path=str(request.get("result_path") or DEFAULT_RESULT),
                            source_packet_path=str(request.get("source_packet_path") or DEFAULT_SOURCE_PACKET),
                            max_targets=int(request.get("max_targets") or 8),
                            retrieve=self.path == "/api/enrich",
                            top_k=int(request.get("top_k") or 3),
                            queries_per_target=int(request.get("queries_per_target") or 1),
                            output_path=str(request.get("output_path") or DEFAULT_OUTPUT)
                            if self.path == "/api/enrich"
                            else None,
                        ),
                    }
                else:
                    parts = [part for part in self.path.split("/") if part]
                    if len(parts) == 6 and parts[:2] == ["api", "runs"] and parts[3] == "stages" and parts[5] == "run":
                        stage_id = parts[4]
                        if stage_id not in stage_guides:
                            raise ValueError(f"unknown stage: {stage_id}")
                        payload = store.run_stage(
                            parts[2],
                            stage_id,  # type: ignore[arg-type]
                            force=bool(request.get("force") or False),
                        )
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self._send_bytes(encoded, content_type="text/html; charset=utf-8")

        def _send_bytes(self, body: bytes, *, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WorkbenchHandler


def run_server(host: str, port: int) -> ThreadingHTTPServer:
    """Start the workbench server and block forever."""

    server = ThreadingHTTPServer((host, port), make_handler())
    bound_host, bound_port = server.server_address[:2]
    if isinstance(bound_host, bytes):
        bound_host = bound_host.decode("utf-8")
    print(f"http://{bound_host}:{bound_port}", flush=True)
    server.serve_forever()
    return server


def _resolve_repo_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _html(stage_guides: dict[str, StageGuide]) -> str:
    stage_guides_json = json.dumps({key: value.model_dump() for key, value in stage_guides.items()}, indent=2)
    stage_order_json = json.dumps([stage_id for stage_id in stage_guides], indent=2)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Process Tracing Workbench</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5f6b7a;
      --line: #d7dde5;
      --band: #f4f7fa;
      --panel: #ffffff;
      --accent: #215a7a;
      --good: #1f7a4d;
      --warn: #a05a00;
      --bad: #a63131;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 58px;
    }}
    h1 {{ font-size: 18px; margin: 0; letter-spacing: 0; }}
    main {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--band);
      padding: 16px;
      min-width: 0;
    }}
    section {{
      padding: 16px 18px;
      min-width: 0;
    }}
    label {{
      display: block;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
      margin: 10px 0 4px;
    }}
    input, select {{
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
    }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .actions {{ display: grid; gap: 8px; margin-top: 12px; }}
    button {{
      min-height: 38px;
      border: 1px solid var(--accent);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 650;
      cursor: pointer;
    }}
    button.secondary {{ background: #fff; color: var(--accent); }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    .status {{ color: var(--muted); font-size: 12px; min-height: 18px; }}
    .card, .target, .hit {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      margin-bottom: 10px;
    }}
    .card h2, .target h2 {{ font-size: 15px; margin: 0 0 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }}
    .panel-title {{
      font-size: 12px;
      font-weight: 750;
      color: var(--muted);
      margin: 0 0 8px;
      text-transform: uppercase;
    }}
    .report-shell {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
      min-height: 62vh;
      margin-bottom: 14px;
    }}
    iframe {{
      width: 100%;
      height: 62vh;
      border: 0;
      display: block;
      background: #fff;
    }}
    .stage-list {{ display: grid; gap: 6px; margin-top: 10px; }}
    .stage-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 8px;
    }}
    .stage-row.active {{ border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }}
    .stage-name {{ display: flex; align-items: center; gap: 8px; min-width: 0; }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--line);
      flex: 0 0 auto;
    }}
    .stage-row.complete .dot {{ background: var(--good); }}
    .stage-row.running .dot {{ background: var(--accent); }}
    .stage-row.blocked .dot {{ background: var(--warn); }}
    .stage-row.failed .dot {{ background: var(--bad); }}
    .stage-row.skipped .dot {{ background: #9aa4b2; }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 12px;
      padding: 2px 7px;
      white-space: nowrap;
    }}
    .monospace {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
      background: #17202a;
      color: #e7edf5;
      padding: 10px;
      border-radius: 7px;
      max-height: 260px;
      overflow: auto;
    }}
    @media (max-width: 1100px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Process Tracing Workbench</h1>
    <span id="status" class="status"></span>
  </header>
  <main>
    <aside>
      <div class="card">
        <h2>Create Run</h2>
        <label for="input-path">Input text</label>
        <input id="input-path" value="input_text/source_packets/18_brumaire_source_packet.txt">
        <label for="packet-path">Source packet</label>
        <input id="packet-path" value="__DEFAULT_SOURCE_PACKET__">
        <label for="theories-path">Theories</label>
        <input id="theories-path" value="input_text/theories/18_brumaire_rival_frameworks.txt">
        <label for="question">Research question</label>
        <input id="question" value="Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup?">
        <div class="row">
          <div>
            <label for="model">Model</label>
            <input id="model" placeholder="configured default">
          </div>
          <div>
            <label for="budget">Budget</label>
            <input id="budget" type="number" step="0.1" min="0" placeholder="1.0">
          </div>
        </div>
        <label><input id="refine" type="checkbox" checked> include refine stage</label>
        <div class="actions">
          <button id="create-run-btn">Create Run</button>
          <button id="reload-run-btn" class="secondary">Reload Run</button>
        </div>
      </div>

      <div class="card">
        <h2>Stages</h2>
        <div id="stage-list" class="stage-list"></div>
      </div>

      <div class="card">
        <h2>Acquisition</h2>
        <label for="result-path">Result JSON</label>
        <input id="result-path" value="__DEFAULT_RESULT__">
        <label for="report-path">Report HTML</label>
        <input id="report-path" value="__DEFAULT_REPORT__">
        <div class="row">
          <div>
            <label for="max-targets">Targets</label>
            <input id="max-targets" type="number" min="1" max="20" value="8">
          </div>
          <div>
            <label for="top-k">Hits</label>
            <input id="top-k" type="number" min="1" max="10" value="3">
          </div>
        </div>
        <label for="enrich-targets">Enrich targets</label>
        <input id="enrich-targets" type="number" min="1" max="5" value="2">
        <label for="output-path">Enrichment output</label>
        <input id="output-path" value="__DEFAULT_OUTPUT__">
        <div class="actions">
          <button id="load-report-btn" class="secondary">Load Report</button>
          <button id="plan-btn" class="secondary">Build Agenda</button>
          <button id="enrich-btn">Enrich Top Targets</button>
        </div>
      </div>
    </aside>

    <section>
      <div class="grid">
        <div class="card">
          <h2>Run State</h2>
          <div id="run-summary" class="status"></div>
          <div id="run-json" class="monospace"></div>
        </div>
        <div class="card" id="guide-card">
          <h2 id="guide-title">Stage Guide</h2>
          <div id="guide-body" class="status">Create a run, then select a stage.</div>
        </div>
      </div>

      <div class="card">
        <h2>Selected Stage</h2>
        <div id="stage-summary" class="status">No stage selected.</div>
        <pre id="artifact-json">{{}}</pre>
      </div>

      <p class="panel-title">Report</p>
      <div class="report-shell">
        <iframe id="report-frame" title="Process tracing report"></iframe>
      </div>

      <div class="grid">
        <div>
          <p class="panel-title">Acquisition Targets</p>
          <div id="targets"></div>
        </div>
        <div>
          <p class="panel-title">Retrieved Sources</p>
          <div id="hits"></div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const STAGE_GUIDES = __STAGE_GUIDES__;
    const STAGE_ORDER = __STAGE_ORDER__;
    let currentRunId = localStorage.getItem("pt-current-run") || "";
    let selectedStageId = "extract";

    const $ = (id) => document.getElementById(id);
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, ch => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[ch]));
    const artifactUrl = (path) => `/artifact?path=${encodeURIComponent(path)}`;
    const requestBody = (enrich=false) => ({
      result_path: $("result-path").value,
      source_packet_path: $("packet-path").value,
      max_targets: enrich ? Number($("enrich-targets").value || 2) : Number($("max-targets").value || 8),
      top_k: Number($("top-k").value || 3),
      queries_per_target: 1,
      output_path: $("output-path").value
    });

    function renderGuide(stageId) {{
      const guide = STAGE_GUIDES[stageId];
      if (!guide) return;
      $("guide-title").textContent = `Stage Guide: ${stageId}`;
      $("guide-body").innerHTML = `
        <div><strong>Purpose:</strong> ${escapeHtml(guide.purpose)}</div>
        <div><strong>Consumes:</strong> ${escapeHtml((guide.consumes || []).join(", "))}</div>
        <div><strong>Produces:</strong> ${escapeHtml((guide.produces || []).join(", "))}</div>
        <div><strong>Audit:</strong> ${escapeHtml((guide.audit_questions || []).join(" | "))}</div>
        <div title="${escapeHtml(guide.tooltip)}"><strong>Tooltip:</strong> ${escapeHtml(guide.tooltip)}</div>
      `;
    }}

    function renderStageList(run) {{
      const stages = run?.stages || [];
      const byId = Object.fromEntries(stages.map(stage => [stage.stage_id, stage]));
      $("stage-list").innerHTML = STAGE_ORDER.map(stageId => {{
        const stage = byId[stageId] || {{}};
        const status = stage.status || (stageId === "setup" ? "complete" : stageId === "extract" ? "ready" : "blocked");
        const summary = stage.summary ? ` - ${escapeHtml(stage.summary)}` : "";
        return `
          <div class="stage-row ${escapeHtml(status)} ${stageId === selectedStageId ? "active" : ""}">
            <div class="stage-name">
              <span class="dot"></span>
              <button class="secondary" data-stage="${escapeHtml(stageId)}" title="${escapeHtml(STAGE_GUIDES[stageId].tooltip)}">${escapeHtml(stageId)}</button>
              <span class="pill">${escapeHtml(status)}</span>
            </div>
            <span class="status">${summary}</span>
          </div>
        `;
      }}).join("");
      [...document.querySelectorAll("button[data-stage]")].forEach(button => {{
        button.addEventListener("click", () => runStage(button.dataset.stage));
      }});
    }}

    function renderRun(run) {{
      currentRunId = run.run_id;
      localStorage.setItem("pt-current-run", currentRunId);
      $("run-summary").textContent = `${run.case_name} | ${run.status} | current: ${run.current_stage} | output: ${run.output_dir}`;
      $("run-json").textContent = JSON.stringify(run, null, 2);
      renderStageList(run);
      renderGuide(selectedStageId);
    }}

    function renderArtifacts(payload) {{
      const artifacts = payload?.artifacts || [];
      const stage = payload?.stage || null;
      if (stage) {{
        $("stage-summary").textContent = `${stage.stage_id} | ${stage.status} | ${stage.summary || ""}`;
      }}
      $("artifact-json").textContent = JSON.stringify({{ stage, artifacts }}, null, 2);
    }}

    async function createRun() {{
      $("status").textContent = "Creating run...";
      try {{
        const body = {{
          input_path: $("input-path").value,
          source_packet_path: $("packet-path").value,
          theories_path: $("theories-path").value,
          research_question: $("question").value,
          model: $("model").value || null,
          refine: $("refine").checked,
          max_budget: $("budget").value ? Number($("budget").value) : null
        }};
        const response = await fetch("/api/runs", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(body)
        }});
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error || "Failed to create run");
        renderRun(payload.run);
        $("status").textContent = `Run ${payload.run.run_id} created`;
      }} catch (error) {{
        $("status").textContent = error.message;
      }}
    }}

    async function reloadRun() {{
      if (!currentRunId) {{
        $("status").textContent = "No run selected";
        return;
      }}
      try {{
        const response = await fetch(`/api/runs/${currentRunId}`);
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error || "Failed to load run");
        renderRun(payload.run);
        $("status").textContent = `Loaded ${payload.run.run_id}`;
      }} catch (error) {{
        $("status").textContent = error.message;
      }}
    }}

    async function runStage(stageId) {{
      if (!currentRunId) {{
        $("status").textContent = "Create a run first";
        return;
      }}
      selectedStageId = stageId;
      renderGuide(stageId);
      $("status").textContent = `Running ${stageId}...`;
      try {{
        const response = await fetch(`/api/runs/${currentRunId}/stages/${stageId}/run`, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ force: false }})
        }});
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error || "Stage failed");
        renderRun(payload.run);
        renderArtifacts(payload);
        $("status").textContent = `${stageId} complete`;
        if (stageId === "synthesize" || stageId === "refine") {{
          const resultPath = payload.run.result_path || `${payload.run.output_dir}/result.json`;
          $("result-path").value = resultPath;
          $("report-path").value = `${payload.run.output_dir}/report.html`;
          loadReport();
        }}
      }} catch (error) {{
        $("status").textContent = error.message;
      }}
    }}

    function loadReport() {{
      $("report-frame").src = artifactUrl($("report-path").value);
    }}

    function renderAcquisition(payload) {{
      const targets = payload.plan?.targets || [];
      $("targets").innerHTML = targets.map(target => `
        <article class="target">
          <h2>${escapeHtml(target.target_id)} <span class="pill">${escapeHtml(target.kind)}</span></h2>
          <div class="meta">
            <span class="pill">${escapeHtml(target.target_source_class)}</span>
            <span class="pill">score ${escapeHtml(target.priority_score)}</span>
          </div>
          <p>${escapeHtml(target.evidence_need)}</p>
          <p>${escapeHtml(target.inferential_payoff)}</p>
          <p class="monospace">${escapeHtml((target.search_queries || [])[0] || "")}</p>
        </article>
      `).join("");
      const retrieval = payload.retrieval || [];
      $("hits").innerHTML = retrieval.flatMap(item => (item.hits || []).map(hit => `
        <article class="hit">
          <div class="meta">
            <span class="pill">${escapeHtml(item.target_id)}</span>
            <span class="pill">${escapeHtml(hit.provider)} #${escapeHtml(hit.rank)}</span>
            <span class="pill">${hit.extracted ? `extracted ${escapeHtml(hit.text_char_count)} chars` : "not extracted"}</span>
          </div>
          <a href="${escapeHtml(hit.url)}" target="_blank" rel="noreferrer">${escapeHtml(hit.title || hit.url)}</a>
          <p>${escapeHtml(hit.snippet || "")}</p>
        </article>
      `)).join("");
    }}

    async function buildAgenda(retrieve=false) {{
      $("status").textContent = retrieve ? "Enriching..." : "Building agenda...";
      try {{
        const path = retrieve ? "/api/enrich" : "/api/acquisition-plan";
        const response = await fetch(path, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(requestBody(retrieve))
        }});
        const payload = await response.json();
        if (!payload.ok) throw new Error(payload.error || "Request failed");
        renderAcquisition(payload);
        if (payload.design_state) {{
          $("artifact-json").textContent = JSON.stringify(payload.design_state, null, 2);
        }}
        $("status").textContent = payload.output_path ? `Saved ${payload.output_path}` : "Ready";
      }} catch (error) {{
        $("status").textContent = error.message;
      }}
    }}

    $("create-run-btn").addEventListener("click", createRun);
    $("reload-run-btn").addEventListener("click", reloadRun);
    $("load-report-btn").addEventListener("click", loadReport);
    $("plan-btn").addEventListener("click", () => buildAgenda(false));
    $("enrich-btn").addEventListener("click", () => buildAgenda(true));
    renderGuide(selectedStageId);
    loadReport();
    if (currentRunId) {{
      reloadRun();
    }} else {{
      buildAgenda(false);
    }}
  </script>
</body>
</html>"""
    return (
        html.replace("__DEFAULT_SOURCE_PACKET__", DEFAULT_SOURCE_PACKET)
        .replace("__DEFAULT_RESULT__", DEFAULT_RESULT)
        .replace("__DEFAULT_REPORT__", DEFAULT_REPORT)
        .replace("__DEFAULT_OUTPUT__", DEFAULT_OUTPUT)
        .replace("__STAGE_GUIDES__", stage_guides_json)
        .replace("__STAGE_ORDER__", stage_order_json)
        .replace("{{", "{")
        .replace("}}", "}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
