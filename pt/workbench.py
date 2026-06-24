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

    class WorkbenchHandler(BaseHTTPRequestHandler):
        server_version = "ProcessTracingWorkbench/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_html(_html())
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
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path not in {"/api/acquisition-plan", "/api/enrich"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                request = self._read_json()
                payload = build_app_payload(
                    result_path=str(request.get("result_path") or DEFAULT_RESULT),
                    source_packet_path=str(request.get("source_packet_path") or DEFAULT_SOURCE_PACKET),
                    max_targets=int(request.get("max_targets") or 8),
                    retrieve=self.path == "/api/enrich",
                    top_k=int(request.get("top_k") or 3),
                    queries_per_target=int(request.get("queries_per_target") or 1),
                    output_path=str(request.get("output_path") or DEFAULT_OUTPUT)
                    if self.path == "/api/enrich"
                    else None,
                )
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            self._send_json({"ok": True, **payload})

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
            body = json.dumps(payload).encode("utf-8")
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


def _html() -> str:
    return f"""<!doctype html>
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
      background: #ffffff;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 14px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{ font-size: 18px; margin: 0; letter-spacing: 0; }}
    main {{ display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 58px); }}
    aside {{ border-right: 1px solid var(--line); padding: 16px; background: var(--band); }}
    section {{ padding: 16px 20px; }}
    label {{ display: block; font-size: 12px; font-weight: 650; color: var(--muted); margin: 12px 0 4px; }}
    input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
      background: #fff;
    }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .actions {{ display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 14px; }}
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
    .status {{ color: var(--muted); font-size: 12px; min-height: 20px; margin-top: 10px; }}
    .target, .hit {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
      background: #fff;
    }}
    .target h2 {{ font-size: 15px; margin: 0 0 6px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; font-size: 12px; color: var(--muted); }}
    .score {{ color: var(--good); font-weight: 700; }}
    .query {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #334155; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }}
    .panel-title {{ font-size: 13px; font-weight: 750; color: var(--muted); margin: 0 0 10px; text-transform: uppercase; }}
    .report-shell {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
      min-height: 68vh;
      margin-bottom: 16px;
    }}
    iframe {{
      width: 100%;
      height: 68vh;
      border: 0;
      display: block;
      background: #fff;
    }}
    a {{ color: var(--accent); overflow-wrap: anywhere; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Process Tracing Workbench</h1>
    <span id="provider-status" class="status"></span>
  </header>
  <main>
    <aside>
      <label for="result-path">Result JSON</label>
      <input id="result-path" value="{DEFAULT_RESULT}">
      <label for="report-path">Report HTML</label>
      <input id="report-path" value="{DEFAULT_REPORT}">
      <label for="packet-path">Source Packet</label>
      <input id="packet-path" value="{DEFAULT_SOURCE_PACKET}">
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
      <label for="enrich-targets">Enrich Targets</label>
      <input id="enrich-targets" type="number" min="1" max="5" value="2">
      <label for="output-path">Enrichment Output</label>
      <input id="output-path" value="{DEFAULT_OUTPUT}">
      <div class="actions">
        <button id="report-btn" class="secondary">Load Report</button>
        <button id="plan-btn" class="secondary">Build Agenda</button>
        <button id="enrich-btn">Enrich Top Targets</button>
      </div>
      <div id="status" class="status"></div>
    </aside>
    <section>
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
    const $ = (id) => document.getElementById(id);
    function artifactUrl(path) {{
      return `/artifact?path=${{encodeURIComponent(path)}}`;
    }}
    function loadReport() {{
      $("report-frame").src = artifactUrl($("report-path").value);
    }}
    function requestBody(enrich=false) {{
      const maxTargets = Number($("max-targets").value || 8);
      return {{
        result_path: $("result-path").value,
        source_packet_path: $("packet-path").value,
        max_targets: enrich ? Number($("enrich-targets").value || 2) : maxTargets,
        top_k: Number($("top-k").value || 3),
        queries_per_target: 1,
        output_path: $("output-path").value
      }};
    }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
    }}
    async function postJson(path) {{
      $("status").textContent = path === "/api/enrich" ? "Enriching..." : "Building...";
      $("plan-btn").disabled = true;
      $("enrich-btn").disabled = true;
      try {{
        const res = await fetch(path, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(requestBody(path === "/api/enrich"))
        }});
        const payload = await res.json();
        if (!payload.ok) throw new Error(payload.error || "Request failed");
        render(payload);
        $("status").textContent = payload.output_path ? `Saved ${{payload.output_path}}` : "Ready";
      }} catch (error) {{
        $("status").textContent = error.message;
      }} finally {{
        $("plan-btn").disabled = false;
        $("enrich-btn").disabled = false;
      }}
    }}
    function render(payload) {{
      const targets = payload.plan.targets || [];
      $("targets").innerHTML = targets.map(target => `
        <article class="target">
          <h2>${{escapeHtml(target.target_id)}} <span class="score">${{target.priority_score}}</span></h2>
          <div class="meta">
            <span class="pill">${{escapeHtml(target.kind)}}</span>
            <span class="pill">${{escapeHtml(target.target_source_class)}}</span>
          </div>
          <p>${{escapeHtml(target.evidence_need)}}</p>
          <p>${{escapeHtml(target.inferential_payoff)}}</p>
          <p class="query">${{escapeHtml((target.search_queries || [])[0] || "")}}</p>
        </article>
      `).join("");
      const retrieval = payload.retrieval || [];
      $("hits").innerHTML = retrieval.flatMap(item => (item.hits || []).map(hit => `
        <article class="hit">
          <div class="meta">
            <span class="pill">${{escapeHtml(item.target_id)}}</span>
            <span class="pill">${{escapeHtml(hit.provider)}} #${{escapeHtml(hit.rank)}}</span>
            <span class="pill">${{hit.extracted ? `extracted ${{hit.text_char_count}} chars` : "not extracted"}}</span>
          </div>
          <a href="${{escapeHtml(hit.url)}}" target="_blank" rel="noreferrer">${{escapeHtml(hit.title || hit.url)}}</a>
          <p>${{escapeHtml(hit.snippet || "")}}</p>
        </article>
      `)).join("");
    }}
    $("report-btn").addEventListener("click", loadReport);
    $("plan-btn").addEventListener("click", () => postJson("/api/acquisition-plan"));
    $("enrich-btn").addEventListener("click", () => postJson("/api/enrich"));
    loadReport();
    postJson("/api/acquisition-plan");
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
