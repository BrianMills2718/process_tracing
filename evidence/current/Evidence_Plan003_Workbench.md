# Evidence - Plan 003 Local Workbench

## Scope

This slice moves the acquisition loop from a static report/CLI-only workflow to
a local server with a human-clickable enrichment action and JSON endpoints for
agents.

## Implementation

- Added `pt.workbench`.
- Added `make workbench`.
- Added `/api/acquisition-plan` for deterministic target generation.
- Added `/api/enrich` for live `open_web_retrieval` search/fetch/extract.
- Added `/artifact?path=...` and embedded the existing `report.html` so the
  workbench wraps the full analytical report instead of replacing it.
- Reused `pt.source_acquisition` for CLI, server, and tests.

## Verification

Focused deterministic tests:

```bash
PYTHONPATH=. pytest tests/test_workbench.py tests/test_source_acquisition.py -q
```

Result: `5 passed`.

Browser verification:

```bash
make workbench WORKBENCH_PORT=8501
```

URL: `http://127.0.0.1:8501`

Puppeteer verified:

- page loads and renders acquisition targets;
- original report loads in the `report-frame` iframe with title
  `Process Tracing Analysis Report`;
- `Enrich Top Targets` invokes the server API;
- enrichment completes and saves `output/source_acquisition/workbench_latest.json`;
- retrieved-source panel renders six hits across the top two targets;
- first hit is Jacques Boudon, *Lucien Bonaparte et le coup d'État de Brumaire*.

Screenshot artifact names:

- `process_tracing_workbench_initial`
- `process_tracing_workbench_enriched_complete`
- `process_tracing_workbench_report_embedded`

## Critique

This is a workbench, not yet a complete research application. It now preserves
the full report as the main review surface and adds enrichment as an action
layer. It still does not let the user approve a retrieved source into the source
packet, append extracted text to the corpus, rerun the trace, or compare before
/ after inference. Those should be the next workflow steps before investing in
more report polish.
