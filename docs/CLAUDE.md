# docs

Maintained documentation for `process_tracing`.

## Active Top-Level Docs

- `PROJECT_THEORY_AND_GOALS.md` - canonical project intent and current
  capability ledger.
- `WHITEPAPER_optimal_automated_process_tracing.md` - quality-optimal
  methodology paper.
- `BUILDPLAN_pragmatic_process_tracing.md` - pragmatic implementation plan and
  compromise record.
- `OUTPUT_QUALITY_RUBRIC.md` - grading rubric for generated `result.json` and
  `report.html` outputs.
- `ontology.md` - current analytic ontology and report-network semantics.
- `FUTURE_WORK.md` - current roadmap after the inference-core rebuild.
- `REVIEWER_WALKTHROUGH.md` - portfolio/reviewer orientation.
- `ARCHIVE_POLICY.md` - current policy for moving retired material out of
  `~/projects/` and maintaining OKF wiki logs.
- `research/PROCESS_TRACING_SOTA_REVIEW_2026.md` - external SOTA map for
  process tracing, mixed methods, and LLM-assisted qualitative/causal work.

## Route By Question

- active implementation planning -> `plans/`
- SOTA/literature/research synthesis -> `research/`
- active deterministic verification -> `../tests/` via `make check`
- historical manual verification helpers -> `testing/`
- historical validation benchmarks and checks -> `validation/`
- legacy sprint/status reports and superseded architecture analyses ->
  `~/archive/process_tracing/`

## Working Rules

- Keep this parent file at routing level only.
- Historical or superseded material belongs in `~/archive/process_tracing/`, not
  in new top-level docs files.
- Do not add a new top-level Markdown file unless it becomes one of the active
  docs listed above.
- When changing pipeline behavior, update the closest methodological or
  validation doc rather than adding loose status notes here.
