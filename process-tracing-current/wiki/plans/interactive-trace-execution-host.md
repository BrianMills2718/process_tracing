---
type: Plan
title: Interactive Trace Execution Host
description: Current plan for a stage-by-stage local process-tracing host.
tags: [process-tracing, workbench, ui, planning]
created: 2026-06-24
updated: 2026-06-24
sources: [../../../docs/plans/005_interactive_trace_execution_host.md]
confidence: high
---

# Summary

Plan 005 defines the next planned UI/workbench slice: an interactive trace
execution host that lets a reviewer create a run, trigger each process-tracing
stage, and inspect the typed outputs before moving to the next stage.

The goal is not merely to display the final report. The host should teach and
audit the method by making each transformation visible: setup, extraction,
hypothesis generation, diagnostic testing, absence analysis, Bayesian support
updating, synthesis, optional refinement, and source acquisition.

# Design Gate

The plan satisfies the design-plan gate by adding a static mockup and a
contract notebook before implementation:

- `../../../docs/plans/005_interactive_trace_execution_host_mockup.html`
- `../../../docs/plans/005_interactive_trace_execution_host_contracts.ipynb`

Implementation should not start until the mockup is reviewed and requested
changes are dispositioned.

# Archive Decision

The existing `pt.workbench` should not be archived immediately. It remains a
working, tested source-acquisition surface. Plan 005 treats it as an active
narrow surface until the new host preserves or replaces equivalent behavior.
Archiving should happen only after replacement passes live E2E and a rationale
is written.

# Citations

[1] `../../../docs/plans/005_interactive_trace_execution_host.md`
