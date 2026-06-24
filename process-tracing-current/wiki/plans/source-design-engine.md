---
type: Plan
title: Source Design Engine
description: Execution-ready child plan for iterative source design, acquisition, and disposition.
tags: [process-tracing, planning, source-design, methodology]
created: 2026-06-24
updated: 2026-06-24
sources: [../../../docs/plans/006_source_design_engine.md]
confidence: high
---

# Summary

Plan 006 is the execution-ready child plan for the next non-UI methodology
slice. It upgrades the current source packet and source-acquisition surfaces
into a fuller source-design engine that can persist acquisition actions, review
outcomes, and gap disposition changes across repeated trace iterations.

The plan exists because the repository's SOTA+ target is not just a static
packet with a retrieval helper. The end-goal methodology requires an active
loop that decides what source class to pursue next, what counts as adjacent but
insufficient evidence, and how unresolved gaps continue to cap claims.

# Design Gate

The plan includes the required pre-implementation seam artifacts:

- `../../../docs/plans/006_source_design_engine_mockup.md`
- `../../../docs/plans/006_source_design_engine_contracts.ipynb`

# Citations

[1] `../../../docs/plans/006_source_design_engine.md`
