---
type: Source
title: Evidence Policy
description: Active policy for current evidence notes versus archived historical evidence.
tags: [evidence, archive, policy, sota]
created: 2026-06-24
updated: 2026-06-24
sources: [../../../docs/EVIDENCE_POLICY.md]
confidence: high
---

# Summary

The evidence policy keeps active evidence small and plan-linked. Current evidence
in the repo is limited to Plan 003 notes under `evidence/current/`. Historical
Phase 23-27, baseline, and legacy architecture evidence moved to
`~/archive/process_tracing/raw/repo-historical-evidence-2026-06-24/`.

# Interpretation

Evidence notes are useful only when they support the current implementation
surface. Retrospective evidence remains recoverable in the archive, but agents
should not search it during normal implementation unless the task is historical
recovery or previous-attempt review.

The practical rule is that evidence should follow active plans, not become a
parallel knowledge base. If a future slice needs durable verification notes, add
them under the active plan's naming convention and keep the manifest narrow. If a
slice retires, move its evidence with the same source-manifest and wiki-summary
process used for historical docs.

# Citations

[1] `../../../docs/EVIDENCE_POLICY.md`
