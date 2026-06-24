---
type: Overview
title: Process Tracing Current Repo Overview
description: Current-state synthesis of the active automated process tracing repo.
tags: [process-tracing, current-state, orientation]
created: 2026-06-24
updated: 2026-06-24
sources: [../../CLAUDE.md, ../../docs/ARCHITECTURE.md, ../../docs/ARCHIVE_POLICY.md]
confidence: medium
---

# Summary

The active repo implements an LLM-first process tracing pipeline that extracts
evidence, generates competing causal hypotheses, tests evidence against those
hypotheses, performs deterministic Bayesian support updating, synthesizes an
analytic narrative, and renders audit-oriented outputs.

The current strategic target is PhD/think-tank quality automated process tracing,
not generic qualitative coding. The repo wiki is a compiled orientation layer for
that active system. It must be checked against active repo sources before any
implementation decision.

# Current Orientation Links

- [Architecture Spine](/wiki/architecture/architecture-spine.md)
- [Archive And Wiki Policy](/wiki/sources/archive-policy.md)
- [Evidence Policy](/wiki/sources/evidence-policy.md)
- [Archive Migration Review 2026-06-24](/wiki/plans/archive-migration-review-2026-06-24.md)

# Open Questions

- Which active docs should be ingested first beyond the initial policy and
  architecture spine?
- Which archive bundles should be moved to `~/archive/process_tracing` after
  human review?

# Citations

[1] `../../CLAUDE.md`
[2] `../../docs/ARCHITECTURE.md`
[3] `../../docs/ARCHIVE_POLICY.md`
