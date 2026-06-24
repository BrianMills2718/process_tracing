---
type: Architecture
title: Architecture Spine
description: Current high-level map of the active process tracing system and its knowledge boundaries.
tags: [architecture, contracts, process-tracing]
created: 2026-06-24
updated: 2026-06-24
sources: [../../../docs/ARCHITECTURE.md, ../../../CLAUDE.md]
confidence: medium
---

# Summary

The active system is organized around a typed LLM-first process tracing pipeline:
source text and optional source packets enter the CLI, pipeline passes produce a
structured `result.json`, deterministic components compute support and audit
signals, and reporting/workbench surfaces expose the result for review and
source acquisition.

This wiki page is a navigation layer. The authoritative architecture document is
`../../../docs/ARCHITECTURE.md`.

# Boundaries

- CLI and workbench are user/agent entrypoints.
- Pipeline passes produce and consume Pydantic models.
- LLM semantic work routes through the project LLM boundary and shared
  `llm_client`.
- Bayesian updating and report/audit calculations are deterministic.
- Source acquisition is a separate planning/enrichment loop over existing
  results and source packets.

# Citations

[1] `../../../docs/ARCHITECTURE.md`
[2] `../../../CLAUDE.md`
