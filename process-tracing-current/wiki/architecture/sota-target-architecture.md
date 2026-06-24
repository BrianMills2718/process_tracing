---
type: Architecture
title: SOTA+ Target Architecture
description: End-goal architecture for the intended process-tracing SOTA+ system.
tags: [process-tracing, architecture, methodology, sota]
created: 2026-06-24
updated: 2026-06-24
sources: [../../../docs/SOTA_PLUS_TARGET_ARCHITECTURE.md]
confidence: high
---

# Summary

This page records the end-goal architecture for the repository's intended
SOTA+ methodology. It complements, rather than replaces, the current-state
architecture spine.

The target architecture makes explicit that the endpoint is not just the
current pipeline plus a nicer report. The intended system includes a source
design engine, trace-production model, dependence/lineage layer, structural
critic loop, frozen benchmark runner, and a gated within-case -> cross-case
causal-model bridge.

# Key Distinction

- Current implemented system:
  [Architecture Spine](/wiki/architecture/architecture-spine.md)
- End-goal target system:
  [SOTA+ Target Architecture](/wiki/architecture/sota-target-architecture.md)

The target architecture is the authority for future-slice boundaries and
contracts. The current architecture is the authority for what is implemented
today.

# Citations

[1] `../../../docs/SOTA_PLUS_TARGET_ARCHITECTURE.md`
