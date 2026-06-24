---
type: Plan
title: Archive Migration Review 2026-06-24
description: Review record for the first archive migration batch.
tags: [archive, migration, cleanup, current-state]
created: 2026-06-24
updated: 2026-06-24
sources: [../../../docs/ARCHIVE_POLICY.md, ../../../../archive/process_tracing/wiki/sources/repo-archive-2026-06-24.md]
confidence: medium
---

# Summary

The first migration batch moved files that were already under explicit archive
paths: active repo `docs/archive/` and top-level `archive/`. The move reduces
agent search clutter inside `~/projects/process_tracing` while preserving raw
sources, original paths, hashes, and archive wiki summaries under
`~/archive/process_tracing`.

# Classification

| Scope | Decision | Rationale |
| --- | --- | --- |
| `docs/archive/` | move-to-archive | Already named as archived/superseded material. |
| top-level `archive/` | move-to-archive | Already outside active docs/code structure and dated as historical. |
| `docs/debug/` | defer | Contains executable/debug artifacts; needs separate active-vs-historical review. |
| `docs/testing/` | defer | Referenced by validation and planning docs; needs separate review. |
| `docs/validation/` | defer | Referenced by validation docs; needs separate review. |
| `docs/phases/` | defer | Some phase docs may still describe future ambitions; needs separate review. |

# Next Review Batch

Review `docs/debug/`, `docs/testing/`, `docs/validation/`, and `docs/phases/`
with the same disposition categories before moving anything else.

# Citations

[1] `../../../docs/ARCHIVE_POLICY.md`
[2] `~/archive/process_tracing/wiki/sources/repo-archive-2026-06-24.md`

