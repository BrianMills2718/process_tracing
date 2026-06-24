---
type: Plan
title: Archive Migration Review 2026-06-24
description: Review record for archive migration batches completed on 2026-06-24.
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

The second migration batch moved historical debug, testing, validation, and
phase-roadmap surfaces. These files either referenced removed legacy APIs,
explicitly described themselves as historical, or represented superseded roadmap
material. Current verification remains centered on `tests/` and `make check`.

# Classification

| Scope | Decision | Rationale |
| --- | --- | --- |
| `docs/archive/` | move-to-archive | Already named as archived/superseded material. |
| top-level `archive/` | move-to-archive | Already outside active docs/code structure and dated as historical. |
| `docs/debug/` | move-to-archive | Legacy debug artifacts referenced removed `core/` APIs and old output paths. |
| `docs/testing/` | move-to-archive | Legacy script tests targeted removed APIs; active tests live in `tests/`. |
| `docs/validation/` | move-to-archive | Directory marked itself historical; active validation is documented in `docs/VALIDATION.md`. |
| `docs/phases/` | move-to-archive | Old phase roadmap material is superseded by active SOTA+ plans. |

# Next Review Batch

No historical doc directories remain in the active repo. The next cleanup pass
should scan for stale references in `evidence/` and decide whether evidence
artifacts need an archive policy of their own.

# Citations

[1] `../../../docs/ARCHIVE_POLICY.md`
[2] `~/archive/process_tracing/wiki/sources/repo-archive-2026-06-24.md`
[3] `~/archive/process_tracing/wiki/sources/repo-historical-surfaces-2026-06-24.md`
