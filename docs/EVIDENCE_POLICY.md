---
status: active
owner: process-tracing
updated: 2026-06-24
---

# Evidence Policy

Current implementation evidence belongs in `evidence/current/` and should be tied
to an active plan, slice, or verification gate. Historical evidence from retired
architectures belongs in `~/archive/process_tracing/`.

## Current Evidence

The active repo currently keeps Plan 003 evidence notes:

- `evidence/current/Evidence_Plan003*.md`

These notes support the active SOTA+ execution plan and current source-packet,
source-acquisition, and workbench slices.

## Historical Evidence

Older Phase 23-27, baseline, and legacy architecture evidence has been archived
at:

`~/archive/process_tracing/raw/repo-historical-evidence-2026-06-24/`

The archive bundle preserves original paths and SHA-256 hashes in its manifest.
Do not restore those files into the active repo unless a specific artifact is
rebuilt against current `pt/` APIs and promoted through active docs/tests.

## Agent Search Policy

Agents may search active Plan 003 evidence during current implementation review.
Agents must not search historical evidence during normal implementation unless
the user asks for recovery, archaeology, or previous-attempt review.

