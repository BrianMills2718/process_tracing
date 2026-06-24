---
status: active
owner: process-tracing
updated: 2026-06-24
---

# Archive And Wiki Policy

This repo keeps active project material in `~/projects/process_tracing`. Retired
material should move out of `~/projects/` to avoid polluting normal agent search.
The canonical archive home is `~/archive/process_tracing`.

## Authority

Active repo files are authoritative for current behavior:

- `CLAUDE.md`
- `docs/ARCHITECTURE.md`
- active docs listed in `docs/CLAUDE.md`
- code in `pt/`
- tests and Make targets

Karpathy/OKF wiki pages are derived orientation layers. If a wiki page conflicts
with active code or docs, the wiki is stale.

## Wiki Layout

Use two OKF-style wiki bundles:

| Bundle | Location | Purpose | Default search policy |
| --- | --- | --- | --- |
| Current repo wiki | `process-tracing-current/` | Compiled orientation for the active repo | Allowed for current-state orientation |
| Archive wiki | `~/archive/process_tracing/` | Compiled history of retired material | Only search for history, recovery, or explicit archive review |

Both bundles follow the OKF convention that the chronicle lives at `wiki/log.md`.
Do not place the log beside `wiki/`.

## Archival Rules

Do not archive by deletion alone. An archive move needs:

1. A reviewed source set.
2. A rationale entry in `~/archive/process_tracing/wiki/log.md`.
3. A source summary page under `~/archive/process_tracing/wiki/sources/`.
4. A disposition: `superseded`, `deprecated`, `deferred`, `mistaken`, or
   `candidate-for-resurrection`.
5. A pointer from active docs only when the archived material is still useful for
   historical recovery.

Archive material is non-authoritative until explicitly promoted back into active
repo docs, plans, code, or tests.

## Agent Search Policy

Agents may read the current repo wiki during orientation, but must verify against
active repo sources before editing behavior or declaring facts about current
implementation.

Agents must not search `~/archive/process_tracing` during normal implementation.
Search it only when the user asks for historical context, recovery, previous
attempts, or archive cleanup.

