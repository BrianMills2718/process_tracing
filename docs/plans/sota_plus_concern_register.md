# Plan 003 - SOTA+ Concern Register

This is the live concern register for
`docs/plans/003_sota_plus_execution_master_plan.md`.

Use it for uncertainties, tensions, audit findings, recommendations, and
methodology risks that surface during implementation. Do not leave durable
concerns only in chat. Add entries as they arise, then triage the register at
every slice boundary before moving on.

Do not delete entries. Append new concerns as new rows; update status and
disposition in place when an item is triaged.

## Disposition Values

- `open` - not yet addressed.
- `resolved` - fixed and verified.
- `mitigated` - reduced enough to proceed, with remaining risk stated.
- `accepted` - intentionally left as known risk, with rationale.
- `escalated` - blocks progress until re-planned or decided.
- `deferred` - assigned to a named future slice.

## Register

| ID | Status | Slice | Concern | Why it matters | Disposition / next action |
|---|---|---|---|---|---|
| C-001 | open | 0 | The independent critique path depends on the assistant harness that Slice 0 is supposed to build. | Until the harness exists, critique must use a fallback such as Claude/Codex through an existing tool path, a human review, or a deliberately separate adversarial run. | During Slice 0, document the critique source used and update Plan 003 if the final assistant wrapper changes the preferred critique path. |
| C-002 | deferred | 1 | The Brumaire source packet is a strong first case, but one case cannot calibrate PhD-quality thresholds. | A single historical case can validate shape and artifact quality, but benchmark thresholds need multiple cases and planted failures. | Defer threshold calibration to Slice 10; use Slice 1 only to validate source-packet contract and corpus-cap behavior. |
| C-003 | open | 0 | Live E2E assistant tests may require provider access, budget, and local Codex/Claude Code runtime availability. | A gate that cannot run in normal development will become performative. | Slice 0 must separate deterministic contract tests from live smoke tests and define what can be verified without provider access. |
| C-004 | open | 2 | Hypothesis partition quality is partly exploratory: exact MECE thresholds may be hard to specify before observing benchmark failures. | Over-tight rules can reject useful historical explanations; loose rules allow broad absorptive winners. | Slice 2 should define structural blockers deductively and use benchmark readouts for threshold calibration. |
