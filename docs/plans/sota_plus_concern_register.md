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
| C-001 | mitigated | 0 | The independent critique path depends on the assistant harness that Slice 0 is supposed to build. | Until the harness exists, critique must use a fallback such as Claude/Codex through an existing tool path, a human review, or a deliberately separate adversarial run. | Slice 0 attempted Claude review, but Claude CLI was not authenticated. Fallback adversarial review is documented in `evidence/current/Evidence_Plan003_Slice0_AssistantHarness.md`; future slices should use the assistant harness or authenticated Claude/Codex review when available. |
| C-002 | deferred | 1 | The Brumaire source packet is a strong first case, but one case cannot calibrate PhD-quality thresholds. | A single historical case can validate shape and artifact quality, but benchmark thresholds need multiple cases and planted failures. | Defer threshold calibration to Slice 10; use Slice 1 only to validate source-packet contract and corpus-cap behavior. |
| C-003 | mitigated | 0 | Live E2E assistant tests may require provider access, budget, and local Codex/Claude Code runtime availability. | A gate that cannot run in normal development will become performative. | Deterministic tests validate delegation, metadata, artifact persistence, and dependency boundaries. Live provider smoke is gated behind `PT_RUN_LIVE_AGENT_TESTS=1` and was not required for default `make check`. |
| C-004 | open | 2 | Hypothesis partition quality is partly exploratory: exact MECE thresholds may be hard to specify before observing benchmark failures. | Over-tight rules can reject useful historical explanations; loose rules allow broad absorptive winners. | Slice 2 should define structural blockers deductively and use benchmark readouts for threshold calibration. |
| C-005 | resolved | 1 | The assistant emits a draft source-packet artifact, but the pipeline does not yet consume a source-packet contract. | Without Slice 1, source-packet drafting can still remain adjacent to inference rather than governing it. | Implemented `pt/source_packet.py`, `--source-packet`, `ProcessTracingResult.source_packet`, report source-packet visibility, and audit distinctions for no packet vs packet gaps vs stale synthesis. Verified by source-packet, CLI, pipeline, report, and audit tests. |
| C-006 | open | 4 | Source-packet acceptance does not yet prove that packet sources were acquired or that extracted evidence covers every source group. | A packet can govern research design while the input text still omits sources or underuses key groups; treating packet metadata as evidence would recreate the overclaim problem. | Future source-acquisition/coverage slice must assemble input from packet sources, record per-source provenance, and report which source groups produced evidence. Until then, reports state that packet metadata is not itself evidence. |
