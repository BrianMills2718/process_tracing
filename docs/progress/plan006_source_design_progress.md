# Mission: Plan 006 - Source Design Engine

## Objective
Turn the static source packet into a typed source-design loop that can emit
acquisition actions, record review decisions, and refresh itself from a real
process-tracing result without confusing retrieved candidates with admitted
evidence.

## Acceptance Criteria
- [x] `SourceDesignState` exists as a typed artifact compatible with the current source packet.
- [x] Acquisition planning emits typed action records with explicit status and stop rules.
- [x] Review/disposition updates are durable and do not silently admit retrieval candidates as evidence.
- [x] A real `ProcessTracingResult` can refresh the design state for the next iteration.
- [x] `make check` passes.

## Constraints
- Retrieval outputs remain candidates until reviewed.
- Live non-mocked E2E remains mandatory before closing the slice.
- No silent fallbacks or untyped boundary changes.

## Current Phase
Completed. The typed source-design state is implemented, live-verified, and documented; the next slice is the interactive host/UI review path.

## Completed
- Plan 006 documented with boundary, contract, and notebook artifacts.
- A dedicated `pt/source_design.py` module has been added.
- Acquisition planning now emits typed `AcquisitionAction` records.
- CLI/workbench payloads now include source-design state when a packet is present.
- Round-trip and refresh tests have been added for the source-design loop.
- Live Brumaire retrieval ran successfully and wrote `output/source_acquisition/live_brumaire_design.json`.
- `make check` passes on the updated tree.
