---
status: active
owner: process-tracing
updated: 2026-06-24
---

# Design-Plan Compliance Audit

This audit checks the active `process_tracing` design artifacts against
`~/projects/.claude/skills/design-plan`.

## Verdict

The active repo satisfies the design-plan requirements for the current
architecture and Plan 003 execution model.

One requirement is enforced prospectively rather than retroactively:
significant future UI surfaces and cross-seam contract changes must include a
synthetic mockup and, for non-trivial contract work, a planning notebook or an
explicit waiver in the slice plan.

## Compliance Matrix

| Requirement | Status | Evidence |
| --- | --- | --- |
| Frame goals and constraints | Satisfied | `docs/ARCHITECTURE.md#Frame`, `docs/PROJECT_THEORY_AND_GOALS.md` |
| Methodology/scope rationale | Satisfied | `docs/adr/0001_process_tracing_methodology_spine.md`, `docs/METHODOLOGY.md` |
| SOTA and borrow-vs-build | Satisfied | `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md`, `docs/ARCHITECTURE.md#Frame` |
| Clean active artifact set | Satisfied | `docs/ARCHIVE_POLICY.md`, `docs/EVIDENCE_POLICY.md`, external archive wiki |
| Modality split | Satisfied | `docs/ARCHITECTURE.md#Modality-Split`, Plan 003 modality diagnosis |
| Boundary diagram | Satisfied and tested | `docs/ARCHITECTURE.md#Boundary-Diagram`, `tests/test_architecture_docs.py` |
| Domain model diagram | Satisfied and tested | `docs/ARCHITECTURE.md#Domain-Model-Diagram`, `tests/test_architecture_docs.py` |
| Data-flow / contract diagram | Satisfied and tested | `docs/ARCHITECTURE.md#Data-Flow-And-Contract-Diagram`, `tests/test_architecture_docs.py` |
| Typed contracts and failure semantics | Satisfied | `docs/ARCHITECTURE.md#Data-Flow-And-Contract-Diagram` contract table |
| Backward runtime pass | Satisfied and tested | `docs/ARCHITECTURE.md#Backward-Runtime-Pass`, `tests/test_architecture_docs.py` |
| Risk-ordered vertical slice roadmap | Satisfied | `docs/plans/003_sota_plus_execution_master_plan.md` |
| Live concern register | Satisfied | `docs/plans/sota_plus_concern_register.md` |
| Audit and cleanup in done-when | Satisfied | Plan 003 universal slice contract |
| Synthetic mockups for UI/significant seams | Forward gate added | Plan 003 universal slice contract; concern C-012 |
| Notebook for non-trivial contract work | Forward gate added | Plan 003 universal slice contract; concern C-012 |

## Notes

The current architecture doc is intentionally a design-plan artifact, not a full
schema dump. The source of truth for executable schema remains the Pydantic
models in `pt/`, especially `pt/schemas.py`, `pt/source_packet.py`, and the
source acquisition/workbench contract surfaces.

The current workbench predates the latest explicit mockup/notebook gate. Future
workbench or significant seam changes must not use that history as precedent:
they need a static mockup and concrete input/output examples before
implementation, or the slice plan must document why the gate is waived.

