# Plan #004: Portfolio Dossier Spine

**Status:** Complete
**Type:** documentation
**Priority:** High
**Blocked By:** None
**Blocks:** Portfolio wiki dossier coverage for `process_tracing`

`trace_evaluable: false # docs-only`

---

## Gap

**Current:** The repo has strong methodology, reviewer, SOTA, and evidence docs,
but no standard dossier spine matching the ecosystem portfolio-wiki policy.

**Target:** The repo has `PROJECT.md`, methodology, artifact, validation,
concern, and ADR surfaces that make the portfolio claim and limits readable
without reconstructing them from chat or scattered plans.

**Why:** This is a core CIA analyst portfolio project. It should be easy for a
reviewer or future agent to see what the project proves, what it does not prove,
and what evidence should be inspected first.

---

## Files Affected

- `PROJECT.md` (create)
- `docs/METHODOLOGY.md` (create)
- `docs/ARTIFACTS.md` (create)
- `docs/VALIDATION.md` (create)
- `docs/CONCERNS.md` (create)
- `docs/adr/0001_process_tracing_methodology_spine.md` (create)
- `docs/wiki_manifest.yaml` (modify)
- `docs/plans/004_portfolio_dossier_spine.md` (create)
- `docs/plans/CLAUDE.md` (modify)

---

## Plan

| Step | What | Status |
|---|---|---|
| 1 | Read existing reviewer, methodology, SOTA, artifact, and concern docs. | Complete |
| 2 | Add dossier spine files with candid portfolio framing and links to canonical sources. | Complete |
| 3 | Add methodology ADR and wiki manifest entries. | Complete |
| 4 | Run focused docs validation. | Complete |
| 5 | Commit only the dossier files, leaving pre-existing active dirt untouched. | Complete |

---

## Acceptance Criteria

- [x] `PROJECT.md` explains goal, portfolio claim, status, artifacts, limits,
  next slices, and concern register.
- [x] `docs/METHODOLOGY.md` includes goals, borrow-vs-build, ADR map, modality
  split, failure modes, and promotion rule.
- [x] `docs/ARTIFACTS.md` lists reviewer-facing artifacts and what claims they
  support.
- [x] `docs/VALIDATION.md` separates engineering validation from missing
  methodological validation.
- [x] `docs/CONCERNS.md` records portfolio-level risks.
- [x] `docs/wiki_manifest.yaml` publishes the dossier surfaces.

---

## Verification Results

- Focused Markdown/link validation was run for the new dossier files and wiki
  manifest.
- YAML parsing was run for `docs/wiki_manifest.yaml`.
- Full `make check` was not run because the working tree had pre-existing
  active implementation dirt outside this docs-only slice.

---

## Notes

Pre-existing dirty files were present before this plan started. This plan does
not physically move, rename, delete, archive, or symlink any repo.
