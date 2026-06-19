# Plan #1: Tentative Maintenance Roadmap

**Status:** Planned
**Type:** design
**Priority:** Medium
**Blocked By:** None
**Blocks:** Future cleanup/refactor implementation plans

---

## Gap

**Current:** The active `pt` pipeline, docs, tests, and Makefile are now aligned, but the repo still has several larger cleanup opportunities that should not be done as ad hoc edits.

**Target:** Keep a ranked, reviewable backlog of high-confidence maintenance work with clear decision gates before implementation.

**Why:** The obvious cleanup pass removed stale active guidance. The next layer includes packaging, dead-code policy, historical-doc layout, and report refactoring. Those are worthwhile, but each has enough blast radius to deserve a small plan or explicit go/no-go decision.

---

## References Reviewed

- `README.md` - current setup and verification commands.
- `Makefile` - project interface, `check`, `clean`, and meta-process targets.
- `docs/CLAUDE.md` - documentation routing rules.
- `docs/testing/CLAUDE.md` - historical testing surface warning.
- `docs/validation/CLAUDE.md` - historical validation surface warning.
- `docs/plans/TEMPLATE.md` - local plan format.
- `pt/report.py` - largest active module and likely future refactor target.
- `meta-process.yaml` - dead-code configuration is not enabled.

---

## Files Affected

This design plan does not authorize implementation changes by itself. Expected future implementation files by workstream:

- Packaging: `pyproject.toml` or `requirements.txt`, `README.md`, `Makefile`
- Dead-code policy: `meta-process.yaml`, `requirements.txt`, `.vulture_whitelist.py` if needed
- Historical docs layout: `docs/testing/`, `docs/validation/`, `docs/archive/`, `docs/CLAUDE.md`
- Report refactor: `pt/report.py`, possible new `pt/report_*` helpers, `tests/test_pipeline_integration.py`
- Evaluation/golden tests: `tests/`, `input_text/`, possible `docs/validation/` updates

---

## Plan

### Workstream A: Packaging Metadata

1. Decide whether this repo should be installable as a package or remain script-first.
2. If installable, add `pyproject.toml` with package metadata, Python version floor, runtime deps, and optional dev deps.
3. Keep `llm_client` as an explicit local editable dependency unless it has a stable package source.
4. Update README setup commands and `make check` only after installation is verified in a clean venv.

**Gate:** Do this only after deciding package name, Python version floor, and dependency bounds.

### Workstream B: Dead-Code Detection

1. Add `vulture` as a dev dependency only if the team wants dead-code scans in normal maintenance.
2. Enable `meta_process.quality.dead_code` with narrow paths first, likely `pt/` and `tests/`.
3. Run the scanner, inspect findings manually, and add a whitelist only for intentional public entry points.
4. Keep `make dead-code` advisory until false positives are understood.

**Gate:** Do not make dead-code strict until the first scan has been triaged.

### Workstream C: Historical Docs Quarantine

1. Decide whether `docs/testing/` and `docs/validation/` should remain in place with warnings or move wholesale into `docs/archive/`.
2. If moving, use `git mv` only, preserve paths in commit history, and update `docs/CLAUDE.md`.
3. Keep active verification guidance centered on `tests/` and `make check`.

**Gate:** Move only if historical path stability is less important than reducing stale-doc discoverability.

### Workstream D: Report Renderer Refactor

1. Add snapshot or structural tests around current `generate_report` output before refactoring.
2. Split `pt/report.py` by responsibility: data preparation, HTML rendering, JavaScript asset construction, and formatting helpers.
3. Preserve the output contract: existing `report.html` generation should remain byte-stable or intentionally documented.
4. Run full tests and inspect at least one generated report manually.

**Gate:** No refactor before adding coverage that catches missing sections, broken network data, or malformed HTML.

### Workstream E: Evaluation Coverage

1. Define a small deterministic golden set for extraction and Bayesian behavior.
2. Keep default tests deterministic; live LLM checks stay opt-in via `PT_RUN_LIVE_LLM_TESTS=1`.
3. Add structural assertions for output quality rather than brittle prose matching.
4. Consider a future `make eval` target separate from `make check`.

**Gate:** Needs explicit acceptance criteria for what counts as "better" output.

---

## Required Tests

### New Tests (TDD)

No immediate tests are required for this design plan. Each implementation workstream should add its own focused tests before changing active behavior.

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `make check` | Verifies active tests, type check, LLM compliance, markdown links, and AGENTS sync |
| `make help` | Verifies project interface discoverability |

---

## Acceptance Criteria

- [ ] Plan is saved under `docs/plans/`
- [ ] Plan index references this roadmap
- [ ] Workstreams are ranked and gated
- [ ] No implementation work is implied without a follow-up decision
- [ ] `make check` passes after adding the plan

---

## Notes

Recommended next implementation order:

1. Packaging metadata, if installability matters now.
2. Dead-code detection, if maintainability automation is preferred.
3. Historical docs quarantine, if stale-doc discoverability remains a problem.
4. Report renderer refactor, only after adding report-output tests.
5. Evaluation coverage, once output-quality criteria are explicit.

Avoid combining these workstreams in one implementation commit. They touch different risk surfaces and should remain independently reviewable.
