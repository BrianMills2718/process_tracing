# Issues

Observed problems, concerns, and technical debt. Items start as **unconfirmed**
observations and get triaged through investigation into confirmed issues, plans,
or dismissed.

**Last reviewed:** 2026-06-26

---

## Status Key

| Status | Meaning | Next Step |
|--------|---------|-----------|
| `unconfirmed` | Observed, needs investigation | Investigate to confirm/dismiss |
| `monitoring` | Confirmed concern, watching for signals | Watch for trigger conditions |
| `confirmed` | Real problem, needs a fix | Create a plan |
| `planned` | Has a plan (link to plan) | Implement |
| `resolved` | Fixed | Record resolution |
| `dismissed` | Investigated, not a real problem | Record reasoning |

---

## Unconfirmed

(Add observations here with enough context to investigate later)

### ISSUE-001: (Title)

**Observed:** (date)
**Status:** `unconfirmed`

(What was observed. Why it might be a problem.)

**To investigate:** (What would confirm or dismiss this.)

---

## Monitoring

(Items confirmed as real but not yet urgent. Include trigger conditions.)

---

## Confirmed

(Items that need a fix but don't have a plan yet.)

### ISSUE-002: `make check` uses the wrong test/runtime surface

**Observed:** 2026-06-26  
**Status:** `confirmed`

`make check` invokes bare `pytest`, which resolved to `/home/brian/.local/bin/pytest` in this session. That run failed 11 critic tests with `ModuleNotFoundError: No module named 'tests.test_pipeline_integration'`. The same suite passed under the repo-local interpreter with `PYTHONPATH=. .venv/bin/python -m pytest tests -q --tb=short` (`385 passed, 2 skipped`).

The repo-local virtualenv also lacks `mypy`, while global `mypy pt --ignore-missing-imports` reports six type errors in `pt/report.py`, `pt/pass_diagnostic.py`, `pt/pipeline.py`, and `pt/multi_pipeline.py`.

**Why it matters:** `make check` is the documented agent-facing verification target. It currently mixes interpreter surfaces and can fail or pass depending on user PATH/global packages rather than repo state.

**Fix direction:** Make the target use the repo-local Python consistently, add/ensure `mypy` in the project dependency surface, and fix the reported type errors. Avoid test-to-test imports from `tests.*`; move shared fixtures to `tests/conftest.py` or a dedicated helper module.

---

## Planned

### ISSUE-003: Workbench-safe `pt_export_v1` artifact

**Observed:** 2026-06-26
**Status:** `planned`
**Plan:** `docs/plans/007_workbench_export_v1.md`

`mixed_methods_workbench` needs process-tracing outputs, but must not parse
internal `result.json` or import `pt.schemas`. The repo needs a versioned public
export that preserves source scope, hypotheses, comparative support, absence
findings, verdicts, run metadata, and caveats without exposing raw internals as
the cross-repo contract.

**Why it matters:** Without this seam, the workbench either couples to PT
internals or flattens process-tracing comparative support into a generic
confidence score.

---

## Resolved

| ID | Description | Resolution | Date |
|----|-------------|------------|------|
| - | - | - | - |

---

## Dismissed

| ID | Description | Why Dismissed | Date |
|----|-------------|---------------|------|
| - | - | - | - |

---

## How to Use This File

1. **Observe something off?** Add under Unconfirmed with context and investigation steps
2. **Investigating?** Update the entry with findings, move to appropriate status
3. **Confirmed and needs a fix?** Create a plan, link it, move to Confirmed/Planned
4. **Not actually a problem?** Move to Dismissed with reasoning
5. **Watching a concern?** Move to Monitoring with trigger conditions
