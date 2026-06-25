# Session Handoff — Process Tracing Pipeline
**Date**: 2026-06-25 | **Status**: Active | **Branch**: master | **HEAD**: ba9f07b

---

## What this project is

An LLM-first automated process tracing pipeline implementing Stephen Van Evera's causal inference methodology. Given a historical text, it runs: Extract → Hypothesize → Partition → Test (likelihood matrix) → Absence evaluation → Bayesian update → Synthesize → (optionally Refine). Output is `result.json` (structured data) + `report.html` (visual network + audit panels). The goal is PhD-quality causal inference on historical texts without manual coding.

---

## What was done this session

This session continued Plan #3 (SOTA+ Execution Master Plan), delivering Slices 3–6:

| Commit | Slice | What changed |
|--------|-------|-------------|
| `1fb9c19` | Slice 3 | 4 Optional provenance fields on `Evidence`: `source_genre`, `source_group`, `date_confidence`, `trace_production_relevance`. Pass 1 prompt updated. Report shows genre/group badges in Evidence Inventory. 13 tests. |
| `4d2cde4` | Slice 4 | `pt/pass_diagnostic.py` — new deterministic pass deriving `DiagnosticMatrix` from likelihood vectors. Identifies which evidence discriminates each rival hypothesis pair (decisive/strong thresholds). Pipeline wired, report section added (collapsed by default), audit cap-82 added. 18 tests. |
| `32c11a2` | Slice 4b | Cap-82 wired into `audit_result_quality.py` enforcement gate — blocks A-grade when any rival pair lacks discriminators. |
| `7f8d9eb` | Slice 5 | `lineage_type: Optional[LineageType]` on `EvidenceCluster` (duplicate/shared_source/same_event/same_mechanism/other). Pass 3 prompt requests lineage_type. Audit cap-84: fires when top-driver evidence items share a source_group but are not clustered. 8 planted-fixture tests proving duplicate inflation and cluster correction. |
| `34d93b7` | Slice 6 | `expected_source_genre` + `expected_source_location` on `AbsenceEvaluation`. Pass 3b prompt updated with "Source acquisition guidance" section. Report adds "Acquire from" column in absence table. 10 tests. E2E: 11/11 absence findings populated with concrete archive targets. |
| `ba9f07b` | Fix | Pass 3b prompt: clarified `expected_source_genre` is the acquisition target, not the current text's genre. gpt-5-mini was returning "overview" (the current text's own genre) before the fix. |

**Total**: 135 deterministic tests pass. All Slice E2E runs on `openrouter/openai/gpt-5-mini` (Gemini free-tier quota exhausted). All E2E audit grades: B (80/100) — expected for broad Wikipedia overview corpus.

**17 commits are unpushed on master.** Run `git push origin master` when ready.

---

## Active source files changed this session

All committed to git — no ephemeral state.

Key files:
- `pt/schemas.py` — Evidence + AbsenceEvaluation + EvidenceCluster + DiagnosticMatrix schema changes
- `pt/pass_diagnostic.py` — NEW: deterministic diagnostic matrix derivation (no LLM)
- `pt/prompts/pass1_extract.yaml` — provenance metadata instructions
- `pt/prompts/pass3_test.yaml` — lineage_type instructions for dependence clusters
- `pt/prompts/pass3b_absence.yaml` — acquisition guidance instructions
- `pt/pipeline.py` — wired diagnostic matrix into pipeline
- `pt/report.py` — genre badges, diagnostic matrix section, "Acquire from" absence column
- `scripts/audit_result_quality.py` — cap-82 (discriminator gap) + cap-84 (source-lineage inflation)
- `tests/test_pass_diagnostic.py` — NEW: 18 tests
- `tests/test_pt_bayesian.py` — 8 new lineage/inflation tests
- `tests/test_extraction_quality.py` — 10 new absence acquisition field tests

---

## Build and run commands

```bash
# Run pipeline (Gemini quota exhausted — use OpenRouter)
PYTHONPATH=. python -m pt input_text/revolutions/french_revolution.txt \
    --output-dir output/test_run --model openrouter/openai/gpt-5-mini --json-only

# Run deterministic tests (fast, no LLM)
PYTHONPATH=. pytest tests/test_pt_bayesian.py tests/test_pt_schemas.py \
    tests/test_extraction_quality.py tests/test_pass_diagnostic.py -q

# Full test suite (3 failures expected from Gemini quota -- pre-existing, not regressions)
PYTHONPATH=. pytest tests/ -q --tb=short

# Audit a result
PYTHONPATH=. python scripts/audit_result_quality.py output/<run>/result.json \
    --report output/<run>/report.html
```

---

## Uncertainties

### U1 — expected_source_genre quality after prompt fix (MEDIUM)

The Pass 3b prompt was fixed (ba9f07b) to clarify `expected_source_genre` is the acquisition target, not the current text's genre. However, **no E2E was run after the fix**. Before the fix, gpt-5-mini returned `"overview"` on all absence findings. The fix should correct this, but it is unverified.

**To verify**: Run a new E2E and inspect:
```bash
python3 -c "
import json
r = json.load(open('output/<your_run>/result.json'))
for ae in r['absence']['evaluations']:
    print(ae['expected_source_genre'], '--', ae.get('expected_source_location','')[:60])
"
```
Expect: genres like `primary_document`, `parliamentary_record`, `memoir` — NOT `overview`.

### U2 — lineage_type null with gpt-5-mini (LOW)

`lineage_type` on `EvidenceCluster` returned `null` on all clusters despite Optional+description schema. gpt-5-mini ignores Optional fields with null defaults. Accepted as a model-tier limitation — Gemini-2.5-flash likely populates it when quota resets. No code change needed unless lineage_type becomes load-bearing for audit logic.

---

## Pending work

### P1 — Audit Slices 3–6 before Slice 7 (HIGH)

User requested a fresh-session audit. Review five dimensions:
1. **Schema coherence** — do new fields on Evidence/AbsenceEvaluation/EvidenceCluster/DiagnosticMatrix hang together? Are Field(description=) entries sufficient to constrain LLM output at decode time?
2. **Prompt quality** — cross-check field names in pass1_extract, pass3_test, pass3b_absence prompts vs schema. Any field missing from its prompt?
3. **Test coverage gaps** — compare slice-spec acceptance criteria vs tests. Any criterion still at grade C/D?
4. **Audit caps** — caps 80, 82, 84, 88: all exercised by French Revolution E2E? Any unreachable cap condition?
5. **Report rendering** — check report.html visually for new columns/sections. No layout issues?

### P2 — Slice 7: Structural Critic Ablation (HIGH, large)

Spec: `docs/plans/003_sota_plus_execution_master_plan.md` §Slice 7.

New critic pass reads causal extraction + likelihood vectors, flags: confounds, missing pathways, void links, too-strong likelihood claims, confirmed links.

Key constraints:
- **No direct LR mutation** — numeric changes route through Pass 3 re-elicitation only
- **On/off config flag** — `--critic` CLI flag
- **Ablation capture** — `result_base.json` + `result_critic.json` + `critic_delta.json`
- **E2E**: Brumaire with critic off and on, both results preserved and audited

### P3 — Push 17 commits (MEDIUM, trivial)

```bash
git push origin master
```

### P4 — Untracked files cleanup (LOW)

Four untracked files to resolve:
- `docs/plans/005d_visual_audit_mockup.html`, `_v2.html`, `_v3.html` — superseded Plan #5 mockups (safe to gitignore or delete)
- `pt/schemas_view.py` — view payload schemas from Plan #5

Check before deleting: `grep -r "schemas_view" pt/ tests/`

---

## Files that must NOT be edited directly

- `docs/plans/CLAUDE.md` — generated by plan-status sync script
- `AGENTS.md` — generated mirror of CLAUDE.md; update canonical CLAUDE.md instead

---

## Quick sanity checks

```bash
# Deterministic suite: expect 135 passed
PYTHONPATH=. pytest tests/test_pt_bayesian.py tests/test_pt_schemas.py \
    tests/test_extraction_quality.py tests/test_pass_diagnostic.py -q

# New schemas load cleanly
python3 -c "from pt.schemas import AbsenceEvaluation, DiagnosticMatrix, EvidenceCluster; print('ok')"

# New pass loads cleanly
python3 -c "from pt.pass_diagnostic import compute_diagnostic_matrix; print('ok')"

# Check untracked files
git status --short
```
