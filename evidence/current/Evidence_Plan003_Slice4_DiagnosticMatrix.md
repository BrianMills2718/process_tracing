# Plan 003 Slice 4 Evidence: Diagnostic Test Matrix

## Implementation Summary

**Date:** 2026-06-25
**Plan:** 003 SOTA+ Execution Master Plan
**Slice:** 4 — Diagnostic Test Matrix

### What was built

1. **`pt/schemas.py`** — Added `DiscriminatorStrength` Literal and three new model classes:
   - `RivalDiscriminator`: one evidence item that discriminates a specific rival pair
     (evidence_id, log_lr_h1_over_h2, favors, strength, diagnostic_type_h1/h2)
   - `RivalPairDiagnostic`: all discriminators for one rival pair + grade_capped flag
   - `DiagnosticMatrix`: full artifact with all rival pair diagnostics, pairs_without_discriminators,
     grade_cap_applied
   - Added `diagnostic_matrix: Optional[DiagnosticMatrix]` to `ProcessTracingResult`

2. **`pt/pass_diagnostic.py`** (new file) — Pure deterministic math. No LLM call.
   `compute_diagnostic_matrix(testing, hypothesis_space, interpretive_evidence_ids)`:
   - For each rival pair (H_i, H_j), iterates all evidence items
   - Computes effective LR ratio log(LR_i/LR_j) using existing `item_lrs()` from bayesian.py
   - Classifies as "decisive" (|log_lr| >= log(5)) or "strong" (>= log(2))
   - Propagates diagnostic_type_h1/h2 from LLM-assigned values in testing matrix
   - Marks pair as grade_capped if discriminator_count == 0
   - Sets grade_cap_applied if any pair is capped

3. **`pt/pipeline.py`** — Wired after Bayesian update in `_run_passes_3_plus()`.
   Returns `diagnostic_matrix` from function; unpacked by both call sites (initial +
   refined). Added `diagnostic_matrix=diagnostic_matrix` to `ProcessTracingResult(...)`.

4. **`pt/report.py`** — Added "Diagnostic Test Matrix" section (collapsed by default)
   between Bayesian and Evidence Inventory sections:
   - Table of rival pairs with discriminator count
   - Top 5 discriminators per pair (sorted by |log_lr|, strong/decisive badges)
   - Red "NO DISCRIMINATORS — CAPPED" badge on pairs with 0 discriminators
   - Red alert box when any pair is capped

5. **`tests/test_pass_diagnostic.py`** (new file) — 18 deterministic tests:
   - `TestDiagnosticMatrixSchema` (6 tests): schema validation, roundtrip, strength rejection
   - `TestComputeDiagnosticMatrix` (12 tests): discriminator detection, threshold classification,
     3-hypothesis coverage, selective capping, low-relevance exclusion, interpretive cap,
     diagnostic_type propagation, empty testing, single hypothesis, favors field, roundtrip

### Test results

```
232 deterministic tests passed (0 failed)
New diagnostic tests: 18/18
```

### E2E result

Applied to existing Slice 3 result (output/slice3_provenance_e2e_openrouter_20260625/result.json):

**Diagnostic matrix output:**
- 15 rival pairs (5 hypotheses → C(5,2)=10; note: 6 hypotheses in this run → C(6,2)=15)
- Grade cap applied: True — h1 vs h6 has 0 discriminators
- Most discriminated pair: h2 vs h3 (8 discriminators)
- Audit grade: B (80/100) — same pre-existing conditional cap

**Key discriminators found:**
- `evi_coup_18_brumaire_details`: decisive (log_lr=2.46 on h3 vs h4) — the most discriminating
- `evi_reign_terror_arrests_executions`: strong across 4 pairs
- `evi_assignat_worth_8pct_1795`: strong across 3 pairs

**Report:**
- Diagnostic Test Matrix section renders with color-coded strength badges ✅
- Cap alert displays for h1 vs h6 ✅

### Success criteria status

| Criterion | Status |
|---|---|
| Structured RivalDiscriminator/RivalPairDiagnostic/DiagnosticMatrix schemas | ✅ |
| Deterministic derivation (no LLM call) | ✅ |
| Coherent likelihood vectors preserved (not replaced by pairwise LRs) | ✅ |
| Pairs without discriminators flagged + grade_capped | ✅ |
| grade_cap_applied when any pair capped | ✅ |
| diagnostic_type propagated from existing test matrix | ✅ |
| Report: Diagnostic Test Matrix section | ✅ |
| Report: A-level cap alert for undiscriminated pairs | ✅ |
| Deterministic tests: 18/18 | ✅ |
| Full deterministic suite: 232/232 | ✅ |
| E2E: diagnostic_matrix in result.json | ✅ |

### Note on A-level cap

The plan says "an A-level claim is impossible without displayed, source-grounded discriminators."
Slice 4 surfaces this information and flags it (grade_capped, grade_cap_applied). The audit
script's grade is still capped by the pre-existing proximity condition (B=80). Wiring the
diagnostic cap into `audit_result_quality.py` as an explicit grade deduction is deferred to
the cleanup step — the data is now in result.json and available for any downstream consumer.
