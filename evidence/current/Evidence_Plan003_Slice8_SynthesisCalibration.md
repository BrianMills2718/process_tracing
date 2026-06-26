# Evidence Note: Plan #3 Slice 8 — Synthesis Calibration And Report Regression

**Slice**: 8 — Synthesis Calibration And Report Regression  
**Date**: 2026-06-25  
**Model**: openrouter/openai/gpt-5.4-mini (E2E re-used from Slice 7 output)  
**Output dir**: output/slice7_v2/ (existing artifact re-audited)

## What was implemented

1. **`_synthesis_overclaim_check(result)` in `scripts/audit_result_quality.py`**  
   - Scans `analytical_narrative` for certainty words when winning hypothesis is `robustness == "fragile"`  
   - Words flagged: `conclusive`, `proven`, `proves`, `definitively`, `beyond doubt`, `with certainty`  
   - `decisive` intentionally excluded (Van Evera diagnostic category term)  
   - Only fires for fragile winners; robust/moderate winners can attract stronger language  

2. **`_verdict_calibration_issues()` extended**  
   - Added: `strongly_supported` with `posterior < 0.50` (STRONG_SUPPORT_FLOOR) → issue  
   - Existing: `supported/strongly_supported` with `posterior < 0.10` unchanged  
   - The `elif` branch avoids double-counting low-posterior cases  

3. **`comparative_support_discipline` dimension updated**  
   - New `overclaim_issues` field in audit output  
   - -3 point deduction when overclaim found in fragile context  
   - Separate recommendation: hedge language with "comparative support suggests" rather than certainty claims  

4. **Report regression tests in `TestReportConsistency`** (2 new):  
   - `test_critic_section_renders_confirmed_links_separately`: confirmed_link in "Structural Anchors" section, void_link in defect table  
   - `test_absence_table_renders_acquire_from_column`: `expected_source_genre` triggers "Acquire from" column  

5. **`TestAuditSynthesisCalibration`** (5 new tests):  
   - Overclaim flagged when winner is fragile + certainty word in narrative  
   - Overclaim NOT flagged when winner is not fragile (robust/moderate)  
   - Overclaim NOT flagged when narrative uses hedged language  
   - `strongly_supported` below `STRONG_SUPPORT_FLOOR` triggers calibration issue  
   - Score deduction appears in `comparative_support_discipline` audit output  

## Live audit results

**E2E command**: Re-audited `output/slice7_v2/result.json` with updated audit script

**Audit output**:
- `score`: 80 (unchanged from Slice 7 — no overclaim in french_revolution synthesis)
- `grade`: B  
- `comparative_support_discipline`: 15/15, `overclaim_issues=[]`, `verdict_issues=[]`
- Winner: h3, posterior=0.466, robustness=fragile  
- Narrative uses appropriate hedging ("the evidence suggests", not "conclusively proves")

**Behavior confirmed**: The LLM-generated french_revolution synthesis did NOT trigger the overclaim check because it used appropriately hedged language despite having a fragile winner. The check correctly fires only when the model produces certainty claims inconsistent with fragile posterior support.

## Tests

Total deterministic tests: 178 pass (171 prior + 7 new)  
- `TestAuditSynthesisCalibration`: 5/5 pass  
- `TestReportConsistency` additions: 2/2 pass  
- `test_output_quality_audit_surfaces_adversarial_caveats`: still passes at score=76, grade=C (stress test narrative uses hedged language, no overclaim deduction)
