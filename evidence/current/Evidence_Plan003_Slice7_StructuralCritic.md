# Evidence Note: Plan #3 Slice 7 — Structural Critic Ablation (Pass 3.7)

**Slice**: 7 — Structural Critic Ablation  
**Date**: 2026-06-25  
**Model**: openrouter/openai/gpt-5.4-mini  
**Output dir**: output/slice7_v2/

## What was implemented

1. **Pass 3.7: Structural critic** (`pt/pass_critic.py`)  
   - `run_critic()` reads extraction, hypothesis_space, testing, diagnostic_matrix, absence  
   - Returns `CriticResult` (findings + summary + re_elicitation_needed)  
   - `re_elicitation_needed` computed deterministically: any `high` severity finding triggers re-run  
   - Compact absence summary injected into prompt to prevent duplication of Pass 3b findings  

2. **Schema: `CriticFinding` and `CriticResult`** (`pt/schemas.py`)  
   - `CriticFinding.target_type`: "evidence" | "hypothesis" | "causal_edge"  
   - `CriticFinding._validate_target_consistency`: model_validator enforcing type/target coherence  
     - `causal_edge` targets must contain exactly one `->` (multi-hop chains rejected)  
     - `evidence` targets must have no `->`; `hypothesis` targets must have no `->`  
   - `CriticResult.re_elicitation_needed`: computed from findings, not set by LLM  
   - Finding types: confound, missing_pathway, void_link, too_strong_claim, confirmed_link  
   - `confirmed_link` is a positive structural note, not a defect  

3. **Pipeline integration** (`pt/pipeline.py`)  
   - `--critic` CLI flag activates Pass 3.7  
   - `_run_core_passes()` helper: runs Pass 3, 3b, Bayesian, 3.6 without synthesis  
   - Ablation pattern: base core passes → synthesis for snapshot (result_base.json) → critic → optional re-elicitation (second core passes with critic_context injected) → conditional synthesis (reuse or re-run) → result.json  
   - `_compute_critic_delta()`: per-hypothesis posterior shift base→critic  
   - `--critic + --refine` incompatibility guard (raises ValueError with explanation)  
   - Absence passed to `run_critic()` for deduplication  

4. **Artifacts written** (all three):  
   - `result_base.json` — pre-critic posterior snapshot  
   - `result.json` — post-critic result (same as base when no re-elicitation)  
   - `critic_delta.json` — per-hypothesis posterior shifts  

5. **Report: `confirmed_link` visual separation** (`pt/report.py`)  
   - Defect findings (all types except confirmed_link) in main critic table  
   - Confirmed links in separate "Structural Anchors" subsection with green styling  
   - Severity badge count excludes confirmed_link  

6. **Tests** (`tests/test_pass_critic.py`)  
   - `TestCriticFindingTargetValidation` (6 tests): model_validator for all target_type/target combos  
   - `TestCriticPipelineOn` (4 tests): synthesis reuse, double core passes, artifact files, critic stored in result  
   - `TestComputeCriticDeltaFindingCounts` (4 tests)  
   - Total: 171 tests pass  

## Review findings addressed (Tasks 1–9, commit 5c87d1f)

1. Dead variable `ev_targets_by_hyp` removed from `_compute_critic_delta`  
2. Broken anchor links in critic section (`id="hyp-{hid}"`, `id="ev-{ev.id}"`) fixed  
3. Stale `result_critic.json` references removed from CLI help and report  
4. `critic_context` kwarg assertion added to re-elicitation test  
5. Multi-hop chain validator added (`count > 1` rejects `A->B->C` targets)  
6. `--critic + --refine` incompatibility guard added  
7. Absence findings passed to critic prompt (deduplication against Pass 3b)  
8. `confirmed_link` visually separated from defects in report  
9. Same-model critic limitation and `missing_pathway` overloading documented in CLAUDE.md  

## E2E run results

**Command**:
```
PYTHONPATH=. python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir output/slice7_v2 --critic \
  --model openrouter/openai/gpt-5.4-mini --json-only
```

**Duration**: 334.9s  
**Passes**: 2.5 partition, 3 testing (2 validation repairs), 3b absence, 4 base synthesis, 3.7 critic, 4 reused  
**Critic output**: 22 findings (0 high, 10 medium, 12 low), re_elicitation_needed=False  
**Finding types**: 11 void_link (10 medium, 1 low), 11 confirmed_link (all low)  
**Critic delta**: 0/7 hypotheses moved (expected — no re-elicitation)  

**Schema validation**: PASS — `ProcessTracingResult.model_validate()` clean; all 22 findings pass model_validator (no invalid target_type/target combinations)  

**Audit grade**: B (80/100)  
- contract_integrity: 15/15  
- comparative_support_discipline: 15/15  
- temporal_and_causal_proximity: 15/15  
- robustness_and_fragility: 15/15  
- evidence_weighting_and_dependence: 15/15  
- hypothesis_discrimination: 10/10  
- source_scope_and_absence: 10/10  
- report_usability_and_safety: 5/5  
- Conditional caps at 80: proximate evidence sparse (4/56), no decisive diagnostic tests — dataset limitation, not a code bug  

**Artifacts**: output/slice7_v2/result.json, result_base.json, critic_delta.json, report.html
