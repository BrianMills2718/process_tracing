# Plan 003 Slice 2 Evidence: Research Question And Hypothesis Partition Gate

## Implementation Summary

**Date:** 2026-06-25  
**Plan:** 003 SOTA+ Execution Master Plan  
**Slice:** 2 — Research Question And Hypothesis Partition Gate

### What was built

1. **`pt/schemas.py`** — Added `PartitionQuality`, `RivalPairAudit`, `PartitionAudit` schemas;
   added `partition_audit: Optional[PartitionAudit]` to `ProcessTracingResult`.

2. **`pt/pass_partition.py`** — New Pass 2.5. Runs after hypothesis generation, before testing.
   Produces `PartitionAudit` with pairwise overlap/complementary/absorptive flags.
   Sets `cap_applied=True` and emits `UserWarning` when `overall_quality == "needs_review"`.

3. **`pt/prompts/pass_partition.yaml`** — Adversarial prompt: pairwise overlap check, complementary
   self-test, absorptive self-test, discriminator count, adversarial summary.

4. **`pt/pipeline.py`** — Wired `run_partition()` after `run_hypothesize()` (and after human review
   checkpoint). Writes `partition.json` to output_dir. Loads partition_audit from `from_result`
   when passes 1-2 are skipped.

5. **`pt/trace_host.py`** — Added `"partition"` stage between `"hypothesize"` and `"test"` in
   `STAGE_ORDER`, `STAGE_GUIDES`, `_execute_stage`, `_default_inputs_for_stage`, and `_build_result`.

6. **`tests/test_pass_partition.py`** — 21 new deterministic tests covering schema validation,
   cap_applied override, warning emission, single-hypothesis edge case, and `ProcessTracingResult`
   field presence.

### Test results

```
make check equivalent: 251 passed, 2 skipped, 1 failed (transient rate-limit in live LLM test)
New partition tests: 21/21 pass
```

The one failure (`test_source_packet_context_reaches_extraction_pass`) is a pre-existing flaky
live-LLM test that hit the Gemini free-tier daily quota (20 req/day) exhausted by the suite run.
Not caused by Slice 2 changes.

### E2E result

**Command:**
```bash
python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir output/slice2_partition_e2e_openrouter_20260625_094256 \
  --research-question "Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup..." \
  --model openrouter/openai/gpt-5-mini
```

**Run directory:** `output/slice2_partition_e2e_openrouter_20260625_094256`  
**Model:** `openrouter/openai/gpt-5-mini`  
**Audit grade:** B (80/100)  
**Pipeline:** Extract→Hypothesize→Partition→Test→Absence→Bayesian→Synthesize (534.7s)

**Partition stage output (Pass 2.5):**
- 5 hypotheses → 10 rival pairs evaluated
- `overall_quality: needs_review`, 4 hypotheses flagged, `cap_applied: True`
- `partition.json` sidecar written ✅
- `partition_audit` present in `result.json` ✅
- UserWarning emitted with adversarial summary:
  > "The dominant methodological risk is absorptive/complementary overlap: the wartime-centralization
  > logic can plausibly incorporate elite action, civilian scheming, individual ambition, and generals'
  > autonomy, blurring rivalry."

**Audit grade explanation:** B (80/100) — capped at 80 due to only 4/41 evidence items being
proximate to the focal outcome. This is a pre-existing issue (Wikipedia article covers the full
French Revolution, not just the Brumaire decision window), not caused by Slice 2 changes.

### Success criteria status

| Criterion | Status |
|---|---|
| Partition artifact validates (schema tests) | ✅ 21/21 |
| cap_applied set on needs_review | ✅ |
| UserWarning emitted for overlap | ✅ |
| Downstream testing cannot proceed silently | ✅ (warning + cap_applied visible) |
| partition in trace_host STAGE_ORDER | ✅ |
| partition_audit in result.json | ✅ |
| partition.json sidecar written | ✅ |
| make audit-result grade | ✅ B (80/100) |

### Strongest PhD-level concern remaining

The partition audit is advisory: it warns but does not block testing. A researcher can proceed
with a flagged partition (overlapping hypotheses inflate support). This is intentional to avoid
breaking existing runs, but means "downstream testing cannot proceed silently" is met only by
warning visibility, not hard blocking. Future work: add `--strict-partition` flag that blocks on
`needs_review`.

The LLM-driven pairwise audit may miss subtle absorptive risk that requires reading the full
evidence against each hypothesis — the prompt is based only on observable_predictions, which
may not reveal absorptive risk until the testing matrix is available.
