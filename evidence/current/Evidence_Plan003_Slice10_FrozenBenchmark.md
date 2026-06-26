# Evidence Note: Plan #3 Slice 10 — Frozen Benchmark Suite And Graduation Gate

**Slice**: 10 — Frozen Benchmark Suite And Graduation Gate
**Date**: 2026-06-25
**Model**: N/A (benchmark uses synthetic fixtures; one real case uses cached result)
**Output dir**: docs/benchmarks/last_scorecard.json

## What was implemented

1. **`docs/benchmarks/benchmark_config.yaml`** — frozen benchmark case definitions:
   - `adversarial_fragile_overclaim` (fixture): 12 weak background items, fragile winner, broad hypothesis — expected grade C, must flag `broad_winning_hypothesis`
   - `adversarial_calibration_mismatch` (fixture): `strongly_supported` label on 0.30-posterior hypothesis — must flag `verdict_calibration_mismatch`
   - `french_revolution_real` (result_file, optional): real E2E output at `output/slice7_v2/result.json` — expected grade B, no overclaim

2. **`scripts/run_benchmark.py`** — benchmark runner:
   - Loads cases from YAML config
   - For `type: fixture`: generates from `_build_adversarial_stress_fixture()` or `_build_calibration_mismatch_fixture()` (planted failure modes, self-contained Python)
   - For `type: result_file`: loads from disk; optional cases skipped if path missing
   - Runs `audit_result()` on each result, checks score range, grade, must_flag, must_not_flag
   - Writes `docs/benchmarks/last_scorecard.json` (machine-readable)
   - Exit 0 if all required cases pass; exit 1 on required failures

3. **`Makefile`**: `make benchmark` target added
   ```bash
   make benchmark  # uses benchmark_config.yaml, writes last_scorecard.json
   ```

4. **`tests/test_benchmark.py`** — 6 deterministic tests:
   - Fixture loading (correct structure and planted mismatch present)
   - All fixture cases pass expectations
   - Optional missing result_file skipped (not failed)
   - Required missing result_file fails (not skipped)
   - Scorecard is valid JSON with required keys

## Live benchmark run results

**Command**: `PYTHONPATH=. python scripts/run_benchmark.py --verbose`

**Output**:
```
Running 3 benchmark case(s)...

Case: adversarial_fragile_overclaim
  [PASS] adversarial_fragile_overclaim: score=76, grade=C, flags=['broad_winning_hypothesis', 'verdict_calibration_mismatch']
  PASS  adversarial_fragile_overclaim: score=76, grade=C

Case: adversarial_calibration_mismatch
  [PASS] adversarial_calibration_mismatch: score=76, grade=C, flags=['verdict_calibration_mismatch']
  PASS  adversarial_calibration_mismatch: score=76, grade=C

Case: french_revolution_real
  [PASS] french_revolution_real: score=82, grade=B, flags=[]
  PASS  french_revolution_real: score=82, grade=B

Benchmark: 3 passed, 0 failed, 0 skipped
```

**Behavior confirmed**:
- Planted failure mode 1 (broad winner): audit correctly flags `broad_winning_hypothesis` and scores C (76/100)
- Planted failure mode 2 (calibration mismatch): audit correctly flags `verdict_calibration_mismatch` 
- Real case (French Revolution, Slice 7 output): grade B (82/100), no overclaim — consistent with Slice 8 evidence note

## Known gap vs. graduation criteria

C-017 (open): The critic pass has not been proven on a frozen case with a planted confound. The benchmark infrastructure now exists, but the specific planted-confound case for the critic needs:
1. A result pre-generated WITH the critic on a text where a confound is known
2. A benchmark case checking that the critic finding categories include `confound`

This gap is registered as C-017 (open). The benchmark config can be extended with a `result_file` case once such a result is generated.

## Tests

Total deterministic tests: 207 pass (201 prior + 6 new benchmark tests)
- `test_benchmark.py`: 6/6 pass
